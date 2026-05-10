"""
Phase 4: KR-FinBERT 2단계 캐스케이드 파인튜닝

Stage A: extreme vs normal (이진 분류) — 전체 train/val 사용
Stage B: extreme_down vs extreme_up (이진 분류) — extreme 샘플만

기반 모델: snunlp/KR-FinBert-SC
손실함수: Focal Loss (gamma=2.0, alpha=sqrt(balanced_weight))
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ──────────────────────────────────────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "training_config.yaml"
TRAIN_PATH = ROOT / "data" / "splits" / "train.csv"
VAL_PATH = ROOT / "data" / "splits" / "val.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 라벨 재매핑
# ──────────────────────────────────────────────────────────────────────────────
def remap_for_stage_a(label_z15: int) -> int | None:
    """label_z1.5: 0(extreme_down), 1(normal), 2(extreme_up) → Stage A 이진 라벨"""
    if label_z15 in (0, 2):
        return 1  # extreme
    if label_z15 == 1:
        return 0  # normal
    return None


def remap_for_stage_b(label_z15: int) -> int | None:
    """Stage B: extreme 샘플만 → down(0) vs up(1)"""
    if label_z15 == 0:
        return 0  # extreme_down
    if label_z15 == 2:
        return 1  # extreme_up
    return None  # normal은 Stage B 제외


# ──────────────────────────────────────────────────────────────────────────────
# 설정 로드
# ──────────────────────────────────────────────────────────────────────────────
def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class RegimeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, label_col: str, max_length: int, max_body_len: int):
        self.labels = df[label_col].astype(int).tolist()
        texts = [
            f"{row['title']} [SEP] {str(row['body'])[:max_body_len]}"
            for _, row in df.iterrows()
        ]
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss & FocalTrainer
# ──────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.to(logits.device) if self.alpha is not None else None
        ce = F.cross_entropy(
            logits, labels, weight=alpha,
            label_smoothing=self.label_smoothing, reduction="none"
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class FocalTrainer(Trainer):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0,
                 label_smoothing: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.focal_loss(outputs.logits, labels.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss


# ──────────────────────────────────────────────────────────────────────────────
# 평가 지표
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics_a(eval_pred):
    """Stage A: extreme 검출 지표"""
    logits = eval_pred.predictions
    preds = np.argmax(logits, axis=1)
    labels = eval_pred.label_ids

    precision, recall, _, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1], zero_division=0
    )
    f2_extreme = fbeta_score(labels, preds, beta=2, pos_label=1, zero_division=0)
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 1].numpy()
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0

    return {
        "extreme_recall": float(recall[1]),
        "extreme_precision": float(precision[1]),
        "f2_extreme": float(f2_extreme),
        "recall_normal": float(recall[0]),
        "balanced_accuracy": float((recall[0] + recall[1]) / 2),
        "roc_auc": float(auc),
    }


def compute_metrics_b(eval_pred):
    """Stage B: 방향성 분류 지표"""
    logits = eval_pred.predictions
    preds = np.argmax(logits, axis=1)
    labels = eval_pred.label_ids

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1], zero_division=0
    )

    return {
        "accuracy": float((preds == labels).mean()),
        "recall_down": float(recall[0]),
        "recall_up": float(recall[1]),
        "precision_down": float(precision[0]),
        "precision_up": float(precision[1]),
        "f1_macro": float(f1.mean()),
        "balanced_accuracy": float((recall[0] + recall[1]) / 2),
        "direction_balance": float(abs(recall[0] - recall[1])),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 학습 곡선 시각화
# ──────────────────────────────────────────────────────────────────────────────
def plot_stage_a_curves(log_history: list, output_path: Path):
    epochs, train_losses, val_losses = [], [], []
    val_ext_recall, val_f2, val_bal_acc = [], [], []

    train_buf: dict[float, float] = {}
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_buf[entry.get("epoch", 0)] = entry["loss"]
        if "eval_loss" in entry:
            ep = entry.get("epoch", 0)
            epochs.append(ep)
            val_losses.append(entry["eval_loss"])
            train_losses.append(train_buf.get(ep, float("nan")))
            val_ext_recall.append(entry.get("eval_extreme_recall", float("nan")))
            val_f2.append(entry.get("eval_f2_extreme", float("nan")))
            val_bal_acc.append(entry.get("eval_balanced_accuracy", float("nan")))

    if not epochs:
        print("  [경고] 학습 로그 없음 — Stage A 곡선 생성 건너뜀")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Stage A: Extreme vs Normal", fontsize=13)

    ax = axes[0]
    ax.plot(epochs, train_losses, "o-", label="Train Loss")
    ax.plot(epochs, val_losses, "s--", label="Val Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
    ax.legend(); ax.grid(True)

    ax = axes[1]
    ax.plot(epochs, val_ext_recall, "o-", label="Val Extreme Recall")
    ax.plot(epochs, val_f2, "s--", label="Val F2 Extreme")
    ax.plot(epochs, val_bal_acc, "^:", label="Val Balanced Acc")
    ax.axhline(0.5, color="red", linestyle=":", alpha=0.6, label="기준선 (0.5)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_title("Stage A Metrics")
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Stage A 학습 곡선 저장: {output_path}")


def plot_stage_b_curves(log_history: list, output_path: Path):
    epochs, train_losses, val_losses = [], [], []
    val_recall_down, val_recall_up, val_bal_acc, val_dir_bal = [], [], [], []

    train_buf: dict[float, float] = {}
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_buf[entry.get("epoch", 0)] = entry["loss"]
        if "eval_loss" in entry:
            ep = entry.get("epoch", 0)
            epochs.append(ep)
            val_losses.append(entry["eval_loss"])
            train_losses.append(train_buf.get(ep, float("nan")))
            val_recall_down.append(entry.get("eval_recall_down", float("nan")))
            val_recall_up.append(entry.get("eval_recall_up", float("nan")))
            val_bal_acc.append(entry.get("eval_balanced_accuracy", float("nan")))
            val_dir_bal.append(entry.get("eval_direction_balance", float("nan")))

    if not epochs:
        print("  [경고] 학습 로그 없음 — Stage B 곡선 생성 건너뜀")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Stage B: Extreme Down vs Up", fontsize=13)

    ax = axes[0]
    ax.plot(epochs, train_losses, "o-", label="Train Loss")
    ax.plot(epochs, val_losses, "s--", label="Val Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
    ax.legend(); ax.grid(True)

    ax = axes[1]
    ax.plot(epochs, val_recall_down, "o-", label="Val Recall Down")
    ax.plot(epochs, val_recall_up, "s--", label="Val Recall Up")
    ax.plot(epochs, val_bal_acc, "^:", label="Val Balanced Acc")
    ax.plot(epochs, val_dir_bal, "D-.", label="Direction Balance (↓좋음)", alpha=0.6)
    ax.axhline(0.55, color="red", linestyle=":", alpha=0.6, label="Balanced Acc 기준 (0.55)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_title("Stage B Metrics")
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Stage B 학습 곡선 저장: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Confusion Matrix 출력
# ──────────────────────────────────────────────────────────────────────────────
def print_cm_2x2(cm: np.ndarray, label_names: list[str]):
    header = f"{'':>18}" + "".join(f"pred_{n:>12}" for n in label_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"  true_{label_names[i]:>12}" + "".join(f"{v:>18}" for v in row))


def print_cm_3x3(cm: np.ndarray):
    names = ["extreme_down", "normal", "extreme_up"]
    header = f"{'':>16}" + "".join(f"{n:>14}" for n in names)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {names[i]:>14}" + "".join(f"{v:>14}" for v in row))


# ──────────────────────────────────────────────────────────────────────────────
# Stage A 학습
# ──────────────────────────────────────────────────────────────────────────────
def train_stage_a(cfg: dict, train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[dict, object, object]:
    """Stage A: extreme vs normal 이진 분류기 학습"""
    scfg = cfg["stage_a"]
    icfg = cfg["input"]
    fl_cfg = cfg["focal_loss"]

    set_seed(scfg["seed"])

    print(f"\n{'='*60}")
    print("  Stage A: Extreme vs Normal 이진 분류")
    print(f"{'='*60}")

    # 라벨 재매핑
    train_a = train_df.copy()
    val_a = val_df.copy()
    train_a["label_a"] = train_a["label_z1.5"].apply(remap_for_stage_a)
    val_a["label_a"] = val_a["label_z1.5"].apply(remap_for_stage_a)
    train_a = train_a.dropna(subset=["label_a"])
    val_a = val_a.dropna(subset=["label_a"])

    # 클래스 분포 출력
    print("\n[Stage A 클래스 분포]")
    for split_name, df_ in [("Train", train_a), ("Val", val_a)]:
        counts = df_["label_a"].value_counts().sort_index()
        print(f"  {split_name}: normal={counts.get(0, 0):,}  extreme={counts.get(1, 0):,}  "
              f"(total={len(df_):,})")

    # 클래스 가중치 (sqrt 완화)
    y_train = train_a["label_a"].astype(int).values
    raw_w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    alpha_a = torch.tensor(np.sqrt(raw_w), dtype=torch.float)
    print(f"\n[Stage A alpha (sqrt_balanced)]  normal={alpha_a[0]:.4f}  extreme={alpha_a[1]:.4f}")

    # 예상 학습 시간
    steps_per_epoch = len(train_a) // scfg["per_device_train_batch_size"]
    print(f"[예상] {steps_per_epoch:,} steps/epoch × {scfg['num_train_epochs']} epochs "
          f"→ 약 {scfg['num_train_epochs'] * 2}~{scfg['num_train_epochs'] * 3}분")

    # 모델 & 토크나이저
    print(f"\n[Stage A 모델 로드] {scfg['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(scfg["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        scfg["base_model"],
        num_labels=2,
        id2label={0: "normal", 1: "extreme"},
        label2id={"normal": 0, "extreme": 1},
        ignore_mismatched_sizes=True,
    )

    # 데이터셋
    print("[데이터셋 토크나이징 중...]")
    train_ds = RegimeDataset(train_a, tokenizer, "label_a", scfg["max_length"], icfg["max_body_len"])
    val_ds = RegimeDataset(val_a, tokenizer, "label_a", scfg["max_length"], icfg["max_body_len"])

    # TrainingArguments
    output_dir = str(ROOT / scfg["output_dir"].lstrip("./"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=scfg["num_train_epochs"],
        per_device_train_batch_size=scfg["per_device_train_batch_size"],
        per_device_eval_batch_size=scfg["per_device_eval_batch_size"],
        learning_rate=scfg["learning_rate"],
        warmup_ratio=scfg["warmup_ratio"],
        weight_decay=scfg["weight_decay"],
        fp16=scfg["fp16"],
        label_smoothing_factor=0.0,  # FocalTrainer.focal_loss에서 직접 처리
        eval_strategy=scfg["eval_strategy"],
        save_strategy=scfg["save_strategy"],
        load_best_model_at_end=scfg["load_best_model_at_end"],
        metric_for_best_model=scfg["metric_for_best_model"],
        greater_is_better=scfg["greater_is_better"],
        logging_dir=str(ROOT / "logs" / "stage_a"),
        logging_steps=50,
        save_total_limit=cfg["output"]["save_total_limit"],
        seed=scfg["seed"],
        report_to="none",
    )

    trainer = FocalTrainer(
        alpha=alpha_a,
        gamma=fl_cfg["gamma"],
        label_smoothing=scfg["label_smoothing_factor"],
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_a,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=scfg["early_stopping_patience"])],
    )

    print("\n[Stage A 학습 시작]")
    train_result = trainer.train()
    runtime_min = train_result.metrics.get("train_runtime", 0) / 60
    print(f"\n[Stage A 완료] 학습 시간: {runtime_min:.1f}분")

    # Val 최종 평가
    print("\n[Stage A Validation 평가]")
    val_metrics = trainer.evaluate()
    print(f"  extreme_recall    : {val_metrics['eval_extreme_recall']:.4f}")
    print(f"  extreme_precision : {val_metrics['eval_extreme_precision']:.4f}")
    print(f"  f2_extreme        : {val_metrics['eval_f2_extreme']:.4f}")
    print(f"  recall_normal     : {val_metrics['eval_recall_normal']:.4f}")
    print(f"  balanced_accuracy : {val_metrics['eval_balanced_accuracy']:.4f}")
    print(f"  roc_auc           : {val_metrics['eval_roc_auc']:.4f}")

    # Confusion Matrix
    print("\n[Stage A Confusion Matrix — Val]")
    preds_out = trainer.predict(val_ds)
    val_preds = np.argmax(preds_out.predictions, axis=1)
    val_labels = preds_out.label_ids
    cm_a = confusion_matrix(val_labels, val_preds, labels=[0, 1])
    print_cm_2x2(cm_a, ["normal", "extreme"])

    # Train-Val gap
    train_preds_out = trainer.predict(train_ds)
    tr_preds = np.argmax(train_preds_out.predictions, axis=1)
    tr_labels = train_preds_out.label_ids
    tr_rec_ext = (tr_preds[tr_labels == 1] == 1).mean() if (tr_labels == 1).sum() > 0 else 0.0
    val_rec_ext = val_metrics["eval_extreme_recall"]
    gap = tr_rec_ext - val_rec_ext
    print(f"\n[Train-Val Gap]  Train extreme_recall={tr_rec_ext:.4f}  Val={val_rec_ext:.4f}  "
          f"Gap={gap:.4f}  {'⚠ 과적합 의심' if gap > 0.15 else '정상'}")

    # 모델 저장
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n[Stage A 모델 저장] {output_dir}")

    # 학습 곡선
    curves_path = ROOT / cfg["output"]["curves_stage_a"].lstrip("./")
    plot_stage_a_curves(trainer.state.log_history, curves_path)

    # best epoch 정보
    best_epoch = getattr(trainer.state, "best_model_checkpoint", "N/A")

    metrics_a = {
        "runtime_min": runtime_min,
        "best_checkpoint": best_epoch,
        "train_extreme_recall": float(tr_rec_ext),
        "val_extreme_recall": float(val_metrics["eval_extreme_recall"]),
        "val_extreme_precision": float(val_metrics["eval_extreme_precision"]),
        "val_f2_extreme": float(val_metrics["eval_f2_extreme"]),
        "val_recall_normal": float(val_metrics["eval_recall_normal"]),
        "val_balanced_accuracy": float(val_metrics["eval_balanced_accuracy"]),
        "val_roc_auc": float(val_metrics["eval_roc_auc"]),
        "train_val_gap": float(gap),
        "alpha": alpha_a.tolist(),
        "confusion_matrix": cm_a.tolist(),
    }

    return metrics_a, model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Stage B 학습
# ──────────────────────────────────────────────────────────────────────────────
def train_stage_b(cfg: dict, train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[dict, object, object]:
    """Stage B: extreme_down vs extreme_up 이진 분류기 (extreme 샘플만)"""
    scfg = cfg["stage_b"]
    icfg = cfg["input"]
    fl_cfg = cfg["focal_loss"]

    set_seed(scfg["seed"])

    print(f"\n{'='*60}")
    print("  Stage B: Extreme Down vs Up 이진 분류 (extreme 샘플만)")
    print(f"{'='*60}")

    # extreme 샘플 필터링 + 라벨 재매핑
    train_b = train_df[train_df["label_z1.5"].isin([0, 2])].copy()
    val_b = val_df[val_df["label_z1.5"].isin([0, 2])].copy()
    train_b["label_b"] = train_b["label_z1.5"].apply(remap_for_stage_b)
    val_b["label_b"] = val_b["label_z1.5"].apply(remap_for_stage_b)
    train_b = train_b.dropna(subset=["label_b"])
    val_b = val_b.dropna(subset=["label_b"])

    # 클래스 분포 출력
    print("\n[Stage B 클래스 분포]")
    for split_name, df_ in [("Train", train_b), ("Val", val_b)]:
        counts = df_["label_b"].value_counts().sort_index()
        print(f"  {split_name}: extreme_down={counts.get(0, 0):,}  extreme_up={counts.get(1, 0):,}  "
              f"(total={len(df_):,})")

    # 클래스 가중치 (sqrt 완화)
    y_train = train_b["label_b"].astype(int).values
    raw_w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    alpha_b = torch.tensor(np.sqrt(raw_w), dtype=torch.float)
    print(f"\n[Stage B alpha (sqrt_balanced)]  down={alpha_b[0]:.4f}  up={alpha_b[1]:.4f}")

    steps_per_epoch = len(train_b) // scfg["per_device_train_batch_size"]
    print(f"[예상] {steps_per_epoch:,} steps/epoch × {scfg['num_train_epochs']} epochs "
          f"→ 약 {scfg['num_train_epochs']}~{scfg['num_train_epochs'] * 2}분")

    # 모델 & 토크나이저 (KR-FinBert-SC에서 fresh start)
    print(f"\n[Stage B 모델 로드] {scfg['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(scfg["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        scfg["base_model"],
        num_labels=2,
        id2label={0: "extreme_down", 1: "extreme_up"},
        label2id={"extreme_down": 0, "extreme_up": 1},
        ignore_mismatched_sizes=True,
    )

    print("[데이터셋 토크나이징 중...]")
    train_ds = RegimeDataset(train_b, tokenizer, "label_b", scfg["max_length"], icfg["max_body_len"])
    val_ds = RegimeDataset(val_b, tokenizer, "label_b", scfg["max_length"], icfg["max_body_len"])

    output_dir = str(ROOT / scfg["output_dir"].lstrip("./"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=scfg["num_train_epochs"],
        per_device_train_batch_size=scfg["per_device_train_batch_size"],
        per_device_eval_batch_size=scfg["per_device_eval_batch_size"],
        learning_rate=scfg["learning_rate"],
        warmup_ratio=scfg["warmup_ratio"],
        weight_decay=scfg["weight_decay"],
        fp16=scfg["fp16"],
        label_smoothing_factor=0.0,  # FocalTrainer.focal_loss에서 직접 처리
        eval_strategy=scfg["eval_strategy"],
        save_strategy=scfg["save_strategy"],
        load_best_model_at_end=scfg["load_best_model_at_end"],
        metric_for_best_model=scfg["metric_for_best_model"],
        greater_is_better=scfg["greater_is_better"],
        logging_dir=str(ROOT / "logs" / "stage_b"),
        logging_steps=50,
        save_total_limit=cfg["output"]["save_total_limit"],
        seed=scfg["seed"],
        report_to="none",
    )

    trainer = FocalTrainer(
        alpha=alpha_b,
        gamma=fl_cfg["gamma"],
        label_smoothing=scfg["label_smoothing_factor"],
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_b,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=scfg["early_stopping_patience"])],
    )

    print("\n[Stage B 학습 시작]")
    train_result = trainer.train()
    runtime_min = train_result.metrics.get("train_runtime", 0) / 60
    print(f"\n[Stage B 완료] 학습 시간: {runtime_min:.1f}분")

    # Val 최종 평가
    print("\n[Stage B Validation 평가]")
    val_metrics = trainer.evaluate()
    print(f"  recall_down       : {val_metrics['eval_recall_down']:.4f}")
    print(f"  recall_up         : {val_metrics['eval_recall_up']:.4f}")
    print(f"  precision_down    : {val_metrics['eval_precision_down']:.4f}")
    print(f"  precision_up      : {val_metrics['eval_precision_up']:.4f}")
    print(f"  f1_macro          : {val_metrics['eval_f1_macro']:.4f}")
    print(f"  balanced_accuracy : {val_metrics['eval_balanced_accuracy']:.4f}")
    print(f"  direction_balance : {val_metrics['eval_direction_balance']:.4f}")

    # Confusion Matrix
    print("\n[Stage B Confusion Matrix — Val]")
    preds_out = trainer.predict(val_ds)
    val_preds = np.argmax(preds_out.predictions, axis=1)
    val_labels = preds_out.label_ids
    cm_b = confusion_matrix(val_labels, val_preds, labels=[0, 1])
    print_cm_2x2(cm_b, ["extreme_down", "extreme_up"])

    # Train-Val balanced_accuracy gap
    train_preds_out = trainer.predict(train_ds)
    tr_preds = np.argmax(train_preds_out.predictions, axis=1)
    tr_labels = train_preds_out.label_ids
    tr_rec_down = (tr_preds[tr_labels == 0] == 0).mean() if (tr_labels == 0).sum() > 0 else 0.0
    tr_rec_up = (tr_preds[tr_labels == 1] == 1).mean() if (tr_labels == 1).sum() > 0 else 0.0
    tr_bal_acc = (tr_rec_down + tr_rec_up) / 2
    val_bal_acc = val_metrics["eval_balanced_accuracy"]
    gap = tr_bal_acc - val_bal_acc
    print(f"\n[Train-Val Gap]  Train balanced_acc={tr_bal_acc:.4f}  Val={val_bal_acc:.4f}  "
          f"Gap={gap:.4f}  {'⚠ 과적합 의심' if gap > 0.15 else '정상'}")

    # 모델 저장
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n[Stage B 모델 저장] {output_dir}")

    curves_path = ROOT / cfg["output"]["curves_stage_b"].lstrip("./")
    plot_stage_b_curves(trainer.state.log_history, curves_path)

    best_epoch = getattr(trainer.state, "best_model_checkpoint", "N/A")

    metrics_b = {
        "runtime_min": runtime_min,
        "best_checkpoint": best_epoch,
        "train_balanced_accuracy": float(tr_bal_acc),
        "val_recall_down": float(val_metrics["eval_recall_down"]),
        "val_recall_up": float(val_metrics["eval_recall_up"]),
        "val_precision_down": float(val_metrics["eval_precision_down"]),
        "val_precision_up": float(val_metrics["eval_precision_up"]),
        "val_f1_macro": float(val_metrics["eval_f1_macro"]),
        "val_balanced_accuracy": float(val_metrics["eval_balanced_accuracy"]),
        "val_direction_balance": float(val_metrics["eval_direction_balance"]),
        "train_val_gap": float(gap),
        "alpha": alpha_b.tolist(),
        "confusion_matrix": cm_b.tolist(),
    }

    return metrics_b, model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# 캐스케이드 통합 평가
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_proba(model, tokenizer, texts: list[str], max_length: int, batch_size: int = 64) -> np.ndarray:
    """주어진 텍스트 리스트에 대해 softmax 확률 반환 (shape: [N, num_labels])"""
    model.eval()
    device = next(model.parameters()).device
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[i: i + batch_size],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)
        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def cascade_evaluate(
    val_df: pd.DataFrame,
    model_a, tokenizer_a,
    model_b, tokenizer_b,
    cfg: dict,
    threshold: float = 0.5,
) -> dict:
    """캐스케이드 통합 평가: Stage A → (조건부) Stage B → 3-class 라벨"""
    icfg = cfg["input"]
    max_length = cfg["stage_a"]["max_length"]

    texts = [
        f"{row['title']} [SEP] {str(row['body'])[:icfg['max_body_len']]}"
        for _, row in val_df.iterrows()
    ]
    true_labels = val_df["label_z1.5"].astype(int).values

    print("\n[캐스케이드 통합 평가 중...]")

    # Stage A 추론
    if torch.cuda.is_available():
        model_a.cuda(); model_b.cuda()
    probs_a = predict_proba(model_a, tokenizer_a, texts, max_length)  # [N, 2]
    probs_b = predict_proba(model_b, tokenizer_b, texts, max_length)  # [N, 2]

    # 캐스케이드 최종 라벨 결정
    final_preds = []
    for i in range(len(texts)):
        p_extreme = probs_a[i, 1]
        if p_extreme < threshold:
            final_preds.append(1)  # normal
        else:
            p_up = probs_b[i, 1]
            p_down = probs_b[i, 0]
            final_preds.append(2 if p_up > p_down else 0)

    final_preds = np.array(final_preds)

    # 3x3 Confusion Matrix
    cm3 = confusion_matrix(true_labels, final_preds, labels=[0, 1, 2])
    print("\n[캐스케이드 3x3 Confusion Matrix — Val]")
    print_cm_3x3(cm3)

    # 클래스별 recall
    recall_down = cm3[0, 0] / cm3[0].sum() if cm3[0].sum() > 0 else 0.0
    recall_normal = cm3[1, 1] / cm3[1].sum() if cm3[1].sum() > 0 else 0.0
    recall_up = cm3[2, 2] / cm3[2].sum() if cm3[2].sum() > 0 else 0.0
    extreme_recall = (recall_down + recall_up) / 2
    min_extreme_recall = min(recall_down, recall_up)

    print(f"\n[캐스케이드 지표]")
    print(f"  recall_down   (cl0): {recall_down:.4f}")
    print(f"  recall_normal (cl1): {recall_normal:.4f}")
    print(f"  recall_up     (cl2): {recall_up:.4f}")
    print(f"  extreme_recall     : {extreme_recall:.4f}")
    print(f"  min_extreme_recall : {min_extreme_recall:.4f}  ← 핵심 지표")

    # 방향성 혼동 (cl0 ↔ cl2)
    cross_confusion = cm3[0, 2] + cm3[2, 0]
    total_extreme = int(cm3[0].sum() + cm3[2].sum())
    print(f"  cl0↔cl2 혼동 비율  : {cross_confusion}/{total_extreme} "
          f"({cross_confusion / max(total_extreme, 1) * 100:.1f}%)")

    return {
        "threshold": threshold,
        "cascade_recall_down": float(recall_down),
        "cascade_recall_normal": float(recall_normal),
        "cascade_recall_up": float(recall_up),
        "cascade_extreme_recall": float(extreme_recall),
        "cascade_min_extreme_recall": float(min_extreme_recall),
        "cross_confusion_count": int(cross_confusion),
        "cross_confusion_pct": float(cross_confusion / max(total_extreme, 1)),
        "confusion_matrix_3x3": cm3.tolist(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 성공/실패 판정
# ──────────────────────────────────────────────────────────────────────────────
def verdict(metrics_a: dict, metrics_b: dict, cascade: dict):
    print(f"\n{'='*60}")
    print("  Phase 4 결과 판정")
    print(f"{'='*60}")

    a_ext_recall = metrics_a["val_extreme_recall"]
    a_bal_acc = metrics_a["val_balanced_accuracy"]
    b_bal_acc = metrics_b["val_balanced_accuracy"]
    b_dir_bal = metrics_b["val_direction_balance"]
    c_min_ext = cascade["cascade_min_extreme_recall"]

    a_pass = a_ext_recall >= 0.5 and a_bal_acc >= 0.6
    b_pass = b_bal_acc >= 0.55 and b_dir_bal <= 0.2
    c_pass = c_min_ext >= 0.3

    print(f"\n  Stage A: extreme_recall={a_ext_recall:.4f}(≥0.5?) "
          f"balanced_acc={a_bal_acc:.4f}(≥0.6?) → {'✅ 통과' if a_pass else '❌ 미달'}")
    print(f"  Stage B: balanced_acc={b_bal_acc:.4f}(≥0.55?) "
          f"direction_balance={b_dir_bal:.4f}(≤0.2?) → {'✅ 통과' if b_pass else '❌ 미달'}")
    print(f"  Cascade: min_extreme_recall={c_min_ext:.4f}(≥0.3?) → {'✅ 통과' if c_pass else '❌ 미달'}")

    print()
    if a_pass and b_pass and c_pass:
        print("  ✅ 성공 — Phase 5 진행 가능")
    elif a_pass and not b_pass:
        print("  ⚠ 부분 성공 — Stage B balanced_accuracy 미달")
        print("    → Stage B 데이터 부족 가능성: z=1.0으로 extreme 확장 검토")
    elif not a_pass:
        print("  ❌ 실패 — Stage A 자체 미달. Focal Loss gamma 조정 또는 근본 재검토 필요")
    else:
        print("  ⚠ 부분 성공 — 캐스케이드 min_extreme_recall 미달")
        print("    → Stage A threshold 조정 또는 Stage B 데이터 확장 검토")

    # 이전 3-class 모델 비교
    archive = ROOT / "results" / "archive_3class_training_metrics.json"
    if archive.exists():
        with open(archive, "r", encoding="utf-8") as f:
            old = json.load(f)
        old_key = "label_z1_5"
        if old_key in old:
            old_ext_rec = old[old_key].get("val_extreme_recall", "N/A")
            print(f"\n  [이전 단일 3-class 모델 대비]")
            print(f"    이전 val_extreme_recall : {old_ext_rec}")
            print(f"    현재 cascade_extreme_recall: {cascade['cascade_extreme_recall']:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    cfg = load_config(CONFIG_PATH)

    print("=" * 60)
    print("  Phase 4: KR-FinBERT 2단계 캐스케이드 파인튜닝")
    print(f"  GPU: {'CUDA ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (경고: 매우 느림)'}")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    print(f"\n데이터 로드: train={len(train_df):,}건  val={len(val_df):,}건")

    # ── Stage A 학습
    metrics_a, model_a, tokenizer_a = train_stage_a(cfg, train_df, val_df)

    # Stage A 메트릭 저장
    metrics_a_path = ROOT / cfg["output"]["metrics_stage_a"].lstrip("./")
    with open(metrics_a_path, "w", encoding="utf-8") as f:
        json.dump(metrics_a, f, ensure_ascii=False, indent=2)
    print(f"\n[Stage A 메트릭 저장] {metrics_a_path}")

    # ── Stage B 학습
    metrics_b, model_b, tokenizer_b = train_stage_b(cfg, train_df, val_df)

    # Stage B 메트릭 저장
    metrics_b_path = ROOT / cfg["output"]["metrics_stage_b"].lstrip("./")
    with open(metrics_b_path, "w", encoding="utf-8") as f:
        json.dump(metrics_b, f, ensure_ascii=False, indent=2)
    print(f"\n[Stage B 메트릭 저장] {metrics_b_path}")

    # ── 캐스케이드 통합 평가
    cascade_metrics = cascade_evaluate(
        val_df, model_a, tokenizer_a, model_b, tokenizer_b, cfg, threshold=0.5
    )

    cascade_path = ROOT / cfg["output"]["cascade_eval"].lstrip("./")
    with open(cascade_path, "w", encoding="utf-8") as f:
        json.dump({"stage_a": metrics_a, "stage_b": metrics_b, "cascade": cascade_metrics},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[캐스케이드 평가 저장] {cascade_path}")

    # ── 성공/실패 판정 및 권장사항
    verdict(metrics_a, metrics_b, cascade_metrics)

    print("\n[Phase 4 캐스케이드 학습 완료]")


if __name__ == "__main__":
    main()
