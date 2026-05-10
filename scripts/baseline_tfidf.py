"""
TF-IDF Baseline 진단 스크립트

목적: BERT 학습 실패가 모델 문제인지 데이터 신호 부족인지 진단

Task A: extreme vs normal (Stage A 동일 설정)
Task B: extreme_down vs extreme_up (Stage B 동일 설정, extreme 샘플만)
Task C: 3-class (참고용)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "splits" / "train.csv"
VAL_PATH   = ROOT / "data" / "splits" / "val.csv"

MAX_BODY_LEN = 400


def make_text(row) -> str:
    return f"{row['title']} {str(row['body'])[:MAX_BODY_LEN]}"


def top_words(vectorizer, clf, class_idx: int, n: int = 20) -> list[tuple[str, float]]:
    feature_names = vectorizer.get_feature_names_out()
    if hasattr(clf, "coef_"):
        coef = clf.coef_[class_idx] if clf.coef_.ndim > 1 else clf.coef_[0]
        top_idx = np.argsort(coef)[-n:][::-1]
        return [(feature_names[i], float(coef[i])) for i in top_idx]
    return []


def bottom_words(vectorizer, clf, class_idx: int, n: int = 20) -> list[tuple[str, float]]:
    feature_names = vectorizer.get_feature_names_out()
    if hasattr(clf, "coef_"):
        coef = clf.coef_[class_idx] if clf.coef_.ndim > 1 else clf.coef_[0]
        bot_idx = np.argsort(coef)[:n]
        return [(feature_names[i], float(coef[i])) for i in bot_idx]
    return []


def run_task(name: str, X_train, y_train, X_val, y_val,
             vectorizer, label_names: list[str], binary: bool = True):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # 클래스 분포
    print("\n[클래스 분포]")
    for split_name, y in [("Train", y_train), ("Val", y_val)]:
        unique, counts = np.unique(y, return_counts=True)
        dist = "  ".join(f"{label_names[int(u)]}={c}" for u, c in zip(unique, counts))
        print(f"  {split_name}: {dist}  (total={len(y)})")

    # 학습
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    # 기본 지표
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    print(f"\n[Val 지표]")
    print(f"  balanced_accuracy : {bal_acc:.4f}")

    # ROC-AUC (이진 task만)
    auc = None
    if binary and len(np.unique(y_val)) == 2:
        proba = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        print(f"  roc_auc           : {auc:.4f}")

    # Confusion matrix
    print(f"\n[Confusion Matrix — Val]")
    labels = list(range(len(label_names)))
    cm = confusion_matrix(y_val, y_pred, labels=labels)
    header = f"{'':>16}" + "".join(f"pred_{label_names[i]:>12}" for i in labels)
    print(header)
    for i, row in enumerate(cm):
        print(f"  true_{label_names[i]:>10}" + "".join(f"{v:>18}" for v in row))

    # Classification report
    print(f"\n[Classification Report]")
    print(classification_report(y_val, y_pred, target_names=label_names, zero_division=0))

    # 영향력 큰 단어 Top 20
    if binary:
        pos_words = top_words(vectorizer, clf, 0, 20)
        neg_words = bottom_words(vectorizer, clf, 0, 20)
        pos_label = label_names[1]
        neg_label = label_names[0]
        print(f"[{pos_label} 예측에 강한 단어 Top 20]")
        for w, c in pos_words:
            print(f"  {w:>20}  coef={c:+.4f}")
        print(f"\n[{neg_label} 예측에 강한 단어 Top 20]")
        for w, c in neg_words:
            print(f"  {w:>20}  coef={c:+.4f}")
    else:
        # 3-class: 각 클래스별 top 10
        for ci, cname in enumerate(label_names):
            top = top_words(vectorizer, clf, ci, 10)
            print(f"\n[{cname} 클래스 Top 10 단어]")
            for w, c in top:
                print(f"  {w:>20}  coef={c:+.4f}")

    return bal_acc, auc


def main():
    print("=" * 60)
    print("  TF-IDF Baseline 진단")
    print("  목적: 데이터 신호 존재 여부 확인")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    print(f"\n데이터 로드: train={len(train_df):,}건  val={len(val_df):,}건")

    # 입력 텍스트 생성
    train_texts = train_df.apply(make_text, axis=1).tolist()
    val_texts   = val_df.apply(make_text, axis=1).tolist()

    # TF-IDF 벡터화 (train fit, val transform)
    print("\n[TF-IDF 벡터화 중...]")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val   = vectorizer.transform(val_texts)
    print(f"  vocab 크기: {len(vectorizer.vocabulary_):,}")

    results = {}

    # ── Task A: extreme vs normal
    def remap_a(x):
        if x in (0.0, 2.0): return 1  # extreme
        if x == 1.0: return 0          # normal
        return None

    y_train_a = train_df["label_z1.5"].apply(remap_a).astype(int)
    y_val_a   = val_df["label_z1.5"].apply(remap_a).astype(int)

    bal_a, auc_a = run_task(
        "Task A: Extreme vs Normal (Stage A 동일)",
        X_train, y_train_a, X_val, y_val_a,
        vectorizer,
        label_names=["normal", "extreme"],
        binary=True,
    )
    results["task_a"] = {"balanced_accuracy": bal_a, "roc_auc": auc_a}

    # ── Task B: extreme_down vs extreme_up (extreme 샘플만)
    train_b = train_df[train_df["label_z1.5"].isin([0.0, 2.0])].copy()
    val_b   = val_df[val_df["label_z1.5"].isin([0.0, 2.0])].copy()

    train_texts_b = train_b.apply(make_text, axis=1).tolist()
    val_texts_b   = val_b.apply(make_text, axis=1).tolist()
    X_train_b = vectorizer.transform(train_texts_b)
    X_val_b   = vectorizer.transform(val_texts_b)

    y_train_b = train_b["label_z1.5"].apply(lambda x: 0 if x == 0.0 else 1).astype(int)
    y_val_b   = val_b["label_z1.5"].apply(lambda x: 0 if x == 0.0 else 1).astype(int)

    bal_b, auc_b = run_task(
        "Task B: Extreme Down vs Up (Stage B 동일, extreme 샘플만)",
        X_train_b, y_train_b, X_val_b, y_val_b,
        vectorizer,
        label_names=["extreme_down", "extreme_up"],
        binary=True,
    )
    results["task_b"] = {"balanced_accuracy": bal_b, "roc_auc": auc_b}

    # ── Task C: 3-class (참고)
    y_train_c = train_df["label_z1.5"].astype(int)
    y_val_c   = val_df["label_z1.5"].astype(int)

    bal_c, _ = run_task(
        "Task C: 3-class (extreme_down / normal / extreme_up, 참고용)",
        X_train, y_train_c, X_val, y_val_c,
        vectorizer,
        label_names=["extreme_down", "normal", "extreme_up"],
        binary=False,
    )
    results["task_c"] = {"balanced_accuracy": bal_c, "roc_auc": None}

    # ── 최종 요약
    print(f"\n{'='*60}")
    print("  최종 요약 및 판정")
    print(f"{'='*60}")
    print(f"\n{'Task':<35} {'balanced_acc':>14} {'roc_auc':>10} {'판정':>8}")
    print("-" * 70)

    for task_key, label in [
        ("task_a", "Task A (extreme vs normal)"),
        ("task_b", "Task B (down vs up)"),
        ("task_c", "Task C (3-class)"),
    ]:
        r = results[task_key]
        bal = r["balanced_accuracy"]
        auc = r["roc_auc"]
        auc_str = f"{auc:.4f}" if auc is not None else "  —   "

        if auc is not None:
            if auc >= 0.55:
                verdict = "BERT 재학습 가치 있음"
            elif auc >= 0.52:
                verdict = "단순 이진 분류 검토"
            else:
                verdict = "데이터 신호 부족"
        else:
            if bal >= 0.55:
                verdict = "신호 존재"
            elif bal >= 0.50:
                verdict = "약한 신호"
            else:
                verdict = "신호 부족"

        print(f"  {label:<33} {bal:>14.4f} {auc_str:>10}  {verdict}")

    print()
    overall_auc_a = results["task_a"]["roc_auc"]
    if overall_auc_a >= 0.55:
        print("  ✅ Task A ROC-AUC ≥ 0.55 → BERT 학습 설정 재조정 후 재시도 권장")
    elif overall_auc_a >= 0.52:
        print("  ⚠ Task A ROC-AUC 0.52~0.55 → 약한 신호. 이진 분류 단순화 또는 z=1.0 확장 검토")
    else:
        print("  ❌ Task A ROC-AUC < 0.52 → 데이터 신호 부족. 연구 방향 재검토 필요")


if __name__ == "__main__":
    main()
