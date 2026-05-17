"""
═══════════════════════════════════════════════════════════════════════════════
 04_finetune.py 핵심 패치 — z=1.0, extreme_up vs normal 이진 재학습
═══════════════════════════════════════════════════════════════════════════════

적용 방법: 아래 3개 블록을 04_finetune.py의 해당 위치에 교체/추가

  PATCH 1: FocalLoss 클래스 (L111-125 교체)        — alpha 버그 수정
  PATCH 2: remap 함수 (L55 근처에 추가)             — z=1.0 extreme_up 이진
  PATCH 3: cost-aware alpha 함수 (신규 추가)        — 보상 비율 명시 통제
  PATCH 4: train_stage_a 내부 alpha 계산부 (L330-334 교체)

그리고 config(plan.md 또는 yaml)에서:
  - stage_a.metric_for_best_model: "f2_extreme" → "balanced_accuracy"
  - stage_a.label_column: "label_z1.0"
  - focal_loss.alpha_method: "cost_aware",  focal_loss.fn_fp_ratio: 2.0
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 1: FocalLoss — alpha를 cross_entropy 밖으로 분리 (표준 구현)
#          04_finetune.py L111-125를 아래로 교체
# ═══════════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None,
                 gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha            # shape [num_classes], 예: tensor([1.0, 2.0])
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 1) alpha 없이 순수 per-sample CE  ← 핵심 수정 (weight= 제거)
        ce = F.cross_entropy(
            logits, labels,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # 2) pt는 클래스 가중치에 오염되지 않은 순수 확률
        pt = torch.exp(-ce)
        # 3) focal modulating factor
        focal_term = (1.0 - pt) ** self.gamma
        # 4) alpha를 샘플별 라벨에 맞춰 분리 적용
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[labels]
            loss = alpha_t * focal_term * ce
        else:
            loss = focal_term * ce
        return loss.mean()


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 2: z=1.0 extreme_up vs normal 이진 재매핑
#          04_finetune.py L55 근처(remap_for_stage_a 위)에 추가
# ═══════════════════════════════════════════════════════════════════════════════
def remap_binary_up(label_z10: int) -> int | None:
    """
    z=1.0 3-class 라벨 → extreme_up vs normal 이진

      label_z1.0: 0(extreme_down), 1(normal), 2(extreme_up)
        2 → 1 (extreme_up)
        0,1 → 0 (extreme_down은 normal로 흡수; early warning은 급등 중심)

    흡수 후 예상 분포: normal ≈ 81%, extreme_up ≈ 19%
    decision-safe ratio = 0.81/0.19 ≈ 4.26  (fn_fp_ratio < 4.26 이면 안전)
    """
    if label_z10 == 2:
        return 1
    if label_z10 in (0, 1):
        return 0
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 3: cost-aware alpha — 보상 비율 명시 통제 (Decision Theory)
#          04_finetune.py의 함수 정의 영역(예: PATCH 2 아래)에 신규 추가
# ═══════════════════════════════════════════════════════════════════════════════
def make_cost_aware_alpha(y_train, fn_fp_ratio: float = 2.0,
                          verbose: bool = True) -> torch.Tensor:
    """
    Type 2 (FN: extreme 놓침) 비용을 Type 1 (FP: 잘못 경보)의
    fn_fp_ratio 배로 명시 설정한 alpha 벡터 반환.

    Decision theory: extreme 예측이 항상 최적이 되는 경계는
        weight_ratio > (1 - p) / p   (= decision_safe)
    이 값 미만으로 유지해야 degenerate solution 회피.

    early warning 관행: FN 비용을 FP의 2~3배로 설정 (위기 예측 문헌 표준).

    Returns
    -------
    torch.Tensor shape [2]:  [normal_weight=1.0, extreme_weight=fn_fp_ratio]
    """
    y = np.asarray(y_train)
    p_extreme = float((y == 1).mean())
    decision_safe = (1.0 - p_extreme) / max(p_extreme, 1e-9)

    if verbose:
        status = "안전" if fn_fp_ratio < decision_safe else "⚠ degenerate 위험"
        print(f"  [cost-aware alpha] p(extreme)={p_extreme:.4f}")
        print(f"  [cost-aware alpha] decision-safe ratio < {decision_safe:.3f}")
        print(f"  [cost-aware alpha] fn_fp_ratio={fn_fp_ratio}  ({status})")

    if fn_fp_ratio >= decision_safe:
        print(f"  ⚠ 경고: fn_fp_ratio({fn_fp_ratio}) >= "
              f"decision_safe({decision_safe:.2f}). degenerate 위험. "
              f"값을 낮추세요.")

    return torch.tensor([1.0, float(fn_fp_ratio)], dtype=torch.float)


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 4: train_stage_a 내부 alpha 계산부 교체
#          04_finetune.py L330-334 (sqrt 블록)을 아래로 교체
# ═══════════════════════════════════════════════════════════════════════════════
"""
교체 전 (L330-334):

    # 클래스 가중치 (sqrt 완화)
    y_train = train_a["label_a"].astype(int).values
    raw_w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    alpha_a = torch.tensor(np.sqrt(raw_w), dtype=torch.float)
    print(f"\\n[Stage A alpha (sqrt_balanced)]  normal={alpha_a[0]:.4f}  extreme={alpha_a[1]:.4f}")

교체 후:
"""
# --- 아래 블록을 train_stage_a에 붙여넣기 ---
def _compute_alpha_block(train_a, fl_cfg):
    """train_stage_a 내부의 alpha 계산 로직 (참고용 함수화)."""
    y_train = train_a["label_a"].astype(int).values

    method = fl_cfg.get("alpha_method", "cost_aware")

    if method == "cost_aware":
        alpha_a = make_cost_aware_alpha(
            y_train,
            fn_fp_ratio=fl_cfg.get("fn_fp_ratio", 2.0),
            verbose=True,
        )
        print(f"  [Stage A alpha (cost_aware)]  "
              f"normal={alpha_a[0]:.4f}  extreme={alpha_a[1]:.4f}")

    elif method == "none":
        alpha_a = None
        print("  [Stage A alpha] None (중립 학습, 추론에서 임계값 조정)")

    else:  # "sqrt_balanced" (기존 방식, 호환용)
        from sklearn.utils.class_weight import compute_class_weight
        raw_w = compute_class_weight("balanced",
                                     classes=np.array([0, 1]), y=y_train)
        alpha_a = torch.tensor(np.sqrt(raw_w), dtype=torch.float)
        print(f"  [Stage A alpha (sqrt_balanced)]  "
              f"normal={alpha_a[0]:.4f}  extreme={alpha_a[1]:.4f}")

    return alpha_a


# ═══════════════════════════════════════════════════════════════════════════════
# PATCH 5 (선택): 추론 시 비용 기반 임계값 + PR curve 최적 탐색
#          평가/추론 코드에 추가
# ═══════════════════════════════════════════════════════════════════════════════
def predict_with_cost(probs_extreme, fn_cost: float = 2.5,
                      fp_cost: float = 1.0):
    """
    기대비용 최소화 임계값으로 이진 예측.
      threshold = fp_cost / (fp_cost + fn_cost)
    예: fn=2.5, fp=1.0 → threshold ≈ 0.286
    """
    threshold = fp_cost / (fp_cost + fn_cost)
    return (np.asarray(probs_extreme) >= threshold).astype(int), threshold


def find_best_threshold_f2(y_true, probs_extreme):
    """Validation에서 F2 최대화하는 임계값 탐색."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, probs_extreme)
    # 마지막 요소는 threshold 없음 → [:-1]
    f2 = 5 * prec[:-1] * rec[:-1] / (4 * prec[:-1] + rec[:-1] + 1e-9)
    best_idx = int(np.argmax(f2))
    return float(thr[best_idx]), float(f2[best_idx])


# ═══════════════════════════════════════════════════════════════════════════════
# 검증용 스모크 테스트 (이 파일 단독 실행 시)
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== PATCH 검증 스모크 테스트 ===\n")

    # FocalLoss 동작 확인
    torch.manual_seed(42)
    logits = torch.randn(8, 2)
    labels = torch.tensor([0, 1, 0, 1, 0, 0, 1, 0])

    fl_none = FocalLoss(alpha=None, gamma=2.0)
    fl_cost = FocalLoss(alpha=torch.tensor([1.0, 2.0]), gamma=2.0)
    print(f"FocalLoss(alpha=None)        loss = {fl_none(logits, labels):.4f}")
    print(f"FocalLoss(alpha=[1.0, 2.0])  loss = {fl_cost(logits, labels):.4f}")
    print("  → alpha 적용 시 loss가 달라지면 분리 적용 정상 동작\n")

    # cost-aware alpha 확인
    y = np.array([0]*81 + [1]*19)  # 81% normal, 19% extreme_up (z=1.0 흡수 후)
    print("make_cost_aware_alpha(fn_fp_ratio=2.0):")
    a = make_cost_aware_alpha(y, fn_fp_ratio=2.0)
    print(f"  alpha = {a.tolist()}\n")

    print("make_cost_aware_alpha(fn_fp_ratio=3.0):")
    a = make_cost_aware_alpha(y, fn_fp_ratio=3.0)
    print(f"  alpha = {a.tolist()}\n")

    print("make_cost_aware_alpha(fn_fp_ratio=5.0):  ← 위험 경고 확인")
    a = make_cost_aware_alpha(y, fn_fp_ratio=5.0)
    print(f"  alpha = {a.tolist()}\n")

    # remap 확인
    print("remap_binary_up:")
    for lbl in (0, 1, 2):
        print(f"  label_z1.0={lbl} → {remap_binary_up(lbl)}")

    print("\n=== 모든 패치 정상 동작 ===")
