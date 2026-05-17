# 보상 함수(Loss) 조정 솔루션 — z=1.0, extreme_up vs normal 이진 재학습

## 진단 요약

현재 `04_finetune.py`에 **세 가지 문제**가 동시에 작동 중:

| # | 문제 | 위치 | 심각도 |
|---|------|------|--------|
| 1 | **Focal Loss alpha 버그** | `04_finetune.py` L120-121 | 치명적 |
| 2 | **metric_for_best_model = "f2_extreme"** | `plan.md` L1002 (Stage A) | 치명적 |
| 3 | class_weight 비대칭 (sqrt로 부분 완화됨) | L332-333 | 중간 |

규영님이 의심한 것은 #3이지만, 분석 결과 #3은 sqrt 완화로 비율이 1.95 (안전선 3.76 미만)까지 내려와 있어 **이론상 degenerate를 직접 유발하진 않음**. 진짜 원인은 #1과 #2.

---

## 문제 1 상세: Focal Loss alpha 버그

### 현재 코드 (잘못됨)

```python
def forward(self, logits, labels):
    alpha = self.alpha.to(logits.device) if self.alpha is not None else None
    ce = F.cross_entropy(
        logits, labels, weight=alpha,        # ← alpha가 CE 안으로
        label_smoothing=self.label_smoothing, reduction="none"
    )
    pt = torch.exp(-ce)                        # ← ce가 이미 가중됨 → pt 오염
    return ((1 - pt) ** self.gamma * ce).mean()
```

### 왜 문제인가

`weight=alpha`가 `cross_entropy` 안에 들어가면:

```
정상 Focal Loss:  FL = α · (1 - p)^γ · (-log p)
현재 코드:        FL = (1 - exp(-α·CE))^γ · (α·CE)
```

`pt = exp(-α·CE)`에서 extreme 클래스(α≈1.54)는 CE가 1.54배 부풀려져
`pt`가 인위적으로 작아짐 → `(1-pt)^γ ≈ 1` 고정 → **focusing 작동 불능**.

결과: Focal Loss가 그냥 "강하게 가중된 CE"로 퇴화 → degenerate 유발.

### 수정 코드 (표준 Focal Loss)

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha          # shape: [num_classes], 예: [1.0, 2.0]
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        # 1) alpha 없이 순수 per-sample CE
        ce = F.cross_entropy(
            logits, labels,
            label_smoothing=self.label_smoothing,
            reduction="none"
        )
        # 2) pt는 오염 없는 순수 확률
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
```

이 한 가지 수정만으로도 큰 개선이 기대됨.

---

## 문제 2 상세: metric_for_best_model = "f2_extreme"

### 현재 (plan.md L1002, Stage A)

```yaml
stage_a:
  metric_for_best_model: "f2_extreme"   # ← 문제
```

### 왜 문제인가

`f2_extreme`은 extreme 클래스의 recall에 4배 가중:

```
f2 = 5·P·R / (4·P + R)

"모두 extreme 예측" → extreme recall = 1.0, precision 낮음
→ f2 ≈ 0.6+ (가짜 best)

balanced_accuracy 기준이면:
"모두 extreme 예측" → (recall_normal 0.02 + recall_extreme 0.98)/2 = 0.5
→ random과 동일 → best로 안 뽑힘 ✓
```

### 수정

```yaml
stage_a:
  metric_for_best_model: "balanced_accuracy"   # f2_extreme → 변경
  greater_is_better: true
```

또는 더 엄격하게 (degenerate를 더 강하게 거부):

`compute_metrics_a`에 `min_recall` 추가:
```python
"min_recall": float(min(recall[0], recall[1])),
```
그리고 `metric_for_best_model: "min_recall"`.

---

## 문제 3: class_weight 보상 함수 조정 (규영님 핵심 질문)

### 현재 방식의 한계

```python
raw_w = compute_class_weight("balanced", classes=[0,1], y=y_train)
alpha_a = torch.tensor(np.sqrt(raw_w), dtype=torch.float)
```

`balanced`는 클래스 빈도 역수로 자동 계산 → **early warning의 비용 구조를
반영 못 함**. sqrt는 임시방편 (왜 sqrt인지 학술적 근거 약함).

### 대안: 비용 비율 명시 통제 (Decision Theory 기반)

규영님이 지적한 "Type 1 / Type 2 비용을 직접 설계"를 코드화:

```python
def make_cost_aware_alpha(y_train, fn_fp_ratio=2.5, verbose=True):
    """
    Type 2 (FN: extreme 놓침) 비용을 Type 1 (FP: 잘못 경보)의
    fn_fp_ratio 배로 명시 설정.

    Decision theory: extreme 예측이 항상 최적이 되는 임계값은
        weight_ratio > (1 - p) / p
    이 값보다 작게 유지해야 degenerate 회피.

    early warning 맥락:
      - FN (위기를 놓침) 비용 >> FP (잘못 경보) 비용
      - 일반적으로 2~3배가 합리적 (선행연구 관행)
    """
    p_extreme = (np.asarray(y_train) == 1).mean()
    decision_safe = (1 - p_extreme) / p_extreme   # 이 값 미만이어야 안전

    if verbose:
        print(f"  [cost-aware alpha] p(extreme)={p_extreme:.3f}, "
              f"decision-safe ratio < {decision_safe:.2f}")
        print(f"  [cost-aware alpha] 설정 fn_fp_ratio={fn_fp_ratio} "
              f"({'안전' if fn_fp_ratio < decision_safe else '⚠ degenerate 위험'})")

    # normal=1.0 기준, extreme=fn_fp_ratio
    alpha = torch.tensor([1.0, float(fn_fp_ratio)], dtype=torch.float)
    return alpha
```

`z=1.0`, extreme_up vs normal 분포 (예상):
- normal: 약 79%
- extreme_up: 약 21%
- decision-safe ratio = 0.79/0.21 ≈ **3.76**

따라서 `fn_fp_ratio`는 **1.5 ~ 3.0 범위에서 실험** (3.76 미만 필수):

| fn_fp_ratio | 의미 | 권장도 |
|-------------|------|--------|
| 1.0 | 비용 대칭 (FN=FP) | recall 낮을 수 있음 |
| **2.0** | FN이 2배 비쌈 | **기본 권장** |
| **2.5** | FN이 2.5배 비쌈 | early warning 적합 |
| 3.0 | FN이 3배 비쌈 (한계) | recall↑ precision↓ |
| 3.76+ | degenerate 발생 | ❌ 금지 |

### 학술적 정당화

이 방식은 sqrt보다 학술적으로 견고:
- **Decision theory**: Bayes 최적 분류의 비용민감 임계값 이론
- **선행연구 관행**: early warning 시스템은 FN 비용을 FP의 2~3배로 설정 (금융 위기 예측 문헌 표준)
- **해석 가능**: "우리는 위기를 놓치는 비용을 잘못 경보의 2.5배로 설계했다"는 명확한 서술 가능

---

## 추론 단계 비용 반영 (가장 우아한 대안 — 선택)

학습은 중립으로 두고, 추론에서 비용 임계값 조정:

```python
def predict_with_cost(probs_extreme, fn_cost=2.5, fp_cost=1.0):
    """
    기대비용 최소화 임계값:
      threshold = fp_cost / (fp_cost + fn_cost)
    fn_cost=2.5, fp_cost=1.0 → threshold ≈ 0.286
    (extreme 확률이 28.6%만 넘으면 경고)
    """
    threshold = fp_cost / (fp_cost + fn_cost)
    return (np.asarray(probs_extreme) >= threshold).astype(int)
```

평가 시 PR curve 전체로 최적 임계값 탐색:

```python
from sklearn.metrics import precision_recall_curve
prec, rec, thr = precision_recall_curve(y_val, probs_extreme)
f2 = 5 * prec * rec / (4 * prec + rec + 1e-9)
best_thr = thr[np.argmax(f2[:-1])]
print(f"F2 최대 임계값: {best_thr:.3f}")
```

**장점**: 학습이 degenerate에 빠질 비용 비대칭 자체가 없음 →
ROC-AUC가 모델의 진짜 판별력을 정직하게 보여줌.

---

## 권장 실행 전략 (우선순위 순)

### 필수 (반드시 함께 적용)

1. **Focal Loss 버그 수정** (문제 1) — alpha를 CE 밖으로 분리
2. **metric_for_best_model 변경** (문제 2) — `f2_extreme` → `balanced_accuracy`

이 둘만 고쳐도 degenerate 대부분 해소될 가능성 높음.

### 권장 (규영님 핵심 아이디어)

3. **cost-aware alpha 도입** (문제 3) — `make_cost_aware_alpha(fn_fp_ratio=2.0)`
   - sqrt 방식 대신 명시적 비용 비율
   - fn_fp_ratio = 2.0 → 2.5 → 3.0 순차 실험

### 선택 (이론적으로 가장 깔끔)

4. **추론 임계값 분리** — 학습 alpha=None, 추론에서 비용 임계값

---

## z=1.0, extreme_up vs normal 재구성

현재 코드는 `label_z1.5` + Stage A(extreme=down∪up) 구조.
규영님 요청은 **z=1.0** + **extreme_up vs normal** 단일 이진.

### 라벨 재매핑 변경

```python
def remap_binary_up(label_z10):
    """
    z=1.0 라벨 → extreme_up vs normal 이진
      label_z1.0: 0(extreme_down), 1(normal), 2(extreme_up)
      → extreme_down은 normal로 흡수 (early warning은 급등 중심)
    """
    if label_z10 == 2:        # extreme_up
        return 1
    elif label_z10 in (0, 1): # extreme_down + normal → normal
        return 0
    return None
```

**주의**: extreme_down을 normal로 흡수하면 분포가 더 불균형해짐
- normal(흡수 후): 약 79% → 약 81%
- extreme_up: 약 21% → 약 19%
- decision-safe ratio ≈ 0.81/0.19 ≈ **4.26** (더 여유 생김)

따라서 fn_fp_ratio는 최대 3.5까지 실험 가능.

### config 변경 (plan.md 또는 training_config.yaml)

```yaml
stage_a:                              # 단일 이진이므로 stage_b 불필요
  num_labels: 2
  label_column: "label_z1.0"          # label_z1.5 → label_z1.0
  remap_function: "remap_binary_up"
  metric_for_best_model: "balanced_accuracy"   # 핵심 변경
  greater_is_better: true
  num_train_epochs: 8
  learning_rate: 1.0e-5
  warmup_ratio: 0.15
  early_stopping_patience: 3

focal_loss:
  gamma: 2.0
  alpha_method: "cost_aware"          # sqrt_balanced → cost_aware
  fn_fp_ratio: 2.0                    # 신규 (1.5~3.5 실험)

# Stage B 섹션 삭제 (단일 이진이므로 불필요)
```

---

## 검증 기준 (재학습 후 확인)

| 지표 | 목표 | 의미 |
|------|------|------|
| **ROC-AUC** | ≥ 0.55 | TF-IDF(0.528) 초과가 핵심 목표 |
| balanced_accuracy | ≥ 0.55 | degenerate 아님 |
| min(recall_0, recall_1) | ≥ 0.40 | 양쪽 모두 잡음 |
| 예측 분포 | 한쪽 ≤ 80% | 쏠림 없음 |

**핵심 판정**:
- ROC-AUC > 0.528 → FinBERT가 TF-IDF 초과 ✓ (규영님 가설 입증)
- ROC-AUC ≈ 0.50 → 비용 수정해도 판별력 없음 → 데이터 본질 한계

---

## 실험 매트릭스 (권장)

| 실험 | Focal 버그 | metric | alpha 방식 | fn_fp_ratio |
|------|-----------|--------|-----------|-------------|
| Exp 1 (baseline 확인) | 수정 | balanced_acc | None | — |
| Exp 2 (cost 약) | 수정 | balanced_acc | cost_aware | 2.0 |
| Exp 3 (cost 중) | 수정 | balanced_acc | cost_aware | 2.5 |
| Exp 4 (cost 강) | 수정 | balanced_acc | cost_aware | 3.0 |

각 실험 약 8~12분. Exp 1로 "버그 수정만으로 얼마나 개선되나" 먼저 확인 권장.
Exp 1의 ROC-AUC가 0.528 넘으면 규영님 가설 입증 완료.
