# 환율 극단 변동 Early Warning 모델 파인튜닝 계획

## 프로젝트 개요

### 연구 가설
> **"극단적 환율 변동이 발생할 환경에는 특정한 뉴스 패턴이 동반된다"**

큰 변동이 일어난 날에는 고조된 표현, 불확실성 언어, 이벤트 키워드가 뉴스에 동반 등장한다. 이 공출현 패턴(co-occurrence pattern)을 학습하면, "오늘 뉴스의 언어적 스타일이 과거 극단 변동일 뉴스와 유사한가?"를 판별할 수 있다.

### 연구 정체성
- ❌ 감성 분류기 (sentiment classifier)
- ✅ **극단 변동 regime 감지기 (extreme movement regime detector)**
- ✅ **Early warning 시스템의 텍스트 신호 추출 모듈**

### 학술적 위치
이벤트 감지(event detection) / 테일 리스크 예측(tail risk forecasting) / Volatility regime classification 영역의 연구. 인과 관계 주장이 아닌 **공출현 패턴 학습**이므로 endogeneity 문제로부터 자유로움.

### 라벨 정의
- **Class 0 (extreme_down)**: 극단적 환율 하락 regime
- **Class 1 (normal)**: 정상 변동 regime
- **Class 2 (extreme_up)**: 극단적 환율 상승 regime

라벨 부여 방식: T일 환율과 T+1~T+5 영업일 평균 환율의 차이를, **롤링 윈도우 표준편차로 정규화한 z-score** 기준으로 임계값 적용.

### 임계값 정책 (교수 자문 후 결정)

**메인 임계값: |z| ≥ 1.5** (1.5 표준편차)

선정 근거:
- IMF 외환위기, 2008 글로벌 금융위기, 2020 코로나 등 환율이 크게 변동하는 **극단 이벤트는 사회적으로 자주 발생하지 않는 희소 사건**
- z=1.0(약 32%)은 "약간 큰 변동"까지 포함하여 본 연구의 "극단 변동 early warning"이라는 task 정의와 정합성 부족
- z=1.5(약 13%)가 "진짜 극단 변동"의 통계적 정의에 부합

**Fallback: |z| ≥ 1.0** (1.0 표준편차)
- z=1.5 모델의 Test 성능이 충족되지 않는 경우(extreme_recall < 0.4 등)에만 사용
- z=1.0은 비교 분석용으로 라벨링은 수행하되, 학습은 z=1.5 결과 이후 결정

**샘플링 규모: 20,000건** (5,000건에서 증대)
- z=1.5 적용 시 5,000건 샘플에서 Val/Test extreme 클래스가 70/101, 62/86 수준으로 학습/평가 안정성 우려
- 4배 증대로 z=1.5 extreme 클래스를 충분히 확보 (예상 Val/Test 각 280/240건 이상)
- 시간 균등 샘플링 비율 유지: 연도별 약 870건 (월별 약 73건)

### 최종 활용
학습된 모델로 11만 건 전체 뉴스의 변동 regime 신호 추출 → Early warning 다운스트림 모델(LSTM 등)의 입력으로 활용 (별도 단계, 본 plan에서는 다루지 않음).

---

## 환경 정보

- **GPU**: RTX 4070 Ti (12GB VRAM)
- **기반 모델**: `snunlp/KR-FinBert-SC` (한국어 금융 도메인 사전학습 BERT)
- **학습 프레임워크**: PyTorch + HuggingFace Transformers
- **환율 데이터**: `FinanceDataReader` (USD/KRW 일별 종가)
- **패키지 관리**: uv

### 추가 의존성
```bash
uv add finance-datareader  # Phase 1 환율 데이터 수집용
```

---

## 디렉토리 구조

```
fx_regime_detection/
├── plan.md                          # 본 문서
├── data/
│   ├── raw/
│   │   └── fx_news_raw.csv          # 원본 11만 건 (토큰화된 뉴스)
│   ├── sampled/
│   │   ├── market_table.csv         # 영업일 환율 테이블 (FDR 수집)
│   │   ├── fx_news_filtered.csv     # 필터링 + 환율 매핑 결과
│   │   ├── fx_news_20000.csv        # 시간 균등 샘플링 결과 (20,000건)
│   │   └── filtering_report.md
│   ├── labeled/
│   │   ├── label_distribution.png
│   │   ├── threshold_comparison.csv
│   │   └── final_labeled.csv
│   ├── splits/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   ├── inference/
│   │   ├── all_articles_signals.csv
│   │   └── daily_signals.csv
│   └── validation/
│       └── human_audit_200.csv      # 사후 spot-check (선택)
├── scripts/
│   ├── 01_filter_and_map.py         # Phase 1: 필터링 + FDR 환율 수집 + 매핑
│   ├── 02_compute_labels.py         # Phase 2: z-score 라벨링
│   ├── 03_split_dataset.py          # Phase 3
│   ├── 04_finetune.py               # Phase 4
│   ├── 05_evaluate.py               # Phase 5
│   └── 06_inference_full.py         # Phase 6
├── utils/
│   ├── __init__.py
│   ├── stopwords.py                 # 불용어 사전
│   ├── text_cleaning.py
│   └── labeling.py                  # z-score 라벨링 함수
├── models/
│   ├── kr-finbert-fx-regime-z1.0/
│   └── kr-finbert-fx-regime-z1.5/
├── results/
│   ├── label_quality_report.md
│   ├── training_metrics.json
│   ├── evaluation_report.md
│   └── confusion_matrices/
└── configs/
    └── training_config.yaml
```

---

## Phase 0: 불용어 사전 모듈 구축

### 목표
기존 `clean_dictionary.py`의 환율 도메인 특화 불용어 사전을 재사용 가능한 모듈로 정리

### 작업 내용

**파일**: `utils/stopwords.py`

기존 `clean_dictionary.py`의 `STOPWORD_KEYWORDS`를 그대로 가져오되, 두 맥락에서 쓸 수 있도록:

1. **N-gram 기반 필터링** (기존 용도, 형태소 분석 후): 그대로 유지
2. **원본 텍스트 필터링** (Phase 1 신규 용도): 문자열 매칭

```python
STOPWORD_KEYWORDS: frozenset = frozenset({
    # ... clean_dictionary.py 그대로 복사 ...
})

def count_stopwords_in_text(text: str) -> dict: ...
def stopword_density(text: str) -> float: ...
```

**파일**: `utils/text_cleaning.py`

```python
def clean_article_body(text: str) -> str: ...
def is_irrelevant_article(title: str, body: str) -> bool: ...
```

**중요**: 본 연구에서는 단순 시황 보도(`is_market_summary`)도 학습 데이터에 포함. 큰 변동일에는 시황 보도조차도 표현이 달라질 수 있음 (긴급, 급등, 급락 등). 시황 보도 필터링 함수는 **만들지 않음**.

### 검증 체크
- [ ] 기존 `clean_dictionary.py`와 동일한 단어 셋 유지
- [ ] 샘플 기사 10건으로 함수 동작 확인

---

## Phase 1: 데이터 필터링 및 샘플링

### 목표
11만 건 원본에서 노이즈를 제거하고, 학습에 적합한 20,000건을 시간 균등 샘플링. T+1~T+5 환율 정합성 검증 포함.

### 입력
- `data/raw/fx_news_raw.csv` (11만 건)
- 필수 컬럼: `date`, `title`, `body`, `close_rate`, T+1~T+5 환율 정보

### 작업 내용
**스크립트**: `scripts/01_filter_and_sample.py`

## Phase 1: 뉴스 필터링 및 환율 매핑

### 목표
11만 건 원본에서 노이즈를 제거하고, **각 기사를 영업일 환율 데이터에 매핑**하여 T-1, T, T+1~T+5 종가 정보를 연결한다. 시간 균등 샘플링으로 20,000건 추출.

### 입력
- `data/raw/fx_news_raw.csv` (11만 건, 토큰화된 뉴스 데이터)
- 필수 컬럼: `publish_date`, `title`, `text`, `tokens`, `press`, `url`
- **환율 데이터는 본 단계에서 별도 수집** (FinanceDataReader)

### 작업 내용
**스크립트**: `scripts/01_filter_and_map.py`

본 Phase는 두 개 트랙으로 진행됨:
- **Track A (Step 1~3)**: 뉴스 데이터 정제 및 필터링
- **Track B (Step 4~6)**: 환율 데이터 수집 및 영업일 테이블 구성
- **Step 7**: 두 트랙 병합 (merge_asof)
- **Step 8**: 시간 균등 샘플링

---

### Track A: 뉴스 데이터 정제

#### Step 1: 본문 정제

```python
CLEANUP_PATTERNS = [
    r"<저작권자.*?>", r"\(c\)\s*연합인포맥스.*",
    r"무단\s*전재.*재배포\s*금지\s*",
    r"\([가-힣]+\s*=\s*연합[가-힣]+\)\s*[가-힣]+\s*기자=?",
    r"※\s*제보.*", r"▶\s*.*?구독.*",
    r"\(끝\)\s*$",
]
```

원본은 `body_raw`에 보관, 정제 결과를 `body`에 저장.

#### Step 2: 형식 기반 필터링

| 조건 | 기준 |
|---|---|
| 본문 길이 | 100자 이상, 3000자 이하 |
| 제목 길이 | 5자 이상 |
| 중복 기사 | title + body[:100] hash로 제거 |
| 결측치 | title, body 모두 존재 |
| 빈 토큰 | `tokens` 컬럼이 `[]`이거나 NaN인 행 제거 |
| 날짜 파싱 | `publish_date` → datetime 변환 실패 행 제거 |

#### Step 3: 콘텐츠 기반 필터링 (불용어 사전 활용)

**핵심 원칙**: fx_keyword (환율/달러/원화 등) 화이트리스트 필터는 **사용하지 않음**. 

이유:
- 본 데이터는 이미 네이버 금융 "환율 섹터"로 분류되어 수집된 기사 풀
- 환율을 크게 움직이는 외생 충격 기사 (FOMC 결정, 무역분쟁, 지정학 리스크 등)는 "환율/달러/원화"를 직접 언급하지 않을 수 있음
- 본 연구 가설("큰 변동일 = 큰 사건일")의 핵심인 외생 충격 기사를 키워드 필터로 거르면 가설 검증이 불가능
- 따라서 **블랙리스트(불용어) 방식**만 채택. 무관한 기사는 거르되, 환율과 직접 관련 없어 보여도 환율 섹터에 포함된 기사는 신뢰

**불용어 밀도 임계값은 percentile 기반 동적 산정**:

`stopword_density = matches / chars * 1000` (본문 1,000자당 불용어 매칭 수)

고정 임계값(예: 8.0)은 위험:
- "딜러", "마감", "전장", "기준율" 등 정상적인 환율 시황 단어도 불용어 사전에 포함됨 (clean_dictionary.py 카테고리 7-3 등)
- 정상 환율 기사도 8/1000자 매칭이 흔함 → 과도한 제거 발생
- 따라서 **데이터 분포 기반 percentile 컷**을 사용하여 robust한 임계값 자동 산정

```python
DENSITY_PERCENTILE = 90  # 상위 10% 노이즈 기사 제거

def filter_by_density(df, percentile=90):
    """
    불용어 밀도 상위 (100-percentile)% 기사 제거.
    
    Returns:
        filtered_df, threshold (실제 적용된 임계값)
    """
    densities = df['body'].apply(stopword_density)
    threshold = densities.quantile(percentile / 100)
    
    print(f"density 분포: median={densities.median():.2f}, "
          f"mean={densities.mean():.2f}, p{percentile}={threshold:.2f}")
    
    keep_mask = densities <= threshold
    return df[keep_mask].copy(), threshold


def is_irrelevant_article(title, body):
    """
    광고/안내 기사 판별 (불용어 밀도와는 별개로 적용).
    
    Returns:
        (is_irrelevant: bool, reason: str)
    """
    text = title + " " + body
    
    ad_indicators = ["바로가기", "구독신청", "이벤트", "회원가입"]
    if sum(text.count(w) for w in ad_indicators) >= 3:
        return True, "advertisement"
    
    return False, "keep"
```

**적용 순서**:
1. `is_irrelevant_article()`로 광고 기사 제거 (절대 기준)
2. `filter_by_density(percentile=90)`로 노이즈 상위 10% 제거 (분포 기반)

**percentile 선택 가이드**:
- `90` (기본값, 권장): 상위 10% 노이즈 제거. 20,000건 샘플링에 충분한 잔존
- `95`: 보수적, 더 많이 보존하되 노이즈 잔존 가능
- `85`: 공격적, 정상 기사도 일부 제거될 위험

**중요**: percentile 기반이므로 `DENSITY_PERCENTILE`을 plan과 코드에 **명시적 상수로 노출**. 결과 재현성을 위해 변경 시 로그에 기록.

#### Step 3 결과: 정제된 뉴스 DataFrame
- 컬럼: `article_id, publish_date, news_date, title, body, body_raw, tokens, press, url`
- `news_date`: `publish_date.dt.normalize()` (시간 절사, 날짜 단위)

---

### Track B: 환율 데이터 수집 및 영업일 테이블 구성

본 트랙은 **첨부 코드 `labeling.py`의 `fetch_exchange_rate()` 및 `build_market_table()` 함수 로직을 그대로 활용**.

#### Step 4: USD/KRW 환율 데이터 수집

```python
import FinanceDataReader as fdr

def fetch_exchange_rate(start="2004-05-01", end="2026-04-30"):
    raw = fdr.DataReader("USD/KRW", start, end)
    fx = raw[["Close"]].copy()
    fx.index = pd.to_datetime(fx.index).normalize()
    fx.index.name = "trade_date"
    fx = fx.sort_index().dropna(subset=["Close"])
    
    # FDR 서버 통신 버그 방어: 동일 날짜 중복 행 제거
    fx = fx[~fx.index.duplicated(keep="last")]
    return fx
```

**수집 기간**: `2004-05-01 ~ 2026-04-30`
- 시작일에 여유 (5월) → 첫 기사 발행일 이전부터 환율 데이터 확보
- 종료일에 여유 (4월) → T+5(20영업일) 후 종가까지 확보

**예상 영업일 수**: 약 5,500일 (23년)

#### Step 5: 영업일 테이블 구성 (T / T-1 / T+1 ~ T+5 매핑)

```python
def build_market_table(fx):
    market = fx.copy().rename(columns={"Close": "close"})
    
    # T-1, T+1 종가
    market["prev_close"] = market["close"].shift(1)        # T-1
    market["next_close"] = market["close"].shift(-1)       # T+1
    
    # T+1 ~ T+5 개별 종가 (z-score 라벨링용)
    for k in range(1, 6):
        market[f"close_T{k}"] = market["close"].shift(-k)
    
    # T+1 ~ T+5 평균 (= avg_1w, 본 연구의 핵심 컬럼)
    market["close_T1_T5_mean"] = sum(
        market["close"].shift(-k) for k in range(1, 6)
    ) / 5
    
    # 다중 horizon 평균 (참고용, 본 plan에서 직접 사용은 안 함)
    market["avg_2w"] = sum(market["close"].shift(-k) for k in range(1, 11)) / 10
    market["avg_3w"] = sum(market["close"].shift(-k) for k in range(1, 21)) / 20
    
    # 첫 행(prev_close 없음) · 마지막 20행(avg_3w 없음) 제거
    market = market.dropna(subset=["prev_close", "next_close", "avg_3w"])
    
    return market.reset_index()
```

**산출 컬럼**:
- `trade_date`: T 영업일
- `close`: T 종가 (= `close_T`)
- `prev_close`: T-1 종가
- `next_close`: T+1 종가
- `close_T1, close_T2, ..., close_T5`: T+1 ~ T+5 개별 종가
- `close_T1_T5_mean`: **T+1 ~ T+5 평균 (Phase 2 라벨링의 핵심 입력)**
- `avg_2w, avg_3w`: 참고용

#### Step 6: 일일 변동성 시계열 사전 계산

Phase 2의 z-score 계산을 위해 본 단계에서 미리 산출:

```python
market["daily_change"] = market["close"].diff()
market["rolling_std_90d"] = market["daily_change"].rolling(
    window=90, min_periods=60
).std()
```

이 컬럼들은 Phase 2에서 그대로 활용됨 (재계산 불필요).

---

### Step 7: 뉴스-환율 매핑 (merge_asof)

본 단계는 **첨부 코드의 `map_news_to_market()` 함수 로직을 그대로 활용**.

```python
def map_news_to_market(news, market):
    news_sorted = news.sort_values("news_date").reset_index(drop=True)
    market_sorted = market.sort_values("trade_date")
    
    merge_cols = [
        "trade_date", "close", "prev_close", "next_close",
        "close_T1", "close_T2", "close_T3", "close_T4", "close_T5",
        "close_T1_T5_mean", "daily_change", "rolling_std_90d",
    ]
    
    merged = pd.merge_asof(
        news_sorted,
        market_sorted[merge_cols],
        left_on="news_date",
        right_on="trade_date",
        direction="backward",   # 직전 영업일로 소급
    )
    
    # 매핑 실패 행 (환율 데이터 범위 외) 제거
    merged = merged.dropna(subset=["trade_date", "next_close"]).reset_index(drop=True)
    return merged
```

**매핑 규칙** (`direction='backward'`):
| 발행일 | 매핑되는 T |
|---|---|
| 영업일 발행 | 당일 T |
| 토요일 발행 | 직전 금요일 T |
| 일요일 발행 | 직전 금요일 T |
| 공휴일 발행 | 직전 영업일 T |

**중요한 결측 처리**:
- `close_T1_T5_mean`이 NaN인 행은 제거 (T+5까지 환율이 없는 최근 기사)
- 즉, 환율 데이터 종료일 - 5영업일 이후 발행 기사는 학습에서 제외

---

### Step 8: 시간 균등 샘플링 (20,000건)

**규모 변경 이력**: 5,000건 → 20,000건 (2026-05-06 결정)

- z=1.5 임계값을 메인으로 사용하기로 결정 (교수 자문)
- 5,000건 샘플에서 z=1.5 Val/Test extreme 클래스가 70/101, 62/86으로 학습/평가 안정성 우려
- 4배 증대로 충분한 extreme 클래스 확보

**샘플링 방식**:
- 잔존 기사 풀(약 92,536건)에서 20,000건 추출
- 2004~2026년 = 23년 → 연도별 약 870건 (월별 약 73건)
- 각 연-월 그룹에서 무작위 추출 (random_state=42, 재현성 유지)
- 특정 연-월에 873건이 안 되면 가능한 만큼만 추출 (월별 가용 기사 수에 따라 자동 조정)

**재시작 정책**:
- Phase 1의 Step 1~7 (정제, 필터링, 환율 매핑) 결과는 **재사용 가능**
- 즉, `data/sampled/fx_news_filtered.csv` (92,536건)이 존재하면 Step 8만 재실행
- Step 1~7 재실행은 비효율적이므로 캐시 활용 권장

### 출력
- `data/sampled/fx_news_filtered.csv` (Track A 결과 + 환율 매핑 결과, 약 92,536건) — **재사용**
- `data/sampled/fx_news_20000.csv` (최종 샘플링 결과, 신규)
- `data/sampled/market_table.csv` (Track B 산출, 영업일 환율 테이블) — **재사용**
- `data/sampled/filtering_report.md` (필터링 통계, 업데이트)

**최종 컬럼**: 
```
article_id, publish_date, news_date, trade_date,
title, body, body_raw, tokens, press, url,
close, prev_close, next_close,
close_T1, close_T2, close_T3, close_T4, close_T5, close_T1_T5_mean,
daily_change, rolling_std_90d
```

### 검증 체크
- [ ] FinanceDataReader 데이터 수집 성공 (영업일 약 5,500일)
- [ ] **merge_asof 키 dtype 통일** (`news_date`, `trade_date` 모두 `datetime64[ns]`)
- [ ] 영업일 테이블에 `close_T1_T5_mean` 결측치 없음 (마지막 20일 제외)
- [ ] merge_asof 매핑 성공률 ≥ 99%
- [ ] 매핑 실패 기사: 환율 데이터 범위 외 발행분만 (예: 2004-04 이전, 2026-04 마지막 5영업일 이후)
- [ ] 주말 발행 기사가 직전 금요일 T에 매핑되었는지 spot-check
- [ ] **불용어 밀도 분포 로그 출력** (median, mean, p90, p95, p99)
- [ ] **DENSITY_PERCENTILE=90 적용 후 잔존 기사 20,000건 샘플링 가능 여부 확인**
- [ ] **density 필터로 제거된 기사 50건 spot-check (false positive 확인)**
- [ ] **연도별 분포: 평균 870 ± 100건 범위 (특정 시기 결손 없음)**
- [ ] **샘플링 규모가 20,000건에 근접한지 확인 (월별 가용 기사 부족으로 일부 결손 허용)**
- [ ] 필터링 통과 기사 50건 spot-check (false negative 확인)

---

## Phase 2: 변동 기반 라벨링 (z-score 방식)

### 목표
T일 환율과 T+1~T+5 평균의 차이를, **시기별 변동성에 정규화한 z-score** 기준으로 3-class 라벨 부여

### 핵심 설계 원칙

#### 원칙 1: 절대 변동량(원) 사용 금지, z-score 사용

**고정 임계값(예: 8원)의 문제**:
- 2008 금융위기: 일일 변동성 20원+ → 거의 모든 날이 "극단"
- 2017~2019 안정기: 일일 변동성 3~4원 → 거의 모든 날이 "정상"
- 시기 편향으로 모델이 "변동성"이 아닌 "어느 시기인가"를 학습할 위험

**해결책**: 직전 90일 일일 변동의 표준편차로 정규화

```python
df['daily_change'] = df['close'].diff()
df['rolling_std_90d'] = df['daily_change'].rolling(90).std()
df['change_T_to_T1T5'] = df['close_T1_T5_mean'] - df['close_T']
df['z_score'] = df['change_T_to_T1T5'] / df['rolling_std_90d']
```

각 시기 기준으로 **"그 당시 변동성 대비 큰 변동"**이 일관되게 정의됨.

#### 원칙 2: 두 가지 임계값 비교 실험

| 임계값 | 예상 분포 | 의미 |
|---|---|---|
| **|z| ≥ 1.0** | [16% / 68% / 16%] | 일반적 큰 변동 |
| **|z| ≥ 1.5** | [7% / 86% / 7%] | 진짜 극단 변동만 |

두 라벨 셋 모두 만들어 Phase 4에서 별도 모델 학습 후 비교.

#### 원칙 3 (선택): 일관된 방향성 조건

T+1~T+5 5일 동안 변동의 일관성 확인:
- 5일 중 4일 이상 같은 방향이면 라벨 부여, 아니면 중립
- "확 튀고 다시 돌아오는" 노이즈 제거 가능
- 일단 옵션으로 구현, 적용 여부는 라벨 분포 보고 결정

### 입력
- `data/sampled/fx_news_20000.csv` (Phase 1 산출, 이미 `daily_change`, `rolling_std_90d` 포함)

### 작업 내용
**스크립트**: `scripts/02_compute_labels.py`
**유틸**: `utils/labeling.py`

> **참고**: Phase 1에서 이미 `daily_change`와 `rolling_std_90d` (window=90, min_periods=60) 계산이 완료되어 있음. Phase 2는 z-score 계산과 라벨 부여에 집중.

#### Step 1: z-score 계산

```python
df['change_T_to_T1T5'] = df['close_T1_T5_mean'] - df['close']
df['z_score'] = df['change_T_to_T1T5'] / df['rolling_std_90d']
```

**`close`**는 Phase 1의 영업일 매핑 결과로, T일 종가에 해당. (Phase 1에서 `close_T`라는 별칭도 존재 가능 — 코드 작성 시 통일 필요)

#### Step 2 (선택): 롤링 윈도우 민감도 분석

본 plan에서는 window=90을 기본값으로 사용하지만, 학술적 robustness를 위해:
- window 60/90/120/180으로 z-score 분포 비교
- 결과를 `results/label_quality_report.md`에 부록으로 첨부

이 단계가 필요하면 Phase 1의 영업일 테이블 생성 단계로 돌아가 `rolling_std_60d`, `rolling_std_120d`, `rolling_std_180d` 컬럼을 추가 산출해야 함.

#### Step 3: 라벨 부여 (두 가지 임계값)

```python
def assign_label(z, threshold=1.0):
    if pd.isna(z):
        return None
    if z >= threshold:
        return 2  # extreme_up
    elif z <= -threshold:
        return 0  # extreme_down
    else:
        return 1  # normal

df['label_z1.0'] = df['z_score'].apply(lambda z: assign_label(z, 1.0))
df['label_z1.5'] = df['z_score'].apply(lambda z: assign_label(z, 1.5))
```

#### Step 4: 라벨 품질 분석

자동 생성: `results/label_quality_report.md`

보고할 지표:

1. **클래스 분포**: z=1.0, z=1.5 각각 [class_0, class_1, class_2] 비율
2. **시기별 분포**: 연도별 각 클래스 비율 (특정 시기 편중 확인)
3. **z-score 분포 시각화**: 히스토그램, QQ-plot
4. **"같은 날 발행 기사 동일 라벨" 비율**: 구조적 한계 명시 → reviewer 사전 대응
5. **롤링 윈도우 효과 확인**: 2008 금융위기, 2020 코로나 시기 라벨 분포가 정상화되었는지 확인

#### Step 5 (선택): 사람 정합성 spot-check

200건 무작위 추출 (극단 100 + 정상 100) → 사람이 다음 질문에 답:

> "이 기사가 환율 큰 변동 시기의 뉴스처럼 읽히는가?"

답변: Yes / No / 애매

**중요**: 이건 IAA가 아니라 **라벨이 사람 직관과 부합하는지 점검하는 보조 분석**. 결과는 라벨 품질 보충 자료.

### 출력
- `data/labeled/final_labeled.csv`
- 컬럼: 모든 Phase 1 컬럼 + `change_T_to_T1T5, z_score, label_z1.0, label_z1.5`
- `data/labeled/label_distribution.png`
- `data/labeled/threshold_comparison.csv`
- `data/validation/human_audit_200.csv` (선택)
- `results/label_quality_report.md`

### 검증 체크
- [ ] z-score 결측치 0건 (Phase 1에서 마지막 20일 제외 처리되었어야 함)
- [ ] z=1.0 라벨 분포 [10~20% / 60~80% / 10~20%]
- [ ] z=1.5 라벨 분포 [5~10% / 80~90% / 5~10%]
- [ ] 연도별 라벨 분포 극단적 편중 없음 (롤링 윈도우 효과 확인)
- [ ] z-score 분포 시각화 (대략 정규분포)
- [ ] (선택) 사람 spot-check 시 극단 라벨 정합성 ≥ 60%

---

## Phase 3: 학습/검증/테스트 분할

### 목표
시간순 분할로 leakage 없는 평가 셋 구성

### 작업 내용
**스크립트**: `scripts/03_split_dataset.py`

#### 시간순 분할

| Split | 기간 | 비율 |
|---|---|---|
| Train | 2004-01 ~ 2020-12 | ~74% |
| Validation | 2021-01 ~ 2023-06 | ~13% |
| Test | 2023-07 ~ 2026-04 | ~13% |

**시간순 분할 필수 이유**:
- 환율 시계열 자기상관 → 랜덤 분할 시 leakage
- 다운스트림 LSTM도 시간순 평가
- Distribution shift 검증 가능

#### 두 가지 라벨 셋 모두 분할

`label_z1.0`, `label_z1.5` 컬럼 모두 보존. Phase 4에서 두 라벨로 각각 학습.

#### 클래스 분포 확인

각 split의 두 라벨 분포 출력. unbalanced 시 Phase 4에서 class_weight 적용.

### 출력
- `data/splits/train.csv`, `val.csv`, `test.csv`
- `data/splits/split_stats.md`

### 검증 체크
- [ ] 시간 범위 겹침 없음
- [ ] 각 split 클래스 분포 출력
- [ ] 두 라벨 셋 모두 보존됨

---

## Phase 4: 2단계 캐스케이드 모델 파인튜닝

### 목표
KR-FinBERT를 기반으로 **2단계 캐스케이드(cascade) 분류기**를 학습한다. 단일 3-class 모델이 방향성 학습에 실패하는 문제를 해결하기 위해 task의 난이도를 분리한다.

### 학습 전략 변경 이력 (2026-05-11)

**초기 시도 (실패)**: 단일 3-class 모델 (extreme_down / normal / extreme_up)
- 문제: class_weight 적용 후 모델이 96%를 extreme_down으로 예측하는 degenerate solution 발생
- 원인: extreme_down과 extreme_up이 모두 ~10% 분포로 거의 같은 weight (3.27 vs 3.06)를 받아, 미세한 weight 차이가 학습 초기 random direction을 결정한 후 자기강화
- 결과: extreme_recall 0.5063은 가짜 신호 (recall_down 0.98이 평균을 끌어올림, recall_up 0.03)

**개선 전략**: 2단계 캐스케이드 학습 + Focal Loss

### 2단계 캐스케이드 구조

**Stage A: 극단 vs 정상 이진 분류**
```
입력: 뉴스 텍스트
출력: P(extreme), P(normal)
라벨 매핑: (extreme_down, extreme_up) → 1 (extreme), normal → 0
```
이 단계는 "극단 변동의 가능성"만 판별. 방향성은 무시.

**Stage B: 극단 내 방향성 분류 (down vs up)**
```
입력: 뉴스 텍스트 (Stage A에서 extreme으로 판정된 샘플만)
출력: P(extreme_down), P(extreme_up)
학습 데이터: extreme 라벨인 샘플만 (extreme_down + extreme_up)
```
이 단계는 "이미 극단으로 판정된 샘플"의 방향성 결정.

### 캐스케이드 추론 파이프라인

```python
def cascade_predict(text, threshold=0.5):
    """
    2단계 캐스케이드 추론
    
    Returns:
        final_label: 0 (extreme_down), 1 (normal), 2 (extreme_up)
        extreme_score: P(extreme) from Stage A
        direction_score: P(extreme_up) - P(extreme_down) from Stage B
    """
    # Stage A: 극단 가능성 점수
    p_extreme = model_A(text)  # softmax[1]
    
    if p_extreme < threshold:
        return 1, p_extreme, 0.0  # normal
    
    # Stage B: 방향성 결정
    p_down, p_up = model_B(text)  # softmax
    direction = p_up - p_down
    
    if p_up > p_down:
        return 2, p_extreme, direction  # extreme_up
    else:
        return 0, p_extreme, direction  # extreme_down
```

### 학술적 정당화

본 접근은 **divide-and-conquer** 패러다임의 표준 사례:
- Multi-class 문제를 두 개의 binary task로 분해
- 각 task의 난이도와 클래스 분포가 독립적으로 최적화 가능
- 추론 시에도 해석 가능성 향상 ("얼마나 극단적인가" + "어느 방향인가")

선행 사례:
- 의료 영상 분류 (먼저 "병변 vs 정상", 다음 "병변 종류")
- 이상 탐지 (anomaly detection → 이상 유형 분류)

### 핵심 설계 결정

#### 결정 1: 기반 모델 선택

**선정**: `snunlp/KR-FinBert-SC` (Stage A, B 모두 동일)

#### 결정 2: Focal Loss 도입 (필수)

이전 학습 실패의 근본 원인은 class_weight + cross_entropy의 degenerate solution. **Focal Loss로 교체**:

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # 클래스별 가중치
        self.gamma = gamma  # focusing parameter
    
    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()
```

**Focal Loss 장점**:
- "쉬운 샘플"의 손실을 자동 감소 → 모델이 어려운 샘플(방향성 구별 등)에 집중
- class_weight를 약하게 써도 잘 작동 → degenerate solution 방지
- `gamma=2.0`이 표준값

#### 결정 3: 클래스 가중치 완화 (sqrt)

```python
balanced_weights = compute_class_weight('balanced', ...)
alpha = torch.tensor(np.sqrt(balanced_weights), dtype=torch.float)
# 예: balanced [4.62, 0.56] → sqrt [2.15, 0.75]
```

너무 강한 weight는 한쪽 쏠림을 유발. sqrt 완화로 균형.

#### 결정 4: 학습 데이터 (z=1.5 유지)

Stage A 학습 데이터:
- normal (label=0): 11,701건 (Train)
- extreme (label=1): 3,123건 (extreme_down 1,512 + extreme_up 1,611)
- 비율: 79% / 21%

Stage B 학습 데이터:
- extreme_down (label=0): 1,512건
- extreme_up (label=1): 1,611건
- 비율: 48% / 52% (거의 균형)

**Stage B의 장점**: 클래스 균형이 거의 1:1 → degenerate solution 위험 낮음.

### 입력
- `data/splits/train.csv`, `val.csv` (label_z1.5 컬럼 사용)

### 작업 내용

#### Step 1: 라벨 재매핑

```python
def remap_for_stage_a(label_z15):
    """Stage A: extreme vs normal"""
    if label_z15 in [0, 2]:  # extreme_down or extreme_up
        return 1
    elif label_z15 == 1:  # normal
        return 0
    return None

def remap_for_stage_b(label_z15):
    """Stage B: extreme 내에서만, down vs up"""
    if label_z15 == 0:  # extreme_down
        return 0
    elif label_z15 == 2:  # extreme_up
        return 1
    return None  # normal은 Stage B 학습/평가에서 제외

# Stage A 데이터: 모든 샘플
train_df['label_a'] = train_df['label_z1.5'].apply(remap_for_stage_a)

# Stage B 데이터: extreme 샘플만 필터링
train_df_b = train_df[train_df['label_z1.5'].isin([0, 2])].copy()
train_df_b['label_b'] = train_df_b['label_z1.5'].apply(remap_for_stage_b)
```

#### Step 2: 하이브리드 입력

```python
def make_input(row, max_body_len=400):
    return f"{row['title']} [SEP] {row['body'][:max_body_len]}"
```

#### Step 3: 모델 로드 (각 Stage별)

```python
# Stage A 모델
model_a = AutoModelForSequenceClassification.from_pretrained(
    "snunlp/KR-FinBert-SC",
    num_labels=2,
    id2label={0: "normal", 1: "extreme"},
    label2id={"normal": 0, "extreme": 1},
)

# Stage B 모델 (별도 인스턴스)
model_b = AutoModelForSequenceClassification.from_pretrained(
    "snunlp/KR-FinBert-SC",
    num_labels=2,
    id2label={0: "extreme_down", 1: "extreme_up"},
    label2id={"extreme_down": 0, "extreme_up": 1},
)
```

#### Step 4: Focal Trainer 구현

```python
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


class FocalTrainer(Trainer):
    def __init__(self, alpha=None, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.focal_loss(outputs.logits, labels.to(model.device))
        return (loss, outputs) if return_outputs else loss
```

#### Step 5: 평가 지표 (Stage별)

**Stage A 지표** (extreme 검출):
```python
def compute_metrics_a(eval_pred):
    logits = eval_pred.predictions
    preds = np.argmax(logits, axis=1)
    labels = eval_pred.label_ids
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1], zero_division=0
    )
    
    extreme_recall = recall[1]
    extreme_precision = precision[1]
    f2_extreme = fbeta_score(labels, preds, beta=2, pos_label=1, zero_division=0)
    
    # 확률 기반 평가
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0
    
    return {
        "extreme_recall": extreme_recall,        # 핵심 지표
        "extreme_precision": extreme_precision,
        "f2_extreme": f2_extreme,                 # metric_for_best_model
        "recall_normal": recall[0],
        "balanced_accuracy": (recall[0] + recall[1]) / 2,
        "roc_auc": auc,
    }
```

**Stage B 지표** (방향성 분류):
```python
def compute_metrics_b(eval_pred):
    logits = eval_pred.predictions
    preds = np.argmax(logits, axis=1)
    labels = eval_pred.label_ids
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1], zero_division=0
    )
    
    return {
        "accuracy": (preds == labels).mean(),
        "recall_down": recall[0],
        "recall_up": recall[1],
        "precision_down": precision[0],
        "precision_up": precision[1],
        "f1_macro": f1.mean(),                    # metric_for_best_model
        "balanced_accuracy": (recall[0] + recall[1]) / 2,  # 핵심 지표
        "direction_balance": abs(recall[0] - recall[1]),   # 0에 가까울수록 좋음
    }
```

#### Step 6: 학습 루프

**Stage A 학습** (extreme vs normal):

```python
# Stage A 데이터
train_a = train_df.copy()
val_a = val_df.copy()
# label_a는 모든 샘플에 부여됨

# Class weights (sqrt 완화)
class_weights_a = compute_class_weight(
    'balanced', classes=np.array([0, 1]),
    y=train_a['label_a'].values
)
alpha_a = torch.tensor(np.sqrt(class_weights_a), dtype=torch.float)

trainer_a = FocalTrainer(
    alpha=alpha_a, gamma=2.0,
    model=model_a, args=training_args_a,
    train_dataset=train_ds_a, eval_dataset=val_ds_a,
    compute_metrics=compute_metrics_a,
)
trainer_a.train()
trainer_a.save_model("./models/kr-finbert-fx-stage-a")
```

**Stage B 학습** (extreme 내 방향):

```python
# Stage B 데이터: extreme 샘플만 필터링
train_b = train_df[train_df['label_z1.5'].isin([0, 2])].copy()
val_b = val_df[val_df['label_z1.5'].isin([0, 2])].copy()
# label_b 부여됨

# Class weights (거의 균형이지만 일관성 유지)
class_weights_b = compute_class_weight(
    'balanced', classes=np.array([0, 1]),
    y=train_b['label_b'].values
)
alpha_b = torch.tensor(np.sqrt(class_weights_b), dtype=torch.float)

trainer_b = FocalTrainer(
    alpha=alpha_b, gamma=2.0,
    model=model_b, args=training_args_b,
    train_dataset=train_ds_b, eval_dataset=val_ds_b,
    compute_metrics=compute_metrics_b,
)
trainer_b.train()
trainer_b.save_model("./models/kr-finbert-fx-stage-b")
```

#### Step 7: 캐스케이드 통합 평가

두 모델 학습 후 Val 셋에 캐스케이드 추론 수행:

```python
def cascade_evaluate(val_df, model_a, model_b, threshold=0.5):
    """
    캐스케이드 통합 평가:
    Stage A → (조건부) Stage B → 최종 3-class 라벨
    """
    final_preds = []
    for text, true_label in zip(val_df['input_text'], val_df['label_z1.5']):
        # Stage A
        p_extreme = predict_proba(model_a, text)[1]
        
        if p_extreme < threshold:
            final_preds.append(1)  # normal
        else:
            # Stage B
            probs_b = predict_proba(model_b, text)
            if probs_b[1] > probs_b[0]:
                final_preds.append(2)  # extreme_up
            else:
                final_preds.append(0)  # extreme_down
    
    # 3-class confusion matrix 계산
    labels = val_df['label_z1.5'].values
    cm = confusion_matrix(labels, final_preds, labels=[0, 1, 2])
    
    # 클래스별 recall
    recall_down = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    recall_normal = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
    recall_up = cm[2, 2] / cm[2].sum() if cm[2].sum() > 0 else 0
    
    return {
        "cascade_recall_down": recall_down,
        "cascade_recall_normal": recall_normal,
        "cascade_recall_up": recall_up,
        "cascade_extreme_recall": (recall_down + recall_up) / 2,
        "cascade_min_extreme_recall": min(recall_down, recall_up),
        "confusion_matrix": cm,
    }
```

### 하이퍼파라미터

```yaml
# configs/training_config.yaml

stage_a:
  base_model: "snunlp/KR-FinBert-SC"
  num_labels: 2
  max_length: 256
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 32
  num_train_epochs: 8                      # 5 → 8 (천천히 학습)
  learning_rate: 1.0e-5                    # 2e-5 → 1e-5
  warmup_ratio: 0.15                       # 0.1 → 0.15
  weight_decay: 0.01
  fp16: true
  label_smoothing_factor: 0.1              # 신규
  eval_strategy: "epoch"
  save_strategy: "epoch"
  load_best_model_at_end: true
  metric_for_best_model: "f2_extreme"
  greater_is_better: true
  early_stopping_patience: 3               # 2 → 3
  seed: 42
  output_dir: "./models/kr-finbert-fx-stage-a"

stage_b:
  base_model: "snunlp/KR-FinBert-SC"
  num_labels: 2
  max_length: 256
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 32
  num_train_epochs: 8
  learning_rate: 1.0e-5
  warmup_ratio: 0.15
  weight_decay: 0.01
  fp16: true
  label_smoothing_factor: 0.1
  eval_strategy: "epoch"
  save_strategy: "epoch"
  load_best_model_at_end: true
  metric_for_best_model: "balanced_accuracy"  # 양방향 균형 핵심
  greater_is_better: true
  early_stopping_patience: 3
  seed: 42
  output_dir: "./models/kr-finbert-fx-stage-b"

focal_loss:
  gamma: 2.0
  alpha_method: "sqrt_balanced"

input:
  max_body_len: 400
  use_title_body_concat: true
```

### 출력
- `models/kr-finbert-fx-stage-a/` (Stage A 모델)
- `models/kr-finbert-fx-stage-b/` (Stage B 모델)
- `results/training_metrics_stage_a.json`
- `results/training_metrics_stage_b.json`
- `results/cascade_evaluation.json` (통합 평가)
- `results/training_curves_stage_a.png`
- `results/training_curves_stage_b.png`

### 검증 체크

**Stage A** (extreme 검출):
- [ ] Validation **extreme_recall ≥ 0.5** (이전 0.4 기준 상향)
- [ ] **balanced_accuracy ≥ 0.6** (degenerate solution 방지)
- [ ] roc_auc ≥ 0.65
- [ ] Train-Val gap ≤ 0.15

**Stage B** (방향성 분류):
- [ ] Validation **balanced_accuracy ≥ 0.55** (random=0.5)
- [ ] **direction_balance ≤ 0.2** (한쪽 쏠림 없음)
- [ ] recall_down과 recall_up 모두 0.4 이상

**캐스케이드 통합**:
- [ ] **min_extreme_recall ≥ 0.3** (양방향 모두 어느 정도 잡음)
- [ ] cascade_extreme_recall ≥ 0.4
- [ ] 3-class confusion matrix에서 클래스 0과 2 혼동 비율 < 30%

### 학습 시간 예상
- Stage A: 약 12~18분 (14,824건 train, 8 epoch, early stop 가능)
- Stage B: 약 4~7분 (3,123건 train, 8 epoch, early stop 가능)
- **합계: 약 16~25분**

### Fallback 전략

만약 Stage A/B 모두 학습 후에도 검증 기준 미달:

**Plan A**: Stage B 데이터 부족이 원인일 수 있음 → z=1.0으로 확장
- z=1.0 기준 Stage B extreme 합 약 7,800건 (z=1.5의 2.5배)
- Stage A는 z=1.5 유지 (task 정의상 z=1.5가 적절)

**Plan B**: Focal Loss gamma 증가 (2.0 → 3.0)
- 더 어려운 샘플에 집중

**Plan C**: 단순 이진 분류로 회귀 (extreme_up vs 나머지)
- Stage B 폐기, extreme_down 정보 손실 감수

---

## Phase 5: 캐스케이드 모델 평가

### 목표
Test set에서 캐스케이드 통합 모델 성능 평가 + Random/Majority baseline 비교 + 단계별 진단

### 작업 내용
**스크립트**: `scripts/05_evaluate.py`

#### 1. 단계별 평가 (Stage A, Stage B 독립)

**Stage A 평가** (Test 전체):
- extreme_recall, extreme_precision, f2_extreme
- ROC-AUC (확률 기반)
- balanced_accuracy

**Stage B 평가** (Test extreme 샘플만):
- recall_down, recall_up
- balanced_accuracy
- direction_balance

#### 2. 캐스케이드 통합 평가 (Test 전체, 3-class)

```python
cascade_metrics = cascade_evaluate(test_df, model_a, model_b, threshold=0.5)
# 출력: 3x3 confusion matrix, 클래스별 recall/precision
```

#### 3. 임계값 sensitivity 분석

Stage A의 threshold(기본 0.5)를 변경하면 P/R trade-off 가능:

| Threshold | Stage A recall | Stage A precision | 의미 |
|---|---|---|---|
| 0.3 | 높음 | 낮음 | "의심스러우면 일단 통과" |
| **0.5** | 균형 | 균형 | **기본값** |
| 0.7 | 낮음 | 높음 | "확실한 것만" |

Test set에서 threshold 0.3, 0.4, 0.5, 0.6, 0.7로 각각 평가하여 최적값 탐색.

#### 4. Baseline 비교 (필수)

| Baseline | 설명 | 기대 성능 |
|---|---|---|
| Majority | 항상 normal 예측 | extreme_recall = 0 |
| Random uniform | 균등 무작위 | extreme_recall ≈ 0.33 |
| Random proportional | 라벨 분포 비례 | extreme_recall ≈ class ratio |

**우리 모델 목표**: Random uniform 대비 extreme_recall +0.1 이상 개선

#### 5. 시기별 분해 평가

Test set(2023.7~2026.4)을 시기별로 쪼개:
- 2023.7~2023.12 (강달러 정점)
- 2024.1~2024.12 (트럼프 당선기 포함)
- 2025.1~2026.4 (최근)

각 시기별 캐스케이드 성능 추적 → concept drift 진단.

#### 6. Error analysis

캐스케이드 오분류 케이스 50건 추출:
- Stage A에서 놓친 extreme (FN at A)
- Stage A 통과했지만 Stage B에서 방향 잘못 예측 (Stage B 실패)
- Stage A 잘못 통과 (FP at A): normal을 extreme으로

각 유형 분석으로 모델 약점 식별.

#### 7. 단어 의존성 진단 (Concept Drift Robustness)

```python
# Permutation Importance
def diagnose_word_dependency(model, text, target_words):
    original_pred = predict(text)
    for word in target_words:
        masked_text = text.replace(word, "[MASK]")
        masked_pred = predict(masked_text)
        impact = abs(original_pred - masked_pred)
```

- 고유명사 (서브프라임, 코로나 등) 마스킹 시 예측 변화
- 추상 어휘 (위기, 급등, 불안) 마스킹 시 예측 변화
- 추상 어휘 영향 > 고유명사 영향 → 일반화 가능 패턴 학습 입증

### 출력
- `results/evaluation_report.md` (메인 리포트)
- `results/cascade_confusion_matrix.png`
- `results/stage_a_metrics.json`, `stage_b_metrics.json`
- `results/cascade_metrics.json`
- `results/threshold_sensitivity.csv`
- `results/error_analysis_50.csv`
- `results/baseline_comparison.csv`
- `results/word_dependency_diagnosis.md`

### 검증 체크
- [ ] **Stage A: extreme_recall ≥ 0.5**, balanced_accuracy ≥ 0.6
- [ ] **Stage B: balanced_accuracy ≥ 0.55**, direction_balance ≤ 0.2
- [ ] **캐스케이드 min_extreme_recall ≥ 0.3**
- [ ] Random baseline 대비 extreme_recall +0.1 이상 개선
- [ ] threshold sensitivity 분석 완료
- [ ] Error analysis 50건 검토
- [ ] 시기별 분해 평가 완료
- [ ] 단어 의존성 진단 완료

---

## Phase 6: 11만 건 전체 추론

### 목표
파인튜닝 모델로 11만 건 전체 뉴스의 변동 regime 신호 추출 → 다운스트림 입력 준비

### 입력
- `data/raw/fx_news_raw.csv` (11만 건, 정제 전)
- Phase 5에서 선택된 더 우수한 모델

### 작업 내용
**스크립트**: `scripts/06_inference_full.py`

#### 1. 전체 데이터 정제

Phase 1의 Step 1, Step 2만 적용 (저작권 제거 + 형식 필터). 불용어 밀도 필터(Step 3)는 미적용 권장 — 다운스트림에서 모든 기사 활용 가능하게.

#### 2. 캐스케이드 추론

```python
from torch.nn.functional import softmax

def cascade_inference(texts, model_a, model_b, tokenizer, batch_size=64, threshold=0.5):
    """
    11만 건에 대한 캐스케이드 추론.
    Stage A 확률, Stage B 확률, 최종 라벨, 신호 모두 저장.
    """
    model_a.eval(); model_b.eval()
    
    all_probs_a = []
    all_probs_b = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch = tokenizer(batch_texts, truncation=True, max_length=256,
                         padding=True, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            # Stage A: extreme vs normal
            probs_a = softmax(model_a(**batch).logits, dim=-1).cpu().numpy()
            # Stage B: extreme_down vs extreme_up (모든 샘플에 추론, 사용은 조건부)
            probs_b = softmax(model_b(**batch).logits, dim=-1).cpu().numpy()
        
        all_probs_a.append(probs_a)
        all_probs_b.append(probs_b)
    
    return np.vstack(all_probs_a), np.vstack(all_probs_b)
```

**중요**: Stage B는 **모든 샘플에 추론**하되, 다운스트림에서 활용 시 Stage A의 extreme 확률로 가중.

#### 3. 결과 저장 (다운스트림 LSTM 입력용)

```python
# Stage A 출력
df['prob_normal'] = probs_a[:, 0]
df['prob_extreme'] = probs_a[:, 1]

# Stage B 출력 (조건부 의미 있음)
df['prob_dir_down'] = probs_b[:, 0]
df['prob_dir_up'] = probs_b[:, 1]

# 캐스케이드 통합 신호 (LSTM에 직접 사용 가능)
df['extreme_signal'] = probs_a[:, 1]                    # 0~1, 극단 가능성
df['directional_signal'] = (probs_a[:, 1] *             # 가중 방향성: -1 ~ +1
                            (probs_b[:, 1] - probs_b[:, 0]))
df['signed_extreme_up'] = probs_a[:, 1] * probs_b[:, 1] # 극단 상승 신호
df['signed_extreme_down'] = probs_a[:, 1] * probs_b[:, 0] # 극단 하락 신호

# 최종 라벨 (캐스케이드 추론)
def assign_final_label(p_extreme, p_down, p_up, threshold=0.5):
    if p_extreme < threshold:
        return 1  # normal
    return 2 if p_up > p_down else 0  # extreme_up or extreme_down

df['predicted_label'] = df.apply(
    lambda r: assign_final_label(r['prob_extreme'], r['prob_dir_down'], r['prob_dir_up']),
    axis=1
)
```

**다운스트림 LSTM에 권장하는 핵심 feature**:
- `extreme_signal` (0~1): 극단 가능성의 강도
- `directional_signal` (-1~+1): 가중 방향 (Stage A 확률로 약화)
- `signed_extreme_up` (0~1): 극단 상승만의 신호 (early warning 핵심)
```

`extreme_signal`과 `directional_signal`은 다운스트림 LSTM에서 직접 사용 가능한 형태.

#### 4. 일별 집계

```python
daily_signals = df.groupby('date').agg({
    'prob_extreme_down': 'mean',
    'prob_normal': 'mean',
    'prob_extreme_up': 'mean',
    'extreme_signal': ['mean', 'max'],
    'directional_signal': 'mean',
    'article_id': 'count',  # 일별 기사 수
})
```

### 출력
- `data/inference/all_articles_signals.csv`
- `data/inference/daily_signals.csv`
- `data/inference/inference_stats.md`

### 검증 체크
- [ ] 11만 건 모두 추론 완료
- [ ] 일별 집계 데이터 형식 검증
- [ ] 일별 신호 시계열 시각화 (실제 환율 변동성과 시각적 상관 확인)
- [ ] 학습에 없던 기간(2026)의 추론도 합리적인지 spot-check

---

## 진행 체크리스트 (요약)

- [x] **Phase 0**: 불용어 사전 모듈화 (✅ 완료)
- [x] **Phase 1**: 뉴스 필터링 + FDR 환율 수집 + merge_asof 매핑 + 20,000건 샘플링 (✅ 완료)
- [x] **Phase 2**: z-score 기반 라벨링 (z=1.0, z=1.5) (✅ 완료)
- [x] **Phase 3**: 시간순 train/val/test 분할 (✅ 완료)
- [ ] **Phase 4**: 2단계 캐스케이드 학습 (Stage A: extreme 검출 + Stage B: 방향 분류)
- [ ] **Phase 5**: 캐스케이드 통합 평가 + baseline 비교 + 단어 의존성 진단
- [ ] **Phase 6**: 11만 건 캐스케이드 추론 + 일별 집계

---

## 위험 요소 및 대응

| 위험 | 대응책 |
|---|---|
| FDR USD/KRW 데이터 수집 실패 | 한국은행 ECOS API 또는 Investing.com 백업, 캐시 파일 보관 |
| FDR 동일 날짜 중복 행 (서버 버그) | `duplicated(keep='last')`로 처리 (첨부 코드 참고) |
| **merge_asof dtype 불일치 에러** | **`news_date`와 `trade_date` 모두 `datetime64[ns]`로 명시 변환** |
| **불용어 밀도 고정 임계값으로 과도 제거** | **DENSITY_PERCENTILE=90 동적 임계값 사용** (정상 기사도 매크로 단어 매칭 흔함) |
| 주말/공휴일 발행 기사 매핑 오류 | merge_asof `direction='backward'`로 직전 영업일 자동 매핑 |
| z=1.5 학습 안정성 (Stage 1 미달) | 2단계 캐스케이드 + Focal Loss로 전환 (현재 채택) |
| 20,000건 샘플링 시 특정 월 결손 | 시간 균등 샘플링 시 가용 기사 수에 자동 적응, 연도별 분포 확인 |
| **3-class 학습 시 한쪽 쏠림 (degenerate solution)** | **2단계 캐스케이드 학습으로 Stage A/B 난이도 분리** (이전 학습 실패 → 캐스케이드 전환) |
| **Stage A degenerate (모두 normal 또는 모두 extreme)** | Focal Loss gamma 조정 (2.0→3.0), label smoothing, sqrt class weight |
| **Stage B degenerate (한쪽 방향)** | direction_balance 모니터링, balanced_accuracy를 metric_for_best_model로 사용 |
| **Stage B 데이터 부족** | z=1.0으로 확장하여 extreme 샘플 2.5배 증대 (fallback 옵션) |
| 시기별 성능 편차 큼 | 학습 데이터에 모든 변동성 regime 포함 확인, 롤링 윈도우 크기 조정 |
| 같은 날 발행 기사 동일 라벨 문제 | label_quality_report에 명시, reviewer 사전 대응 |
| 사후 보도 기사 편향 | T+1~T+5 평균 사용으로 영향 완화, Error analysis에서 비율 보고 |
| Train-Val 성능 차이 큼 | early stopping, dropout 증가 |
| 4070 Ti VRAM OOM | batch_size 8 축소, gradient_accumulation 2 |

---

## 학술적 방어 포인트

본 연구가 받을 수 있는 비판에 대한 답변 준비:

### 비판 1: "타겟 변수에서 라벨을 추출하면 순환 논리 아닌가"

**답변**:
- 본 연구는 **"감성 → 환율 예측"** 인과 가설이 아닌 **"극단 변동 regime의 공출현 언어 패턴"** 가설 검증
- 텍스트 패턴이 변동 regime과 동시에 나타나는 통계적 관계 식별 (contemporaneous association)
- 다운스트림에서 **시간적 분리** 적용 (T시점 신호 → T+1~T+5 변동)
- Volatility forecasting 및 regime classification 문헌의 표준 접근

### 비판 2: "라벨이 같은 날 모든 기사에 동일하게 부여되는데 의미 있나"

**답변**:
- 본 연구의 목적은 개별 기사 단위 인과 추론이 아닌 **집합적 언어 패턴 식별**
- 다운스트림에서 일별 집계 사용 → 개별 라벨 노이즈는 평균화
- label_quality_report에서 동일 라벨 비율 명시 보고

### 비판 3: "z-score 임계값 선택의 자의성"

**답변**:
- z=1.0, z=1.5 두 임계값으로 비교 실험
- 선택 근거를 학습 가능성과 다운스트림 활용도 기준으로 명시
- 롤링 윈도우 크기 민감도 분석 보고

### 비판 4: "사후 보도 기사가 라벨링되어 의미 없는 신호 학습"

**답변**:
- T+1~T+5 평균으로 단발성 사후 보도 영향 완화
- 본 연구의 가설은 "큰 사건 = 큰 변동" 공출현 패턴이므로, 사후 보도조차도 "큰 사건 시기 언어 패턴"의 일부로 해석 가능
- Error analysis에서 사후 보도성 기사 비율 보고
- (확장 가능) 발행 시각 정보 활용 시 외환시장 마감 이후 기사 제외 옵션 — 본 plan 범위 외

---

## Claude Code 작업 시 주의사항

1. **데이터 파일 git 커밋 금지** (`.gitignore`에 `data/` 포함)
2. **각 Phase 완료 후 결과 commit**
3. **랜덤 시드 모든 스크립트에서 고정** (재현성)
4. **중간 산출물 csv로 저장** (검수 가능)
5. **`감성`이라는 단어 사용 금지** — 반드시 `변동 regime`, `extreme movement` 등으로 표현
6. **모든 평가에서 Macro-F1보다 Extreme Recall / F-beta=2 우선 보고**

---

## 다음 단계 (본 plan 이후, 별도 작업)

이 plan 완료 후 별도 단계로 진행할 작업:

1. **다운스트림 LSTM 모델 설계**
   - Baseline: 환율 시계열만 (GARCH, ARIMA 비교)
   - Treatment: 환율 시계열 + 일별 BERT 신호
2. **Early warning 평가 프레임워크**
   - VaR (Value-at-Risk) 정확도
   - 극단 변동일 사전 경고 lead time
   - Precision/Recall on tail events
3. **Robustness 검증**
   - 외생 충격일 분리 평가
   - Out-of-distribution 테스트

본 plan에서는 다루지 않음. Phase 6의 일별 집계 파일이 다음 단계의 입력이 됨.
