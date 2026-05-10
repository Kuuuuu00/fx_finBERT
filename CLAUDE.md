# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

환율 뉴스 감성 분석 파인튜닝 프로젝트. KR-FinBERT(`DataWizardd/finbert-sentiment-ko`)를 기반으로 3-class 분류기(긍정=0, 중립=1, 부정=2)를 학습한 뒤, 11만 건 전체 뉴스에 감성 추론하여 LSTM 환율 예측 모델의 feature로 활용한다.

## Commands

```bash
# 의존성 설치
uv sync
uv sync --dev          # 개발 의존성 포함

# 스크립트 실행
uv run python scripts/01_sample_articles.py
uv run python clean_dictionary.py --prefix tone_dictionary --top-n 20

# 린트
uv run ruff check .
uv run ruff format .

# 테스트
uv run pytest
uv run pytest tests/test_stopwords.py  # 단일 파일
```

## PyTorch / CUDA 설치 주의사항

torch는 별도 인덱스(`download.pytorch.org/whl/cu121`)를 사용하며, `pyproject.toml`에 `index-strategy = "unsafe-best-match"` 가 설정되어 있다. **CUDA 버전이 다른 경우** `[[tool.uv.index]]` url의 `cu121`을 교체 후 `uv sync`:
- CUDA 11.8: `cu118`
- CPU only: `cpu`

## Architecture

### Phase 순서 (plan.md 참조)

| Phase | 스크립트 | 입력 | 출력 |
|---|---|---|---|
| 0 | _(수동)_ | `clean_dictionary.py` | `utils/stopwords.py`, `utils/text_cleaning.py` |
| 1 | `scripts/01_sample_articles.py` | `data/raw/fx_news_raw.csv` | `data/sampled/fx_news_5000.csv` |
| 2 | _(수동 라벨링)_ | `fx_news_5000.csv` 200건 | `data/labeled/gold_set_200.csv` |
| 3 | `scripts/02_llm_labeling.py` | `fx_news_5000.csv` | `data/labeled/llm_labeled_5000.csv` (GPT-4o-mini) |
| 4 | `scripts/03_iaa_evaluation.py` | gold_set + llm_labeled | `results/iaa_report.md` |
| 5 | `scripts/04_split_dataset.py` | llm_labeled | `data/splits/{train,val,test}.csv` |
| 6 | `scripts/05_finetune.py` | splits | `models/kr-finbert-fx-final/` |
| 7 | `scripts/06_inference_full.py` | 11만 건 전체 | `data/inference/daily_sentiment.csv` |

### 데이터 흐름

```
data/raw/fx_news_raw.csv (11만 건)
  └─ Phase 1 필터링 ──→ data/sampled/fx_news_filtered.csv (~8~9만 건)
       └─ 시간 균등 샘플링 ──→ data/sampled/fx_news_5000.csv (5,000건)
            ├─ Phase 2 수동 라벨링 ──→ data/labeled/gold_set_200.csv
            └─ Phase 3 LLM 라벨링 (GPT-4o-mini) ──→ data/labeled/llm_labeled_5000.csv
                 └─ Phase 4 IAA 검증 (Kappa ≥ 0.6 통과 시만 진행)
                      └─ Phase 5 시간순 분할 ──→ data/splits/{train,val,test}.csv
                           └─ Phase 6 파인튜닝 ──→ models/kr-finbert-fx-final/
                                └─ Phase 7 전체 추론 ──→ data/inference/daily_sentiment.csv
```

### utils 패키지

- `utils/stopwords.py`: `clean_dictionary.py`의 `STOPWORD_KEYWORDS`를 그대로 이식. 추가로 `count_stopwords_in_text()` / `stopword_density()` 함수를 제공하여 Phase 1 콘텐츠 필터링에 사용
- `utils/text_cleaning.py`: `clean_article_body()` (저작권/꼬리말 제거), `is_market_summary()` (시황 마감 보도 판별), `is_irrelevant_article()` (불용어 밀도 기반 무관 기사 판별)

### clean_dictionary.py (기존 도구)

`tone_dictionary_*w.csv` (N-gram 어조 사전)에서 불용어 포함 행을 제거하는 독립 스크립트. `STOPWORD_KEYWORDS`가 `utils/stopwords.py`의 원본 소스다. 처리 대상: `*_1w.csv`, `*_2w.csv`, `*_3w.csv` → `*_cleaned_*w.csv`.

### 모델 입력 형식

```python
f"{row['title']} [SEP] {row['body'][:400]}"  # max_length=256 토크나이즈
```

### 학습 설정 기준값 (RTX 4070 Ti 12GB)

`configs/training_config.yaml` 기준: `batch=16`, `fp16=True`, `lr=2e-5`, `epochs=4`, `early_stopping_patience=2`, `metric=f1_macro`

## Label Schema

| 값 | 의미 | 예시 |
|---|---|---|
| 0 | 긍정 (원화 강세 / 환율 하락 요인) | 무역흑자 확대, 외국인 자금 유입 |
| 1 | 중립 (방향성 불분명 / 시황 보도) | "오늘 환율 1,320원 마감" |
| 2 | 부정 (원화 약세 / 환율 상승 요인) | FOMC 금리인상, 달러 강세 |

## IAA 통과 기준

Phase 4에서 Cohen's Kappa < 0.6이면 Phase 5로 진행하지 않고 `configs/labeling_prompt.txt` 수정 후 Phase 3을 재실행한다.

## 환경 변수

`.env.example`을 복사해 `.env` 생성:
```bash
cp .env.example .env
# OPENAI_API_KEY 입력 (Phase 3 GPT-4o-mini 라벨링용)
```

Phase 3 라벨링은 OpenAI Structured Outputs (`response_format` JSON Schema)를 사용한다. 프롬프트는 `configs/labeling_prompt.txt`에서 system/user 블록을 `=== SYSTEM ===` / `=== USER ===` 구분자로 로드한다.
