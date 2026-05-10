# Claude Code 프롬프트: 프로젝트 초기 세팅

아래 프롬프트를 Claude Code에 그대로 복사하여 실행하세요.

---

## 프롬프트

```
환율 뉴스 감성 분석 모델 파인튜닝 프로젝트의 초기 세팅을 진행해줘.
plan.md를 먼저 읽고 전체 계획을 파악한 뒤, 다음 작업을 순차적으로 수행해줘.

## 1. uv 프로젝트 초기화

현재 디렉토리에서 Python 3.11 기반으로 uv 프로젝트를 초기화한다.

```bash
uv init --python 3.11
```

생성된 pyproject.toml의 메타데이터를 다음과 같이 수정한다:
- name: "fx-sentiment"
- version: "0.1.0"
- description: "환율 뉴스 감성 분석 모델 파인튜닝 (KR-FinBERT 기반)"
- requires-python: ">=3.11,<3.13"

## 2. 의존성 추가

다음 패키지들을 카테고리별로 분리하여 설치한다.

### 핵심 의존성 (uv add)

```bash
# 데이터 처리
uv add pandas numpy pyarrow

# ML / 딥러닝 (CUDA 12.1 가정, 4070 Ti 환경)
uv add torch --index https://download.pytorch.org/whl/cu121
uv add transformers datasets accelerate evaluate

# 평가 및 통계
uv add scikit-learn

# LLM 라벨링용 (Phase 3에서 GPT-4o-mini 사용)
uv add openai python-dotenv

# 한국어 형태소 분석 (불용어 사전 호환)
uv add konlpy

# 진행 상황 표시
uv add tqdm

# 시각화 (필터링 리포트, 학습 곡선용)
uv add matplotlib seaborn
```

### 개발 의존성 (uv add --dev)

```bash
uv add --dev jupyter ipykernel ruff pytest
```

### CUDA 환경 확인 메모

torch 설치 시 CUDA 12.1 wheel을 사용한다. 만약 사용자의 CUDA 버전이
다르면 (예: 11.8) 다음과 같이 변경하라고 안내해줘:
- CUDA 11.8: `https://download.pytorch.org/whl/cu118`
- CPU only: `https://download.pytorch.org/whl/cpu`

## 3. 디렉토리 구조 생성

plan.md에 정의된 디렉토리 구조를 그대로 생성한다.

```bash
mkdir -p data/raw data/sampled data/labeled data/splits data/inference
mkdir -p scripts utils models results configs logs
```

각 핵심 디렉토리에 .gitkeep 파일을 생성하여 빈 디렉토리도 git에서 추적되게 한다.

## 4. utils 패키지 초기화

`utils/__init__.py`를 생성한다 (빈 파일 OK).

`utils/stopwords.py`와 `utils/text_cleaning.py`는 Phase 0에서 작성할
예정이므로 이 단계에서는 생성하지 않는다.

## 5. .gitignore 작성

다음 내용으로 `.gitignore`를 생성한다:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
env/

# uv
.uv-cache/

# IDE
.vscode/
.idea/
*.swp

# Jupyter
.ipynb_checkpoints/

# 환경변수 (API 키)
.env
.env.local

# 데이터 (대용량/민감)
data/raw/
data/sampled/
data/labeled/
data/splits/
data/inference/
*.csv
*.parquet
!data/.gitkeep
!**/.gitkeep

# 모델 체크포인트
models/
*.safetensors
*.bin
*.pt
*.pth

# 로그
logs/
*.log
wandb/
runs/

# 결과물 (선택적 추적)
results/*.png
results/*.json
!results/*.md

# OS
.DS_Store
Thumbs.db
```

## 6. .env.example 작성

API 키 관리용 템플릿을 작성한다:

```
# OpenAI API (Phase 3 LLM 라벨링용 - GPT-4o-mini)
OPENAI_API_KEY=your_api_key_here

# (선택) Weights & Biases - 학습 모니터링용
WANDB_API_KEY=your_wandb_key_here
WANDB_PROJECT=fx-sentiment
```

`.env`는 .gitignore에 포함되어 있으므로 사용자가 직접 복사하여 키를 넣어야
함을 README에 명시한다.

## 7. configs/ 초기 파일

### configs/training_config.yaml

```yaml
# Phase 6 파인튜닝 하이퍼파라미터
model:
  base_model: "DataWizardd/finbert-sentiment-ko"
  num_labels: 3

training:
  max_length: 256
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 32
  num_train_epochs: 4
  learning_rate: 2.0e-5
  warmup_ratio: 0.1
  weight_decay: 0.01
  fp16: true
  eval_strategy: "epoch"
  save_strategy: "epoch"
  load_best_model_at_end: true
  metric_for_best_model: "f1_macro"
  greater_is_better: true
  early_stopping_patience: 2
  seed: 42

input:
  # 하이브리드 입력: title [SEP] body[:max_body_len]
  max_body_len: 400
  use_title_body_concat: true

output:
  output_dir: "./models/kr-finbert-fx-final"
  logging_dir: "./logs/training"
  save_total_limit: 2
```

### configs/labeling_prompt.txt

GPT-4o-mini의 Structured Outputs 방식에 맞춰 system/user를 분리하여 저장한다.

```
=== SYSTEM ===
당신은 한국 외환시장 전문 애널리스트입니다.
뉴스 기사가 원/달러 환율에 미칠 영향을 판단하여 분류하세요.

분류 기준:
- 긍정 (0): 원화가치 상승 / 환율 하락 요인
  예) 한국 무역흑자 확대, 외국인 자금 유입, 미 연준 완화 시그널
- 중립 (1): 환율 방향성 불분명 / 단순 시황 보도
  예) 단순 시세 전달, 혼재된 신호, 영향 미미
- 부정 (2): 원화가치 하락 / 환율 상승 요인
  예) 미 금리인상, 한국 경상수지 악화, 지정학 리스크

주의사항:
- 단순 환율 시황 보도("오늘 환율 ~원 마감")는 중립
- 인용문의 화자 입장이 아닌, 사건 자체의 방향성으로 판단
- 영향이 애매하면 중립

=== USER ===
기사 제목: {title}
기사 본문: {body}
```

JSON 형식 강제는 코드 내에서 `response_format` 파라미터(JSON Schema)로 처리하므로
프롬프트 텍스트에 JSON 출력 지시를 명시할 필요 없음.

## 8. README.md 작성

프로젝트 루트에 README.md를 생성한다. 다음 내용을 포함:

- 프로젝트 개요 (plan.md에서 발췌)
- 환경 설정 방법
  - uv sync 명령
  - .env 설정
  - GPU 환경 (CUDA 12.1 권장)
- 디렉토리 구조 설명
- Phase별 실행 순서 (plan.md 참조 안내)
- 주요 파일 설명

## 9. ruff 설정

pyproject.toml에 ruff 설정 추가:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "C4", "SIM"]
ignore = ["E501"]  # line too long (line-length으로 제어)

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

## 10. 검증 및 마무리

마지막으로 다음을 수행:

```bash
# 의존성 설치 확인
uv sync

# 환경 작동 확인
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

uv run python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

uv run python -c "import openai; print(f'OpenAI SDK: {openai.__version__}')"
```

## 11. git 초기화 (이미 안 되어 있다면)

```bash
git init
git add .
git status  # 사용자가 추적될 파일 확인하도록
```

처음 커밋은 사용자가 직접 하도록 안내만 해줘. 자동 커밋 금지.

---

## 작업 완료 후 보고할 것

1. 생성된 디렉토리 구조 (`tree -L 2 -a`)
2. uv sync 결과 (성공/실패)
3. CUDA 인식 여부 (torch.cuda.is_available)
4. 다음 단계 안내:
   - 사용자에게 .env 파일 생성 (.env.example 복사) 요청
   - data/raw/fx_news_raw.csv 배치 위치 안내
   - "Phase 0 시작 준비 완료" 메시지

## 작업 시 주의사항

- 모든 파일 생성 후 내용을 보여주고 진행
- 의존성 설치 중 에러 발생 시 즉시 멈추고 사용자에게 보고
- pyproject.toml 수정 시 기존 내용 백업 후 진행
- 절대 데이터 파일을 만지지 않음 (사용자가 직접 배치할 예정)
- API 키 관련 파일을 절대 자동 커밋하지 않음
```

---

## 사용 방법

1. **이 파일을 프로젝트 루트에 배치**
   ```bash
   mkdir fx_sentiment && cd fx_sentiment
   # 위 plan.md도 같이 배치
   mv ~/Downloads/plan.md .
   mv ~/Downloads/setup_prompt.md .
   ```

2. **uv가 설치되어 있는지 확인**
   ```bash
   uv --version
   # 없다면: curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Claude Code 실행**
   ```bash
   claude
   ```

4. **위 프롬프트 영역(```으로 감싸인 부분)을 복사하여 Claude Code에 붙여넣기**

5. **세팅 완료 후 수동 작업**
   ```bash
   # API 키 설정
   cp .env.example .env
   # .env 파일을 열어 OPENAI_API_KEY 입력
   # OpenAI API 키는 https://platform.openai.com/api-keys 에서 발급
   
   # 원본 데이터 배치
   cp /path/to/fx_news_raw.csv data/raw/
   
   # Claude Code에서 다음 작업 시작
   # "plan.md의 Phase 0를 시작해줘. clean_dictionary.py를 utils/stopwords.py로 변환해줘"
   ```

---

## 트러블슈팅

### CUDA 버전이 12.1이 아닌 경우
프롬프트 수정 후 재실행:
- CUDA 11.8: `--index https://download.pytorch.org/whl/cu118`
- CPU only: `--index https://download.pytorch.org/whl/cpu`

### konlpy 설치 실패 (Java 의존성)
```bash
# Ubuntu/Debian
sudo apt-get install default-jdk
# macOS
brew install openjdk
```

설치 후 `uv add konlpy` 재시도.

### uv sync 시 오래 걸리는 경우
PyTorch는 약 2GB로 큰 패키지. 첫 설치 시 5~10분 소요 정상.
