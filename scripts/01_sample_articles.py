"""
Phase 1: 데이터 필터링 및 시간 균등 샘플링
Input:  data/raw/fx_news_raw.csv  (114,192건)
Output: data/sampled/fx_news_filtered.csv  — 콘텐츠 필터링 통과 전체
        data/sampled/fx_news_5000.csv      — 시간 균등 샘플 5,000건
        data/sampled/filtering_report.md   — 단계별 통계
"""

import hashlib
import logging
import math
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.stopwords import stopword_density
from utils.text_cleaning import clean_article_body

# ── 로거 ──────────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/phase1.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── 경로 ──────────────────────────────────────────────────────────────────────
RAW_PATH      = Path("data/raw/fx_news_raw.csv")
FILTERED_PATH = Path("data/sampled/fx_news_filtered.csv")
SAMPLED_PATH  = Path("data/sampled/fx_news_5000.csv")
REPORT_PATH   = Path("data/sampled/filtering_report.md")

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
TARGET_SAMPLE      = 5_000
MIN_BODY_LEN       = 100
MAX_BODY_LEN       = 3_000
MIN_TITLE_LEN      = 5
DENSITY_PERCENTILE = 90    # 상위 N% 고밀도 기사 제거 (plan.md 권장: 5~10%)
AD_THRESHOLD       = 3     # 광고 지시어 N개 이상이면 제거
RANDOM_STATE       = 42

_FX_KEYWORDS   = frozenset(["환율", "원/달러", "원달러", "달러", "원화", "외환", "외화"])
_AD_INDICATORS = ["바로가기", "구독신청", "이벤트", "회원가입"]


def time_balanced_sample(df: pd.DataFrame, target: int, random_state: int = 42) -> pd.DataFrame:
    """연-월 균등 샘플링. 부족한 시기의 쿼터는 후속 시기에서 동적으로 보충."""
    groups  = dict(list(df.groupby("ym")))
    ym_list = sorted(groups.keys())
    n       = len(ym_list)

    quotas: dict = {}
    remaining = target

    for i, ym in enumerate(ym_list):
        if remaining <= 0:
            quotas[ym] = 0
            continue
        groups_left = n - i
        quota = math.ceil(remaining / groups_left)
        take  = min(len(groups[ym]), quota)
        quotas[ym] = take
        remaining -= take

    parts = [
        groups[ym].sample(take, random_state=random_state)
        for ym, take in quotas.items()
        if take > 0
    ]
    return pd.concat(parts).sort_values("date").reset_index(drop=True)


def write_report(
    raw_n: int,
    step2_n: int,
    step3_n: int,
    step2_reasons: dict,
    step3_reasons: dict,
    filtered_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    density_threshold: float = 0.0,
) -> None:
    lines = [
        "# Phase 1 필터링 리포트",
        "",
        "## 단계별 잔존 건수",
        "",
        "| 단계 | 건수 | 잔존율 |",
        "|---|---:|---:|",
        f"| 원본 | {raw_n:,} | 100.0% |",
        f"| Step 2 형식 필터링 후 | {step2_n:,} | {step2_n/raw_n*100:.1f}% |",
        f"| Step 3 콘텐츠 필터링 후 | {step3_n:,} | {step3_n/raw_n*100:.1f}% |",
        f"| 최종 샘플 | {len(sampled_df):,} | - |",
        "",
        "## Step 2 제거 사유",
        "",
        "| 사유 | 건수 |",
        "|---|---:|",
    ]
    for reason, cnt in step2_reasons.items():
        if cnt > 0:
            lines.append(f"| {reason} | {cnt:,} |")

    lines += [
        "",
        f"density 임계값 ({DENSITY_PERCENTILE}th percentile): **{density_threshold:.1f}**",
        "",
        "## Step 3 제거 사유",
        "",
        "| 사유 | 건수 |",
        "|---|---:|",
    ]
    for reason, cnt in step3_reasons.items():
        lines.append(f"| {reason} | {cnt:,} |")

    # 연도별 분포
    raw_yr     = filtered_df.groupby("year").size().rename("filtered")
    sample_yr  = sampled_df.groupby("year").size().rename("sampled")
    year_df    = pd.concat([raw_yr, sample_yr], axis=1).fillna(0).astype(int)

    lines += [
        "",
        "## 연도별 분포",
        "",
        "| 연도 | 필터링 통과 | 샘플링 |",
        "|---|---:|---:|",
    ]
    for yr, row in year_df.iterrows():
        lines.append(f"| {yr} | {row['filtered']:,} | {row['sampled']:,} |")

    # 본문 길이
    q = sampled_df["body_len"]
    lines += [
        "",
        "## 샘플 본문 길이 분포",
        "",
        f"- 평균: {q.mean():.0f}자",
        f"- 중앙값: {q.median():.0f}자",
        f"- 25%: {q.quantile(0.25):.0f}자",
        f"- 75%: {q.quantile(0.75):.0f}자",
        f"- 최소: {q.min():.0f}자 / 최대: {q.max():.0f}자",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log.info("리포트 저장: %s", REPORT_PATH)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("Phase 1 시작")

    # ── 로드 ───────────────────────────────────────────────────────────────────
    df = pd.read_csv(RAW_PATH, parse_dates=["publish_date"])
    raw_n = len(df)
    log.info("원본 로드: %d건", raw_n)

    df = df.rename(columns={"text": "body", "publish_date": "date"})
    df["article_id"] = range(len(df))
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["ym"]    = df["date"].dt.to_period("M")

    # ── Step 1: 본문 정제 (기사 제거 없음) ────────────────────────────────────
    tqdm.pandas(desc="[Step 1] 본문 정제")
    df["body_raw"] = df["body"]
    df["body"]     = df["body"].progress_apply(clean_article_body)
    df["body_len"] = df["body"].str.len()
    df["title_len"] = df["title"].str.len()
    log.info("Step 1 완료")

    # ── Step 2: 형식 기반 필터링 ───────────────────────────────────────────────
    step2_reasons: dict[str, int] = {}

    null_m   = df["title"].isna() | df["body"].isna()
    short_m  = df["body_len"] < MIN_BODY_LEN
    long_m   = df["body_len"] > MAX_BODY_LEN
    stitle_m = df["title_len"] < MIN_TITLE_LEN

    step2_reasons["null"]              = int(null_m.sum())
    step2_reasons["body_too_short"]    = int(short_m.sum())
    step2_reasons["body_too_long"]     = int(long_m.sum())
    step2_reasons["title_too_short"]   = int(stitle_m.sum())

    fmt_mask = ~(null_m | short_m | long_m | stitle_m)

    # 중복: 형식 통과 건에 한해서만 hash 계산
    df["_hash"] = (df["title"].astype(str) + df["body"].str[:100]).apply(
        lambda s: hashlib.md5(s.encode()).hexdigest()
    )
    dup_m = df["_hash"].duplicated(keep="first") & fmt_mask
    step2_reasons["duplicate"] = int(dup_m.sum())

    df = df[fmt_mask & ~dup_m].drop(columns=["_hash"]).reset_index(drop=True)
    step2_n = len(df)
    log.info(
        "Step 2 완료: %d건 (제거 %d건, %.1f%%)",
        step2_n, raw_n - step2_n, (raw_n - step2_n) / raw_n * 100,
    )

    # ── Step 3: 콘텐츠 기반 필터링 ────────────────────────────────────────────
    step3_reasons: dict[str, int] = {}

    # 3-a: 환율 키워드 부재
    text_ser = df["title"] + " " + df["body"]
    has_fx = text_ser.apply(lambda t: any(kw in t for kw in _FX_KEYWORDS))
    step3_reasons["no_fx_keyword"] = int((~has_fx).sum())
    df = df[has_fx].reset_index(drop=True)

    # 3-b: 불용어 밀도 — 90th percentile 자동 임계값 (plan.md: 상위 5~10% 컷)
    tqdm.pandas(desc="[Step 3b] density 계산")
    df["_density"] = df["body"].progress_apply(stopword_density)
    density_threshold = float(df["_density"].quantile(DENSITY_PERCENTILE / 100))
    log.info("density %d%%ile 임계값: %.1f", DENSITY_PERCENTILE, density_threshold)

    high_d = df["_density"] > density_threshold
    step3_reasons["high_stopword_density"] = int(high_d.sum())
    df = df[~high_d].drop(columns=["_density"]).reset_index(drop=True)

    # 3-c: 광고 지시어
    ad_mask = (df["title"] + df["body"]).apply(
        lambda t: sum(t.count(w) for w in _AD_INDICATORS) >= AD_THRESHOLD
    )
    step3_reasons["advertisement"] = int(ad_mask.sum())
    df = df[~ad_mask].reset_index(drop=True)

    step3_n = len(df)
    log.info(
        "Step 3 완료: %d건 (제거 %d건, %.1f%%)",
        step3_n, step2_n - step3_n, (step2_n - step3_n) / step2_n * 100,
    )
    log.info("  제거 사유: %s", step3_reasons)

    # ── 필터링 전체 결과 저장 ──────────────────────────────────────────────────
    Path("data/sampled").mkdir(parents=True, exist_ok=True)
    filtered_cols = [
        "article_id", "date", "year", "month", "ym",
        "title", "body", "body_raw", "body_len", "title_len", "press",
    ]
    df[filtered_cols].to_csv(FILTERED_PATH, index=False, encoding="utf-8-sig")
    log.info("저장: %s  (%d건)", FILTERED_PATH, len(df))

    # ── Step 4: 시간 균등 샘플링 ──────────────────────────────────────────────
    sampled = time_balanced_sample(df, TARGET_SAMPLE, RANDOM_STATE)
    log.info("Step 4 완료: %d건 시간 균등 샘플링", len(sampled))

    sample_cols = [
        "article_id", "date", "year", "month",
        "title", "body", "body_raw", "body_len", "title_len",
    ]
    sampled[sample_cols].to_csv(SAMPLED_PATH, index=False, encoding="utf-8-sig")
    log.info("저장: %s  (%d건)", SAMPLED_PATH, len(sampled))

    # ── 리포트 ────────────────────────────────────────────────────────────────
    write_report(raw_n, step2_n, step3_n, step2_reasons, step3_reasons, df, sampled,
                 density_threshold)

    log.info("=" * 60)
    log.info("Phase 1 완료  |  원본 %d → 필터링 %d → 샘플 %d", raw_n, step3_n, len(sampled))


if __name__ == "__main__":
    main()
