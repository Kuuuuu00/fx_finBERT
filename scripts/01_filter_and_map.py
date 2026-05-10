"""
Phase 1: 뉴스 필터링 + FDR 환율 수집 + merge_asof 매핑 + 시간 균등 샘플링

Track A (Step 1~3):  뉴스 본문 정제 및 콘텐츠 필터링
Track B (Step 4~6):  FinanceDataReader USD/KRW 수집 → 영업일 테이블 (T-1, T, T+1~T+5)
Step 7:              merge_asof(direction='backward')로 뉴스↔환율 매핑
Step 8:              시간 균등 샘플링 20,000건

Inputs
------
data/raw/fx_news_raw.csv   (114,192건 / columns: title, press, publish_date, url, text)

Outputs
-------
data/sampled/market_table.csv         — 영업일 환율 테이블 (Track B)
data/sampled/fx_news_filtered.csv     — Track A + Step 7 결과 (수만~10만 건)
data/sampled/fx_news_20000.csv        — 최종 시간 균등 샘플
data/sampled/filtering_report.md      — 단계별 통계
"""

from __future__ import annotations

import hashlib
import logging
import math
import sys
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.stopwords import stopword_density  # noqa: E402
from utils.text_cleaning import clean_article_body, is_irrelevant_article  # noqa: E402

# NOTE: utils.text_cleaning.is_market_summary 는 plan.md Phase 0 지침에 따라
# 본 Phase 1에서 호출하지 않는다. 큰 변동일에는 시황 보도 표현도 달라지므로
# 시황 보도 필터링은 의도적으로 적용하지 않는다.

# ── 로거 ──────────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/phase1.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── 경로 ──────────────────────────────────────────────────────────────────────
RAW_PATH       = Path("data/raw/fx_news_raw.csv")
MARKET_PATH    = Path("data/sampled/market_table.csv")
FILTERED_PATH  = Path("data/sampled/fx_news_filtered.csv")
SAMPLED_PATH   = Path("data/sampled/fx_news_20000.csv")
REPORT_PATH    = Path("data/sampled/filtering_report.md")

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
TARGET_SAMPLE      = 20_000
MIN_BODY_LEN       = 100
MAX_BODY_LEN       = 3_000
MIN_TITLE_LEN      = 5
# plan.md: 고정 임계값 대신 percentile 기반 동적 임계값 사용 (분포 기반 robust 컷)
DENSITY_PERCENTILE = 90     # 상위 10% 노이즈 기사 제거
RANDOM_STATE       = 42

FX_START   = "2004-05-01"
FX_END     = "2026-04-30"
ROLL_WINDOW   = 90
ROLL_MIN_OBS  = 60


# ══════════════════════════════════════════════════════════════════════════════
# Track B — Step 4: USD/KRW 환율 수집
# ══════════════════════════════════════════════════════════════════════════════

def fetch_exchange_rate(start: str = FX_START, end: str = FX_END) -> pd.DataFrame:
    log.info("[Track B / Step 4] FDR USD/KRW 수집  (%s ~ %s)", start, end)
    raw = fdr.DataReader("USD/KRW", start, end)
    fx = raw[["Close"]].copy()
    fx.index = pd.to_datetime(fx.index).normalize()
    fx.index.name = "trade_date"
    fx = fx.sort_index().dropna(subset=["Close"])

    # FDR 동일 날짜 중복 행 (서버 버그) 방어
    n_before = len(fx)
    fx = fx[~fx.index.duplicated(keep="last")]
    n_dup = n_before - len(fx)
    if n_dup:
        log.warning("  FDR 중복 행 %d건 제거 (keep=last)", n_dup)

    log.info(
        "  영업일 %d일 | %s ~ %s | 종가 %.2f ~ %.2f",
        len(fx), fx.index.min().date(), fx.index.max().date(),
        fx["Close"].min(), fx["Close"].max(),
    )
    return fx


# ══════════════════════════════════════════════════════════════════════════════
# Track B — Step 5+6: 영업일 테이블 (T-1, T, T+1~T+5) + 일일 변동성
# ══════════════════════════════════════════════════════════════════════════════

def build_market_table(fx: pd.DataFrame) -> pd.DataFrame:
    log.info("[Track B / Step 5-6] 영업일 테이블 + rolling_std_90d 산출")
    market = fx.copy().rename(columns={"Close": "close"})

    # T-1 / T+1
    market["prev_close"] = market["close"].shift(1)
    market["next_close"] = market["close"].shift(-1)

    # T+1 ~ T+5 개별 종가
    for k in range(1, 6):
        market[f"close_T{k}"] = market["close"].shift(-k)

    # T+1 ~ T+5 평균 (Phase 2 라벨링 핵심 입력)
    market["close_T1_T5_mean"] = sum(
        market["close"].shift(-k) for k in range(1, 6)
    ) / 5

    # 다중 horizon 평균 (참고용)
    market["avg_2w"] = sum(market["close"].shift(-k) for k in range(1, 11)) / 10
    market["avg_3w"] = sum(market["close"].shift(-k) for k in range(1, 21)) / 20

    # Step 6: 일일 변동성 시계열 (Phase 2에서 그대로 사용)
    market["daily_change"]    = market["close"].diff()
    market["rolling_std_90d"] = (
        market["daily_change"].rolling(window=ROLL_WINDOW, min_periods=ROLL_MIN_OBS).std()
    )

    # 첫 행(prev_close NaN) · 마지막 20행(avg_3w NaN) 제거
    market = market.dropna(subset=["prev_close", "next_close", "avg_3w"])
    log.info("  영업일 테이블: %d행 (양끝 결측 제거 후)", len(market))
    return market.reset_index()


# ══════════════════════════════════════════════════════════════════════════════
# Step 7: 뉴스 ↔ 환율 매핑 (merge_asof, direction='backward')
# ══════════════════════════════════════════════════════════════════════════════

def map_news_to_market(news: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    log.info("[Step 7] merge_asof 매핑 (direction='backward')")
    # merge_asof 키 dtype 통일 (plan.md 검증 체크)
    news_sorted = news.copy()
    news_sorted["news_date"] = pd.to_datetime(news_sorted["news_date"]).astype("datetime64[ns]")
    news_sorted = news_sorted.sort_values("news_date").reset_index(drop=True)

    market_sorted = market.copy()
    market_sorted["trade_date"] = pd.to_datetime(market_sorted["trade_date"]).astype("datetime64[ns]")
    market_sorted = market_sorted.sort_values("trade_date")

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
        direction="backward",
    )

    n_total      = len(merged)
    n_no_trade   = merged["trade_date"].isna().sum()
    n_no_close5  = merged["close_T1_T5_mean"].isna().sum()
    log.info(
        "  매핑 전 %d건 | trade_date NaN %d건 | close_T1_T5_mean NaN %d건",
        n_total, n_no_trade, n_no_close5,
    )

    # close_T1_T5_mean NaN 행 = 환율 데이터 종료일 - 5영업일 이후 발행 기사 → 제거
    merged = merged.dropna(subset=["trade_date", "close_T1_T5_mean"]).reset_index(drop=True)
    log.info("  매핑 후 %d건 (성공률 %.2f%%)", len(merged), len(merged) / n_total * 100)
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Step 8: 시간 균등 샘플링
# ══════════════════════════════════════════════════════════════════════════════

def time_balanced_sample(
    df: pd.DataFrame, target: int, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """연-월 단위 균등 샘플링. 부족한 시기의 쿼터는 후속 시기에서 동적 보충."""
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
    return pd.concat(parts).sort_values("news_date").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 리포트 작성
# ══════════════════════════════════════════════════════════════════════════════

def write_report(
    raw_n: int,
    step2_n: int,
    step3_n: int,
    mapped_n: int,
    step2_reasons: dict,
    step3_reasons: dict,
    market_n: int,
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
        f"| Step 7 환율 매핑 후 | {mapped_n:,} | {mapped_n/raw_n*100:.1f}% |",
        f"| Step 8 시간 균등 샘플 | {len(sampled_df):,} | - |",
        "",
        f"환율 영업일 테이블: **{market_n:,}일**  (FDR USD/KRW {FX_START} ~ {FX_END})",
        "",
        "## Step 2 제거 사유 (형식 기반)",
        "",
        "| 사유 | 건수 |",
        "|---|---:|",
    ]
    for reason, cnt in step2_reasons.items():
        if cnt > 0:
            lines.append(f"| {reason} | {cnt:,} |")

    lines += [
        "",
        f"불용어 밀도: DENSITY_PERCENTILE={DENSITY_PERCENTILE} → 적용 임계값 **{density_threshold:.2f}** (분포 기반)",
        "",
        "## Step 3 제거 사유 (콘텐츠 기반)",
        "",
        "| 사유 | 건수 |",
        "|---|---:|",
    ]
    for reason, cnt in step3_reasons.items():
        lines.append(f"| {reason} | {cnt:,} |")

    # 연도별 분포
    sample_yr = sampled_df.groupby(sampled_df["news_date"].dt.year).size()
    lines += [
        "",
        "## Step 8 샘플의 연도별 분포",
        "",
        "| 연도 | 샘플 수 |",
        "|---|---:|",
    ]
    for yr, n in sample_yr.items():
        lines.append(f"| {yr} | {n:,} |")

    # 본문 길이
    body_len = sampled_df["body"].str.len()
    lines += [
        "",
        "## 샘플 본문 길이 분포",
        "",
        f"- 평균: {body_len.mean():.0f}자",
        f"- 중앙값: {body_len.median():.0f}자",
        f"- 25%: {body_len.quantile(0.25):.0f}자 / 75%: {body_len.quantile(0.75):.0f}자",
        f"- 최소: {body_len.min():.0f}자 / 최대: {body_len.max():.0f}자",
        "",
        "## 주말 발행 기사 매핑 spot-check (앞 5건)",
        "",
        "| news_date (요일) | trade_date (요일) |",
        "|---|---|",
    ]
    weekend = sampled_df[sampled_df["news_date"].dt.dayofweek.isin([5, 6])].head(5)
    for _, row in weekend.iterrows():
        nd = row["news_date"]
        td = row["trade_date"]
        lines.append(
            f"| {nd.strftime('%Y-%m-%d')} ({nd.strftime('%a')}) | "
            f"{td.strftime('%Y-%m-%d')} ({td.strftime('%a')}) |"
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log.info("리포트 저장: %s", REPORT_PATH)


def update_report_step8(sampled_df: pd.DataFrame, market_n: int) -> None:
    """캐시 사용 시 기존 리포트에 Step 8 재실행 결과를 덧붙임."""
    existing = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else ""
    marker = "\n## Step 8 재실행"
    if marker in existing:
        existing = existing[:existing.index(marker)]

    sample_yr = sampled_df.groupby(sampled_df["news_date"].dt.year).size()
    body_len = sampled_df["body"].str.len()

    lines = [
        "",
        f"## Step 8 재실행 ({TARGET_SAMPLE:,}건 샘플링 — 캐시 사용)",
        "",
        f"샘플링 규모 변경: 5,000건 → {TARGET_SAMPLE:,}건",
        "변경 사유: z=1.5 Val/Test extreme 클래스 표본 수 확보 (교수님 미팅, 2026-05-08)",
        "",
        f"환율 영업일 테이블: **{market_n:,}일**  (캐시 재사용)",
        "",
        "### Step 8 샘플의 연도별 분포",
        "",
        "| 연도 | 샘플 수 |",
        "|---|---:|",
    ]
    for yr, n in sample_yr.items():
        lines.append(f"| {yr} | {n:,} |")

    lines += [
        "",
        "### 샘플 본문 길이 분포",
        "",
        f"- 평균: {body_len.mean():.0f}자",
        f"- 중앙값: {body_len.median():.0f}자",
        f"- 25%: {body_len.quantile(0.25):.0f}자 / 75%: {body_len.quantile(0.75):.0f}자",
        f"- 최소: {body_len.min():.0f}자 / 최대: {body_len.max():.0f}자",
    ]

    REPORT_PATH.write_text(existing + "\n".join(lines), encoding="utf-8")
    log.info("리포트 업데이트: %s", REPORT_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# 메인 파이프라인
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=" * 64)
    log.info("Phase 1 시작 (filter + FDR map + sample)")
    log.info("=" * 64)

    Path("data/sampled").mkdir(parents=True, exist_ok=True)

    out_cols = [
        "article_id", "publish_date", "news_date", "trade_date",
        "title", "body", "body_raw", "press", "url",
        "close", "prev_close", "next_close",
        "close_T1", "close_T2", "close_T3", "close_T4", "close_T5",
        "close_T1_T5_mean", "daily_change", "rolling_std_90d",
    ]

    use_cache = FILTERED_PATH.exists() and MARKET_PATH.exists()

    if use_cache:
        # ── 캐시 경로: Step 1~7 건너뜀 ─────────────────────────────────────
        log.info("캐시 감지: %s → Step 1~7 건너뜀", FILTERED_PATH)
        mapped = pd.read_csv(
            FILTERED_PATH,
            parse_dates=["publish_date", "news_date", "trade_date"],
        )
        mapped_n = len(mapped)
        market = pd.read_csv(MARKET_PATH)
        market_n = len(market)
        log.info("  캐시 로드 완료: mapped %d건 | market %d행", mapped_n, market_n)
    else:
        # ── 풀 파이프라인: Track A Step 1~3 ─────────────────────────────────
        log.info("[Track A] 뉴스 데이터 로드: %s", RAW_PATH)
        df = pd.read_csv(RAW_PATH, parse_dates=["publish_date"], encoding="utf-8-sig")
        raw_n = len(df)
        log.info("  원본: %d건  | 컬럼: %s", raw_n, df.columns.tolist())

        df = df.rename(columns={"text": "body"})
        df["article_id"] = range(len(df))

        if df["publish_date"].isna().any():
            before = len(df)
            df = df.dropna(subset=["publish_date"]).reset_index(drop=True)
            log.warning("  publish_date 파싱 실패 %d건 제거", before - len(df))

        df["news_date"] = df["publish_date"].dt.normalize()
        df["ym"]        = df["news_date"].dt.to_period("M")

        # ── Step 1: 본문 정제 ────────────────────────────────────────────────
        tqdm.pandas(desc="[Step 1] body 정제")
        df["body_raw"] = df["body"]
        df["body"]     = df["body"].fillna("").progress_apply(clean_article_body)
        log.info("Step 1 완료")

        # ── Step 2: 형식 기반 필터링 ─────────────────────────────────────────
        title_str = df["title"].fillna("").astype(str)
        body_str  = df["body"].astype(str)
        body_len  = body_str.str.len()
        title_len = title_str.str.len()

        null_m   = df["title"].isna() | df["body"].isna() | (body_len == 0)
        short_m  = body_len < MIN_BODY_LEN
        long_m   = body_len > MAX_BODY_LEN
        stitle_m = title_len < MIN_TITLE_LEN

        step2_reasons = {
            "null":            int(null_m.sum()),
            "body_too_short":  int(short_m.sum()),
            "body_too_long":   int(long_m.sum()),
            "title_too_short": int(stitle_m.sum()),
        }
        fmt_mask = ~(null_m | short_m | long_m | stitle_m)

        h = (title_str + body_str.str[:100]).apply(
            lambda s: hashlib.md5(s.encode()).hexdigest()
        )
        dup_m = h.duplicated(keep="first") & fmt_mask
        step2_reasons["duplicate"] = int(dup_m.sum())

        df = df[fmt_mask & ~dup_m].reset_index(drop=True)
        step2_n = len(df)
        log.info(
            "Step 2 완료: %d건 (제거 %d건, %.1f%%) | %s",
            step2_n, raw_n - step2_n, (raw_n - step2_n) / raw_n * 100, step2_reasons,
        )

        # ── Step 3: 콘텐츠 기반 필터링 ───────────────────────────────────────
        step3_reasons: dict = {}

        tqdm.pandas(desc="[Step 3a] 광고 판별")
        irrelevant = df.progress_apply(
            lambda r: is_irrelevant_article(r["title"], r["body"])[0], axis=1
        )
        step3_reasons["advertisement"] = int(irrelevant.sum())
        df = df[~irrelevant].reset_index(drop=True)

        tqdm.pandas(desc="[Step 3b] 밀도 계산")
        density = df["body"].progress_apply(stopword_density)
        density_threshold = float(density.quantile(DENSITY_PERCENTILE / 100))
        log.info(
            "  density 분포: median=%.2f, mean=%.2f, p90=%.2f, p95=%.2f, p99=%.2f",
            density.median(), density.mean(),
            density.quantile(0.90), density.quantile(0.95), density.quantile(0.99),
        )
        log.info(
            "  DENSITY_PERCENTILE=%d → 적용 임계값=%.2f", DENSITY_PERCENTILE, density_threshold
        )
        high_d = density > density_threshold
        step3_reasons["high_stopword_density"] = int(high_d.sum())
        df = df[~high_d].reset_index(drop=True)

        step3_n = len(df)
        log.info(
            "Step 3 완료: %d건 (제거 %d건, %.1f%%) | %s",
            step3_n, step2_n - step3_n, (step2_n - step3_n) / step2_n * 100, step3_reasons,
        )

        # ── Track B Step 4~6: 환율 수집 + 영업일 테이블 ──────────────────────
        fx     = fetch_exchange_rate()
        market = build_market_table(fx)
        market_n = len(market)

        market.to_csv(MARKET_PATH, index=False, encoding="utf-8-sig")
        log.info("저장: %s  (%d행)", MARKET_PATH, market_n)

        # ── Step 7: merge_asof 매핑 ───────────────────────────────────────────
        mapped = map_news_to_market(df, market)
        mapped_n = len(mapped)

        mapped[out_cols].to_csv(FILTERED_PATH, index=False, encoding="utf-8-sig")
        log.info("저장: %s  (%d건)", FILTERED_PATH, mapped_n)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 8: 시간 균등 샘플링 (캐시 사용 여부 무관하게 항상 실행)
    # ──────────────────────────────────────────────────────────────────────────
    log.info("[Step 8] 시간 균등 샘플링 (target=%d)", TARGET_SAMPLE)
    sampler = mapped.copy()
    sampler["ym"] = sampler["news_date"].dt.to_period("M")
    sampled = time_balanced_sample(sampler, TARGET_SAMPLE, RANDOM_STATE)
    log.info("  샘플: %d건", len(sampled))

    sampled[out_cols].to_csv(SAMPLED_PATH, index=False, encoding="utf-8-sig")
    log.info("저장: %s  (%d건)", SAMPLED_PATH, len(sampled))

    # ──────────────────────────────────────────────────────────────────────────
    # 리포트
    # ──────────────────────────────────────────────────────────────────────────
    if use_cache:
        update_report_step8(sampled, market_n)
        log.info("=" * 64)
        log.info("Phase 1 완료 (캐시) | mapped %d → sample %d", mapped_n, len(sampled))
    else:
        write_report(
            raw_n=raw_n,
            step2_n=step2_n,
            step3_n=step3_n,
            mapped_n=mapped_n,
            step2_reasons=step2_reasons,
            step3_reasons=step3_reasons,
            market_n=market_n,
            sampled_df=sampled,
            density_threshold=density_threshold,
        )
        log.info("=" * 64)
        log.info(
            "Phase 1 완료 | raw %d → fmt %d → content %d → mapped %d → sample %d",
            raw_n, step2_n, step3_n, mapped_n, len(sampled),
        )
    log.info("=" * 64)


if __name__ == "__main__":
    main()
