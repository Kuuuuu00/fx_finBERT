"""
Phase 2: z-score 기반 변동 regime 라벨링

방향성 일관성 조건 미적용 (단순 z-score 임계값만 사용).
z=1.0, z=1.5 두 라벨 셋 모두 생성 → Phase 4에서 별도 모델 학습 후 비교.

Inputs
------
data/sampled/fx_news_20000.csv

Outputs
-------
data/labeled/final_labeled.csv         — z-score + 두 라벨 셋
data/labeled/label_distribution.png   — 클래스 분포 + 연도별 분포 + z-score 히스토그램
data/labeled/threshold_comparison.csv — z=1.0 vs z=1.5 분포 수치
results/label_quality_report.md
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.labeling import assign_label, compute_z_score  # noqa: E402

# ── 로거 ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── 경로 ──────────────────────────────────────────────────────────────────────
INPUT_PATH   = Path("data/sampled/fx_news_20000.csv")
LABELED_PATH = Path("data/labeled/final_labeled.csv")
PLOT_PATH    = Path("data/labeled/label_distribution.png")
COMP_PATH    = Path("data/labeled/threshold_comparison.csv")
REPORT_PATH  = Path("results/label_quality_report.md")

THRESHOLDS = [1.0, 1.5]
CLASS_NAMES = {0: "extreme_down", 1: "normal", 2: "extreme_up"}


# ══════════════════════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════════════════════

def _plot_class_dist(ax: plt.Axes, counts: pd.Series, title: str) -> None:
    colors = ["#e74c3c", "#95a5a6", "#27ae60"]
    labels = [CLASS_NAMES[i] for i in range(3)]
    bars = ax.bar(labels, [counts.get(i, 0) for i in range(3)], color=colors)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("건수")
    for bar, cnt in zip(bars, [counts.get(i, 0) for i in range(3)]):
        total = counts.sum()
        pct = cnt / total * 100 if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{cnt}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)


def _plot_year_dist(ax: plt.Axes, df: pd.DataFrame, col: str, title: str) -> None:
    year_label = df.groupby("year")[col].value_counts(normalize=True).unstack(fill_value=0)
    for c in [0, 1, 2]:
        if c not in year_label.columns:
            year_label[c] = 0.0
    year_label = year_label[[0, 1, 2]]
    colors = ["#e74c3c", "#95a5a6", "#27ae60"]
    bottoms = np.zeros(len(year_label))
    for idx, (c, color) in enumerate(zip([0, 1, 2], colors)):
        vals = year_label[c].values
        ax.bar(year_label.index, vals, bottom=bottoms, color=color,
               label=CLASS_NAMES[c] if idx == 0 or c != 1 else None, width=0.8)
        bottoms += vals
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("비율")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=7)
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)


def make_plots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    # 행 0: 클래스 분포
    for col_idx, thr in enumerate([1.0, 1.5]):
        col = f"label_z{thr}"
        counts = df[col].dropna().astype(int).value_counts()
        _plot_class_dist(axes[0][col_idx], counts, f"클래스 분포 (z={thr})")

    # 행 1: 연도별 스택드 바
    for col_idx, thr in enumerate([1.0, 1.5]):
        col = f"label_z{thr}"
        sub = df.dropna(subset=[col]).copy()
        sub[col] = sub[col].astype(int)
        _plot_year_dist(axes[1][col_idx], sub, col, f"연도별 라벨 비율 (z={thr})")

    # 행 2: z-score 히스토그램 (전체 / 확대)
    z = df["z_score"].dropna()
    axes[2][0].hist(z, bins=80, color="steelblue", alpha=0.7, edgecolor="none")
    axes[2][0].axvline(0, color="black", lw=0.8, linestyle="--")
    for thr in [1.0, 1.5]:
        axes[2][0].axvline(thr,  color="green", lw=0.8, linestyle=":")
        axes[2][0].axvline(-thr, color="red",   lw=0.8, linestyle=":")
    axes[2][0].set_title("z-score 분포 (전체)", fontsize=10)
    axes[2][0].set_xlabel("z-score")
    axes[2][0].set_ylabel("건수")

    axes[2][1].hist(z.clip(-5, 5), bins=80, color="steelblue", alpha=0.7, edgecolor="none")
    axes[2][1].axvline(0, color="black", lw=0.8, linestyle="--")
    for thr in [1.0, 1.5]:
        axes[2][1].axvline(thr,  color="green", lw=0.8, linestyle=":", label=f"z=±{thr}")
        axes[2][1].axvline(-thr, color="red",   lw=0.8, linestyle=":")
    axes[2][1].set_title("z-score 분포 ([-5, 5] 클립)", fontsize=10)
    axes[2][1].set_xlabel("z-score")
    axes[2][1].legend(fontsize=8)

    plt.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("저장: %s", PLOT_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# 리포트
# ══════════════════════════════════════════════════════════════════════════════

def write_report(df: pd.DataFrame) -> None:
    lines: list[str] = [
        "# Phase 2 라벨 품질 리포트",
        "",
        "## 1. z-score 기술통계",
        "",
    ]

    z = df["z_score"].dropna()
    lines += [
        f"- 전체 z-score 계산 건수: **{len(z):,}건** (NaN {df['z_score'].isna().sum()}건 제외)",
        f"- mean={z.mean():.4f}, std={z.std():.4f}",
        f"- min={z.min():.3f}, p5={z.quantile(0.05):.3f}, median={z.median():.3f}, "
        f"p95={z.quantile(0.95):.3f}, max={z.max():.3f}",
        "",
    ]

    # 클래스 분포
    lines += ["## 2. 클래스 분포", ""]
    comp_rows = []
    for thr in THRESHOLDS:
        col = f"label_z{thr}"
        labeled = df[col].dropna().astype(int)
        total = len(labeled)
        cnt = {c: int((labeled == c).sum()) for c in range(3)}
        pct = {c: cnt[c] / total * 100 for c in range(3)}
        lines += [
            f"### z={thr} (|z|≥{thr} → extreme)",
            "",
            f"총 라벨 건수: **{total:,}건**",
            "",
            "| 클래스 | 건수 | 비율 |",
            "|---|---:|---:|",
        ]
        for c in range(3):
            lines.append(f"| {CLASS_NAMES[c]} ({c}) | {cnt[c]:,} | {pct[c]:.1f}% |")
        lines.append("")
        comp_rows.append({
            "threshold": thr,
            "n_extreme_down": cnt[0], "pct_extreme_down": round(pct[0], 2),
            "n_normal":       cnt[1], "pct_normal":       round(pct[1], 2),
            "n_extreme_up":   cnt[2], "pct_extreme_up":   round(pct[2], 2),
            "n_total_labeled": total,
            "n_nan": int(df[col].isna().sum()),
        })
    pd.DataFrame(comp_rows).to_csv(COMP_PATH, index=False, encoding="utf-8-sig")
    log.info("저장: %s", COMP_PATH)

    # 연도별 분포
    lines += ["## 3. 연도별 라벨 분포 (정규화 검증)", ""]
    for thr in THRESHOLDS:
        col = f"label_z{thr}"
        sub = df.dropna(subset=[col]).copy()
        sub[col] = sub[col].astype(int)
        yr = sub.groupby("year")[col].value_counts().unstack(fill_value=0)
        for c in range(3):
            if c not in yr.columns:
                yr[c] = 0
        yr = yr[[0, 1, 2]]
        yr_pct = yr.div(yr.sum(axis=1), axis=0).round(3)

        lines += [f"### z={thr}", "", "| 연도 | extreme_down% | normal% | extreme_up% |", "|---|---:|---:|---:|"]
        for year_idx, row in yr_pct.iterrows():
            lines.append(
                f"| {year_idx} | {row[0]*100:.1f}% | {row[1]*100:.1f}% | {row[2]*100:.1f}% |"
            )
        lines.append("")

    # 같은 날 동일 라벨 비율
    lines += ["## 4. 같은 날 발행 기사 동일 라벨 비율 (구조적 한계)", ""]
    for thr in THRESHOLDS:
        col = f"label_z{thr}"
        sub = df.dropna(subset=[col]).copy()
        sub[col] = sub[col].astype(int)
        # 같은 trade_date 내 기사들의 라벨 일치율
        same_day = sub.groupby("trade_date")[col]
        ratios = []
        for _, grp in same_day:
            if len(grp) > 1:
                mode_cnt = grp.value_counts().iloc[0]
                ratios.append(mode_cnt / len(grp))
        avg_ratio = np.mean(ratios) if ratios else 1.0
        multi_day_articles = sum(len(g) for _, g in same_day if len(g) > 1)
        lines += [
            f"**z={thr}**: 같은 영업일 내 다수 기사 날짜 {len(ratios):,}일 "
            f"({multi_day_articles:,}건) → 최빈 라벨 평균 일치율 **{avg_ratio*100:.1f}%**",
            "",
            "> 이는 구조적 한계이며, 다운스트림에서 일별 집계 시 평균화됨.",
            "",
        ]

    # 방어 포인트
    lines += [
        "## 5. 학술적 방어 포인트",
        "",
        "- **라벨 정의**: T일 종가 → T+1~T+5 평균 변동의 z-score (시기별 변동성 정규화)",
        "- **방향성 조건 미적용**: extreme 라벨 수 보전 및 '변동 발생 자체'에 관심",
        "- **rolling window=90, min_periods=60**: 초기 31건 NaN 제외 처리",
        "- **FDR 데이터 공백 (2008-07-30 ~ 2008-08-25)**: 11건 영향, 전체의 0.22%로 무시 가능",
        "- **같은 날 기사 동일 라벨**: 위 §4 참조 — 개별 기사 인과 추론이 아닌 집합적 패턴 학습",
        "",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log.info("저장: %s", REPORT_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=" * 64)
    log.info("Phase 2 시작 (z-score 라벨링)")
    log.info("=" * 64)

    Path("data/labeled").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, parse_dates=["news_date", "trade_date", "publish_date"])
    log.info("입력: %d건 | 컬럼: %s", len(df), df.columns.tolist())

    # ── Step 1: z-score 계산 ────────────────────────────────────────────────
    df["change_T_to_T1T5"] = df["close_T1_T5_mean"] - df["close"]
    df["z_score"] = compute_z_score(df)

    n_nan = df["z_score"].isna().sum()
    log.info(
        "z_score 계산 완료: %d건 유효, %d건 NaN (rolling_std 결측 또는 0)",
        len(df) - n_nan, n_nan,
    )
    log.info(
        "z_score 분포: mean=%.4f, std=%.4f, min=%.3f, median=%.3f, max=%.3f",
        df["z_score"].mean(), df["z_score"].std(),
        df["z_score"].min(), df["z_score"].median(), df["z_score"].max(),
    )

    # ── Step 2: 라벨 부여 (z=1.0, z=1.5) ───────────────────────────────────
    for thr in THRESHOLDS:
        col = f"label_z{thr}"
        df[col] = df["z_score"].apply(lambda z, t=thr: assign_label(z, t))

        labeled = df[col].dropna()
        total = len(labeled)
        for c in range(3):
            cnt = int((labeled.astype(int) == c).sum())
            log.info("  z=%.1f | %s: %d건 (%.1f%%)", thr, CLASS_NAMES[c], cnt, cnt/total*100)

    df["year"] = df["news_date"].dt.year

    # ── 출력 ────────────────────────────────────────────────────────────────
    out_cols = [
        "article_id", "publish_date", "news_date", "trade_date", "year",
        "title", "body", "body_raw", "press", "url",
        "close", "prev_close", "next_close",
        "close_T1", "close_T2", "close_T3", "close_T4", "close_T5",
        "close_T1_T5_mean", "daily_change", "rolling_std_90d",
        "change_T_to_T1T5", "z_score", "label_z1.0", "label_z1.5",
    ]
    df[out_cols].to_csv(LABELED_PATH, index=False, encoding="utf-8-sig")
    log.info("저장: %s (%d건)", LABELED_PATH, len(df))

    make_plots(df)
    write_report(df)

    log.info("=" * 64)
    log.info("Phase 2 완료")
    log.info("=" * 64)


if __name__ == "__main__":
    main()
