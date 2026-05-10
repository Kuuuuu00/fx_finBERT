"""
Phase 3: 시간순 train/val/test 분할
  Train: 2004-01 ~ 2020-12
  Val:   2021-01 ~ 2023-06
  Test:  2023-07 ~ 2026-04

입력: data/labeled/final_labeled.csv
출력: data/splits/{train,val,test}.csv, data/splits/split_stats.md
"""

from pathlib import Path

import pandas as pd

INPUT = Path("data/labeled/final_labeled.csv")
SPLITS_DIR = Path("data/splits")

# 5,000건 기준 비교용 참조값 (data/splits/archive/split_stats_5000.md)
_PREV_5000 = {
    "total": 4969,
    "z1.5": {
        "train": {"n": 3791, "c0": 379, "c1": 3020, "c2": 392},
        "val":   {"n": 570,  "c0": 70,  "c1": 399,  "c2": 101},
        "test":  {"n": 608,  "c0": 62,  "c1": 460,  "c2": 86},
    },
    "z1.0": {
        "train": {"n": 3791, "c0": 675, "c1": 2419, "c2": 697},
        "val":   {"n": 570,  "c0": 107, "c1": 283,  "c2": 180},
        "test":  {"n": 608,  "c0": 114, "c1": 347,  "c2": 147},
    },
}

TRAIN_END = "2020-12-31"
VAL_START = "2021-01-01"
VAL_END = "2023-06-30"
TEST_START = "2023-07-01"


def class_dist_str(series: pd.Series, label_col: str) -> str:
    counts = series[label_col].value_counts().sort_index()
    total = counts.sum()
    rows = []
    for cls, n in counts.items():
        rows.append(f"  Class {int(cls)} : {n:5d} ({n / total * 100:5.1f}%)")
    return "\n".join(rows)


def main() -> None:
    df = pd.read_csv(INPUT)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["publish_date"] = pd.to_datetime(df["publish_date"])

    # 두 라벨 모두 None인 행 제외 (rolling_std_90d NaN 31건)
    valid = df.dropna(subset=["label_z1.0", "label_z1.5"]).copy()
    dropped = len(df) - len(valid)
    print(f"NaN 라벨 제외: {dropped}건 → 잔존 {len(valid)}건")

    # 시간순 분할 (trade_date 기준)
    train = valid[valid["trade_date"] <= TRAIN_END].reset_index(drop=True)
    val = valid[
        (valid["trade_date"] >= VAL_START) & (valid["trade_date"] <= VAL_END)
    ].reset_index(drop=True)
    test = valid[valid["trade_date"] >= TEST_START].reset_index(drop=True)

    # ── 검증 ──────────────────────────────────────────────────────────────────
    total = len(train) + len(val) + len(test)
    assert total == len(valid), f"합계 불일치: {total} ≠ {len(valid)}"

    # 시간 범위 겹침 없음
    assert train["trade_date"].max() < pd.Timestamp(VAL_START), "Train-Val 겹침"
    assert val["trade_date"].max() < pd.Timestamp(TEST_START), "Val-Test 겹침"

    # 두 라벨 컬럼 보존 확인
    for col in ("label_z1.0", "label_z1.5"):
        assert col in train.columns, f"{col} 컬럼 누락"

    # ── 저장 ──────────────────────────────────────────────────────────────────
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(SPLITS_DIR / "train.csv", index=False)
    val.to_csv(SPLITS_DIR / "val.csv", index=False)
    test.to_csv(SPLITS_DIR / "test.csv", index=False)
    print(f"저장 완료: train={len(train)}, val={len(val)}, test={len(test)}")

    # ── 보고서 ────────────────────────────────────────────────────────────────
    lines = [
        "# Split Stats",
        "",
        "## 샘플링 규모 변경 이력",
        "",
        "| 버전 | 샘플 수 | 변경 사유 | 날짜 |",
        "|------|---------|-----------|------|",
        "| v1 (5,000건) | 4,969건 (NaN 31건 제외) | 초기 파이롯 | 2026-05-06 |",
        f"| v2 ({len(df):,}건) | {len(valid):,}건 (NaN {dropped}건 제외) | z=1.5 extreme 클래스 표본 확보 | 2026-05-08 |",
        "",
        f"- 원본 기사 수: {len(df):,}",
        f"- NaN 라벨 제외: {dropped}건 (rolling_std_90d NaN)",
        f"- 분할 대상: {len(valid):,}건",
        "",
        "## Split 크기 및 시간 범위",
        "",
        "| Split | 기사 수 | 비율 | 시작일 | 종료일 |",
        "|-------|---------|------|--------|--------|",
    ]
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        pct = len(split) / len(valid) * 100
        min_d = split["trade_date"].min().date()
        max_d = split["trade_date"].max().date()
        lines.append(f"| {name} | {len(split):,} | {pct:.1f}% | {min_d} | {max_d} |")

    lines += [
        "",
        "## 시간 범위 겹침 검증",
        "",
        f"- Train 최대일: {train['trade_date'].max().date()}",
        f"- Val   최소일: {val['trade_date'].min().date()}",
        f"- Val   최대일: {val['trade_date'].max().date()}",
        f"- Test  최소일: {test['trade_date'].min().date()}",
        "- **겹침: 0건 (검증 통과)**",
        "",
        "## 클래스 분포 (label_z1.0)",
        "",
    ]
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        lines.append(f"### {name}")
        lines.append(class_dist_str(split, "label_z1.0"))
        lines.append("")

    lines += ["## 클래스 분포 (label_z1.5)", ""]
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        lines.append(f"### {name}")
        lines.append(class_dist_str(split, "label_z1.5"))
        lines.append("")

    # Distribution shift 요약 (비율 비교)
    lines += ["## Distribution Shift 확인 (split 간 클래스 비율 비교)", ""]
    for lbl_col in ("label_z1.0", "label_z1.5"):
        lines.append(f"### {lbl_col}")
        lines.append("")
        lines.append("| Split | Class 0 % | Class 1 % | Class 2 % |")
        lines.append("|-------|-----------|-----------|-----------|")
        for name, split in [("Train", train), ("Val", val), ("Test", test)]:
            vc = split[lbl_col].value_counts(normalize=True).sort_index()
            c0 = vc.get(0.0, 0) * 100
            c1 = vc.get(1.0, 0) * 100
            c2 = vc.get(2.0, 0) * 100
            lines.append(f"| {name} | {c0:.1f}% | {c1:.1f}% | {c2:.1f}% |")
        lines.append("")

    # ── 5,000건 vs 현재 비교 ──────────────────────────────────────────────────
    lines += [
        "## 5,000건 → 현재 비교 (extreme 클래스 표본 수)",
        "",
        "### z=1.5 (주요 임계값)",
        "",
        "| Split | Metric | 5,000건 기준 | 현재 | 배율 |",
        "|-------|--------|-------------|------|------|",
    ]
    prev = _PREV_5000["z1.5"]
    for name, split, prev_key in [("Train", train, "train"), ("Val", val, "val"), ("Test", test, "test")]:
        col = "label_z1.5"
        cur = split[col].dropna().astype(int)
        c0_cur = int((cur == 0).sum())
        c2_cur = int((cur == 2).sum())
        ext_cur = c0_cur + c2_cur
        ext_prev = prev[prev_key]["c0"] + prev[prev_key]["c2"]
        ratio = ext_cur / ext_prev if ext_prev > 0 else float("inf")
        lines.append(
            f"| {name} | extreme (c0+c2) | {ext_prev:,} | {ext_cur:,} | {ratio:.1f}x |"
        )
        lines.append(
            f"| {name} | extreme_down (c0) | {prev[prev_key]['c0']:,} | {c0_cur:,} | "
            f"{c0_cur/prev[prev_key]['c0']:.1f}x |"
        )
        lines.append(
            f"| {name} | extreme_up (c2) | {prev[prev_key]['c2']:,} | {c2_cur:,} | "
            f"{c2_cur/prev[prev_key]['c2']:.1f}x |"
        )

    lines += [
        "",
        "### z=1.0 (fallback 임계값)",
        "",
        "| Split | Metric | 5,000건 기준 | 현재 | 배율 |",
        "|-------|--------|-------------|------|------|",
    ]
    prev10 = _PREV_5000["z1.0"]
    for name, split, prev_key in [("Train", train, "train"), ("Val", val, "val"), ("Test", test, "test")]:
        col = "label_z1.0"
        cur = split[col].dropna().astype(int)
        c0_cur = int((cur == 0).sum())
        c2_cur = int((cur == 2).sum())
        ext_cur = c0_cur + c2_cur
        ext_prev = prev10[prev_key]["c0"] + prev10[prev_key]["c2"]
        ratio = ext_cur / ext_prev if ext_prev > 0 else float("inf")
        lines.append(
            f"| {name} | extreme (c0+c2) | {ext_prev:,} | {ext_cur:,} | {ratio:.1f}x |"
        )

    lines.append("")

    report = "\n".join(lines)
    (SPLITS_DIR / "split_stats.md").write_text(report, encoding="utf-8")
    print("split_stats.md 저장 완료")

    # 콘솔 요약
    print("\n" + "=" * 60)
    print(report)


if __name__ == "__main__":
    main()
