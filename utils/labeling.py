"""z-score 기반 변동 regime 라벨링 유틸리티."""

from __future__ import annotations

import pandas as pd


def compute_z_score(df: pd.DataFrame) -> pd.Series:
    """T → T+1~T+5 평균 변동을 90일 롤링 표준편차로 정규화한 z-score.

    Args:
        df: close, close_T1_T5_mean, rolling_std_90d 컬럼 포함 DataFrame
    Returns:
        z_score Series (rolling_std_90d == 0 또는 NaN인 행은 NaN)
    """
    change = df["close_T1_T5_mean"] - df["close"]
    std = df["rolling_std_90d"].replace(0, float("nan"))
    return change / std


def assign_label(z: float, threshold: float = 1.0) -> int | None:
    """z-score → 3-class 라벨.

    Returns:
        0 (extreme_down) | 1 (normal) | 2 (extreme_up) | None (NaN)
    """
    if pd.isna(z):
        return None
    if z >= threshold:
        return 2
    if z <= -threshold:
        return 0
    return 1
