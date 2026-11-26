"""Детерминированный базовый прогноз временного ряда."""

import math


def predict(values: list[float], horizon: int) -> float:
    """Прогноз по последнему шагу; вход: 4–128 чисел, горизонт 1–30."""
    if not 4 <= len(values) <= 128:
        raise ValueError("Invalid sequence length")
    if not 1 <= horizon <= 30:
        raise ValueError("Invalid horizon")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Non-finite observation")
    result = values[-1] + horizon * (values[-1] - values[-2])
    if not math.isfinite(result):
        raise ValueError("Non-finite prediction")
    return result
