"""Statistics for comparing evaluation runs, in plain Python (no scipy).

All comparisons are paired: both runs answered the same videos, so differences come from the
models, prompts or rules rather than from different inputs. Videos are the resampling unit,
because the topics of one video are not independent of each other.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import comb

DEFAULT_ITERATIONS = 10_000
DEFAULT_SEED = 20261013


@dataclass(frozen=True)
class Counts:
    """Topic decisions for one video: correct, extra and missed topics."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


@dataclass(frozen=True)
class PRF:
    precision: float | None
    recall: float | None
    f1: float | None


def micro_prf(counts: Sequence[Counts]) -> PRF:
    """Micro-averaged precision, recall and F1: sum the decisions of all videos, then divide.

    ``None`` where a ratio is undefined (no predicted topics, or no expected topics).
    """
    total = sum(counts, Counts())
    precision = total.tp / (total.tp + total.fp) if total.tp + total.fp else None
    recall = total.tp / (total.tp + total.fn) if total.tp + total.fn else None
    if precision is None or recall is None:
        return PRF(precision, recall, None)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF(precision, recall, f1)


def micro_f1(counts: Sequence[Counts]) -> float:
    return micro_prf(counts).f1 or 0.0


def micro_precision(counts: Sequence[Counts]) -> float:
    return micro_prf(counts).precision or 0.0


PairedStatistic = Callable[[Sequence[Counts], Sequence[Counts]], float]


def f1_difference(a: Sequence[Counts], b: Sequence[Counts]) -> float:
    return micro_f1(b) - micro_f1(a)


def precision_difference(a: Sequence[Counts], b: Sequence[Counts]) -> float:
    return micro_precision(b) - micro_precision(a)


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


def paired_bootstrap(
    a: Sequence[Counts],
    b: Sequence[Counts],
    statistic: PairedStatistic,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> Interval:
    """Percentile bootstrap CI of ``statistic(a, b)``, resampling whole videos (pairs kept together)."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and the same length")
    rng = random.Random(seed)
    n = len(a)
    values = []
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        values.append(statistic([a[i] for i in idx], [b[i] for i in idx]))
    values.sort()
    tail = (1 - confidence) / 2
    low = values[int(tail * iterations)]
    high = values[min(iterations - 1, int((1 - tail) * iterations))]
    return Interval(statistic(a, b), low, high)


def paired_permutation_p(
    a: Sequence[Counts],
    b: Sequence[Counts],
    statistic: PairedStatistic,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
) -> float:
    """Two-sided p-value: how often randomly swapping A and B within videos gives a difference
    at least as large as the observed one (Monte Carlo, with the +1 correction)."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and the same length")
    rng = random.Random(seed)
    observed = abs(statistic(a, b))
    extreme = 0
    for _ in range(iterations):
        swapped_a, swapped_b = [], []
        for x, y in zip(a, b, strict=True):
            if rng.random() < 0.5:
                x, y = y, x
            swapped_a.append(x)
            swapped_b.append(y)
        if abs(statistic(swapped_a, swapped_b)) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (iterations + 1)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test from the discordant pairs (b: only A succeeded, c: only B)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / (1 << n)
    return min(1.0, 2 * tail)


def _binomial_cdf(k: int, n: int, p: float) -> float:
    return float(sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1)))


def _bisect(f: Callable[[float], float], target: float, *, increasing: bool) -> float:
    low, high = 0.0, 1.0
    for _ in range(100):
        mid = (low + high) / 2
        if (f(mid) < target) == increasing:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def clopper_pearson(k: int, n: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval for k successes in n trials."""
    if not 0 <= k <= n or n == 0:
        raise ValueError("need 0 <= k <= n and n > 0")
    alpha = 1 - confidence
    # lower: P(X >= k | p) = alpha/2; upper: P(X <= k | p) = alpha/2
    low = 0.0 if k == 0 else _bisect(lambda p: 1 - _binomial_cdf(k - 1, n, p), alpha / 2, increasing=True)
    high = 1.0 if k == n else _bisect(lambda p: _binomial_cdf(k, n, p), alpha / 2, increasing=False)
    return low, high
