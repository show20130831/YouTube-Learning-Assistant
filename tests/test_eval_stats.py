from __future__ import annotations

import pytest

from yla.eval.stats import (
    Counts,
    clopper_pearson,
    f1_difference,
    mcnemar_exact_p,
    micro_prf,
    paired_bootstrap,
    paired_permutation_p,
)


def test_micro_prf_sums_decisions_before_dividing() -> None:
    # video 1: 2 correct, 1 extra; video 2: 1 correct; video 3: no expected topic, 1 extra
    prf = micro_prf([Counts(2, 1, 0), Counts(1, 0, 0), Counts(0, 1, 0)])
    assert prf.precision == pytest.approx(0.6)
    assert prf.recall == pytest.approx(1.0)
    assert prf.f1 == pytest.approx(0.75)


def test_micro_prf_undefined_ratios() -> None:
    assert micro_prf([Counts(0, 0, 2)]).precision is None  # nothing predicted
    assert micro_prf([Counts(0, 3, 0)]).recall is None  # nothing expected
    assert micro_prf([Counts(0, 1, 1)]).f1 == 0.0


def test_mcnemar_exact_matches_hand_computed_binomial() -> None:
    # 1 vs 7 discordant pairs: 2 * (C(8,0) + C(8,1)) / 2^8 = 18/256
    assert mcnemar_exact_p(1, 7) == pytest.approx(18 / 256)
    assert mcnemar_exact_p(4, 4) == 1.0
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(0, 6) == pytest.approx(2 / 64)


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        (0, 24, 0.0, 1 - 0.025 ** (1 / 24)),  # closed form when k = 0
        (24, 24, 0.025 ** (1 / 24), 1.0),
        (5, 20, 0.0866, 0.4910),  # standard table value
    ],
)
def test_clopper_pearson(k: int, n: int, low: float, high: float) -> None:
    lo, hi = clopper_pearson(k, n)
    assert lo == pytest.approx(low, abs=1e-4)
    assert hi == pytest.approx(high, abs=1e-4)


def test_clopper_pearson_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        clopper_pearson(3, 2)


A = [Counts(1, 1, 0), Counts(0, 0, 1), Counts(2, 0, 0), Counts(1, 0, 1)]
BETTER = [Counts(1, 0, 0), Counts(1, 0, 0), Counts(2, 0, 0), Counts(2, 0, 0)]


def test_identical_runs_show_no_difference() -> None:
    interval = paired_bootstrap(A, A, f1_difference, iterations=500)
    assert (interval.estimate, interval.low, interval.high) == (0.0, 0.0, 0.0)
    assert paired_permutation_p(A, A, f1_difference, iterations=500) == 1.0


def test_bootstrap_is_reproducible_and_brackets_the_estimate() -> None:
    first = paired_bootstrap(A, BETTER, f1_difference, iterations=2000)
    again = paired_bootstrap(A, BETTER, f1_difference, iterations=2000)
    assert first == again  # fixed seed
    assert first.low <= first.estimate <= first.high
    assert first.estimate > 0


def test_permutation_p_value_is_a_probability() -> None:
    p = paired_permutation_p(A, BETTER, f1_difference, iterations=2000)
    assert 0 < p <= 1


def test_paired_samples_must_match() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap(A, A[:2], f1_difference)
