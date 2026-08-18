"""Tests for the complexity detective.

These tests feed it data whose true growth curve is known in advance and check
that it names the right one. Counted work is used rather than real timings,
because a test that depends on how busy the machine is would fail at random.
"""

import math

import pytest

from dsalab.complexity import GROWTH_CURVES, detect, doubling_sizes, fit_curve, measure_steps


def synthetic(curve: str, sizes, scale: float = 3.0, offset: float = 7.0):
    """Work counts generated from a known curve, so the right answer is known."""
    basis = GROWTH_CURVES[curve]
    return [scale * basis(n) + offset for n in sizes]


@pytest.mark.parametrize("curve", ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)"])
def test_detects_the_curve_the_data_was_generated_from(curve):
    sizes = doubling_sizes(8, 8)
    verdict = detect(sizes, synthetic(curve, sizes))

    assert verdict.best.curve == curve
    assert verdict.best.r_squared > 0.999


def test_a_perfect_fit_reports_confidence():
    sizes = doubling_sizes(16, 7)
    verdict = detect(sizes, synthetic("O(n^2)", sizes))

    assert verdict.is_confident
    assert "O(n^2)" in verdict.summary()


def test_pure_noise_is_reported_as_inconclusive_rather_than_guessed_at():
    sizes = [10, 20, 40, 80, 160, 320]
    # Values with no relationship to n at all. A tempting bug here would be to
    # always name whichever curve scored highest, however badly it scored.
    verdict = detect(sizes, [5.0, 1.0, 9.0, 2.0, 8.0, 3.0])

    assert not verdict.is_confident
    assert "too noisy" in verdict.summary() or "almost" in verdict.summary()


def test_measurements_that_do_not_move_are_read_as_constant_time():
    sizes = doubling_sizes(4, 6)
    verdict = detect(sizes, [42.0] * len(sizes))

    assert verdict.best.curve == "O(1)"
    assert verdict.best.r_squared == 1.0


def test_the_fitted_scale_recovers_the_real_multiplier():
    sizes = doubling_sizes(8, 7)
    verdict = detect(sizes, synthetic("O(n)", sizes, scale=2.5, offset=11.0))

    assert verdict.best.scale == pytest.approx(2.5, rel=1e-6)
    assert verdict.best.offset == pytest.approx(11.0, rel=1e-6)


def test_a_constant_overhead_does_not_change_the_detected_curve():
    sizes = doubling_sizes(8, 7)
    cheap = detect(sizes, synthetic("O(n log n)", sizes, offset=0.0))
    expensive = detect(sizes, synthetic("O(n log n)", sizes, offset=5000.0))

    assert cheap.best.curve == expensive.best.curve == "O(n log n)"


def test_measure_steps_reads_a_real_nested_loop_as_quadratic():
    def nested_loop_work(n: int) -> float:
        return sum(1 for _ in range(n) for _ in range(n))

    verdict = measure_steps(nested_loop_work, doubling_sizes(8, 6))

    assert verdict.best.curve == "O(n^2)"
    assert verdict.is_confident


def test_measure_steps_reads_repeated_halving_as_logarithmic():
    def halving_work(n: int) -> float:
        steps = 0
        while n > 1:
            n //= 2
            steps += 1
        return steps

    verdict = measure_steps(halving_work, doubling_sizes(64, 8))

    assert verdict.best.curve == "O(log n)"


def test_measure_steps_reads_merge_sort_shaped_work_as_linearithmic():
    def merge_sort_work(n: int) -> float:
        # The recurrence T(n) = 2T(n/2) + n, evaluated directly.
        if n <= 1:
            return 0.0
        return n + 2 * merge_sort_work(n // 2)

    verdict = measure_steps(merge_sort_work, doubling_sizes(32, 8))

    assert verdict.best.curve == "O(n log n)"


def test_neighbouring_curves_over_a_narrow_range_are_reported_as_ambiguous():
    # n and n log n over a short span of sizes genuinely cannot be told apart.
    # The detective should admit that rather than pretending to be sure.
    sizes = [100, 110, 120, 130, 140]
    verdict = detect(sizes, [float(n) * math.log2(n) for n in sizes])

    assert not verdict.is_confident, "a near tie over a narrow range is not a confident answer"


def test_fits_are_returned_ranked_best_first():
    sizes = doubling_sizes(8, 7)
    verdict = detect(sizes, synthetic("O(n^2)", sizes))
    scores = [fit.r_squared for fit in verdict.all_fits]

    assert scores == sorted(scores, reverse=True)
    assert len(verdict.all_fits) == len(GROWTH_CURVES)


def test_fitting_needs_at_least_three_points():
    with pytest.raises(ValueError):
        fit_curve([1, 2], [1.0, 2.0], "O(n)")


def test_mismatched_input_lengths_are_rejected():
    with pytest.raises(ValueError):
        fit_curve([1, 2, 3], [1.0, 2.0], "O(n)")


def test_an_unknown_curve_name_is_rejected():
    with pytest.raises(ValueError):
        fit_curve([1, 2, 3], [1.0, 2.0, 3.0], "O(n!)")


def test_doubling_sizes_doubles():
    assert doubling_sizes(10, 5) == [10, 20, 40, 80, 160]


def test_doubling_sizes_rejects_nonsense_arguments():
    with pytest.raises(ValueError):
        doubling_sizes(0, 5)
    with pytest.raises(ValueError):
        doubling_sizes(10, 0)


def test_exponential_growth_is_detected_at_small_sizes():
    sizes = [2, 4, 6, 8, 10, 12, 14]
    verdict = detect(sizes, [float(2**n) for n in sizes])

    assert verdict.best.curve == "O(2^n)"


def test_huge_sizes_do_not_crash_the_exponential_curve():
    # 2^n overflows to infinity well before n=1000. The fit should score that
    # curve zero and carry on rather than raising.
    sizes = [100, 200, 400, 800]
    verdict = detect(sizes, [float(n) for n in sizes])

    assert verdict.best.curve == "O(n)"
    assert any(fit.curve == "O(2^n)" and fit.r_squared == 0.0 for fit in verdict.all_fits)
