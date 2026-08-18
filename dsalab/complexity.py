"""The complexity detective: measure how something scales, do not just claim it.

Textbooks tell you that binary search is O(log n) and that bubble sort is O(n^2).
This module checks. It runs a function across a range of input sizes, records how
much work each run took, then fits those measurements against every standard
growth curve and reports which one the data actually matches.

It answers a question that a plain benchmark cannot: not "how fast is this" but
"what shape is this". Two implementations can have the same speed at n=100 and
completely different futures at n=100000, and the shape is what tells you which.

How the fitting works, in plain English:

For each candidate curve f (constant, log n, n, n log n, n^2, n^3, 2^n) we look
for the best possible fit of the form

    measured_work = a * f(n) + b

`a` is a scale factor, because we do not care whether an operation takes 3
nanoseconds or 3 milliseconds, only how the total grows. `b` is a fixed overhead,
because every measurement includes some constant setup cost. Finding the best a
and b for a given f is ordinary least squares, which has a closed form solution,
so no fitting library is needed.

Then each fit is scored with R squared, which is the fraction of the variation in
the measurements that the curve explains. 1.0 is a perfect fit and 0.0 means the
curve is no better than a flat line through the average. The curve with the
highest score wins.

Two honest warnings, because this is measurement and measurement lies:

1. Timing on a shared machine is noisy. Garbage collection, other processes and
   CPU frequency scaling all show up in the numbers. Counting operations instead
   of seconds is far more stable, which is why `measure_steps` exists and why the
   tests use it.
2. Neighbouring curves are hard to tell apart over a small range of n. n and
   n log n look nearly identical from n=10 to n=100, because log n barely moves.
   Use a wide range of sizes, ideally doubling each time, and treat a close
   second place as a real ambiguity rather than noise.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# The candidate growth curves, in increasing order of how fast they grow. Each
# is a plain function of n, and each name is what the report prints.
GROWTH_CURVES: dict[str, Callable[[float], float]] = {
    "O(1)": lambda n: 1.0,
    "O(log n)": lambda n: math.log2(n) if n > 1 else 1.0,
    "O(n)": lambda n: float(n),
    "O(n log n)": lambda n: n * math.log2(n) if n > 1 else 1.0,
    "O(n^2)": lambda n: float(n) ** 2,
    "O(n^3)": lambda n: float(n) ** 3,
    "O(2^n)": lambda n: float(2**n) if n < 64 else float("inf"),
}


@dataclass(frozen=True)
class Fit:
    """How well one growth curve explains a set of measurements.

    Attributes:
        curve: The name of the curve, for example "O(n log n)".
        r_squared: How much of the variation the curve explains, from 0 to 1.
        scale: The fitted multiplier a, which is roughly the cost of one unit of
            work in whatever unit was measured.
        offset: The fitted constant b, the fixed overhead per run.
    """

    curve: str
    r_squared: float
    scale: float
    offset: float


@dataclass(frozen=True)
class Verdict:
    """The detective's conclusion about a set of measurements.

    Attributes:
        best: The winning fit.
        runner_up: The second best fit, kept because a near tie is information.
        all_fits: Every fit, best first, so the app can show the full table.
        sizes: The input sizes that were measured.
        measurements: What each size cost.
    """

    best: Fit
    runner_up: Fit | None
    all_fits: list[Fit]
    sizes: list[int]
    measurements: list[float]

    @property
    def is_confident(self) -> bool:
        """True when the winner is a good fit and clearly ahead of second place.

        The two thresholds are judgement calls, not laws, and they were picked
        by measuring rather than guessed. A winning R squared below 0.95 means
        the measurements are too noisy to conclude anything.

        The gap threshold needed more care. The first attempt demanded a gap of
        0.02 between first and second place, which turned out to reject clear
        answers: a perfect quadratic fit scores 1.0 while the cubic curve still
        scores about 0.985 on the same data, a gap of only 0.015, because a
        cubic can be bent to sit fairly close to a quadratic. Meanwhile the case
        this threshold is really meant to catch, n against n log n over a narrow
        range of sizes, has a gap of about 0.00007. Those two cases are three
        orders of magnitude apart, so 0.001 sits comfortably between them and
        separates a genuine winner from a genuine tie.
        """
        if self.best.r_squared < 0.95:
            return False
        if self.runner_up is None:
            return True
        return self.best.r_squared - self.runner_up.r_squared >= 0.001

    def summary(self) -> str:
        """One line of plain English suitable for printing under a chart."""
        if not self.is_confident:
            if self.runner_up is not None and self.best.r_squared >= 0.95:
                return (
                    f"Looks like {self.best.curve}, but {self.runner_up.curve} fits almost "
                    f"as well over this range of sizes. Measure a wider range to separate them."
                )
            return (
                "The measurements are too noisy to name a growth curve. "
                "Try more repeats, larger sizes, or counting operations instead of time."
            )
        return f"Measured growth matches {self.best.curve} (R squared {self.best.r_squared:.4f})."


def fit_curve(sizes: Sequence[int], measurements: Sequence[float], curve: str) -> Fit:
    """Fit one named growth curve to the measurements by least squares.

    Solves for the a and b that minimise the squared error of
    measurement = a * curve(n) + b, then scores the result with R squared.
    """
    if len(sizes) != len(measurements):
        raise ValueError("there must be exactly one measurement per input size")
    if len(sizes) < 3:
        raise ValueError("at least three sizes are needed before a fit means anything")
    if curve not in GROWTH_CURVES:
        raise ValueError(f"unknown curve {curve!r}, expected one of {list(GROWTH_CURVES)}")

    basis = GROWTH_CURVES[curve]
    xs = [basis(n) for n in sizes]
    ys = list(measurements)

    if any(math.isinf(x) for x in xs):
        # 2^n overflows into infinity for large n, which means this curve simply
        # cannot describe the data. Score it zero rather than crashing.
        return Fit(curve, 0.0, 0.0, 0.0)

    count = len(xs)
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count

    # Least squares in its textbook form: the slope is the covariance of x and y
    # divided by the variance of x, and the line passes through the two means.
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    variance = sum((x - mean_x) ** 2 for x in xs)

    if variance == 0:
        # Every x is identical, which happens for the O(1) curve since it is flat.
        # The best possible fit is then a horizontal line at the average y.
        scale, offset = 0.0, mean_y
    else:
        scale = covariance / variance
        offset = mean_y - scale * mean_x

    predicted = [scale * x + offset for x in xs]
    residual = sum((y - p) ** 2 for y, p in zip(ys, predicted, strict=True))
    total = sum((y - mean_y) ** 2 for y in ys)

    if total == 0:
        # Every measurement is identical. A flat line explains that perfectly,
        # which is genuinely constant behaviour rather than a fitting failure.
        r_squared = 1.0 if residual == 0 else 0.0
    else:
        r_squared = 1 - residual / total

    return Fit(curve, max(0.0, r_squared), scale, offset)


def detect(sizes: Sequence[int], measurements: Sequence[float]) -> Verdict:
    """Fit every candidate curve to the measurements and rank them."""
    fits = sorted(
        (fit_curve(sizes, measurements, curve) for curve in GROWTH_CURVES),
        key=lambda fit: fit.r_squared,
        reverse=True,
    )
    return Verdict(
        best=fits[0],
        runner_up=fits[1] if len(fits) > 1 else None,
        all_fits=fits,
        sizes=list(sizes),
        measurements=list(measurements),
    )


def measure_time(
    action: Callable[[int], None],
    sizes: Sequence[int],
    repeats: int = 5,
) -> Verdict:
    """Time `action` at each input size and work out how it scales.

    `action` is called as action(n) and should do one full run at size n. Any
    setup that should not be timed must happen inside `action` before the work,
    which is not ideal, so for anything sensitive prefer `measure_steps`.

    Each size is run `repeats` times and the fastest run is kept. The fastest run
    is used rather than the average because interference from the operating
    system can only ever make a run slower, never faster, so the minimum is the
    measurement least polluted by things that have nothing to do with the code.
    """
    timings: list[float] = []
    for n in sizes:
        best = math.inf
        for _ in range(repeats):
            start = time.perf_counter()
            action(n)
            best = min(best, time.perf_counter() - start)
        timings.append(best)
    return detect(sizes, timings)


def measure_steps(work: Callable[[int], float], sizes: Sequence[int]) -> Verdict:
    """Count operations at each input size and work out how it scales.

    `work` is called as work(n) and returns how much work that run took, usually
    a count of steps from a traced algorithm. Counting is deterministic, so this
    gives a far cleaner answer than timing does, and it is what the tests use.
    """
    counts = [float(work(n)) for n in sizes]
    return detect(sizes, counts)


def doubling_sizes(start: int, count: int) -> list[int]:
    """Input sizes that double each time, which is the right shape for this.

    Doubling is what separates the curves. Ten evenly spaced sizes from 100 to
    200 tell you almost nothing, because every curve looks like a straight line
    over a narrow range. Sizes of 100, 200, 400, 800 and 1600 make the difference
    between n and n^2 obvious immediately.
    """
    if start < 1 or count < 1:
        raise ValueError("start and count must both be at least 1")
    return [start * 2**i for i in range(count)]
