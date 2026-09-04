"""The benchmark harness: measuring what the docstrings claim.

Every complexity claim in this project is written down somewhere. This runs them
against reality: time each implementation across growing input sizes, hand the
measurements to the complexity detective from day 3, and report whether the
measured growth curve matches the claimed one.

That last step is what makes this more than a stopwatch. A benchmark that prints
"0.42 seconds" tells you almost nothing. A benchmark that says "claimed O(log n),
measured O(log n), R squared 0.998" is checking the thing that was actually
asserted.

Three deliberate choices, each with a reason:

**The fastest run is kept, not the average.** Interference from the operating
system, other processes and garbage collection can only ever make a run slower,
never faster, so the minimum is the measurement least polluted by things that have
nothing to do with the code.

**Garbage collection is disabled during timing.** A collection pause landing inside
one measurement and not another adds noise unrelated to the algorithm. It is
switched back on afterwards, in a `finally`, so an exception cannot leave it off.

**Setup is done outside the timed region.** Building the input is often more
expensive than the operation being measured, and including it would drown the
signal entirely.

Where the comparison is against a Python standard library equivalent, the result
will usually be embarrassing, and that is worth stating plainly: `list.sort` is
written in C and this project's merge sort is written in Python, so a fifty times
gap says nothing about the algorithms. What is comparable is the **shape** of the
curve, which is why the fitted curve matters more than the seconds.
"""

from __future__ import annotations

import gc
import json
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dsalab.complexity import Verdict, detect

RESULTS_DIRECTORY = Path(__file__).parent / "results"


@dataclass
class Measurement:
    """One implementation measured at one input size."""

    name: str
    size: int
    seconds: float
    operations: float | None = None


@dataclass
class BenchmarkResult:
    """Everything measured about one implementation across all sizes.

    Attributes:
        name: What was measured.
        claimed: The complexity the code claims, as written in its docstring.
        sizes: The input sizes used.
        timings: The best time at each size.
        verdict: What the complexity detective made of those timings.
        notes: Anything worth saying about the measurement itself.
    """

    name: str
    claimed: str
    sizes: list[int]
    timings: list[float]
    verdict: Verdict
    notes: str = ""

    @property
    def matches_claim(self) -> bool:
        """Whether the measured curve is the claimed one.

        A near miss between neighbouring curves is treated as agreement, because
        n and n log n are genuinely hard to separate by timing over a limited
        range of sizes, and pretending otherwise would make the report dishonest
        in the opposite direction.
        """
        if self.verdict.best.curve == self.claimed:
            return True

        neighbours = {
            ("O(n)", "O(n log n)"),
            ("O(n log n)", "O(n)"),
            ("O(1)", "O(log n)"),
            ("O(log n)", "O(1)"),
        }
        return (self.claimed, self.verdict.best.curve) in neighbours

    def summary(self) -> str:
        verdict = "matches" if self.matches_claim else "DISAGREES with"
        line = (
            f"{self.name}: claimed {self.claimed}, measured {self.verdict.best.curve} "
            f"(R squared {self.verdict.best.r_squared:.3f}), which {verdict} the claim"
        )
        return f"{line}. {self.notes}" if self.notes else line + "."

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "claimed": self.claimed,
            "measured": self.verdict.best.curve,
            "r_squared": round(self.verdict.best.r_squared, 6),
            "confident": self.verdict.is_confident,
            "matches_claim": self.matches_claim,
            "sizes": self.sizes,
            "seconds": [round(value, 9) for value in self.timings],
            "notes": self.notes,
        }


@dataclass
class Comparison:
    """Two or more implementations measured on identical inputs."""

    title: str
    results: list[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "results": [result.to_dict() for result in self.results]}

    def report(self) -> str:
        lines = [self.title, "=" * len(self.title)]
        lines.extend(f"  {result.summary()}" for result in self.results)
        return "\n".join(lines)


def time_once(action: Callable[[], Any]) -> float:
    """Time a single call as cleanly as this environment allows."""
    collecting = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter()
        action()
        return time.perf_counter() - start
    finally:
        if collecting:
            gc.enable()


def measure(
    name: str,
    claimed: str,
    build_input: Callable[[int], Any],
    operation: Callable[[Any], Any],
    sizes: Sequence[int],
    repeats: int = 5,
    notes: str = "",
) -> BenchmarkResult:
    """Time `operation` across input sizes and work out how it really scales.

    `build_input` is called outside the timed region, so the cost of constructing
    a million element list does not get counted against the algorithm that sorts
    it. This separation is the single most common way benchmarks go wrong.
    """
    timings: list[float] = []

    for size in sizes:
        best = float("inf")
        for _ in range(repeats):
            prepared = build_input(size)
            # `prepared` is bound as a default argument rather than captured, so
            # the lambda cannot pick up a later value. It happens to be called
            # immediately here, but a closure over a loop variable is a trap worth
            # not leaving lying around.
            best = min(best, time_once(lambda ready=prepared: operation(ready)))
        timings.append(best)

    return BenchmarkResult(name, claimed, list(sizes), timings, detect(sizes, timings), notes)


def measure_operations(
    name: str,
    claimed: str,
    count_work: Callable[[int], float],
    sizes: Sequence[int],
    notes: str = "",
) -> BenchmarkResult:
    """Measure by counting operations rather than by timing.

    Deterministic, so it gives the same answer every run, which makes it the right
    tool for anything that has to pass a test. The tracing layer means counting is
    free: the number of `compare` steps in a sort is a real measurement with no
    timer and no counters added to the algorithm.

    The timings list holds the counts, since everything downstream only cares
    about how the numbers grow.
    """
    counts = [float(count_work(size)) for size in sizes]
    return BenchmarkResult(name, claimed, list(sizes), counts, detect(sizes, counts), notes)


def save(comparisons: list[Comparison], directory: Path = RESULTS_DIRECTORY) -> Path:
    """Write the results as JSON, which is always possible.

    Deliberately separate from plotting. The numbers are the result; the picture is
    a convenience, and it needs matplotlib, which the library itself does not. A
    benchmark that produces nothing when a plotting library is missing would be a
    poor design.
    """
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "benchmarks.json"

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": _python_description(),
        "comparisons": [comparison.to_dict() for comparison in comparisons],
    }
    destination.write_text(json.dumps(payload, indent=2))
    return destination


def _python_description() -> str:
    import platform

    return f"{platform.python_implementation()} {platform.python_version()} on {platform.system()}"


def plot(comparisons: list[Comparison], directory: Path = RESULTS_DIRECTORY) -> list[Path]:
    """Draw one chart per comparison, if matplotlib is installed.

    Returns the files written, or an empty list when matplotlib is unavailable.
    Returning a list rather than raising means `python -m benchmarks.run` still
    does its real work on a machine without plotting libraries.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # no display needed, which matters on a server
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for comparison in comparisons:
        figure, axes = plt.subplots(figsize=(8, 5))

        for result in comparison.results:
            axes.plot(result.sizes, result.timings, marker="o",
                      label=f"{result.name} ({result.verdict.best.curve})")

        axes.set_xlabel("input size")
        axes.set_ylabel("seconds (best of several runs)")
        axes.set_title(comparison.title)
        axes.set_xscale("log", base=2)
        axes.set_yscale("log")
        axes.grid(True, which="both", alpha=0.3)
        axes.legend()

        name = comparison.title.lower().replace(" ", "-").replace(",", "")
        destination = directory / f"{name}.png"
        figure.tight_layout()
        figure.savefig(destination, dpi=120)
        plt.close(figure)
        written.append(destination)

    return written


def relative_speed(results: list[BenchmarkResult]) -> dict[str, float]:
    """How much slower each implementation is than the fastest, at the largest size.

    Reported as a ratio rather than in seconds, because seconds mean nothing
    without knowing the machine, and a ratio between two implementations measured
    on the same machine in the same run is a fair comparison.
    """
    if not results:
        return {}

    largest = [result.timings[-1] for result in results]
    fastest = min(largest)
    return {
        result.name: (time / fastest if fastest else float("inf"))
        for result, time in zip(results, largest, strict=True)
    }


def stable_enough(timings: list[float], tolerance: float = 0.25) -> bool:
    """Whether repeated measurements are consistent enough to draw conclusions from.

    A benchmark on a busy machine can produce numbers that vary by more than the
    difference being measured. This gives the caller a way to notice rather than
    quietly reporting noise as a finding.
    """
    if len(timings) < 2:
        return True
    average = statistics.fmean(timings)
    if average == 0:
        return True
    return statistics.pstdev(timings) / average <= tolerance
