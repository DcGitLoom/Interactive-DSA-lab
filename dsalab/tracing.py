"""The step recording system that the whole lab is built on.

The idea in one paragraph:

Every algorithm in this project is written once, as a Python generator that
yields a `Step` object each time something interesting happens (a comparison, a
swap, a node visit). Nothing else changes about the algorithm. From that single
implementation we get three things for free:

1. The plain answer, by running the generator to the end and ignoring the steps.
2. A full replay of the run, by collecting every step into a list. The
   Streamlit app uses this to draw the animation.
3. A cheap way to count work, because the number of steps of a given kind is a
   real measurement of how much the algorithm did.

Writing the algorithm twice (one fast version, one animated version) is the
usual approach in visualiser projects, and it is how the two copies drift apart
and start disagreeing. One implementation, two ways to consume it, keeps the
animation honest: what you see on screen is the code that the tests run.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Step:
    """One moment in the life of a running algorithm.

    Attributes:
        kind: A short machine readable label, for example "compare" or "swap".
            The visualiser uses this to decide how to colour the frame, and the
            tests use it to count work.
        note: One sentence of plain English describing what just happened. This
            is what a learner reads under the animation.
        data: Anything the visualiser needs to draw this frame, for example the
            two indices being compared and the current state of the array.
    """

    kind: str
    note: str
    data: dict[str, Any] = field(default_factory=dict)


# A traced algorithm yields Steps while it works and finally returns an answer
# of type T. Read `Traced[list[int]]` as "yields steps, ends up returning a
# list of ints".
Traced = Generator[Step, None, T]


def run(traced: Traced[T]) -> T:
    """Run a traced algorithm to completion and return only its answer.

    Use this when you want the result and do not care about the animation, which
    is the case in most of the unit tests and in every benchmark.
    """
    while True:
        try:
            next(traced)
        except StopIteration as stop:
            return stop.value


def record(traced: Traced[T]) -> tuple[T, list[Step]]:
    """Run a traced algorithm and keep every step it produced.

    Returns a pair of (answer, steps). The Streamlit app calls this once and
    then lets you scrub back and forth through the list of steps.

    Be aware that this holds the whole run in memory, so it is meant for the
    small inputs you actually want to watch, not for a million element sort.
    """
    steps: list[Step] = []
    while True:
        try:
            steps.append(next(traced))
        except StopIteration as stop:
            return stop.value, steps


# How large a collection may be before a step stops carrying a copy of it.
#
# This number exists because of a bug the benchmarks found on the last day. Every
# sorting step used to carry a full copy of the array so the visualiser could draw
# that frame. Copying an n element list on each of n log n steps is O(n^2 log n)
# work, so merge sort *timed* as quadratic while its comparison count was still
# n log n. The tracing layer was quietly distorting the very thing being measured.
#
# The fix is to snapshot only when the collection is small enough to be worth
# animating. Nobody watches a ten thousand element sort frame by frame, and the
# benchmarks never look at the data, so above this size the field is simply None
# and the step still carries the indices that changed.
SNAPSHOT_LIMIT = 256


def snapshot(values: Any) -> list[Any] | None:
    """A copy of a collection for the animation, or None when it is too large.

    Callers building a Step should use this rather than `list(values)`, so that
    tracing stays cheap on the large inputs the benchmarks use. See
    `SNAPSHOT_LIMIT` for the measurement that made this necessary.
    """
    return list(values) if len(values) <= SNAPSHOT_LIMIT else None


def count_kinds(steps: list[Step]) -> dict[str, int]:
    """Count how many steps of each kind happened.

    This is how the benchmarks report comparisons and swaps without adding
    counters to the algorithms themselves.
    """
    counts: dict[str, int] = {}
    for step in steps:
        counts[step.kind] = counts.get(step.kind, 0) + 1
    return counts
