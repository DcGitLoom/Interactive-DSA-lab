"""The logic behind the visualiser, with no Streamlit in it.

This module exists because of a testing problem. A Streamlit app is awkward to
test: it needs a browser, a server and a session, and the checks end up being about
widgets rather than about behaviour.

So everything that can be decided without a screen lives here as ordinary
functions: which frames to draw, which cells to highlight, what to compare in a
race, how to describe a step. `app/main.py` is then a thin layer that calls these
and draws the result.

That split means the interesting part of the app is covered by the same test suite
as the rest of the project, and the untested part is small enough to check by
looking at it. It is the same instinct as separating the tracing from the drawing:
**keep the decisions in code you can test, and let the drawing be dumb.**
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from dsalab.algorithms.sorting import COMPARISON_SORTS
from dsalab.complexity import Verdict, detect, doubling_sizes
from dsalab.tracing import Step, count_kinds, record

# How each kind of step should be coloured. Keeping this here rather than in the
# drawing code means the visual language stays consistent across every animation,
# and it can be tested.
STEP_COLOURS: dict[str, str] = {
    "compare": "#f4a261",
    "swap": "#e76f51",
    "shift": "#e76f51",
    "place": "#2a9d8f",
    "insert": "#2a9d8f",
    "merged": "#2a9d8f",
    "split": "#8ab6d6",
    "pivot": "#9d4edd",
    "done": "#2a9d8f",
    "default": "#adb5bd",
}

MAXIMUM_ANIMATED_SIZE = 60


@dataclass
class Frame:
    """One drawable moment of a run.

    Attributes:
        index: Which step this is, counting from zero.
        values: The data as it stood at that moment.
        highlighted: Positions the step was working on.
        colour: What colour to draw the highlight.
        note: The plain English caption.
        kind: The step kind, for the counters.
    """

    index: int
    values: list[Any]
    highlighted: list[int]
    colour: str
    note: str
    kind: str


def colour_for(kind: str) -> str:
    return STEP_COLOURS.get(kind, STEP_COLOURS["default"])


def build_frames(values: Sequence[Any], algorithm: str) -> list[Frame]:
    """Run a sorting algorithm and turn its steps into drawable frames.

    The array snapshot in a step can be None, because day 20's benchmark work made
    snapshots stop above a size limit. The visualiser is only ever used on small
    inputs, so that limit is never reached here, but carrying the last known array
    forward means a missing snapshot degrades into a still picture rather than a
    crash.
    """
    if algorithm not in COMPARISON_SORTS:
        raise ValueError(f"unknown algorithm {algorithm!r}, expected one of "
                         f"{sorted(COMPARISON_SORTS)}")

    _, steps = record(COMPARISON_SORTS[algorithm](list(values)))

    frames: list[Frame] = []
    current = list(values)

    for index, step in enumerate(steps):
        snapshot = step.data.get("array")
        if snapshot is not None:
            current = list(snapshot)

        highlighted = step.data.get("indices", [])
        if not highlighted and "index" in step.data:
            highlighted = [step.data["index"]]

        frames.append(Frame(
            index=index,
            values=list(current),
            highlighted=list(highlighted),
            colour=colour_for(step.kind),
            note=step.note,
            kind=step.kind,
        ))

    return frames


@dataclass
class RaceResult:
    """One competitor in a race between two algorithms on identical input."""

    name: str
    frames: list[Frame]
    comparisons: int
    swaps: int
    total_steps: int

    @property
    def work(self) -> int:
        """Comparisons plus moves, which is the fair single number to race on.

        Not seconds. Timing in a browser session measures the machine and the
        interpreter as much as the algorithm, while counted operations are exactly
        what the complexity claims are about.
        """
        return self.comparisons + self.swaps


@dataclass
class Race:
    """Two algorithms run on the same input, for side by side comparison."""

    values: list[Any]
    left: RaceResult
    right: RaceResult

    @property
    def length(self) -> int:
        """How many frames the animation runs for: the longer of the two.

        The shorter competitor holds its final frame once finished, which is what
        makes the finish visible rather than the animation simply stopping.
        """
        return max(len(self.left.frames), len(self.right.frames))

    def winner(self) -> str | None:
        """Whichever did less work, or None on a tie."""
        if self.left.work == self.right.work:
            return None
        return self.left.name if self.left.work < self.right.work else self.right.name

    def verdict(self) -> str:
        """A sentence explaining the result, which is the point of the race."""
        winner = self.winner()
        if winner is None:
            return (
                f"Both did exactly {self.left.work} units of work on this input, which "
                "happens more often than people expect on small or nearly sorted data."
            )

        loser = self.right if winner == self.left.name else self.left
        champion = self.left if winner == self.left.name else self.right
        ratio = loser.work / champion.work if champion.work else float("inf")

        return (
            f"{champion.name} did {champion.work} units of work and {loser.name} did "
            f"{loser.work}, so {champion.name} was {ratio:.1f} times cheaper on this "
            "input. Try sorted input to see the ranking change."
        )


def race(values: Sequence[Any], left: str, right: str) -> Race:
    """Run two sorting algorithms on identical input and compare the work done.

    Identical input is the whole point. Racing on two different random arrays
    measures the arrays as much as the algorithms, which is the mistake that makes
    most informal benchmarks meaningless.
    """
    def compete(name: str) -> RaceResult:
        frames = build_frames(values, name)
        kinds = count_kinds([
            Step(frame.kind, frame.note) for frame in frames
        ])
        return RaceResult(
            name=name,
            frames=frames,
            comparisons=kinds.get("compare", 0),
            swaps=kinds.get("swap", 0) + kinds.get("shift", 0),
            total_steps=len(frames),
        )

    return Race(list(values), compete(left), compete(right))


def frame_at(result: RaceResult, index: int) -> Frame | None:
    """The frame at a point in time, holding the last one once finished.

    Returning the final frame rather than None past the end is what lets a race
    show the winner sitting still while the loser is still working, which is the
    most legible way to display the difference.
    """
    if not result.frames:
        return None
    return result.frames[min(index, len(result.frames) - 1)]


def measure_algorithm(algorithm: str, sizes: Sequence[int] | None = None) -> Verdict:
    """Run the complexity detective over one sorting algorithm.

    Counts comparison steps rather than timing, because a browser session is a
    hostile place to measure time and because counting is deterministic, so the
    same input always gives the same verdict.
    """
    sizes = list(sizes or doubling_sizes(32, 5))

    def comparisons(size: int) -> float:
        import random

        values = list(range(size))
        random.Random(size).shuffle(values)
        _, steps = record(COMPARISON_SORTS[algorithm](values))
        return float(count_kinds(steps).get("compare", 0))

    return detect(sizes, [comparisons(size) for size in sizes])


@dataclass
class InvariantPanel:
    """What the app shows beside a self checking structure.

    Attributes:
        rules: Every rule the structure states, in the order it states them.
        broken: The rules currently broken, with the detail.
        healthy: Whether everything holds.
    """

    rules: list[str]
    broken: dict[str, str] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return not self.broken

    def status_of(self, rule: str) -> str:
        """A short marker for one rule, for the app to draw next to it."""
        return "broken" if rule in self.broken else "holds"


def inspect(structure: Any, known_rules: Sequence[str] = ()) -> InvariantPanel:
    """Ask a structure for its rules and which of them currently hold.

    `known_rules` lets the app list rules that are currently satisfied, since a
    structure only reports the ones it has **broken**. Without it the panel could
    only ever show failures, and showing all five red black rules with four ticks
    and one cross is far more useful than showing one cross alone.
    """
    violations = structure.check_invariants()
    broken = {violation.rule: violation.detail for violation in violations}

    rules = list(known_rules)
    for rule in broken:
        if rule not in rules:
            rules.append(rule)

    return InvariantPanel(rules=rules, broken=broken)


RED_BLACK_RULES = [
    "rule 1: every node is red or black",
    "rule 2: the root is black",
    "rule 3: every empty position counts as black",
    "rule 4: a red node never has a red child",
    "rule 5: every path has the same number of black nodes",
    "search order holds all the way down",
]

AVL_RULES = [
    "search order holds all the way down",
    "every node's stored height is correct",
    "no node's subtrees differ in height by more than one",
    "the recorded size matches the tree",
]


def tree_layout(tree: Any, kind: str = "bst") -> list[dict[str, Any]]:
    """Positions for drawing a binary tree, one entry per node.

    An inorder walk gives each node its horizontal position, which is the
    standard trick for laying out a tree without any nodes overlapping: inorder
    visits left to right, so the visit number **is** the x coordinate. Depth gives
    the vertical position.

    Returns dictionaries rather than a class because this crosses into the drawing
    code, and a plain mapping is what every chart library wants.
    """
    nodes: list[dict[str, Any]] = []
    position = 0

    def walk(node: Any, depth: int, parent: Any) -> None:
        nonlocal position
        if node is None or (kind == "red_black" and node is getattr(tree, "nil", None)):
            return

        walk(node.left, depth + 1, node)
        nodes.append({
            "value": node.value,
            "x": position,
            "y": -depth,
            "colour": getattr(node, "colour", None),
            "parent": parent.value if parent is not None else None,
        })
        position += 1
        walk(node.right, depth + 1, node)

    walk(tree.root, 0, None)
    return nodes


def describe_step(step: Step) -> str:
    """One line for the caption under an animation."""
    return step.note


def available_algorithms() -> list[str]:
    """The sorting algorithms the app can animate, in a sensible teaching order."""
    order = ["bubble", "selection", "insertion", "shell", "merge", "quick"]
    return [name for name in order if name in COMPARISON_SORTS]


def suggest_input(kind: str, size: int = 20, seed: int = 20260904) -> list[int]:
    """Sample inputs worth trying, each of which shows something different.

    The choice of input is often more instructive than the choice of algorithm,
    which is why these are offered as named options rather than leaving the user to
    type numbers.
    """
    import random

    rng = random.Random(seed)

    if kind == "random":
        values = list(range(1, size + 1))
        rng.shuffle(values)
        return values
    if kind == "sorted":
        return list(range(1, size + 1))
    if kind == "reversed":
        return list(range(size, 0, -1))
    if kind == "nearly sorted":
        values = list(range(1, size + 1))
        for _ in range(max(1, size // 10)):
            index = rng.randrange(size - 1)
            values[index], values[index + 1] = values[index + 1], values[index]
        return values
    if kind == "few distinct":
        return [rng.choice([1, 2, 3]) for _ in range(size)]

    raise ValueError(f"unknown input kind {kind!r}")


INPUT_KINDS = ["random", "sorted", "reversed", "nearly sorted", "few distinct"]

INPUT_EXPLANATIONS = {
    "random": "The usual case, and the one average complexity describes.",
    "sorted": "The best case for insertion sort and the worst case for a quick sort "
              "that takes the first element as its pivot.",
    "reversed": "The worst case for insertion sort: every item travels the full width.",
    "nearly sorted": "Where adaptive sorts shine and merge sort gains nothing, because "
                     "it does the same work whatever the input.",
    "few distinct": "Hard on Lomuto partitioning, where equal values all go to one side.",
}
