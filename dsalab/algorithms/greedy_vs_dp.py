"""The counterexample finder: proving greedy wrong with an actual example.

Every textbook says greedy algorithms are not always optimal. Almost none of them
hand you an input where it fails, and being told a thing is a different experience
from watching it happen.

This module searches for those inputs. Give it a problem where a greedy method and
a correct method disagree, and it hunts through small inputs until it finds one
where they differ, then reports both answers and the gap between them.

Why this is worth building rather than quoting the two examples everybody knows:

1. **A found counterexample is convincing in a way a claim is not.** "Greedy coin
   change is not optimal" is a sentence you nod at. "Coins of 1, 3 and 4 making 6
   takes three coins greedily and two properly" is something you remember.
2. **It generalises.** The same search works for any pair of a fast heuristic and a
   correct method, which is a genuinely useful thing to have when you have written
   a heuristic and want to know whether it is safe.
3. **It fails informatively.** When no counterexample is found after a thorough
   search, that is weak evidence the greedy method might be right, and the report
   says exactly how hard it looked. Weak evidence honestly labelled beats a
   confident guess.

The search is deliberately smallest first. A counterexample with three coins and a
target of 6 teaches something; one with fifty random coins and a target of 9,417
teaches nothing, even though both prove the same point.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any

from dsalab.algorithms.dp import coin_change, knapsack_01
from dsalab.algorithms.greedy import (
    Activity,
    Item,
    activity_selection,
    activity_selection_by_shortest,
    greedy_coin_change,
    greedy_knapsack_01,
)


@dataclass
class Counterexample:
    """One input where the quick method and the correct method disagree.

    Attributes:
        problem: What the problem was.
        inputs: A readable description of the failing input.
        greedy_answer: What the quick method produced.
        correct_answer: What the correct method produced.
        greedy_score: The quality of the greedy answer, as a number.
        correct_score: The quality of the correct answer.
        explanation: One or two sentences of plain English about why greedy lost.
    """

    problem: str
    inputs: str
    greedy_answer: Any
    correct_answer: Any
    greedy_score: float
    correct_score: float
    explanation: str

    @property
    def gap(self) -> float:
        """How much worse greedy was, as a fraction of the correct answer."""
        if self.correct_score == 0:
            return 0.0
        return abs(self.greedy_score - self.correct_score) / abs(self.correct_score)

    def __str__(self) -> str:
        return (
            f"{self.problem}\n"
            f"  Input:   {self.inputs}\n"
            f"  Greedy:  {self.greedy_answer!r} (score {self.greedy_score})\n"
            f"  Correct: {self.correct_answer!r} (score {self.correct_score})\n"
            f"  Why:     {self.explanation}"
        )


@dataclass
class SearchReport:
    """What a search found, or did not find, and how hard it looked.

    A search that finds nothing is not the same as a proof that nothing exists, and
    this type exists so the difference cannot be quietly lost.
    """

    problem: str
    tried: int
    counterexamples: list[Counterexample] = field(default_factory=list)

    @property
    def found_any(self) -> bool:
        return bool(self.counterexamples)

    @property
    def smallest(self) -> Counterexample | None:
        """The first one found, which is the smallest since the search is ordered."""
        return self.counterexamples[0] if self.counterexamples else None

    def summary(self) -> str:
        if not self.counterexamples:
            return (
                f"{self.problem}: no counterexample after {self.tried} inputs. That is "
                "weak evidence the greedy method may be right here, not a proof. A proof "
                "needs an exchange argument, not a search."
            )
        return (
            f"{self.problem}: found {len(self.counterexamples)} counterexample(s) in "
            f"{self.tried} inputs. The smallest is:\n{self.smallest}"
        )


def find_coin_change_counterexample(
    max_coin: int = 8, max_coins_in_system: int = 3, max_amount: int = 20
) -> SearchReport:
    """Search for a coin system where taking the largest coin first is not optimal.

    Enumerates every coin system up to the given size, smallest first, and every
    amount up to the limit. The classic answer, coins of 1, 3 and 4 making 6, falls
    out within the first few dozen attempts.

    A coin of 1 is always included, so that every amount is makeable and any failure
    is about the **number** of coins rather than about being unable to pay at all.
    The separate `find_coin_change_failure` below covers that second, more dramatic
    kind of failure.
    """
    report = SearchReport("Greedy coin change (fewest coins)", 0)

    for size in range(2, max_coins_in_system + 1):
        for others in itertools.combinations(range(2, max_coin + 1), size - 1):
            coins = [1, *others]

            for amount in range(1, max_amount + 1):
                report.tried += 1

                greedy = greedy_coin_change(coins, amount)
                correct = coin_change(coins, amount)

                if greedy is None or correct is None:
                    continue

                if len(greedy) > len(correct):
                    report.counterexamples.append(Counterexample(
                        problem="Greedy coin change is not optimal",
                        inputs=f"coins {coins}, making {amount}",
                        greedy_answer=greedy,
                        correct_answer=correct,
                        greedy_score=len(greedy),
                        correct_score=len(correct),
                        explanation=(
                            f"Taking the largest coin that fits ({max(greedy)}) leaves a "
                            f"remainder that needs {len(greedy) - 1} more coins. Refusing "
                            f"the biggest coin and using {correct} instead needs only "
                            f"{len(correct)}. The greedy choice is locally best and "
                            "globally worse."
                        ),
                    ))
                    if len(report.counterexamples) >= 5:
                        return report

    return report


def find_coin_change_failure(max_coin: int = 9, max_amount: int = 30) -> SearchReport:
    """Search for a case where greedy finds no answer at all, though one exists.

    This is the worse failure and the more surprising one. Without a coin of 1,
    greedy can take a large coin and strand itself on a remainder it cannot make,
    then report the amount as impossible when it is not. Coins of 3 and 4 making 6
    is the smallest case: greedy takes a 4, cannot make 2, and gives up, while 3 and
    3 works.

    A method that returns a worse answer is a nuisance. A method that returns "no
    answer exists" when one does is a different category of wrong, and it is worth
    seeing that a greedy algorithm can do it.
    """
    report = SearchReport("Greedy coin change failing entirely", 0)

    for size in (2, 3):
        for coins in itertools.combinations(range(2, max_coin + 1), size):
            for amount in range(1, max_amount + 1):
                report.tried += 1

                greedy = greedy_coin_change(list(coins), amount)
                correct = coin_change(list(coins), amount)

                if greedy is None and correct is not None:
                    report.counterexamples.append(Counterexample(
                        problem="Greedy coin change reports failure when an answer exists",
                        inputs=f"coins {list(coins)}, making {amount}",
                        greedy_answer=None,
                        correct_answer=correct,
                        greedy_score=float("inf"),
                        correct_score=len(correct),
                        explanation=(
                            f"Greedy takes the largest coin that fits and strands itself "
                            f"on a remainder it cannot make. {correct} works perfectly "
                            "well. Reporting no answer when one exists is a worse failure "
                            "than reporting a poor answer."
                        ),
                    ))
                    if len(report.counterexamples) >= 5:
                        return report

    return report


def find_knapsack_counterexample(
    attempts: int = 2000, seed: int = 20260903
) -> SearchReport:
    """Search for a 0/1 knapsack where sorting by value per weight is not optimal.

    Greedy is provably optimal for the **fractional** version, where items can be
    split. For whole items it can be arbitrarily bad, because taking the densest
    item can leave space that nothing fits into.

    The search is random rather than exhaustive, because the space of weights and
    values is far too large to enumerate, but the sizes are kept tiny so that any
    counterexample found is small enough to check by hand.
    """
    rng = random.Random(seed)
    report = SearchReport("Greedy 0/1 knapsack (highest value per weight first)", 0)

    for _ in range(attempts):
        report.tried += 1

        count = rng.randint(2, 4)
        capacity = rng.randint(4, 12)
        items = [
            Item(f"item{index}", float(rng.randint(1, capacity)), float(rng.randint(1, 20)))
            for index in range(count)
        ]

        greedy_value, greedy_items = greedy_knapsack_01(items, capacity)
        best_value, best_items = knapsack_01(items, capacity)

        if best_value > greedy_value:
            described = ", ".join(
                f"{item.name}(w={item.weight:g}, v={item.value:g})" for item in items
            )
            report.counterexamples.append(Counterexample(
                problem="Greedy 0/1 knapsack is not optimal",
                inputs=f"capacity {capacity}, items: {described}",
                greedy_answer=greedy_items,
                correct_answer=best_items,
                greedy_score=greedy_value,
                correct_score=best_value,
                explanation=(
                    "Greedy took the item with the best value per unit weight, which left "
                    "space too small for anything else. Passing over that item allows a "
                    f"combination worth {best_value} instead of {greedy_value}. Splitting "
                    "items would make greedy optimal, and being unable to split is exactly "
                    "what breaks it."
                ),
            ))
            if len(report.counterexamples) >= 5:
                return report

    return report


def find_activity_selection_counterexample(
    attempts: int = 2000, seed: int = 20260903
) -> SearchReport:
    """Search for a schedule where "shortest activity first" loses to "earliest finish".

    Both rules are greedy and only one is optimal, which makes this the sharpest
    demonstration in the module: the failure is not greed itself, it is the
    **choice of greedy rule**.

    Picking the shortest activity sounds obviously sensible. It fails when a short
    activity straddles the boundary between two longer ones and blocks both, so
    taking it gains one slot and costs two.
    """
    rng = random.Random(seed)
    report = SearchReport("Activity selection with the wrong greedy rule", 0)

    for _ in range(attempts):
        report.tried += 1

        count = rng.randint(3, 5)
        activities = []
        for index in range(count):
            start = rng.randint(0, 12)
            activities.append(Activity(f"a{index}", start, start + rng.randint(1, 6)))

        best = activity_selection(activities)
        shortest_first = activity_selection_by_shortest(activities)

        if len(shortest_first) < len(best):
            described = ", ".join(
                f"{item.name}[{item.start},{item.finish}]" for item in activities
            )
            report.counterexamples.append(Counterexample(
                problem="Shortest activity first is not optimal",
                inputs=described,
                greedy_answer=[item.name for item in shortest_first],
                correct_answer=[item.name for item in best],
                greedy_score=len(shortest_first),
                correct_score=len(best),
                explanation=(
                    "The shortest activity straddles the boundary between two others and "
                    "blocks both, so taking it gains one slot and costs two. Choosing the "
                    "activity that finishes earliest instead leaves the most room for "
                    "whatever comes next, which is why that rule is the optimal one."
                ),
            ))
            if len(report.counterexamples) >= 5:
                return report

    return report


def run_all_searches() -> list[SearchReport]:
    """Every search, for the app and for a quick look from the command line."""
    return [
        find_coin_change_counterexample(),
        find_coin_change_failure(),
        find_knapsack_counterexample(),
        find_activity_selection_counterexample(),
    ]


if __name__ == "__main__":
    for report in run_all_searches():
        print(report.summary())
        print()
