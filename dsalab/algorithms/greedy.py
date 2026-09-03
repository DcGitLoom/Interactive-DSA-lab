"""Greedy algorithms: take the best looking option now and never reconsider.

A greedy algorithm makes the choice that looks best at this moment and never goes
back. That is the fastest thing you can do, and it is right surprisingly often, and
wrong often enough that you must always ask why it works.

Greedy is provably optimal only when the problem has two properties:

1. **The greedy choice property.** Some optimal solution contains the choice that
   looks best right now. This is what the exchange argument proves: take any
   optimal solution, swap the greedy choice in, and show the result is no worse.
2. **Optimal substructure.** After making that choice, what remains is a smaller
   version of the same problem.

The minimum spanning trees on day 18 have both, which is why Prim and Kruskal are
provably correct. The problems here split into two groups, and the split is the
lesson:

**Greedy is optimal**: activity selection, fractional knapsack, Huffman coding.
Each has a proof, sketched where it lives.

**Greedy is not optimal**: 0/1 knapsack, coin change with arbitrary coin systems.
These look almost identical to the ones above, and the difference is subtle. That
is what `greedy_vs_dp.py` exists for: rather than describing the failure, it
searches for a concrete input where greedy loses and shows you both answers.
"""

from __future__ import annotations

from dataclasses import dataclass

from dsalab.structures.heap import BinaryHeap
from dsalab.tracing import Step, Traced, run


@dataclass(frozen=True)
class Activity:
    """Something that occupies a room or a machine from `start` until `finish`."""

    name: str
    start: int
    finish: int


def activity_selection(activities: list[Activity]) -> list[Activity]:
    return run(activity_selection_traced(activities))


def activity_selection_traced(activities: list[Activity]) -> Traced[list[Activity]]:
    """Fit as many non overlapping activities as possible into one room. O(n log n).

    The greedy rule is **always take the activity that finishes earliest** among
    those that still fit. That is optimal, and the proof is worth knowing because
    it is the cleanest exchange argument there is:

    Take any optimal schedule. Look at its first activity. Replace it with the one
    that finishes earliest overall. That one finishes no later, so it cannot
    conflict with anything the original schedule had after it, and the count is
    unchanged. So there is an optimal schedule starting with the greedy choice.
    Repeat on what remains.

    The rules that sound just as reasonable and are all wrong:

    * **Shortest activity first**: a short activity can straddle the boundary
      between two others and block both.
    * **Earliest start first**: one activity starting at 9 and running all day
      blocks everything.
    * **Fewest conflicts first**: better, and still not optimal.

    That four way comparison is the real content of this problem. "Greedy works
    here" is not the lesson; "greedy works here **with this particular rule**" is.
    """
    ordered = sorted(activities, key=lambda activity: activity.finish)
    chosen: list[Activity] = []
    free_from = float("-inf")

    for activity in ordered:
        if activity.start >= free_from:
            chosen.append(activity)
            free_from = activity.finish
            yield Step(
                "take",
                f"{activity.name!r} runs {activity.start} to {activity.finish} and the "
                f"room is free, so it is taken. Finishing earliest leaves the most room "
                "for whatever comes next.",
                {"activity": activity.name, "free_from": free_from, "count": len(chosen)},
            )
        else:
            yield Step(
                "skip",
                f"{activity.name!r} starts at {activity.start} but the room is busy until "
                f"{free_from}, so it cannot be taken.",
                {"activity": activity.name, "free_from": free_from},
            )

    return chosen


def activity_selection_by_shortest(activities: list[Activity]) -> list[Activity]:
    """The tempting wrong rule, kept so the failure can be demonstrated.

    Picking the shortest activity first sounds sensible and is not optimal. A short
    activity that straddles the boundary between two longer ones blocks both, so
    taking it costs one slot rather than gaining one.
    """
    chosen: list[Activity] = []
    taken: list[Activity] = []

    for activity in sorted(activities, key=lambda item: item.finish - item.start):
        if all(activity.finish <= other.start or activity.start >= other.finish
               for other in taken):
            taken.append(activity)
            chosen.append(activity)

    return sorted(chosen, key=lambda activity: activity.start)


@dataclass(frozen=True)
class Item:
    """Something with a weight and a value, for the knapsack problems."""

    name: str
    weight: float
    value: float

    @property
    def density(self) -> float:
        """Value per unit of weight, which is what the greedy rule sorts by."""
        return self.value / self.weight if self.weight else float("inf")


def fractional_knapsack(items: list[Item], capacity: float) -> tuple[float, dict[str, float]]:
    return run(fractional_knapsack_traced(items, capacity))


def fractional_knapsack_traced(
    items: list[Item], capacity: float
) -> Traced[tuple[float, dict[str, float]]]:
    """Fill a bag of limited capacity, allowed to take fractions of items. O(n log n).

    Greedy is optimal here: **take items in order of value per unit weight**, and
    fill the last bit of space with a fraction of the next item.

    The proof: if any space in the bag is filled with something of lower density
    while something of higher density is available, swapping a unit of weight
    between them increases the total. So no better arrangement exists.

    Now the important part. **The 0/1 version, where items cannot be split, is not
    solvable this way at all.** It is NP hard, and the same greedy rule can be
    badly wrong, because the last item may not fit and the space is wasted.

    The two problems differ by one word, "fractional", and that word is the entire
    difference between an O(n log n) greedy solution and a problem with no known
    efficient exact algorithm. `greedy_vs_dp.py` finds inputs where the greedy rule
    loses on the 0/1 version, and `dp.py` solves it properly.
    """
    if capacity < 0:
        raise ValueError("capacity cannot be negative")

    taken: dict[str, float] = {}
    total = 0.0
    space = capacity

    for item in sorted(items, key=lambda item: item.density, reverse=True):
        if space <= 0:
            break

        if item.weight <= space:
            taken[item.name] = 1.0
            total += item.value
            space -= item.weight
            yield Step(
                "take-all",
                f"{item.name!r} is worth {item.density:.2f} per unit, the best available, "
                f"and it fits, so all of it goes in. {space:.2f} space left.",
                {"item": item.name, "fraction": 1.0, "total": total},
            )
        else:
            fraction = space / item.weight
            taken[item.name] = fraction
            total += item.value * fraction
            yield Step(
                "take-part",
                f"Only {space:.2f} space is left, so {fraction:.0%} of {item.name!r} goes "
                "in. Being able to split the item is exactly why greedy is optimal here.",
                {"item": item.name, "fraction": fraction, "total": total},
            )
            space = 0

    return total, taken


def greedy_knapsack_01(items: list[Item], capacity: float) -> tuple[float, list[str]]:
    """The same greedy rule applied to the 0/1 problem, where it is not optimal.

    Kept deliberately, so `greedy_vs_dp.py` has something to compare against the
    correct dynamic programming answer, and so the failure can be shown with a
    concrete input rather than described.
    """
    taken: list[str] = []
    total = 0.0
    space = capacity

    for item in sorted(items, key=lambda item: item.density, reverse=True):
        if item.weight <= space:
            taken.append(item.name)
            total += item.value
            space -= item.weight

    return total, taken


def greedy_coin_change(coins: list[int], amount: int) -> list[int] | None:
    return run(greedy_coin_change_traced(coins, amount))


def greedy_coin_change_traced(coins: list[int], amount: int) -> Traced[list[int] | None]:
    """Make an amount from the fewest coins, always taking the largest that fits.

    This is how every human makes change, and for the coin systems in actual use
    (1, 2, 5, 10, 20, 50 and so on) it is optimal. Such systems are called
    canonical, and they are designed that way on purpose.

    For an arbitrary set of coins it is **not** optimal, and the standard example
    is coins of 1, 3 and 4 making 6: greedy takes 4 then 1 then 1, three coins,
    while 3 and 3 is two. There is nothing exotic about that coin system, which is
    the point.

    Worse, greedy can fail to find any answer at all when one exists. With coins of
    3 and 4 making 6, it takes a 4 and then cannot make the remaining 2, and
    reports failure even though 3 and 3 works.

    `greedy_vs_dp.py` searches for these cases automatically rather than relying on
    the two everybody quotes.
    """
    if amount < 0:
        raise ValueError("cannot make a negative amount")

    used: list[int] = []
    remaining = amount

    for coin in sorted(coins, reverse=True):
        while coin <= remaining:
            used.append(coin)
            remaining -= coin
            yield Step(
                "take",
                f"Took a {coin}, leaving {remaining} to make.",
                {"coin": coin, "remaining": remaining, "used": len(used)},
            )

    if remaining != 0:
        yield Step(
            "stuck",
            f"{remaining} left over and no coin fits. Greedy has failed, which does not "
            "prove the amount cannot be made, only that this method cannot make it.",
            {"remaining": remaining},
        )
        return None

    return used


@dataclass
class HuffmanNode:
    """A node of the Huffman tree: either a symbol or a merge of two subtrees."""

    weight: int
    symbol: str | None = None
    left: HuffmanNode | None = None
    right: HuffmanNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.symbol is not None


def huffman_coding(text: str) -> tuple[dict[str, str], HuffmanNode | None]:
    return run(huffman_coding_traced(text))


def huffman_coding_traced(text: str) -> Traced[tuple[dict[str, str], HuffmanNode | None]]:
    """Build the optimal prefix code for a piece of text. O(n log n).

    The problem: give each symbol a binary code so that the total encoded length is
    as short as possible, with no code being a prefix of another (so the decoder
    never needs a separator).

    The greedy rule: **repeatedly merge the two least frequent symbols.** They end
    up deepest in the tree, so they get the longest codes, which is right because
    they are used least.

    The proof that this is optimal is the same exchange argument as before. In any
    optimal tree the two deepest nodes are siblings, and swapping the two least
    frequent symbols into those positions cannot make the total longer.

    This is a greedy algorithm that is optimal, sitting next to greedy coin change,
    which is not. The difference is that Huffman's exchange argument goes through
    and coin change's does not, and there is no way to tell by looking. **You have
    to do the proof.** That is the real lesson of the day.

    Used in ZIP, JPEG, MP3 and essentially every compression format, usually as one
    stage of several.
    """
    if not text:
        return {}, None

    frequency: dict[str, int] = {}
    for character in text:
        frequency[character] = frequency.get(character, 0) + 1

    if len(frequency) == 1:
        # One distinct symbol: it still needs one bit, since a zero bit code would
        # make the length of the message unrecoverable.
        symbol = next(iter(frequency))
        yield Step("single", f"Only {symbol!r} appears, so it gets the single bit code 0.",
                   {"symbol": symbol})
        return {symbol: "0"}, HuffmanNode(frequency[symbol], symbol)

    # A tie breaking counter keeps the result deterministic. Without it, two nodes
    # of equal weight are ordered by whatever the heap happens to do, and the
    # codes change between runs while staying equally optimal, which makes the
    # output impossible to test.
    order = 0
    heap = BinaryHeap(key=lambda entry: (entry[0], entry[1]))
    for symbol, count in sorted(frequency.items()):
        heap.push((count, order, HuffmanNode(count, symbol)))
        order += 1

    while len(heap) > 1:
        left_weight, _, left = heap.pop()
        right_weight, _, right = heap.pop()

        merged = HuffmanNode(left_weight + right_weight, None, left, right)
        heap.push((merged.weight, order, merged))
        order += 1

        yield Step(
            "merge",
            f"The two rarest remaining groups weigh {left_weight} and {right_weight}, so "
            f"they merge into one of weight {merged.weight}. Being merged first means "
            "sitting deepest, which means the longest codes for the rarest symbols.",
            {"left": left_weight, "right": right_weight, "weight": merged.weight},
        )

    _, _, root = heap.pop()
    codes: dict[str, str] = {}

    def assign(node: HuffmanNode, prefix: str) -> None:
        if node.is_leaf:
            codes[node.symbol] = prefix or "0"
            return
        if node.left:
            assign(node.left, prefix + "0")
        if node.right:
            assign(node.right, prefix + "1")

    assign(root, "")
    return codes, root


def huffman_encoded_length(text: str) -> int:
    """How many bits the text takes with its Huffman code, for comparison."""
    codes, _ = huffman_coding(text)
    return sum(len(codes[character]) for character in text)


def fixed_width_length(text: str) -> int:
    """How many bits the same text takes with equal length codes.

    The comparison that shows what Huffman buys. Fixed width needs
    ceil(log2(distinct symbols)) bits for every symbol; Huffman spends fewer on the
    common ones and more on the rare ones, and comes out ahead whenever the
    frequencies are uneven.
    """
    import math

    distinct = len(set(text))
    if distinct <= 1:
        return len(text)
    return len(text) * math.ceil(math.log2(distinct))


def huffman_decode(bits: str, root: HuffmanNode) -> str:
    """Decode a bit string using the tree, which is why prefix freedom matters.

    Walk down from the root one bit at a time and emit a symbol whenever a leaf is
    reached. No code is a prefix of another, so the moment a leaf is reached the
    symbol is unambiguous and no lookahead or separator is needed. That property is
    the whole reason the codes are built as a tree.
    """
    if root.is_leaf:
        return root.symbol * len(bits)

    decoded: list[str] = []
    node = root

    for bit in bits:
        node = node.left if bit == "0" else node.right
        if node is None:
            raise ValueError("the bit string does not match this tree")
        if node.is_leaf:
            decoded.append(node.symbol)
            node = root

    return "".join(decoded)
