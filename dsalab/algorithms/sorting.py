"""Sorting: the same problem solved eleven ways.

Sorting is the best subject in the course for one reason: the problem is trivial
to state, so every difference between the algorithms is a pure difference in
*method*. Nothing is hidden behind the problem being complicated.

The map of what follows:

**The quadratic three.** Bubble, selection and insertion sort. All O(n^2) and all
worth knowing, because insertion sort genuinely beats the clever algorithms on
small or nearly sorted input, and real library sorts fall back to it.

**The n log n three.** Merge, quick and heap sort. This is the best possible
complexity for a comparison based sort, and the proof of that is in
docs/15-sorting.md. They differ in memory, stability and worst case, and those
differences decide which one a library actually uses.

**The linear ones.** Counting, radix and bucket sort. These beat n log n, which
sounds impossible until you notice they do not compare values at all. They use the
values as addresses instead, so the lower bound simply does not apply to them.

**Shell sort**, which sits between the two groups and is a genuinely clever idea
that is hard to analyse.

Every one of these is traced, so the visualiser can animate them and the tests can
count comparisons and swaps rather than timing them.

## Stability

A sort is **stable** when equal elements keep their original relative order. It
sounds pedantic until you sort a table by one column having already sorted it by
another: with a stable sort the second sort preserves the first as a tiebreak, and
with an unstable one it does not.

Stable here: bubble, insertion, merge, counting, radix.
Not stable: selection, quick, heap, shell.

Stability is usually decided by a single comparison operator. Writing `<` where
`<=` belongs, or the reverse, silently loses it, so there is a test for every
algorithm that claims it.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

from dsalab.tracing import Step, Traced, run

Key = Callable[[Any], Any]


def _identity(value: Any) -> Any:
    return value


# The quadratic sorts


def bubble_sort(values: Sequence[Any], key: Key | None = None) -> list[Any]:
    return run(bubble_sort_traced(values, key))


def bubble_sort_traced(values: Sequence[Any], key: Key | None = None) -> Traced[list[Any]]:
    """Repeatedly swap adjacent items that are out of order.

    O(n^2) comparisons, O(n^2) swaps, O(1) extra memory, stable.

    Its one redeeming feature is the early exit: if a whole pass makes no swaps,
    the list is sorted and the algorithm stops. That makes it **O(n) on already
    sorted input**, which is the best best case of any sort here. It is still the
    wrong choice essentially always, because insertion sort is O(n) on sorted
    input too and much better on everything else.

    Worth noticing: after pass k, the largest k items are in their final places at
    the end, which is why the inner loop shortens each time. Forgetting that makes
    the sort correct but does twice the work.
    """
    key = key or _identity
    items = list(values)

    for outer in range(len(items)):
        swapped = False

        for index in range(len(items) - outer - 1):
            yield Step(
                "compare",
                f"Is {items[index]!r} bigger than {items[index + 1]!r}?",
                {"indices": [index, index + 1], "array": list(items)},
            )
            if key(items[index]) > key(items[index + 1]):
                items[index], items[index + 1] = items[index + 1], items[index]
                swapped = True
                yield Step(
                    "swap",
                    f"Yes, so they swap. {items[index]!r} moves left.",
                    {"indices": [index, index + 1], "array": list(items)},
                )

        if not swapped:
            yield Step(
                "done",
                "A whole pass with no swaps means everything is already in order, "
                "so there is nothing left to do.",
                {"array": list(items)},
            )
            break

    return items


def selection_sort(values: Sequence[Any], key: Key | None = None) -> list[Any]:
    return run(selection_sort_traced(values, key))


def selection_sort_traced(values: Sequence[Any], key: Key | None = None) -> Traced[list[Any]]:
    """Find the smallest remaining item and swap it into place.

    O(n^2) comparisons always, even on sorted input, but only **O(n) swaps**,
    which is the fewest of any sort here.

    That is its one real use. When comparing is cheap but moving is expensive,
    for example sorting large records in place or writing to flash memory where
    writes wear the hardware out, the swap count is what matters and this wins.

    It is **not stable**, and the reason is the long distance swap: moving the
    minimum into position i jumps over everything in between and can throw an
    equal item past its twin. Insertion sort avoids this by shuffling one step at
    a time, which is exactly why it stays stable.
    """
    key = key or _identity
    items = list(values)

    for position in range(len(items)):
        smallest = position

        for index in range(position + 1, len(items)):
            yield Step(
                "compare",
                f"Is {items[index]!r} smaller than the smallest so far, "
                f"{items[smallest]!r}?",
                {"indices": [index, smallest], "array": list(items)},
            )
            if key(items[index]) < key(items[smallest]):
                smallest = index

        if smallest != position:
            items[position], items[smallest] = items[smallest], items[position]
            yield Step(
                "swap",
                f"{items[position]!r} was the smallest left, so it swaps into "
                f"position {position}.",
                {"indices": [position, smallest], "array": list(items)},
            )

    return items


def insertion_sort(values: Sequence[Any], key: Key | None = None) -> list[Any]:
    return run(insertion_sort_traced(values, key))


def insertion_sort_traced(values: Sequence[Any], key: Key | None = None) -> Traced[list[Any]]:
    """Take each item and slide it back into the sorted part behind it.

    O(n^2) worst case, **O(n) on nearly sorted input**, O(1) memory, stable.

    This is the quadratic sort that earns its keep. Two reasons:

    1. It is **adaptive**. The inner loop stops as soon as the item is in place,
       so the cost is proportional to how far out of order the input is, not to
       its size. On data that is almost sorted it is close to linear.
    2. It has tiny constants. No recursion, no extra memory, one comparison and
       one move per step, and it walks memory in order.

    Together those make it the fastest sort available for small inputs, which is
    why real library sorts (Python's Timsort, C++'s introsort) switch to insertion
    sort once a partition drops below a few dozen elements. Day 15's benchmarks
    measure the crossover.

    The comparison is `>` rather than `>=`, which is what keeps it stable: an item
    equal to the one behind it stops immediately instead of sliding past.
    """
    key = key or _identity
    items = list(values)

    for position in range(1, len(items)):
        current = items[position]
        index = position - 1

        while index >= 0:
            yield Step(
                "compare",
                f"Does {items[index]!r} need to move right to make room for "
                f"{current!r}?",
                {"indices": [index, index + 1], "array": list(items)},
            )
            if key(items[index]) <= key(current):
                break

            items[index + 1] = items[index]
            yield Step(
                "shift",
                f"{items[index]!r} shifts one place right.",
                {"indices": [index, index + 1], "array": list(items)},
            )
            index -= 1

        items[index + 1] = current
        yield Step(
            "place",
            f"{current!r} settles into position {index + 1}.",
            {"index": index + 1, "array": list(items)},
        )

    return items


def binary_insertion_sort(values: Sequence[Any], key: Key | None = None) -> list[Any]:
    """Insertion sort that binary searches for the position instead of scanning.

    This looks like a clear improvement and mostly is not, which makes it a good
    lesson. It cuts **comparisons** from O(n^2) to O(n log n), but the **moves**
    are still O(n^2), because the item still has to be physically shifted into
    place. Since moving is usually as expensive as comparing, the total barely
    changes.

    It genuinely helps in one case: when comparison is expensive relative to
    moving, for example sorting long strings or comparing by a computed key.

    It also gives up the adaptive behaviour. Plain insertion sort does one
    comparison per already sorted item; this one does log n regardless, so it is
    actually **slower on nearly sorted input**, which is insertion sort's whole
    reason for existing.
    """
    key = key or _identity
    items = list(values)

    for position in range(1, len(items)):
        current = items[position]
        low, high = 0, position

        while low < high:
            middle = (low + high) // 2
            if key(items[middle]) <= key(current):
                low = middle + 1
            else:
                high = middle

        for index in range(position, low, -1):
            items[index] = items[index - 1]
        items[low] = current

    return items


def shell_sort(values: Sequence[Any], key: Key | None = None) -> list[Any]:
    return run(shell_sort_traced(values, key))


def shell_sort_traced(values: Sequence[Any], key: Key | None = None) -> Traced[list[Any]]:
    """Insertion sort on items spaced a gap apart, with the gap shrinking to 1.

    The idea behind it is genuinely clever. Insertion sort is slow because an item
    far from its home moves one place at a time. Sorting items that are a large gap
    apart first lets things travel a long way cheaply, so by the time the gap
    reaches 1 the array is nearly sorted, and insertion sort is fast on nearly
    sorted input.

    The complexity depends entirely on the gap sequence, and this is the famous
    part: **nobody knows the best one.** With Shell's original halving sequence it
    is O(n^2). With Knuth's 3k+1 sequence, used here, it is O(n^1.5). The best
    known sequences reach about O(n^1.33), and the true optimum is an open problem
    decades later. That is a rare thing to be able to say about an algorithm this
    simple.

    Not stable, because comparing across a gap can jump equal items past each other.
    """
    key = key or _identity
    items = list(values)

    # Knuth's sequence: 1, 4, 13, 40, 121, each 3k+1.
    gap = 1
    while gap < len(items) // 3:
        gap = gap * 3 + 1

    while gap >= 1:
        yield Step("gap", f"Sorting items that are {gap} apart.", {"gap": gap,
                                                                   "array": list(items)})

        for position in range(gap, len(items)):
            current = items[position]
            index = position

            while index >= gap:
                yield Step(
                    "compare",
                    f"Comparing {items[index - gap]!r} and {current!r}, {gap} apart.",
                    {"indices": [index - gap, index], "gap": gap, "array": list(items)},
                )
                if key(items[index - gap]) <= key(current):
                    break
                items[index] = items[index - gap]
                yield Step(
                    "shift",
                    f"{items[index]!r} jumps {gap} places, which is the point of the gap.",
                    {"indices": [index - gap, index], "array": list(items)},
                )
                index -= gap

            items[index] = current

        gap //= 3

    return items


# The n log n sorts


def merge_sort(values: Sequence[Any], key: Key | None = None) -> list[Any]:
    return run(merge_sort_traced(values, key))


def merge_sort_traced(values: Sequence[Any], key: Key | None = None) -> Traced[list[Any]]:
    """Split in half, sort each half, merge them back together.

    O(n log n) always, best case and worst case alike, stable, and it needs **O(n)
    extra memory** for the merging.

    That guaranteed n log n is the point. Quick sort is usually faster but can
    degrade to O(n^2); merge sort never does, which is why it is chosen where a
    predictable bound matters, and why Python's Timsort is built on it.

    Two other things it is uniquely good at, both consequences of merging being
    sequential:

    * **Sorting linked lists**, where it needs no extra memory at all because
      merging just relinks nodes, as day 5 showed.
    * **Sorting data too big for memory**, by sorting chunks that fit, writing
      them out, and merging the files in one streaming pass. This is what
      "external sorting" means and merge sort is essentially the only option.

    Stability comes from `<=` in the merge: when two items are equal the one from
    the left half is taken first, which is the half that came first originally.
    """
    key = key or _identity
    items = list(values)

    def sort(low: int, high: int) -> Traced[None]:
        if high - low <= 1:
            return

        middle = (low + high) // 2
        yield Step(
            "split",
            f"Splitting positions {low} to {high - 1} into two halves at {middle}.",
            {"low": low, "middle": middle, "high": high, "array": list(items)},
        )
        yield from sort(low, middle)
        yield from sort(middle, high)
        yield from merge(low, middle, high)

    def merge(low: int, middle: int, high: int) -> Traced[None]:
        left = items[low:middle]
        right = items[middle:high]
        a = b = 0
        write = low

        while a < len(left) and b < len(right):
            yield Step(
                "compare",
                f"Which goes next, {left[a]!r} or {right[b]!r}?",
                {"left": left[a], "right": right[b], "array": list(items)},
            )
            # <= rather than < is what makes this stable.
            if key(left[a]) <= key(right[b]):
                items[write] = left[a]
                a += 1
            else:
                items[write] = right[b]
                b += 1
            write += 1

        while a < len(left):
            items[write] = left[a]
            a += 1
            write += 1
        while b < len(right):
            items[write] = right[b]
            b += 1
            write += 1

        yield Step(
            "merged",
            f"Positions {low} to {high - 1} are now sorted: {items[low:high]!r}.",
            {"low": low, "high": high, "array": list(items)},
        )

    yield from sort(0, len(items))
    return items


def merge_sort_iterative(values: Sequence[Any], key: Key | None = None) -> list[Any]:
    """Merge sort with no recursion, merging runs of 1, then 2, then 4, and so on.

    Same complexity, same stability, and no call stack, which matters when the
    input could be large enough to hit the recursion limit. It is also the form
    that generalises to external sorting, since each pass reads and writes
    sequentially.

    The bottom up view is a good way to see what merge sort is really doing:
    treating the array as n sorted runs of length 1 and repeatedly halving the
    number of runs, which is where the log n comes from.
    """
    key = key or _identity
    items = list(values)
    size = len(items)
    width = 1

    while width < size:
        for low in range(0, size, width * 2):
            middle = min(low + width, size)
            high = min(low + width * 2, size)

            left, right = items[low:middle], items[middle:high]
            a = b = 0
            write = low

            while a < len(left) and b < len(right):
                if key(left[a]) <= key(right[b]):
                    items[write] = left[a]
                    a += 1
                else:
                    items[write] = right[b]
                    b += 1
                write += 1
            while a < len(left):
                items[write] = left[a]
                a += 1
                write += 1
            while b < len(right):
                items[write] = right[b]
                b += 1
                write += 1

        width *= 2

    return items


def quick_sort(
    values: Sequence[Any],
    key: Key | None = None,
    pivot: str = "median",
    seed: int | None = None,
) -> list[Any]:
    return run(quick_sort_traced(values, key, pivot, seed))


def quick_sort_traced(
    values: Sequence[Any],
    key: Key | None = None,
    pivot: str = "median",
    seed: int | None = None,
) -> Traced[list[Any]]:
    """Pick a pivot, put smaller items left and larger items right, recurse.

    O(n log n) on average, **O(n^2) in the worst case**, O(log n) stack memory,
    not stable.

    Despite the bad worst case it is the fastest general sort in practice, and the
    reason is memory rather than mathematics: partitioning scans straight through
    the array in both directions, which the CPU cache and prefetcher handle
    perfectly. Merge sort's extra array and heap sort's scattered jumps both cost
    more than the complexity suggests. Day 15's benchmark shows this.

    **The pivot choice is the whole algorithm**, and it is why this function takes
    a strategy:

    * `first`: the classic teaching version, and a trap. On already sorted input
      every partition is maximally unbalanced and the sort is O(n^2). Since sorted
      input is extremely common, this was a real source of production outages
      before people understood it.
    * `random`: picks uniformly, so no particular input is bad. An adversary who
      does not know the seed cannot construct a slow case. This is the honest fix.
    * `median`: median of the first, middle and last elements. Cheap, and it makes
      sorted and reverse sorted input the *best* case rather than the worst, since
      the middle element is then the true median. It is what most real
      implementations use.

    The partition here is Lomuto's, which is easier to read. Hoare's original does
    about three times fewer swaps and is in `_hoare_partition` below for
    comparison.
    """
    key = key or _identity
    items = list(values)
    rng = random.Random(seed)

    def choose(low: int, high: int) -> int:
        if pivot == "first":
            return low
        if pivot == "random":
            return rng.randint(low, high)
        if pivot == "median":
            middle = (low + high) // 2
            trio = sorted([(key(items[low]), low), (key(items[middle]), middle),
                           (key(items[high]), high)])
            return trio[1][1]
        raise ValueError(f"unknown pivot strategy {pivot!r}")

    def sort(low: int, high: int) -> Traced[None]:
        if low >= high:
            return

        chosen = choose(low, high)
        items[chosen], items[high] = items[high], items[chosen]
        pivot_value = items[high]
        yield Step(
            "pivot",
            f"Chose {pivot_value!r} as the pivot for positions {low} to {high}.",
            {"pivot": pivot_value, "low": low, "high": high, "array": list(items)},
        )

        boundary = low
        for index in range(low, high):
            yield Step(
                "compare",
                f"Is {items[index]!r} smaller than the pivot {pivot_value!r}?",
                {"indices": [index, high], "array": list(items)},
            )
            if key(items[index]) < key(pivot_value):
                if boundary != index:
                    items[boundary], items[index] = items[index], items[boundary]
                    yield Step(
                        "swap",
                        f"Yes, so {items[boundary]!r} moves to the left side.",
                        {"indices": [boundary, index], "array": list(items)},
                    )
                boundary += 1

        items[boundary], items[high] = items[high], items[boundary]
        yield Step(
            "place",
            f"The pivot {pivot_value!r} lands at position {boundary}, and it is now in "
            "its final place forever.",
            {"index": boundary, "array": list(items)},
        )

        yield from sort(low, boundary - 1)
        yield from sort(boundary + 1, high)

    yield from sort(0, len(items) - 1)
    return items


def quick_sort_hoare(values: Sequence[Any], key: Key | None = None) -> list[Any]:
    """Quick sort using Hoare's original two pointer partition.

    Two pointers walk towards each other, swapping pairs that are both on the
    wrong side. It does about three times fewer swaps than Lomuto's version and
    handles arrays full of equal elements far better, where Lomuto degrades to
    O(n^2) because every element goes to the same side.

    The catch is that the pivot does **not** end up in its final position, so the
    recursion has to include the split point on one side. That off by one is the
    classic way to write an infinite loop here, which is why the simpler Lomuto
    version is what gets taught.
    """
    key = key or _identity
    items = list(values)

    def partition(low: int, high: int) -> int:
        pivot = key(items[(low + high) // 2])
        left, right = low - 1, high + 1

        while True:
            left += 1
            while key(items[left]) < pivot:
                left += 1
            right -= 1
            while key(items[right]) > pivot:
                right -= 1
            if left >= right:
                return right
            items[left], items[right] = items[right], items[left]

    def sort(low: int, high: int) -> None:
        if low < high:
            split = partition(low, high)
            sort(low, split)
            sort(split + 1, high)

    if items:
        sort(0, len(items) - 1)
    return items


# The linear sorts, which do not compare at all


def counting_sort(values: Sequence[int]) -> list[int]:
    return run(counting_sort_traced(values))


def counting_sort_traced(values: Sequence[int]) -> Traced[list[int]]:
    """Count how many of each value there are, then write them out in order.

    O(n + k) time and O(n + k) memory, where k is the range of the values. Stable.

    This beats the n log n lower bound, which sounds impossible. It is allowed
    because **it never compares two values**. It uses each value as an array
    index, which is a completely different kind of operation and is not covered by
    the decision tree argument in docs/15-sorting.md.

    The cost is the assumption. Values must be integers in a known, reasonably
    small range. Sorting a thousand values that range up to a billion needs a
    billion counters and is far worse than any comparison sort. The rule of thumb
    is that this wins when k is around n or smaller.

    The stability is not automatic. It comes from converting the counts into
    running totals and then walking the **input backwards** when placing items.
    Walking forwards produces the same sorted values with equal items reversed,
    which is a bug you cannot see without a stability test.
    """
    items = list(values)
    if not items:
        return items

    low, high = min(items), max(items)
    counts = [0] * (high - low + 1)

    for value in items:
        counts[value - low] += 1
    yield Step(
        "count",
        f"Counted every value in the range {low} to {high}. No comparisons were "
        "made at all, which is why this can beat n log n.",
        {"counts": list(counts), "offset": low},
    )

    for index in range(1, len(counts)):
        counts[index] += counts[index - 1]
    yield Step(
        "totals",
        "Turned the counts into running totals, so each one now says where that "
        "value's block ends in the output.",
        {"totals": list(counts)},
    )

    output = [0] * len(items)
    for value in reversed(items):
        counts[value - low] -= 1
        output[counts[value - low]] = value
        yield Step(
            "place",
            f"Placed {value!r} at position {counts[value - low]}. Walking the input "
            "backwards is what keeps equal values in their original order.",
            {"value": value, "index": counts[value - low], "array": list(output)},
        )

    return output


def radix_sort(values: Sequence[int], base: int = 10) -> list[int]:
    return run(radix_sort_traced(values, base))


def radix_sort_traced(values: Sequence[int], base: int = 10) -> Traced[list[int]]:
    """Sort by one digit at a time, least significant first.

    O(d times (n + base)) where d is the number of digits, so effectively O(n) for
    fixed width numbers. Stable, and it must be: **the stability is what makes the
    algorithm work at all.**

    That is the insight worth carrying away. Sorting by the last digit and then by
    the second to last only produces a correct result if the second pass preserves
    the order the first pass established for ties. Use an unstable sort for the
    per digit pass and radix sort silently produces wrong answers.

    Negative numbers are handled by splitting them out, sorting their absolute
    values, and reversing that half. Most textbook versions quietly assume
    non negative input and produce nonsense otherwise, which is worth calling out.
    """
    items = list(values)
    if len(items) <= 1:
        return items

    negatives = [-value for value in items if value < 0]
    positives = [value for value in items if value >= 0]

    def sort_non_negative(numbers: list[int]) -> Traced[list[int]]:
        if not numbers:
            return []

        result = list(numbers)
        place = 1
        largest = max(result)

        while largest // place > 0:
            buckets: list[list[int]] = [[] for _ in range(base)]
            for value in result:
                buckets[(value // place) % base].append(value)

            result = [value for bucket in buckets for value in bucket]
            yield Step(
                "digit",
                f"Sorted by the digit in the {place}s column, giving {result!r}. This "
                "pass must be stable or the earlier passes are undone.",
                {"place": place, "array": list(result)},
            )
            place *= base

        return result

    sorted_positives = yield from sort_non_negative(positives)
    sorted_negatives = yield from sort_non_negative(negatives)

    return [-value for value in reversed(sorted_negatives)] + sorted_positives


def bucket_sort(values: Sequence[float], buckets: int | None = None) -> list[float]:
    """Spread values into buckets by range, sort each bucket, concatenate.

    O(n) expected when the values are spread evenly, O(n^2) when they all land in
    one bucket. That gap makes it the most input dependent sort here.

    It is the right choice for values known to be roughly uniform over a range,
    such as random floats between 0 and 1 or sensor readings with a known
    distribution. For unknown data it is a gamble, and the failure is not graceful.

    Each bucket is finished with insertion sort, which is the correct choice
    precisely because buckets are small and nearly sorted, exactly where insertion
    sort is at its best.
    """
    items = list(values)
    if len(items) <= 1:
        return items

    count = buckets or len(items)
    low, high = min(items), max(items)

    if low == high:
        return items

    spread: list[list[float]] = [[] for _ in range(count)]
    for value in items:
        index = int((value - low) / (high - low) * (count - 1))
        spread[index].append(value)

    return [value for bucket in spread for value in insertion_sort(bucket)]


# A registry, so the app and the benchmarks can offer them all without repetition

COMPARISON_SORTS: dict[str, Callable[..., Traced[list[Any]]]] = {
    "bubble": bubble_sort_traced,
    "selection": selection_sort_traced,
    "insertion": insertion_sort_traced,
    "shell": shell_sort_traced,
    "merge": merge_sort_traced,
    "quick": quick_sort_traced,
}

ALL_SORTS: dict[str, Callable[[Sequence[Any]], list[Any]]] = {
    "bubble": bubble_sort,
    "selection": selection_sort,
    "insertion": insertion_sort,
    "binary insertion": binary_insertion_sort,
    "shell": shell_sort,
    "merge": merge_sort,
    "merge (iterative)": merge_sort_iterative,
    "quick": quick_sort,
    "quick (Hoare)": quick_sort_hoare,
}

STABLE_SORTS = {"bubble", "insertion", "binary insertion", "merge", "merge (iterative)"}
