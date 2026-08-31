"""Searching: binary search and the variants people actually need.

Binary search is famous for being easy to describe and hard to write. Jon Bentley
found that around ninety percent of professional programmers could not write a
correct one when asked, and the version in Java's standard library carried an
overflow bug for nine years before anyone noticed.

The two classic mistakes are both here in the comments where they belong:

1. **The overflow.** `(low + high) // 2` overflows when both are large, in any
   language with fixed width integers. `low + (high - low) // 2` cannot.
   Python's integers are unbounded so it cannot happen here, and the correct form
   is used anyway, because the habit is what transfers.
2. **The boundary.** Whether the loop is `while low <= high` or `while low < high`,
   and whether the update is `middle - 1` or `middle`, decides both correctness
   and termination. Get it wrong by one and you either miss the answer or loop
   forever.

The variants matter more than the plain search does. Real questions are rarely
"is this value present". They are "where would it go", "what is the first entry
after this timestamp", "how many are below this threshold". Those need the
boundary variants below, and writing them by hand each time is exactly how the off
by one bugs get in.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dsalab.tracing import Step, Traced, run


def binary_search(values: Sequence[Any], target: Any) -> int:
    return run(binary_search_traced(values, target))


def binary_search_traced(values: Sequence[Any], target: Any) -> Traced[int]:
    """Find `target` in a sorted sequence, returning its index or -1.

    O(log n) comparisons, O(1) memory.

    The requirement people forget: **the input must already be sorted**. Binary
    search on unsorted data does not fail loudly, it quietly returns wrong
    answers, which is far worse. If you are going to sort first just to search
    once, you have spent O(n log n) to save O(n), and a linear scan was the better
    choice. Binary search pays off when you search many times over the same data.
    """
    low, high = 0, len(values) - 1

    while low <= high:
        # Written this way rather than (low + high) // 2 out of habit. In C or
        # Java the obvious form overflows for large arrays, which is the bug that
        # sat in the JDK for nine years.
        middle = low + (high - low) // 2

        yield Step(
            "probe",
            f"Looking at position {middle}, which holds {values[middle]!r}. "
            f"{high - low + 1} candidate(s) remain.",
            {"low": low, "middle": middle, "high": high, "value": values[middle]},
        )

        if values[middle] == target:
            yield Step("found", f"{target!r} found at position {middle}.",
                       {"index": middle, "value": target})
            return middle

        if values[middle] < target:
            low = middle + 1
            yield Step(
                "discard",
                f"{values[middle]!r} is too small, so everything up to and including "
                f"position {middle} can be thrown away.",
                {"low": low, "high": high},
            )
        else:
            high = middle - 1
            yield Step(
                "discard",
                f"{values[middle]!r} is too big, so everything from position {middle} "
                "onwards can be thrown away.",
                {"low": low, "high": high},
            )

    yield Step("missing", f"The range is empty, so {target!r} is not here.", {"target": target})
    return -1


def binary_search_recursive(values: Sequence[Any], target: Any) -> int:
    """The same search written recursively, for comparison.

    Identical complexity in time, but it uses O(log n) stack space where the loop
    uses none. Binary search is the standard example of a recursion that should be
    a loop: there is nothing to do on the way back up, so the stack frames carry
    no information and exist only to be unwound.

    Recognising that shape is useful in general. **When a recursive function's last
    action is the recursive call and nothing happens afterwards, the recursion is
    doing no work that a loop could not do.**
    """

    def search(low: int, high: int) -> int:
        if low > high:
            return -1
        middle = low + (high - low) // 2
        if values[middle] == target:
            return middle
        if values[middle] < target:
            return search(middle + 1, high)
        return search(low, middle - 1)

    return search(0, len(values) - 1)


def lower_bound(values: Sequence[Any], target: Any) -> int:
    """The index of the first element not smaller than `target`.

    Equivalently: where `target` would be inserted to keep the list sorted, taking
    the earliest such position. This is Python's `bisect_left`.

    This is the variant that answers the questions people actually have:

    * The first log entry at or after a timestamp.
    * How many values are strictly below a threshold, which is just this index.
    * Where to insert while keeping duplicates in their original order.

    Note the loop condition is `low < high` and the upper update is `high = middle`
    rather than `middle - 1`. That is not a stylistic choice. The answer can be
    the position after the last element, so `high` starts at `len(values)` and
    must be allowed to remain a candidate.
    """
    low, high = 0, len(values)

    while low < high:
        middle = low + (high - low) // 2
        if values[middle] < target:
            low = middle + 1
        else:
            high = middle

    return low


def upper_bound(values: Sequence[Any], target: Any) -> int:
    """The index of the first element strictly greater than `target`.

    Python's `bisect_right`. The only difference from `lower_bound` is `<=`
    instead of `<`, and that single character decides which side of a run of equal
    values you land on.

    Together the pair gives you the range of every occurrence of a value:
    `lower_bound` to `upper_bound`, so the count of a value is the difference
    between them, computed in O(log n) instead of a linear count.
    """
    low, high = 0, len(values)

    while low < high:
        middle = low + (high - low) // 2
        if values[middle] <= target:
            low = middle + 1
        else:
            high = middle

    return low


def count_occurrences(values: Sequence[Any], target: Any) -> int:
    """How many times `target` appears, in O(log n) rather than O(n)."""
    return upper_bound(values, target) - lower_bound(values, target)


def first_true(values: Sequence[Any], predicate) -> int:
    """The index of the first element where `predicate` becomes true, or len(values).

    This is the general form that all the variants above are special cases of, and
    it is the one worth remembering.

    The requirement is that the predicate is **monotonic**: once true, it stays
    true for the rest of the sequence. Under that condition, the sequence looks
    like false, false, ..., false, true, ..., true, and binary search can find the
    boundary even when nothing is being compared for equality at all.

    Thinking of binary search as "find the boundary of a monotonic predicate"
    rather than "find a value in a sorted array" is what makes it usable on
    problems that have no array in them: the smallest capacity that fits a
    schedule, the first version that fails a test, the minimum speed that finishes
    in time. Binary searching over an answer space is a standard technique and it
    is exactly this function.
    """
    low, high = 0, len(values)

    while low < high:
        middle = low + (high - low) // 2
        if predicate(values[middle]):
            high = middle
        else:
            low = middle + 1

    return low


def exponential_search(values: Sequence[Any], target: Any) -> int:
    """Find a bound by doubling, then binary search inside it.

    O(log i) where i is the position of the answer, rather than O(log n) where n
    is the length. When the target is near the front of a huge collection, that is
    a real saving, and it is the only option when the collection has **no known
    length**, such as a stream or a paginated API where you can ask for item k but
    not for the total.

    The doubling is the same idea as the dynamic array's growth on day 4: reaching
    position i by doubling takes log i steps, and the total work is dominated by
    the last step.
    """
    if not values:
        return -1
    if values[0] == target:
        return 0

    bound = 1
    while bound < len(values) and values[bound] <= target:
        bound *= 2

    low = bound // 2
    high = min(bound, len(values) - 1)
    window = values[low : high + 1]
    found = binary_search(window, target)

    return low + found if found >= 0 else -1


def interpolation_search(values: Sequence[float], target: float) -> int:
    """Guess where the value should be, rather than always splitting in the middle.

    Binary search always guesses the midpoint. If the values are spread evenly,
    you can do better: looking for 950 in a list from 1 to 1000, the answer is
    obviously near the end, so guess there.

    O(log log n) on uniformly distributed data, which is remarkably fast: about
    five probes for a million items against twenty for binary search. But it
    degrades to **O(n) on skewed data**, because a bad guess barely shrinks the
    range.

    So it is a genuine gamble on the shape of the data, in the same way bucket
    sort is. Binary search's O(log n) holds whatever the distribution, which is
    usually worth more than a better average.
    """
    low, high = 0, len(values) - 1

    while low <= high and values[low] <= target <= values[high]:
        if values[low] == values[high]:
            return low if values[low] == target else -1

        # The interpolation: how far along the value range the target sits,
        # applied to the index range.
        span = (target - values[low]) / (values[high] - values[low])
        guess = low + int(span * (high - low))
        guess = max(low, min(high, guess))

        if values[guess] == target:
            return guess
        if values[guess] < target:
            low = guess + 1
        else:
            high = guess - 1

    return -1


def ternary_search_maximum(low: float, high: float, function, tolerance: float = 1e-9) -> float:
    """Find the peak of a function that rises and then falls. O(log((high-low)/tolerance)).

    Binary search needs a sorted sequence, meaning a monotonic predicate. A
    function with a single peak is not monotonic, so binary search cannot be used
    directly, but the same halving idea still works with a different comparison.

    Take two points a third and two thirds along. If the first is lower than the
    second, the peak cannot be to the left of the first, so that part is discarded.
    Each round removes a third of the range, so it converges just as surely as
    binary search does, only slightly slower.

    This is where "unimodal" is a genuinely useful word: it means exactly the
    property this needs, one peak with no local bumps to get stuck on.
    """
    while high - low > tolerance:
        first = low + (high - low) / 3
        second = high - (high - low) / 3

        if function(first) < function(second):
            low = first
        else:
            high = second

    return (low + high) / 2


def linear_search(values: Sequence[Any], target: Any) -> int:
    """Look at everything, in order. O(n).

    Included because it is the honest baseline, and because it beats binary search
    more often than people expect. Binary search needs sorted data, and if you sort
    just to search once you have spent more than a scan would have cost. For small
    collections the constant factors also favour the scan, since it walks memory in
    order while binary search jumps around it.
    """
    for index, value in enumerate(values):
        if value == target:
            return index
    return -1
