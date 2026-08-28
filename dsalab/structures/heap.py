"""Binary heap: a tree that lives in an array and answers one question fast.

A heap gives up most of what a search tree offers. It cannot find an arbitrary
value in less than O(n), it has no ordering you can walk, and it answers no range
queries at all. In exchange it does one thing extremely well: **hand back the
smallest (or largest) item, repeatedly, in O(log n), while accepting new items
just as fast.**

That is exactly the shape of a priority queue, and priority queues turn up
everywhere: Dijkstra's shortest paths, Prim's minimum spanning tree, Huffman
coding, task schedulers, event simulations, the k largest of a stream.

The heap property is deliberately weak:

    every node is no larger than its children (for a min heap)

Notice what it does not say. It says nothing about the relationship between
siblings, or between a node and its cousins. A search tree orders everything; a
heap orders only along each root to leaf path. That weakness is the point, because
a weaker invariant is cheaper to restore, and the minimum is still pinned at the
root where it can be read instantly.

The other half of the design is the array layout from day 8. A heap is always a
**complete** tree, meaning every level is full except the last, which fills from
the left. Complete trees have no gaps, so they fit an array perfectly with no
wasted slots and no pointers at all:

    parent of i = (i - 1) // 2      left child = 2i + 1      right child = 2i + 2

No node objects, no pointer chasing, and a parent sits near its children in
memory, which the CPU cache likes. This is the payoff for accepting a weaker
ordering.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from dsalab.invariants import Violation, verify_if_checking
from dsalab.tracing import Step, Traced, run


class BinaryHeap:
    """A binary heap in an array, min heap by default.

    | Operation | Cost | Why |
    | - | - | - |
    | peek at the best item | O(1) | It is always at index 0 |
    | push | O(log n) | Sift up at most the height of the tree |
    | pop the best item | O(log n) | Sift down at most the height |
    | build from n items | **O(n)** | Not O(n log n). See `heapify` |
    | find an arbitrary value | O(n) | The heap has no useful order for this |
    | change a known item's priority | O(log n) | Sift in whichever direction is needed |

    The `key` argument turns this into a max heap or orders by any field, without
    a second class or any duplicated logic.
    """

    def __init__(
        self,
        items: Iterable[Any] | None = None,
        key: Callable[[Any], Any] | None = None,
    ) -> None:
        self._key = key or (lambda value: value)
        self._items: list[Any] = list(items or ())
        if self._items:
            run(self.heapify_traced())

    def __len__(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return not self._items

    def __repr__(self) -> str:
        return f"BinaryHeap({self._items!r})"

    def to_list(self) -> list[Any]:
        """The raw array, level by level. Not sorted, and not meant to be."""
        return list(self._items)

    # Navigation, which is arithmetic rather than pointer following

    @staticmethod
    def _parent(index: int) -> int:
        return (index - 1) // 2

    @staticmethod
    def _left(index: int) -> int:
        return 2 * index + 1

    @staticmethod
    def _right(index: int) -> int:
        return 2 * index + 2

    def _beats(self, a: int, b: int) -> bool:
        """True when the item at a should sit above the item at b."""
        return self._key(self._items[a]) < self._key(self._items[b])

    # The two repair operations, which are the whole structure

    def _sift_up(self, index: int) -> Traced[None]:
        """Move an item up until its parent is no larger than it.

        Used after a push. The new item goes at the end of the array, which is
        the only place that keeps the tree complete, and then climbs. It travels
        at most the height of the tree, so O(log n).
        """
        while index > 0:
            parent = self._parent(index)
            if not self._beats(index, parent):
                yield Step(
                    "settle",
                    f"{self._items[index]!r} is not smaller than its parent "
                    f"{self._items[parent]!r}, so it has found its place.",
                    {"index": index, "value": self._items[index]},
                )
                return

            yield Step(
                "swap",
                f"{self._items[index]!r} is smaller than its parent "
                f"{self._items[parent]!r}, so they swap.",
                {"from": index, "to": parent, "value": self._items[index]},
            )
            self._items[index], self._items[parent] = self._items[parent], self._items[index]
            index = parent

    def _sift_down(self, index: int) -> Traced[None]:
        """Move an item down until both its children are no smaller than it.

        Used after a pop, and by `heapify`. The item must swap with the **better
        of its two children**, not just any child that beats it. Swapping with
        the larger child would put a bigger value above a smaller one and leave
        the heap broken, which is the classic mistake here and is silent: the
        heap still returns plausible values, just not always the right ones.
        """
        size = len(self._items)

        while True:
            best = index
            for child in (self._left(index), self._right(index)):
                if child < size and self._beats(child, best):
                    best = child

            if best == index:
                return

            yield Step(
                "swap",
                f"{self._items[index]!r} is larger than its best child "
                f"{self._items[best]!r}, so they swap.",
                {"from": index, "to": best, "value": self._items[index]},
            )
            self._items[index], self._items[best] = self._items[best], self._items[index]
            index = best

    # The public operations

    def push(self, value: Any) -> None:
        run(self.push_traced(value))

    def push_traced(self, value: Any) -> Traced[None]:
        """Add an item. O(log n)."""
        self._items.append(value)
        yield Step(
            "append",
            f"{value!r} placed at the end of the array, which is the only spot that "
            "keeps the tree complete. Now it climbs to where it belongs.",
            {"index": len(self._items) - 1, "value": value},
        )
        yield from self._sift_up(len(self._items) - 1)
        verify_if_checking(self)

    def pop(self) -> Any:
        return run(self.pop_traced())

    def pop_traced(self) -> Traced[Any]:
        """Remove and return the best item. O(log n).

        The trick is in how the hole at the root is filled. You cannot promote
        the smaller child, because that leaves a hole one level down and the tree
        stops being complete. Instead the **last** item in the array is moved to
        the root, which keeps the tree complete by construction, and then sinks to
        its proper place.

        So the shape is fixed first and the ordering repaired second. That order
        of priorities is what keeps the array representation valid at all times.
        """
        if not self._items:
            raise IndexError("cannot take from an empty heap")

        best = self._items[0]
        last = self._items.pop()

        if self._items:
            self._items[0] = last
            yield Step(
                "replace",
                f"Took {best!r} from the root and moved the last item {last!r} up to fill "
                "the hole, which keeps the tree complete. Now it sinks.",
                {"removed": best, "moved": last},
            )
            yield from self._sift_down(0)
        else:
            yield Step("empty", f"Took the last item {best!r}, the heap is now empty.",
                       {"removed": best})

        verify_if_checking(self)
        return best

    def peek(self) -> Any:
        """Look at the best item without removing it. O(1)."""
        if not self._items:
            raise IndexError("cannot look at an empty heap")
        return self._items[0]

    def push_pop(self, value: Any) -> Any:
        """Push then pop in one operation, which is cheaper than doing both.

        If the new value is already the best, it is simply returned without ever
        entering the heap. Otherwise it goes straight to the root and sinks once.
        Either way there is one sift instead of two, which halves the work in the
        inner loop of "keep the k largest of a stream", where this is called once
        per item.
        """
        if not self._items or not self._key(self._items[0]) < self._key(value):
            return value

        best = self._items[0]
        self._items[0] = value
        run(self._sift_down(0))
        return best

    def heapify(self) -> None:
        run(self.heapify_traced())

    def heapify_traced(self) -> Traced[None]:
        """Turn an arbitrary array into a heap in **O(n)**, not O(n log n).

        This surprises people, so here is the argument in full.

        The method: sift down every node that has children, working from the last
        such node backwards to the root. Working backwards matters, because when
        a node is sifted down both of its subtrees are already heaps, which is
        exactly the precondition sift down needs.

        Now the cost. The obvious estimate says n nodes times log n each, giving
        O(n log n). That estimate is far too pessimistic, because **almost every
        node is near the bottom, and nodes near the bottom barely move.**

        In a complete tree of n nodes:

        * about n/2 nodes are leaves and are never sifted at all,
        * about n/4 sit one level up and can fall at most 1,
        * about n/8 can fall at most 2,
        * and so on.

        Total work is at most the sum of (n / 2^(h+1)) times h over all heights h,
        which is n times the sum of h / 2^(h+1). That sum converges to 1, so the
        total is O(n).

        The intuition to keep: the many nodes are cheap and the expensive nodes
        are few. Pushing n items one at a time really is O(n log n), because there
        every item can climb the full height. Building all at once is genuinely
        linear, and the tests measure both to show the difference is real.
        """
        for index in range(len(self._items) // 2 - 1, -1, -1):
            yield Step(
                "sift",
                f"Sifting down from index {index}. Both subtrees below it are already "
                "heaps, which is what makes one sift enough.",
                {"index": index, "value": self._items[index]},
            )
            yield from self._sift_down(index)
        verify_if_checking(self)

    def check_invariants(self) -> list[Violation]:
        """The single heap rule: no node is smaller than its parent."""
        violations: list[Violation] = []

        for index in range(1, len(self._items)):
            parent = self._parent(index)
            if self._key(self._items[index]) < self._key(self._items[parent]):
                violations.append(Violation(
                    "no child sits above its parent",
                    f"index {index} holds {self._items[index]!r}, which is smaller than "
                    f"its parent {self._items[parent]!r} at index {parent}",
                ))

        return violations


class PriorityQueue:
    """A priority queue with a way to change an item's priority after the fact.

    A plain heap can push and pop, which is enough for most uses. Dijkstra's
    algorithm on day 18 wants one more thing: **lower the priority of something
    already in the queue**, when a shorter route to a place is discovered.

    Doing that on a plain heap means finding the item, which is O(n), and that
    would dominate the whole algorithm. So this keeps a dictionary from item to
    its position in the array, updated on every swap, which makes finding an item
    O(1) and changing its priority O(log n).

    The cost of that dictionary is real: extra memory, and every single swap now
    has to keep it correct. Forgetting one of those updates gives a queue that
    works perfectly until it silently starts looking in the wrong slot. There is a
    test that checks every recorded position after every operation.

    The usual alternative is the **lazy deletion** trick: never update anything,
    just push the improved entry as a duplicate and ignore stale entries when they
    come out. That is simpler and often faster in practice, at the cost of a queue
    that can hold more entries than there are items. Both are legitimate; this one
    is here because it makes the decrease operation explicit rather than hiding it.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[Any, Any]] = []  # (priority, item)
        self._position: dict[Any, int] = {}

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return not self._heap

    def __contains__(self, item: Any) -> bool:
        return item in self._position

    def __repr__(self) -> str:
        return f"PriorityQueue({[item for _, item in self._heap]!r})"

    def _swap(self, a: int, b: int) -> None:
        """Swap two entries and keep the position table honest."""
        self._heap[a], self._heap[b] = self._heap[b], self._heap[a]
        self._position[self._heap[a][1]] = a
        self._position[self._heap[b][1]] = b

    def _sift_up(self, index: int) -> None:
        while index > 0:
            parent = (index - 1) // 2
            if self._heap[parent][0] <= self._heap[index][0]:
                return
            self._swap(index, parent)
            index = parent

    def _sift_down(self, index: int) -> None:
        size = len(self._heap)
        while True:
            best = index
            for child in (2 * index + 1, 2 * index + 2):
                if child < size and self._heap[child][0] < self._heap[best][0]:
                    best = child
            if best == index:
                return
            self._swap(index, best)
            index = best

    def push(self, item: Any, priority: Any) -> None:
        """Add an item, or improve the priority of one already present.

        Pushing something already in the queue with a worse priority is ignored
        rather than treated as an error, because that is what every caller wants:
        Dijkstra offers a route to a place it has already reached, and the
        shorter of the two should simply win.
        """
        if item in self._position:
            self.decrease_priority(item, priority)
            return

        self._heap.append((priority, item))
        self._position[item] = len(self._heap) - 1
        self._sift_up(len(self._heap) - 1)

    def decrease_priority(self, item: Any, priority: Any) -> bool:
        """Lower an item's priority. Returns whether anything changed. O(log n)."""
        if item not in self._position:
            raise KeyError(f"{item!r} is not in the queue")

        index = self._position[item]
        if priority >= self._heap[index][0]:
            return False

        self._heap[index] = (priority, item)
        self._sift_up(index)
        return True

    def pop(self) -> tuple[Any, Any]:
        """Remove and return the (priority, item) pair with the lowest priority."""
        if not self._heap:
            raise IndexError("cannot take from an empty queue")

        priority, item = self._heap[0]
        last = self._heap.pop()
        del self._position[item]

        if self._heap:
            self._heap[0] = last
            self._position[last[1]] = 0
            self._sift_down(0)

        return priority, item

    def peek(self) -> tuple[Any, Any]:
        if not self._heap:
            raise IndexError("cannot look at an empty queue")
        return self._heap[0]

    def priority_of(self, item: Any) -> Any:
        """The current priority of an item. O(1), thanks to the position table."""
        if item not in self._position:
            raise KeyError(f"{item!r} is not in the queue")
        return self._heap[self._position[item]][0]

    def check_invariants(self) -> list[Violation]:
        """The heap rule, plus every recorded position being correct."""
        violations: list[Violation] = []

        for index in range(1, len(self._heap)):
            parent = (index - 1) // 2
            if self._heap[index][0] < self._heap[parent][0]:
                violations.append(Violation(
                    "no entry sits above one with a better priority",
                    f"index {index} has priority {self._heap[index][0]} above "
                    f"{self._heap[parent][0]} at index {parent}",
                ))

        for index, (_, item) in enumerate(self._heap):
            if self._position.get(item) != index:
                violations.append(Violation(
                    "the position table matches the array",
                    f"{item!r} is at index {index} but the table says "
                    f"{self._position.get(item)}",
                ))

        if len(self._position) != len(self._heap):
            violations.append(Violation(
                "the position table matches the array",
                f"{len(self._heap)} entries but {len(self._position)} recorded positions",
            ))

        return violations


def heap_sort(values: list[Any], reverse: bool = False) -> list[Any]:
    return run(heap_sort_traced(values, reverse))


def heap_sort_traced(values: list[Any], reverse: bool = False) -> Traced[list[Any]]:
    """Sort by building a heap and then repeatedly taking the best item.

    O(n log n) always, best case and worst case alike, and it sorts **in place**
    with O(1) extra memory. That combination is rare: merge sort matches the time
    but needs O(n) extra memory, and quick sort matches the memory but has an
    O(n^2) worst case.

    The in place trick is the elegant part. Build a **max** heap, then swap the
    root with the last item and shrink the heap by one. The largest item is now in
    its final position at the end of the array, and the heap occupies the shrinking
    prefix. Repeat, and the array sorts itself from the back forwards, with the
    heap and the sorted region sharing one array and never overlapping.

    Given all that, why is quick sort still the usual choice? Cache behaviour. Heap
    sort's sift down jumps between index i and index 2i+1, which for a large array
    means a different cache line almost every step. Quick sort scans linearly,
    which the hardware prefetches perfectly. Heap sort typically loses by a factor
    of two or three despite identical complexity, which is a good reminder that
    complexity is not the whole story. Day 15 measures this.
    """
    items = list(values)
    size = len(items)
    if size <= 1:
        return items

    # A max heap for ascending order, because the largest item goes to the back.
    def beats(a: Any, b: Any) -> bool:
        return a > b if not reverse else a < b

    def sift_down(start: int, limit: int) -> Traced[None]:
        index = start
        while True:
            best = index
            for child in (2 * index + 1, 2 * index + 2):
                if child < limit and beats(items[child], items[best]):
                    best = child
            if best == index:
                return
            yield Step(
                "swap",
                f"{items[index]!r} sinks below {items[best]!r} to restore the heap.",
                {"from": index, "to": best, "array": list(items)},
            )
            items[index], items[best] = items[best], items[index]
            index = best

    for start in range(size // 2 - 1, -1, -1):
        yield from sift_down(start, size)
    yield Step("built", "The whole array is now a heap, built in linear time.",
               {"array": list(items)})

    for end in range(size - 1, 0, -1):
        yield Step(
            "extract",
            f"{items[0]!r} is the best remaining item, so it swaps into position {end} "
            "and the heap shrinks by one.",
            {"value": items[0], "to": end, "array": list(items)},
        )
        items[0], items[end] = items[end], items[0]
        yield from sift_down(0, end)

    return items


def k_smallest(values: Iterable[Any], k: int) -> list[Any]:
    """The k smallest values, using a heap of size k rather than sorting everything.

    Sorting costs O(n log n) and O(n) memory. This costs O(n log k) and O(k)
    memory, which is the difference between possible and impossible when the input
    is a stream of a billion items and k is ten.

    The trick is counter intuitive: to find the k **smallest**, keep a **max**
    heap of the best k so far. The largest of those k sits at the root, so each new
    value is compared against it in O(1), and only values that beat it get in. The
    heap never grows past k.
    """
    if k < 0:
        raise ValueError("k cannot be negative")
    if k == 0:
        return []

    largest_first = BinaryHeap(key=lambda value: -value)
    for value in values:
        if len(largest_first) < k:
            largest_first.push(value)
        elif value < largest_first.peek():
            largest_first.push_pop(value)

    return sorted(largest_first.to_list())
