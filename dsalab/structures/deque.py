"""Double ended queue: add and remove at both ends, all in constant time.

A deque is the general case that a stack and a queue are both special cases of.
Use only the front and it is a stack. Add at one end and take from the other and
it is a queue. Because it can do both, it is what you reach for when the access
pattern is not known in advance, or when it is genuinely both.

The real uses are more specific than "a flexible list":

* Sliding window problems, where a monotonic deque gives the maximum of every
  window of size k in O(n) total rather than O(n times k). That algorithm is
  included below, because it is the best argument for the structure existing.
* Undo and redo together, which needs a stack at each end.
* Work stealing schedulers, where a thread takes its own work from one end and
  other threads steal from the other, so the two rarely collide.

The implementation is a doubly linked list, reusing day 5's structure. Every
operation is genuinely O(1) worst case with no copying and no capacity limit. The
alternative, a circular buffer with a front and back pointer, has better memory
behaviour and amortised rather than worst case guarantees. Same trade as the two
stacks on day 6.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from dsalab.structures.linked_list import DoublyLinkedList
from dsalab.tracing import Step, Traced, run


class DequeEmptyError(IndexError):
    """Raised when taking from a deque with nothing in it."""


class Deque:
    """A double ended queue.

    | Operation | Cost |
    | - | - |
    | push_front, push_back | O(1) |
    | pop_front, pop_back | O(1) |
    | peek_front, peek_back | O(1) |
    | len, is_empty | O(1) |
    | anything by index | O(n) |
    """

    def __init__(self, items: Iterable[Any] | None = None) -> None:
        self._items = DoublyLinkedList()
        for item in items or ():
            self.push_back(item)

    def __len__(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def push_front(self, value: Any) -> None:
        self._items.push_front(value)

    def push_back(self, value: Any) -> None:
        self._items.push_back(value)

    def pop_front(self) -> Any:
        if self.is_empty():
            raise DequeEmptyError("cannot take from an empty deque")
        return self._items.pop_front()

    def pop_back(self) -> Any:
        if self.is_empty():
            raise DequeEmptyError("cannot take from an empty deque")
        return self._items.pop_back()

    def peek_front(self) -> Any:
        if self.is_empty():
            raise DequeEmptyError("cannot look at an empty deque")
        return self._items.head.value

    def peek_back(self) -> Any:
        if self.is_empty():
            raise DequeEmptyError("cannot look at an empty deque")
        return self._items.tail.value

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def to_list(self) -> list[Any]:
        return self._items.to_list()

    def __repr__(self) -> str:
        return f"Deque({self.to_list()!r})"


def sliding_window_maximum(values: list[Any], window: int) -> list[Any]:
    return run(sliding_window_maximum_traced(values, window))


def sliding_window_maximum_traced(values: list[Any], window: int) -> Traced[list[Any]]:
    """The maximum of every window of `window` consecutive values, in O(n) total.

    The obvious method looks at all k values in each of the n - k + 1 windows,
    which is O(n times k). For a million values and a window of a thousand that is
    a billion comparisons, and it is far too slow.

    This version is O(n): each value is added to the deque once and removed at
    most once, so the total work is at most 2n regardless of the window size.

    The trick is what the deque holds. It stores *positions*, kept so that their
    values are always in decreasing order. Two rules maintain that:

    1. Before adding a new position, throw away every position at the back whose
       value is smaller than or equal to the new one. Those can never be the
       maximum of any future window, because the new value is both larger and
       stays in the window longer. This is the key insight: they are not merely
       currently unhelpful, they are permanently useless, so discarding them
       loses nothing.
    2. Before reading the answer, throw away the position at the front if it has
       fallen out of the current window.

    With those rules the front of the deque is always the maximum of the current
    window, and reading it is O(1).

    A deque is exactly the right structure because rule 1 works at the back and
    rule 2 works at the front. No other structure gives O(1) at both ends.
    """
    if window < 1:
        raise ValueError("the window must hold at least one value")
    if window > len(values):
        raise ValueError(f"a window of {window} does not fit in {len(values)} values")

    positions = Deque()
    maximums: list[Any] = []

    for index, value in enumerate(values):
        while not positions.is_empty() and values[positions.peek_back()] <= value:
            dropped = positions.pop_back()
            yield Step(
                "discard",
                f"Position {dropped} holds {values[dropped]}, which is no bigger than the "
                f"new {value} and leaves the window sooner, so it can never win again.",
                {"dropped": dropped, "because_of": index},
            )
        positions.push_back(index)

        if not positions.is_empty() and positions.peek_front() <= index - window:
            expired = positions.pop_front()
            yield Step(
                "expire",
                f"Position {expired} has fallen out of the window, so it is dropped.",
                {"expired": expired},
            )

        if index >= window - 1:
            best = values[positions.peek_front()]
            maximums.append(best)
            yield Step(
                "report",
                f"The window covering positions {index - window + 1} to {index} "
                f"has maximum {best}.",
                {"start": index - window + 1, "end": index, "maximum": best},
            )

    return maximums
