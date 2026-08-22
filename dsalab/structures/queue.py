"""Queues: first in, first out.

A queue is the other half of the pair with the stack. A stack hands back the most
recent arrival; a queue hands back the oldest. That one difference decides which
problems each is good for. A stack goes deep, which is why depth first search uses
one. A queue goes wide, which is why breadth first search uses one and why it
finds the shortest path in an unweighted graph.

Queues also appear anywhere work has to be served fairly: print jobs, request
handling, task scheduling, keystrokes waiting to be processed.

Three implementations are here and the differences matter:

* `CircularQueue` is a fixed size ring buffer, which is what you actually find in
  operating systems and network buffers. It never allocates after construction.
* `DynamicQueue` is the same ring that doubles when it fills up.
* `LinkedQueue` uses nodes, so it has no capacity limit and never copies.

The interesting design story is the circular one, and it is written up in the
class itself.
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterable, Iterator
from typing import Any

from dsalab.structures.linked_list import Node
from dsalab.structures.stack import ArrayStack


class QueueEmptyError(IndexError):
    """Raised when taking from a queue with nothing in it."""


class QueueFullError(OverflowError):
    """Raised when adding to a fixed size queue that is already full."""


class CircularQueue:
    """A fixed size queue that reuses its slots by wrapping around.

    Why circular, and not just an array with a front and a back?

    Take a plain array queue of capacity 5. Add five items, then remove three.
    The front pointer is now at index 3 and the back is at index 5, so there are
    three free slots at the beginning of the array and none at the end. Adding a
    sixth item has two bad options: refuse, even though the array is mostly
    empty, or shift everything back to the start, which is O(n) on every add.

    A circular queue fixes this by letting the back wrap around to index 0 when
    it runs off the end. The three free slots at the front are simply used. No
    shifting, no waste, and every operation stays O(1).

    All of it comes from one operator. `(index + 1) % capacity` is what makes the
    array behave like a ring.

    ### The full and empty problem

    A ring has one genuinely awkward detail. When front and back point at the same
    slot, the queue could be completely empty or completely full, and the pointers
    look identical either way. There are two standard fixes:

    1. Deliberately waste one slot, so full means the back is one behind the
       front. Costs one slot, needs no extra state.
    2. Keep a separate count of how many items there are. Costs an integer, and
       every add and remove has to keep it correct.

    This class uses the count. The reason is that the count is useful on its own,
    since `len()` should be O(1) and reporting how full a buffer is turns out to
    be exactly what monitoring code wants. Wasting a slot would make `len()` a
    calculation involving a modulo and a special case, which is more code for
    less information.

    | Operation | Cost |
    | - | - |
    | enqueue, dequeue, peek | O(1) |
    | len, is_empty, is_full | O(1) |
    | memory | fixed at construction, never grows |
    """

    def __init__(self, capacity: int, items: Iterable[Any] | None = None) -> None:
        if capacity < 1:
            raise ValueError("a queue needs room for at least one item")
        self._capacity = capacity
        self._store = (ctypes.py_object * capacity)()
        self._front = 0
        self._count = 0
        for item in items or ():
            self.enqueue(item)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return self._count

    def is_empty(self) -> bool:
        return self._count == 0

    def is_full(self) -> bool:
        return self._count == self._capacity

    def enqueue(self, value: Any) -> None:
        """Add to the back of the queue. O(1)."""
        if self.is_full():
            raise QueueFullError(f"the queue is full at {self._capacity} items")
        back = (self._front + self._count) % self._capacity
        self._store[back] = value
        self._count += 1

    def dequeue(self) -> Any:
        """Remove and return the oldest item. O(1)."""
        if self.is_empty():
            raise QueueEmptyError("cannot take from an empty queue")
        value = self._store[self._front]
        self._release(self._front)
        self._front = (self._front + 1) % self._capacity
        self._count -= 1
        return value

    def peek(self) -> Any:
        """Look at the oldest item without removing it. O(1)."""
        if self.is_empty():
            raise QueueEmptyError("cannot look at an empty queue")
        return self._store[self._front]

    def _release(self, index: int) -> None:
        """Drop the reference in a vacated slot, including the ctypes shadow copy.

        Same trap as the dynamic array on day 4: a ctypes py_object array keeps
        its own `_objects` dictionary, so assigning None to the slot is not enough
        to let the object be collected.
        """
        self._store[index] = None
        objects = getattr(self._store, "_objects", None)
        if isinstance(objects, dict):
            objects.pop(str(index), None)

    def __iter__(self) -> Iterator[Any]:
        """Front to back, which is the order things will come out."""
        for offset in range(self._count):
            yield self._store[(self._front + offset) % self._capacity]

    def to_list(self) -> list[Any]:
        return list(self)

    def __repr__(self) -> str:
        return f"CircularQueue({self.to_list()!r}, capacity={self._capacity})"


class DynamicQueue:
    """A circular queue that doubles instead of refusing when it fills up.

    Everything from `CircularQueue` applies, plus the growth policy from the
    dynamic array on day 4: double the capacity, giving amortised O(1) enqueues.

    Growing a ring is slightly more involved than growing a plain array, because
    the items may be split across the wrap point. The copy has to walk them in
    logical order rather than memory order, and the rebuilt buffer starts at index
    0 again with the front reset. Copying the raw slots in memory order would
    reorder the queue, which is a bug that only appears once the ring has wrapped
    at least once, so it survives any test that just fills a fresh queue.
    """

    def __init__(self, items: Iterable[Any] | None = None, initial_capacity: int = 4) -> None:
        self._ring = CircularQueue(max(1, initial_capacity))
        for item in items or ():
            self.enqueue(item)

    @property
    def capacity(self) -> int:
        return self._ring.capacity

    def __len__(self) -> int:
        return len(self._ring)

    def is_empty(self) -> bool:
        return self._ring.is_empty()

    def enqueue(self, value: Any) -> None:
        """Add to the back. O(1) amortised."""
        if self._ring.is_full():
            bigger = CircularQueue(self._ring.capacity * 2)
            # Iterating the old ring walks it in logical order, front first,
            # which is exactly what makes the wrap point a non issue.
            for item in self._ring:
                bigger.enqueue(item)
            self._ring = bigger
        self._ring.enqueue(value)

    def dequeue(self) -> Any:
        return self._ring.dequeue()

    def peek(self) -> Any:
        return self._ring.peek()

    def __iter__(self) -> Iterator[Any]:
        return iter(self._ring)

    def to_list(self) -> list[Any]:
        return self._ring.to_list()

    def __repr__(self) -> str:
        return f"DynamicQueue({self.to_list()!r}, capacity={self.capacity})"


class LinkedQueue:
    """A queue of linked nodes, taking from the head and adding at the tail.

    Which end is which matters. Adding at the tail and taking from the head are
    both O(1) with a tail pointer. Doing it the other way round would make
    removal O(n), because a singly linked list cannot find the node before its
    tail.

    No capacity limit and no copying, at the cost of one node object per item.
    """

    def __init__(self, items: Iterable[Any] | None = None) -> None:
        self._head: Node | None = None
        self._tail: Node | None = None
        self._count = 0
        for item in items or ():
            self.enqueue(item)

    def __len__(self) -> int:
        return self._count

    def is_empty(self) -> bool:
        return self._head is None

    def enqueue(self, value: Any) -> None:
        node = Node(value)
        if self._tail is None:
            self._head = self._tail = node
        else:
            self._tail.next = node
            self._tail = node
        self._count += 1

    def dequeue(self) -> Any:
        if self._head is None:
            raise QueueEmptyError("cannot take from an empty queue")
        node = self._head
        self._head = node.next
        if self._head is None:
            self._tail = None
        self._count -= 1
        return node.value

    def peek(self) -> Any:
        if self._head is None:
            raise QueueEmptyError("cannot look at an empty queue")
        return self._head.value

    def __iter__(self) -> Iterator[Any]:
        current = self._head
        while current is not None:
            yield current.value
            current = current.next

    def to_list(self) -> list[Any]:
        return list(self)

    def __repr__(self) -> str:
        return f"LinkedQueue({self.to_list()!r})"


class QueueFromTwoStacks:
    """A queue built out of two stacks, which is a lovely piece of amortised analysis.

    The idea: push arrivals onto an inbox stack. When something is needed, if the
    outbox is empty, tip the entire inbox into the outbox. Tipping reverses the
    order, so the oldest item ends up on top of the outbox, which is exactly what
    a queue should hand back.

    A single dequeue can therefore cost O(n), when it triggers a tip. Yet the
    amortised cost is O(1), and the argument is worth understanding because it is
    the cleanest example of amortised reasoning there is:

    **Every item is moved exactly twice in its whole life.** Once from the inbox
    to the outbox, and once out of the outbox. It can never be tipped twice,
    because tipping only happens when the outbox is empty, and once an item is in
    the outbox it stays there until it leaves. So n operations do at most 2n
    moves, which is O(1) each on average, no matter what order the calls come in.

    This is not an average over inputs. It is a guarantee over any sequence, and
    that is precisely the difference between amortised and average case.

    In real code this pattern shows up whenever you have a cheap append only
    structure and need FIFO behaviour out of it, for example a log that is written
    in one order and consumed in another.
    """

    def __init__(self, items: Iterable[Any] | None = None) -> None:
        self._inbox = ArrayStack()
        self._outbox = ArrayStack()
        for item in items or ():
            self.enqueue(item)

    def __len__(self) -> int:
        return len(self._inbox) + len(self._outbox)

    def is_empty(self) -> bool:
        return len(self) == 0

    def enqueue(self, value: Any) -> None:
        """Always cheap: straight onto the inbox. O(1)."""
        self._inbox.push(value)

    def _tip(self) -> None:
        """Move everything from the inbox to the outbox, reversing the order."""
        while not self._inbox.is_empty():
            self._outbox.push(self._inbox.pop())

    def dequeue(self) -> Any:
        """O(1) amortised, occasionally O(n) when a tip is needed."""
        if self._outbox.is_empty():
            if self._inbox.is_empty():
                raise QueueEmptyError("cannot take from an empty queue")
            self._tip()
        return self._outbox.pop()

    def peek(self) -> Any:
        if self._outbox.is_empty():
            if self._inbox.is_empty():
                raise QueueEmptyError("cannot look at an empty queue")
            self._tip()
        return self._outbox.peek()

    def to_list(self) -> list[Any]:
        """Front to back. The outbox is already reversed, the inbox is not."""
        return list(self._outbox) + self._inbox.to_list()

    def __repr__(self) -> str:
        return f"QueueFromTwoStacks({self.to_list()!r})"
