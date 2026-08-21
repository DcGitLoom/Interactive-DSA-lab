"""Stacks: last in, first out.

A stack allows exactly three things: push a value on top, pop the top value off,
and look at the top without removing it. That is a deliberate restriction, not a
missing feature. By refusing to let anything reach into the middle, the structure
guarantees that whatever comes out is always the most recent thing that went in,
and a surprising number of problems are exactly that shape:

* Undo history. The last action taken is the first one undone.
* Matching brackets. The bracket you must close next is always the most recent
  one you opened.
* Function calls. The call stack is a real stack, and it is why a function always
  returns to whoever called it most recently.
* Depth first search, and every recursive algorithm rewritten as a loop.

Two implementations are given because the choice genuinely matters. The array
backed version has better memory behaviour and amortised O(1) pushes. The linked
version has true worst case O(1) pushes with no copying, at the cost of an object
per element. The comparison is in docs/06-stacks.md.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from dsalab.structures.dynamic_array import DynamicArray
from dsalab.structures.linked_list import Node


class StackEmptyError(IndexError):
    """Raised when popping or peeking at a stack with nothing in it.

    It subclasses IndexError so that code expecting the usual Python behaviour
    still works, while code that wants to catch this specific mistake can.
    """


class ArrayStack:
    """A stack built on the dynamic array from day 4.

    The top of the stack is the end of the array, which is deliberate: appending
    and popping at the end are the array's cheap operations, while anything at the
    front would be O(n). Choosing which end is the top is the whole design.

    | Operation | Cost |
    | - | - |
    | push | O(1) amortised |
    | pop | O(1) amortised |
    | peek | O(1) |
    | is_empty, size | O(1) |
    """

    def __init__(self, items: Iterable[Any] | None = None) -> None:
        self._items = DynamicArray()
        for item in items or ():
            self.push(item)

    def push(self, value: Any) -> None:
        self._items.append(value)

    def pop(self) -> Any:
        if self.is_empty():
            raise StackEmptyError("cannot pop from an empty stack")
        return self._items.pop()

    def peek(self) -> Any:
        """Look at the top without removing it."""
        if self.is_empty():
            raise StackEmptyError("cannot peek at an empty stack")
        return self._items[-1]

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Any]:
        """Walk from the top down, which is the order things will come off."""
        for index in range(len(self._items) - 1, -1, -1):
            yield self._items[index]

    def to_list(self) -> list[Any]:
        """Bottom to top, which is how a stack is usually drawn."""
        return self._items.to_list()

    def __repr__(self) -> str:
        return f"ArrayStack(bottom to top: {self.to_list()!r})"


class LinkedStack:
    """A stack built from linked nodes, pushing and popping at the head.

    Every push allocates one node and every pop releases one, so there is never a
    copy and never any wasted capacity. That gives a true worst case O(1) push,
    where the array version has occasional O(n) pauses when it grows.

    It costs more memory per element and loses the cache friendliness of an array,
    so for most uses the array version is the better default. This one earns its
    place where a single slow operation is unacceptable, for example in code with
    a hard latency budget, or where the maximum size is unknown and very large.
    """

    def __init__(self, items: Iterable[Any] | None = None) -> None:
        self._top: Node | None = None
        self._length = 0
        for item in items or ():
            self.push(item)

    def push(self, value: Any) -> None:
        self._top = Node(value, self._top)
        self._length += 1

    def pop(self) -> Any:
        if self._top is None:
            raise StackEmptyError("cannot pop from an empty stack")
        node = self._top
        self._top = node.next
        self._length -= 1
        return node.value

    def peek(self) -> Any:
        if self._top is None:
            raise StackEmptyError("cannot peek at an empty stack")
        return self._top.value

    def is_empty(self) -> bool:
        return self._top is None

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[Any]:
        current = self._top
        while current is not None:
            yield current.value
            current = current.next

    def to_list(self) -> list[Any]:
        """Bottom to top, to match ArrayStack so the two are interchangeable."""
        return list(self)[::-1]

    def __repr__(self) -> str:
        return f"LinkedStack(bottom to top: {self.to_list()!r})"
