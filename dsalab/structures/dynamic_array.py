"""A dynamic array built on genuinely raw memory.

This is the structure Python calls `list`, C++ calls `vector` and Java calls
`ArrayList`. It gives you constant time access by index like a fixed size array,
while still being able to grow.

To make the growth real rather than pretend, the storage here is a `ctypes`
array of object pointers: a fixed size block of memory allocated once, with no
ability to resize itself. That is exactly what a real array is at the machine
level. When it fills up, the only option is to allocate a bigger block and copy
every element across, which is the whole point of the exercise. Storing the
items in a Python list instead would hide the copying behind someone else's
implementation and teach nothing.

The two design decisions worth arguing about, growth factor and shrink
threshold, are explained where they are made.
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterable, Iterator
from typing import Any

from dsalab.tracing import Step, Traced, run

# How much bigger the new block is when the array fills up. Doubling is the
# standard choice and the reasoning is in docs/04-arrays.md: it is what makes
# append cost O(1) amortised. Growing by a fixed number of slots instead would
# make appending n items cost O(n^2) in total.
GROWTH_FACTOR = 2

# Below this fraction full, the array releases memory by halving its block.
# The obvious choice of 1/2 is a trap, and the reason is explained in the
# `_maybe_shrink` docstring.
SHRINK_THRESHOLD = 0.25

INITIAL_CAPACITY = 4


class DynamicArray:
    """A growable array with amortised constant time append.

    Cost of each operation, with the reasoning in docs/04-arrays.md:

    | Operation | Cost | Why |
    | - | - | - |
    | get and set by index | O(1) | Address arithmetic, no searching |
    | append | O(1) amortised | Usually a write, occasionally a full copy |
    | pop from the end | O(1) amortised | Same argument in reverse |
    | insert at position i | O(n) | Everything to the right shifts up one slot |
    | delete at position i | O(n) | Everything to the right shifts down one slot |
    | search by value | O(n) | Nothing is sorted, so every slot may be checked |
    """

    def __init__(self, items: Iterable[Any] | None = None) -> None:
        self._length = 0
        self._capacity = INITIAL_CAPACITY
        self._store = self._make_block(self._capacity)
        for item in items or ():
            self.append(item)

    # Memory handling

    @staticmethod
    def _make_block(capacity: int):
        """Allocate a raw block that can hold `capacity` object pointers.

        `ctypes.py_object * capacity` builds a C array type of that size, and
        calling it allocates one. The slots start empty, and reading a slot that
        was never written raises ValueError rather than returning garbage, which
        is a useful safety net while developing.
        """
        return (ctypes.py_object * capacity)()

    def _resize(self, new_capacity: int) -> None:
        """Move every element into a freshly allocated block of a new size.

        This is the expensive operation the whole structure is arranged around.
        It is O(n) because every single element has to be copied by hand.
        """
        bigger = self._make_block(new_capacity)
        for index in range(self._length):
            bigger[index] = self._store[index]
        self._store = bigger
        self._capacity = new_capacity

    def _release_slot(self, index: int) -> None:
        """Stop holding on to whatever was in `index`, so it can be collected.

        Setting the slot to None looks like it should be enough, and with a
        plain Python list it would be. It is not enough here, and finding out
        why was the most interesting hour of this day.

        A `ctypes` array of `py_object` keeps a second, hidden bookkeeping
        dictionary called `_objects`, mapping slot number to the Python object
        stored there. It exists so that ctypes can keep a reference alive on
        your behalf, since the raw memory only holds a pointer and a pointer
        does not stop the garbage collector. Overwriting the slot with None
        updates the raw memory but leaves the old entry sitting in `_objects`,
        which quietly keeps the removed object alive forever.

        So every removal has to clear both places. The test that catches this
        uses a weak reference to check the popped object is really gone, because
        nothing about the visible behaviour of the array would ever show it.
        """
        self._store[index] = None
        objects = getattr(self._store, "_objects", None)
        if isinstance(objects, dict):
            objects.pop(str(index), None)

    def _maybe_shrink(self) -> None:
        """Release memory when the array has emptied out, but not too eagerly.

        The tempting rule is to halve the block as soon as it is half empty.
        That rule has a nasty failure mode called thrashing. Picture an array of
        capacity 8 holding 4 items. Delete one, it shrinks to capacity 4 holding
        3. Append one, it is full at 4 so it doubles back to 8. Append and delete
        the same element repeatedly and every single operation triggers a full
        O(n) copy, so the amortised guarantee is destroyed by a two element loop.

        Waiting until the array is only a quarter full leaves a gap between the
        shrink point and the grow point. After a shrink the array is half full,
        so it takes n/2 appends to trigger a grow or n/4 deletes to trigger
        another shrink. Either way an expensive copy is paid for by a linear
        number of cheap operations, and the amortised bound survives.
        """
        if self._capacity <= INITIAL_CAPACITY:
            return
        if self._length <= self._capacity * SHRINK_THRESHOLD:
            self._resize(max(INITIAL_CAPACITY, self._capacity // GROWTH_FACTOR))

    # Reading

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[Any]:
        for index in range(self._length):
            yield self._store[index]

    def __getitem__(self, index: int) -> Any:
        """Read the item at `index`. O(1), because the address is computed.

        Negative indices count from the end, matching Python's own convention.
        """
        index = self._normalise(index)
        return self._store[index]

    def __setitem__(self, index: int, value: Any) -> None:
        index = self._normalise(index)
        self._store[index] = value

    def _normalise(self, index: int) -> int:
        """Turn a possibly negative index into a real one, or complain."""
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(f"index out of range: the array holds {self._length} items")
        return index

    def __repr__(self) -> str:
        return f"DynamicArray({list(self)!r}, capacity={self._capacity})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DynamicArray):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other, strict=True))

    @property
    def capacity(self) -> int:
        """How many items the current block can hold before it must grow.

        Exposed because the visualiser draws the empty slots, and because the
        tests check the growth policy directly rather than trusting it.
        """
        return self._capacity

    # Writing. Each of these is implemented once, as a generator that reports
    # what it is doing, with the plain method driving it to the end.

    def append(self, value: Any) -> None:
        run(self.append_traced(value))

    def append_traced(self, value: Any) -> Traced[None]:
        """Add `value` to the end. O(1) amortised, O(n) on the runs that grow."""
        if self._length == self._capacity:
            old = self._capacity
            self._resize(self._capacity * GROWTH_FACTOR)
            yield Step(
                "grow",
                f"The block of {old} slots was full, so a block of {self._capacity} "
                f"was allocated and all {self._length} items were copied across.",
                {"old_capacity": old, "new_capacity": self._capacity, "copied": self._length},
            )

        self._store[self._length] = value
        self._length += 1
        yield Step(
            "append",
            f"Wrote {value!r} into slot {self._length - 1}. "
            f"{self._capacity - self._length} slots still free.",
            {"index": self._length - 1, "value": value, "length": self._length},
        )

    def insert(self, index: int, value: Any) -> None:
        run(self.insert_traced(index, value))

    def insert_traced(self, index: int, value: Any) -> Traced[None]:
        """Insert `value` so that it ends up at `index`. O(n).

        The cost is not the insert itself, it is the shifting. Every element from
        `index` onwards has to move one slot to the right to open a gap, and that
        is why inserting at the front of a big array is slow while appending to
        the end is fast.
        """
        if index < 0:
            index += self._length
        if not 0 <= index <= self._length:
            raise IndexError(f"cannot insert at {index}, the array holds {self._length} items")

        if self._length == self._capacity:
            old = self._capacity
            self._resize(self._capacity * GROWTH_FACTOR)
            yield Step(
                "grow",
                f"Full at {old} slots, so grew to {self._capacity} before inserting.",
                {"old_capacity": old, "new_capacity": self._capacity},
            )

        for position in range(self._length, index, -1):
            self._store[position] = self._store[position - 1]
            yield Step(
                "shift",
                f"Moved the item in slot {position - 1} right into slot {position} "
                f"to open a gap.",
                {"from": position - 1, "to": position},
            )

        self._store[index] = value
        self._length += 1
        yield Step(
            "insert",
            f"Placed {value!r} into the gap at slot {index}.",
            {"index": index, "value": value, "length": self._length},
        )

    def pop(self, index: int | None = None) -> Any:
        return run(self.pop_traced(index))

    def pop_traced(self, index: int | None = None) -> Traced[Any]:
        """Remove and return the item at `index`, or the last one by default.

        Popping the end is O(1). Popping anywhere else is O(n) for the same
        reason inserting is: everything after it shifts down to close the gap.
        """
        if self._length == 0:
            raise IndexError("cannot pop from an empty array")

        index = self._length - 1 if index is None else self._normalise(index)
        value = self._store[index]

        for position in range(index, self._length - 1):
            self._store[position] = self._store[position + 1]
            yield Step(
                "shift",
                f"Moved the item in slot {position + 1} left into slot {position} "
                f"to close the gap.",
                {"from": position + 1, "to": position},
            )

        self._length -= 1
        # Release the slot that is no longer in use. Skipping this is a real
        # memory leak and an easy one to miss, because every test of the visible
        # behaviour still passes. See `_release_slot` for the ctypes detail that
        # makes it more involved than assigning None.
        self._release_slot(self._length)
        yield Step(
            "remove",
            f"Removed {value!r} from slot {index}. The array now holds {self._length} items.",
            {"index": index, "value": value, "length": self._length},
        )

        before = self._capacity
        self._maybe_shrink()
        if self._capacity != before:
            yield Step(
                "shrink",
                f"Only a quarter full, so the block shrank from {before} slots "
                f"to {self._capacity} and the memory was released.",
                {"old_capacity": before, "new_capacity": self._capacity},
            )

        return value

    def remove(self, value: Any) -> None:
        """Remove the first occurrence of `value`. O(n) to find, O(n) to close up."""
        position = self.index(value)
        if position < 0:
            raise ValueError(f"{value!r} is not in the array")
        self.pop(position)

    def index(self, value: Any) -> int:
        """Return the position of the first `value`, or -1 if it is absent. O(n)."""
        return run(self.index_traced(value))

    def index_traced(self, value: Any) -> Traced[int]:
        """Linear search. Nothing is sorted, so there is no shortcut available."""
        for position in range(self._length):
            found = self._store[position] == value
            yield Step(
                "compare",
                f"Is slot {position} equal to {value!r}? {'Yes.' if found else 'No.'}",
                {"index": position, "value": self._store[position], "match": found},
            )
            if found:
                return position
        return -1

    def __contains__(self, value: Any) -> bool:
        return self.index(value) >= 0

    def clear(self) -> None:
        """Empty the array and release the memory back to the starting size."""
        self._length = 0
        self._capacity = INITIAL_CAPACITY
        self._store = self._make_block(self._capacity)

    def to_list(self) -> list[Any]:
        """A plain Python list of the contents, for tests and for display."""
        return list(self)
