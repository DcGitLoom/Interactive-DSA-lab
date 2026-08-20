"""Linked lists: the opposite trade to arrays.

An array buys fast indexing by demanding one continuous block of memory, and pays
for it whenever something has to be inserted or removed in the middle, because
everything after it has to physically move.

A linked list gives up the continuous block. Each element sits in its own node
somewhere in memory, and each node holds a pointer to the next one. That single
change flips every cost around:

* Inserting or deleting at a known position becomes O(1), because it is a couple
  of pointer assignments and nothing moves.
* Reading the element at position i becomes O(n), because there is no arithmetic
  that can find it. You have to walk from the front, one node at a time.

Neither structure is better. They answer different questions, and choosing
between them means knowing which operation your code does most.

There is also a hidden cost that complexity notation does not show. Every node
carries a pointer as well as its value, so a linked list of small values can
easily use twice the memory of an array holding the same data. Worse, the nodes
are scattered, so walking a list jumps around memory and defeats the CPU cache,
while walking an array reads straight through it. In practice this makes arrays
faster than their complexity suggests for a great many real workloads, which is
why languages default to array backed lists rather than linked ones.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from dsalab.tracing import Step, Traced, run


class Node:
    """One link of a singly linked list: a value and a pointer to the next node.

    `__slots__` is used here for a reason worth knowing. By default every Python
    object carries a dictionary for its attributes, which is flexible and costs
    around fifty bytes per object. Declaring `__slots__` replaces that dictionary
    with fixed storage and cuts the per node overhead substantially. For a
    structure whose whole point is one object per element, that is a real saving.
    """

    __slots__ = ("value", "next")

    def __init__(self, value: Any, next: Node | None = None) -> None:
        self.value = value
        self.next = next

    def __repr__(self) -> str:
        return f"Node({self.value!r})"


class SinglyLinkedList:
    """A chain of nodes, each pointing at the next one.

    A tail pointer is kept alongside the head. Without it, appending means
    walking the whole list to find the end, which is O(n) and turns building a
    list of n items into O(n^2) work. With it, appending is O(1). The cost is one
    extra pointer to keep correct, and every method that can change the last node
    has to remember to update it, which is exactly the kind of bookkeeping the
    tests below are designed to catch.

    | Operation | Cost | Why |
    | - | - | - |
    | prepend | O(1) | Point the new node at the old head |
    | append | O(1) | Only because a tail pointer is kept |
    | pop from the front | O(1) | Move the head on by one |
    | pop from the back | O(n) | Must walk to find the node before the tail |
    | get by index | O(n) | No address arithmetic is possible |
    | insert or delete at a known node | O(1) | Two pointer assignments |
    | search | O(n) | Walk until found |
    """

    def __init__(self, items: Iterable[Any] | None = None) -> None:
        self.head: Node | None = None
        self.tail: Node | None = None
        self._length = 0
        for item in items or ():
            self.append(item)

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[Any]:
        current = self.head
        while current is not None:
            yield current.value
            current = current.next

    def __repr__(self) -> str:
        return f"SinglyLinkedList({list(self)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SinglyLinkedList):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other, strict=True))

    def to_list(self) -> list[Any]:
        return list(self)

    # Adding

    def prepend(self, value: Any) -> None:
        """Put a value at the front. O(1), and the cheapest thing a list can do."""
        self.head = Node(value, self.head)
        if self.tail is None:
            self.tail = self.head
        self._length += 1

    def append(self, value: Any) -> None:
        """Put a value at the back. O(1), thanks to the tail pointer."""
        node = Node(value)
        if self.tail is None:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node
        self._length += 1

    def insert(self, index: int, value: Any) -> None:
        run(self.insert_traced(index, value))

    def insert_traced(self, index: int, value: Any) -> Traced[None]:
        """Insert so the value ends up at `index`. O(n) to walk, O(1) to link.

        The walk is the expensive part, not the insertion. This is the mirror
        image of the array, where the finding is instant and the shifting is what
        costs. Once you already hold a pointer to the right place, a linked list
        insert really is constant time, which is what makes it valuable inside
        other structures like the LRU cache built later.
        """
        if not 0 <= index <= self._length:
            raise IndexError(f"cannot insert at {index}, the list holds {self._length} items")

        if index == 0:
            self.prepend(value)
            yield Step("insert", f"Put {value!r} at the front, no walking needed.", {"index": 0})
            return
        if index == self._length:
            self.append(value)
            yield Step("insert", f"Put {value!r} at the back using the tail pointer.",
                       {"index": index})
            return

        before = self.head
        for position in range(index - 1):
            yield Step(
                "walk",
                f"Stepping past position {position} on the way to {index}.",
                {"index": position, "value": before.value},
            )
            before = before.next

        before.next = Node(value, before.next)
        self._length += 1
        yield Step(
            "insert",
            f"Linked {value!r} in after {before.value!r}. Nothing else had to move.",
            {"index": index, "value": value},
        )

    # Removing

    def pop_front(self) -> Any:
        """Remove and return the first value. O(1)."""
        if self.head is None:
            raise IndexError("cannot pop from an empty list")

        node = self.head
        self.head = node.next
        if self.head is None:
            # The list is now empty, so the tail pointer has to be cleared too.
            # Forgetting this leaves the tail pointing at a removed node, and the
            # next append silently links onto rubbish. There is a test for it.
            self.tail = None
        self._length -= 1
        return node.value

    def pop_back(self) -> Any:
        """Remove and return the last value. O(n), and this is the weak spot.

        Even with a tail pointer, removing the last node means finding the node
        *before* it, and in a singly linked list there is no way back. You have
        to walk from the head. This one operation is the reason the doubly linked
        list exists.
        """
        if self.head is None:
            raise IndexError("cannot pop from an empty list")

        if self.head is self.tail:
            value = self.head.value
            self.head = self.tail = None
            self._length -= 1
            return value

        current = self.head
        while current.next is not self.tail:
            current = current.next

        value = self.tail.value
        current.next = None
        self.tail = current
        self._length -= 1
        return value

    def remove(self, value: Any) -> bool:
        """Remove the first node holding `value`. Returns whether it was found.

        Uses the standard trick of tracking the previous node while walking,
        because unlinking a node needs the node before it and a singly linked
        list cannot look backwards.
        """
        previous: Node | None = None
        current = self.head

        while current is not None:
            if current.value == value:
                if previous is None:
                    self.head = current.next
                else:
                    previous.next = current.next
                if current is self.tail:
                    self.tail = previous
                self._length -= 1
                return True
            previous, current = current, current.next

        return False

    # Reading

    def get(self, index: int) -> Any:
        """Return the value at `index`. O(n), because it has to be walked to."""
        if not 0 <= index < self._length:
            raise IndexError(f"index out of range: the list holds {self._length} items")

        current = self.head
        for _ in range(index):
            current = current.next
        return current.value

    def find(self, value: Any) -> int:
        return run(self.find_traced(value))

    def find_traced(self, value: Any) -> Traced[int]:
        """Return the index of the first `value`, or -1. O(n)."""
        current = self.head
        index = 0
        while current is not None:
            found = current.value == value
            yield Step(
                "compare",
                f"Node at position {index} holds {current.value!r}. "
                f"{'Match.' if found else 'Not it, moving on.'}",
                {"index": index, "value": current.value, "match": found},
            )
            if found:
                return index
            current = current.next
            index += 1
        return -1

    def __contains__(self, value: Any) -> bool:
        return self.find(value) >= 0

    def middle(self) -> Any:
        """Return the middle value in a single pass, using two pointers.

        The obvious method walks the list once to count, then walks half of it
        again. That is two passes, and it does not work at all on a stream you
        can only read once.

        The tortoise and hare trick does it in one pass. Move one pointer a step
        at a time and another two steps at a time. When the fast one falls off
        the end, the slow one is exactly halfway. Same O(n) complexity, half the
        walking, and it works on data you cannot rewind.
        """
        if self.head is None:
            raise IndexError("an empty list has no middle")

        slow = fast = self.head
        while fast.next is not None and fast.next.next is not None:
            slow = slow.next
            fast = fast.next.next
        return slow.value

    # Rearranging

    def reverse(self) -> None:
        run(self.reverse_traced())

    def reverse_traced(self) -> Traced[None]:
        """Reverse the list in place. O(n) time and O(1) extra memory.

        The three pointer dance is the thing to memorise, and the order of the
        four lines inside the loop is what makes it work:

            save the next node    (or the rest of the list is lost forever)
            point current backwards
            move previous forward
            move current forward

        Get the first line wrong and the moment you reassign `current.next` you
        have dropped every node after it, with no way to reach them again.
        """
        previous: Node | None = None
        current = self.head
        self.tail = self.head

        while current is not None:
            following = current.next
            current.next = previous
            yield Step(
                "reverse",
                f"Turned the pointer on {current.value!r} to face backwards.",
                {"value": current.value},
            )
            previous, current = current, following

        self.head = previous

    def has_cycle(self) -> bool:
        """Detect whether the list loops back on itself. O(n) time, O(1) memory.

        This is Floyd's cycle detection, the tortoise and hare again. Move one
        pointer one step at a time and another two steps at a time. If there is
        no loop, the fast one reaches the end and the answer is no. If there is a
        loop, both pointers end up inside it, and since the fast one gains one
        position per step it must eventually land on the slow one.

        The obvious alternative is to record every node visited in a set and stop
        when one repeats. That also works and is easier to think about, but it
        costs O(n) memory. Floyd's method costs two pointers, which is why it is
        the one worth knowing.

        Note that a list built through the public methods here can never have a
        cycle. This exists because cycles do happen in real code, usually by
        accident, and because the technique is used again for finding duplicates
        and for the middle of a list.
        """
        slow = fast = self.head
        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return True
        return False


class DoublyNode:
    """A node that knows both its neighbours."""

    __slots__ = ("value", "prev", "next")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.prev: DoublyNode | None = None
        self.next: DoublyNode | None = None

    def __repr__(self) -> str:
        return f"DoublyNode({self.value!r})"


class DoublyLinkedList:
    """A linked list where every node points both forwards and backwards.

    The extra pointer per node buys three things: removing from the back becomes
    O(1) instead of O(n), the list can be walked in either direction, and a node
    can be unlinked when all you hold is that node itself, with no need to know
    what came before it.

    That last property is the one that matters most in practice. It is what makes
    an LRU cache possible: the hash table hands you a node directly, and you can
    pull it out of the list in constant time. A singly linked list cannot do that
    at any price.

    The cost is one more pointer per node, and roughly twice as many pointer
    updates on every change, each of which is a chance to get it wrong.

    | Operation | Cost |
    | - | - |
    | push and pop at either end | O(1) |
    | unlink a node you already hold | O(1) |
    | get by index | O(n) |
    | search | O(n) |
    """

    def __init__(self, items: Iterable[Any] | None = None) -> None:
        self.head: DoublyNode | None = None
        self.tail: DoublyNode | None = None
        self._length = 0
        for item in items or ():
            self.push_back(item)

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[Any]:
        current = self.head
        while current is not None:
            yield current.value
            current = current.next

    def __repr__(self) -> str:
        return f"DoublyLinkedList({list(self)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DoublyLinkedList):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other, strict=True))

    def to_list(self) -> list[Any]:
        return list(self)

    def to_list_backwards(self) -> list[Any]:
        """Walk from the tail to the head, which a singly linked list cannot do."""
        values = []
        current = self.tail
        while current is not None:
            values.append(current.value)
            current = current.prev
        return values

    def push_front(self, value: Any) -> DoublyNode:
        """Add at the front and return the new node. O(1).

        The node is returned rather than discarded so callers can hold on to it
        and later unlink it in constant time. The LRU cache depends on this.
        """
        node = DoublyNode(value)
        node.next = self.head
        if self.head is not None:
            self.head.prev = node
        self.head = node
        if self.tail is None:
            self.tail = node
        self._length += 1
        return node

    def push_back(self, value: Any) -> DoublyNode:
        """Add at the back and return the new node. O(1)."""
        node = DoublyNode(value)
        node.prev = self.tail
        if self.tail is not None:
            self.tail.next = node
        self.tail = node
        if self.head is None:
            self.head = node
        self._length += 1
        return node

    def pop_front(self) -> Any:
        """Remove and return the first value. O(1)."""
        if self.head is None:
            raise IndexError("cannot pop from an empty list")
        return self.unlink(self.head)

    def pop_back(self) -> Any:
        """Remove and return the last value. O(1), unlike the singly linked case."""
        if self.tail is None:
            raise IndexError("cannot pop from an empty list")
        return self.unlink(self.tail)

    def unlink(self, node: DoublyNode) -> Any:
        """Remove a node you already hold, in constant time.

        This is the operation that justifies the whole structure. Because the
        node knows both neighbours, they can be stitched together directly with
        no searching at all.

        The two `if` statements handle the node being at either end, where one of
        the neighbours does not exist. Getting those wrong is the classic doubly
        linked list bug, and it usually shows up as a list that looks right going
        forwards and is broken going backwards, which is why the tests here check
        both directions after every removal.
        """
        if node.prev is not None:
            node.prev.next = node.next
        else:
            self.head = node.next

        if node.next is not None:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

        # Clear the removed node's own pointers so it cannot be used to walk back
        # into a list it no longer belongs to, and so it stops holding its former
        # neighbours alive.
        node.prev = node.next = None
        self._length -= 1
        return node.value

    def move_to_front(self, node: DoublyNode) -> None:
        """Take an existing node and make it the head. O(1).

        Exactly what an LRU cache needs on every access, and the reason that
        cache can promise constant time operations.
        """
        if node is self.head:
            return
        self.unlink(node)
        node.next = self.head
        if self.head is not None:
            self.head.prev = node
        self.head = node
        if self.tail is None:
            self.tail = node
        self._length += 1


class CircularLinkedList:
    """A singly linked list whose last node points back at the first.

    There is no end to fall off, so anything that walks it has to count instead
    of checking for None, and forgetting that is an easy way to write an infinite
    loop.

    Where this earns its place is round robin behaviour: taking turns among a
    fixed set of participants, where after the last one you want the first again
    with no special case. Operating system schedulers and the Josephus problem
    both work this way, and the `josephus` method below solves that problem
    directly by walking the circle and removing as it goes.

    Only the tail is kept, not the head. That is not an arbitrary choice: from
    the tail you can reach the head in one step, since `tail.next` is the head,
    so a single pointer gives O(1) access to both ends.
    """

    def __init__(self, items: Iterable[Any] | None = None) -> None:
        self.tail: Node | None = None
        self._length = 0
        for item in items or ():
            self.append(item)

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[Any]:
        """Walk the circle exactly once, using the length rather than a None check."""
        if self.tail is None:
            return
        current = self.tail.next
        for _ in range(self._length):
            yield current.value
            current = current.next

    def __repr__(self) -> str:
        return f"CircularLinkedList({list(self)!r})"

    @property
    def head(self) -> Node | None:
        """The first node, which is always one step past the tail."""
        return self.tail.next if self.tail is not None else None

    def to_list(self) -> list[Any]:
        return list(self)

    def append(self, value: Any) -> None:
        """Add at the end of the circle. O(1)."""
        node = Node(value)
        if self.tail is None:
            node.next = node  # a circle of one points at itself
            self.tail = node
        else:
            node.next = self.tail.next
            self.tail.next = node
            self.tail = node
        self._length += 1

    def prepend(self, value: Any) -> None:
        """Add at the front of the circle. O(1).

        This is exactly the same linking work as `append`, with one line left
        out: the tail pointer does not move. In a circle the only difference
        between the front and the back is where you consider the tail to be, so
        the same two pointer assignments produce either result.
        """
        if self.tail is None:
            self.append(value)
            return

        node = Node(value, self.tail.next)
        self.tail.next = node
        self._length += 1

    def remove(self, value: Any) -> bool:
        """Remove the first node holding `value`. Returns whether it was found."""
        if self.tail is None:
            return False

        previous = self.tail
        current = self.tail.next
        for _ in range(self._length):
            if current.value == value:
                if current is current.next:
                    self.tail = None
                else:
                    previous.next = current.next
                    if current is self.tail:
                        self.tail = previous
                self._length -= 1
                return True
            previous, current = current, current.next
        return False

    def josephus(self, step: int) -> Traced[Any]:
        """Solve the Josephus problem: repeatedly remove every `step`th person.

        Stand n people in a circle and count round, removing every step'th one,
        until a single survivor is left. This is the classic use of a circular
        list, because after the last position the count simply carries on into
        the first with no special handling at all.

        Costs O(n times step) here, since each removal walks that far round the
        circle. There is a well known O(n) formula for the survivor's position,
        but the simulation is what shows the structure doing its job.
        """
        if self.tail is None:
            raise IndexError("nobody is in the circle")
        if step < 1:
            raise ValueError("the counting step must be at least 1")

        while self._length > 1:
            previous = self.tail
            for _ in range(step - 1):
                previous = previous.next
            doomed = previous.next
            yield Step(
                "eliminate",
                f"Counted {step} places round and removed {doomed.value!r}. "
                f"{self._length - 1} left in the circle.",
                {"removed": doomed.value, "remaining": self._length - 1},
            )
            previous.next = doomed.next
            self.tail = previous
            self._length -= 1

        return self.tail.value


def merge_sorted(left: SinglyLinkedList, right: SinglyLinkedList) -> SinglyLinkedList:
    return run(merge_sorted_traced(left, right))


def merge_sorted_traced(
    left: SinglyLinkedList, right: SinglyLinkedList
) -> Traced[SinglyLinkedList]:
    """Merge two already sorted lists into one sorted list. O(n + m).

    This is the same merge that sits at the heart of merge sort, and it is the
    operation linked lists are genuinely better at than arrays. Merging two
    sorted arrays needs somewhere to put the result, so it costs O(n + m) extra
    memory. Merging two linked lists just relinks the existing nodes, so it costs
    nothing extra at all.

    That is why merge sort on a linked list is a genuinely in place O(n log n)
    sort, while merge sort on an array is not.
    """
    merged = SinglyLinkedList()
    a, b = left.head, right.head

    while a is not None and b is not None:
        if a.value <= b.value:
            yield Step(
                "compare",
                f"{a.value!r} is not larger than {b.value!r}, so it goes next.",
                {"chosen": a.value, "from": "left"},
            )
            merged.append(a.value)
            a = a.next
        else:
            yield Step(
                "compare",
                f"{b.value!r} is smaller than {a.value!r}, so it goes next.",
                {"chosen": b.value, "from": "right"},
            )
            merged.append(b.value)
            b = b.next

    # One list is now empty. Everything left in the other is already sorted and
    # already larger than everything placed so far, so it can be appended as is.
    for remaining, side in ((a, "left"), (b, "right")):
        while remaining is not None:
            yield Step(
                "drain",
                f"The other list is finished, so {remaining.value!r} follows directly.",
                {"chosen": remaining.value, "from": side},
            )
            merged.append(remaining.value)
            remaining = remaining.next

    return merged
