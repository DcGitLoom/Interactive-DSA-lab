"""Segment trees and Fenwick trees: answering questions about ranges, fast.

The problem both solve: given an array, answer many queries like "what is the sum
of positions 3 to 17" while the array is also being changed.

Two obvious approaches, and both are bad in the same way:

* **Scan on every query.** Updates are O(1), queries are O(n). Fine if you rarely
  query.
* **Keep a running prefix sum array.** Queries become O(1), because the sum of a
  range is one prefix sum minus another. But a single update invalidates every
  prefix after it, so updates become O(n).

Each is fast at one operation and slow at the other. Both structures here get
**O(log n) for both**, and they do it by storing partial answers for nested blocks
rather than for single positions or for the whole array.

The two are not interchangeable:

* A **segment tree** works for any operation that combines associatively:
  sum, minimum, maximum, greatest common divisor, matrix product. It also supports
  range updates with lazy propagation.
* A **Fenwick tree** only works for operations with an inverse, in practice sums.
  In exchange it is a third of the code, uses n slots instead of 4n, and is
  noticeably faster in practice because of a very tight inner loop.

The rule of thumb: reach for a Fenwick tree if you only need sums, and a segment
tree the moment you need minimum, maximum or range updates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from dsalab.invariants import Violation, verify_if_checking
from dsalab.tracing import Step, Traced, run


class SegmentTree:
    """A tree of partial answers over ranges, in an array.

    Each node holds the answer for one range. The root covers everything, its two
    children cover the two halves, and so on down to leaves covering one position
    each. A query for an arbitrary range breaks into at most 2 log n of those
    prebuilt blocks, which is where the logarithmic cost comes from.

    | Operation | Cost |
    | - | - |
    | build | O(n) |
    | query a range | O(log n) |
    | update one position | O(log n) |
    | memory | 4n slots |

    Why 4n and not 2n: the tree is only perfectly balanced when n is a power of
    two. For any other n the recursion can reach a depth one greater, and the safe
    bound that covers every n is 4n. Allocating 2n works for many inputs and then
    quietly writes out of bounds on others, which is a nasty bug to chase, so the
    generous bound is the standard choice.
    """

    def __init__(
        self,
        values: Sequence[Any],
        combine: Callable[[Any, Any], Any] | None = None,
        identity: Any = 0,
    ) -> None:
        """`combine` must be associative. `identity` is its neutral value.

        Associativity is the requirement, not commutativity. The tree always
        combines neighbouring ranges in order, so a non commutative operation like
        matrix multiplication works fine, while an operation where grouping
        changes the answer, such as subtraction, does not.
        """
        self._values = list(values)
        self._combine = combine or (lambda a, b: a + b)
        self._identity = identity
        self._size = len(self._values)
        self._tree: list[Any] = [identity] * (4 * max(1, self._size))

        if self._size:
            self._build(1, 0, self._size - 1)

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"SegmentTree({self._values!r})"

    def _build(self, node: int, low: int, high: int) -> None:
        if low == high:
            self._tree[node] = self._values[low]
            return

        middle = (low + high) // 2
        self._build(node * 2, low, middle)
        self._build(node * 2 + 1, middle + 1, high)
        self._tree[node] = self._combine(self._tree[node * 2], self._tree[node * 2 + 1])

    def query(self, low: int, high: int) -> Any:
        return run(self.query_traced(low, high))

    def query_traced(self, low: int, high: int) -> Traced[Any]:
        """Combine everything from `low` to `high` inclusive. O(log n).

        The recursion has three cases at each node, and they are the whole
        algorithm:

        * **No overlap** with the wanted range: contribute the identity and stop.
        * **Fully inside** the wanted range: return this node's stored answer
          without going any deeper. This is where the saving comes from.
        * **Partial overlap**: ask both children and combine.

        The second case is the one to understand. A query never visits more than
        about 2 log n nodes, because at each level at most two nodes can be
        partially overlapping, and everything else is answered whole or skipped.
        """
        if self._size == 0:
            return self._identity
        if not (0 <= low <= high < self._size):
            raise IndexError(f"the range {low} to {high} is outside an array of {self._size}")

        def search(node: int, node_low: int, node_high: int) -> Traced[Any]:
            if high < node_low or node_high < low:
                yield Step(
                    "skip",
                    f"The block covering {node_low} to {node_high} is outside the query, "
                    "so it is skipped entirely.",
                    {"low": node_low, "high": node_high},
                )
                return self._identity

            if low <= node_low and node_high <= high:
                yield Step(
                    "use",
                    f"The block covering {node_low} to {node_high} sits entirely inside "
                    f"the query, so its stored answer {self._tree[node]!r} is used without "
                    "looking any deeper.",
                    {"low": node_low, "high": node_high, "value": self._tree[node]},
                )
                return self._tree[node]

            middle = (node_low + node_high) // 2
            yield Step(
                "split",
                f"The block covering {node_low} to {node_high} only partly overlaps, so "
                "both halves are asked.",
                {"low": node_low, "high": node_high},
            )
            left = yield from search(node * 2, node_low, middle)
            right = yield from search(node * 2 + 1, middle + 1, node_high)
            return self._combine(left, right)

        return (yield from search(1, 0, self._size - 1))

    def update(self, index: int, value: Any) -> None:
        run(self.update_traced(index, value))

    def update_traced(self, index: int, value: Any) -> Traced[None]:
        """Set one position and repair every range that contains it. O(log n).

        Exactly one path from leaf to root contains the changed position, and only
        the nodes on that path can have a different answer. Everything else is
        untouched, which is why an update is logarithmic rather than linear.
        """
        if not 0 <= index < self._size:
            raise IndexError(f"index {index} is outside an array of {self._size}")

        self._values[index] = value

        def repair(node: int, low: int, high: int) -> Traced[None]:
            if low == high:
                self._tree[node] = value
                yield Step("set", f"Position {index} is now {value!r}.",
                           {"index": index, "value": value})
                return

            middle = (low + high) // 2
            if index <= middle:
                yield from repair(node * 2, low, middle)
            else:
                yield from repair(node * 2 + 1, middle + 1, high)

            self._tree[node] = self._combine(self._tree[node * 2], self._tree[node * 2 + 1])
            yield Step(
                "recombine",
                f"The block covering {low} to {high} contains the changed position, so its "
                f"answer is recomputed as {self._tree[node]!r}.",
                {"low": low, "high": high, "value": self._tree[node]},
            )

        yield from repair(1, 0, self._size - 1)
        verify_if_checking(self)

    def to_list(self) -> list[Any]:
        return list(self._values)

    def check_invariants(self) -> list[Violation]:
        """Every internal node must equal the combination of its two children."""
        violations: list[Violation] = []

        if self._size == 0:
            return violations

        def check(node: int, low: int, high: int) -> None:
            if low == high:
                if self._tree[node] != self._values[low]:
                    violations.append(Violation(
                        "each leaf holds the value at its position",
                        f"the leaf for position {low} holds {self._tree[node]!r} but the "
                        f"array holds {self._values[low]!r}",
                    ))
                return

            middle = (low + high) // 2
            check(node * 2, low, middle)
            check(node * 2 + 1, middle + 1, high)

            expected = self._combine(self._tree[node * 2], self._tree[node * 2 + 1])
            if self._tree[node] != expected:
                violations.append(Violation(
                    "each node is the combination of its two children",
                    f"the block covering {low} to {high} holds {self._tree[node]!r} but its "
                    f"children combine to {expected!r}",
                ))

        check(1, 0, self._size - 1)
        return violations


class LazySegmentTree:
    """A segment tree that can add a value to a whole range in O(log n).

    Without lazy propagation, adding 5 to positions 3 through 100000 means 99998
    separate updates. The idea that fixes it is beautifully simple:

    **When a range update covers a node completely, do not push it down. Record it
    at that node and stop.** The stored note says "everything below me has this
    much added, I just have not told them yet". The note is only pushed one level
    down when something actually needs to look inside that node.

    So work is deferred until it is unavoidable, and if nobody ever queries inside
    that range the work is never done at all. This is the same instinct as the
    tombstones on day 14 and the amortised copying on day 4: **do the expensive
    thing only when someone forces you to.**

    Restricted to sums with range addition here, which is the common case. The
    general version needs the update operation to compose with itself in a way the
    node can record, which is a real constraint: range assignment works, range
    multiplication works, but not every operation does.
    """

    def __init__(self, values: Sequence[int]) -> None:
        self._size = len(values)
        self._values = list(values)
        self._tree = [0] * (4 * max(1, self._size))
        self._pending = [0] * (4 * max(1, self._size))
        if self._size:
            self._build(1, 0, self._size - 1)

    def __len__(self) -> int:
        return self._size

    def _build(self, node: int, low: int, high: int) -> None:
        if low == high:
            self._tree[node] = self._values[low]
            return
        middle = (low + high) // 2
        self._build(node * 2, low, middle)
        self._build(node * 2 + 1, middle + 1, high)
        self._tree[node] = self._tree[node * 2] + self._tree[node * 2 + 1]

    def _push_down(self, node: int, low: int, high: int) -> None:
        """Hand this node's pending addition to its two children.

        Called only when a query or update needs to look inside, which is what
        keeps the deferred work deferred.
        """
        if self._pending[node] == 0:
            return

        amount = self._pending[node]
        middle = (low + high) // 2

        for child, span in (
            (node * 2, middle - low + 1),
            (node * 2 + 1, high - middle),
        ):
            self._tree[child] += amount * span
            self._pending[child] += amount

        self._pending[node] = 0

    def add_to_range(self, low: int, high: int, amount: int) -> None:
        run(self.add_to_range_traced(low, high, amount))

    def add_to_range_traced(self, low: int, high: int, amount: int) -> Traced[None]:
        """Add `amount` to every position from low to high inclusive. O(log n)."""
        if self._size == 0:
            return
        if not (0 <= low <= high < self._size):
            raise IndexError(f"the range {low} to {high} is outside an array of {self._size}")

        def apply(node: int, node_low: int, node_high: int) -> Traced[None]:
            if high < node_low or node_high < low:
                return

            if low <= node_low and node_high <= high:
                span = node_high - node_low + 1
                self._tree[node] += amount * span
                self._pending[node] += amount
                yield Step(
                    "defer",
                    f"The block covering {node_low} to {node_high} is entirely inside the "
                    f"update, so its total goes up by {amount * span} and a note is left "
                    "for its children instead of visiting them.",
                    {"low": node_low, "high": node_high, "pending": amount},
                )
                return

            self._push_down(node, node_low, node_high)
            middle = (node_low + node_high) // 2
            yield from apply(node * 2, node_low, middle)
            yield from apply(node * 2 + 1, middle + 1, node_high)
            self._tree[node] = self._tree[node * 2] + self._tree[node * 2 + 1]

        yield from apply(1, 0, self._size - 1)

    def query(self, low: int, high: int) -> int:
        """Sum of the range, pushing down any pending notes on the way. O(log n)."""
        if self._size == 0:
            return 0
        if not (0 <= low <= high < self._size):
            raise IndexError(f"the range {low} to {high} is outside an array of {self._size}")

        def search(node: int, node_low: int, node_high: int) -> int:
            if high < node_low or node_high < low:
                return 0
            if low <= node_low and node_high <= high:
                return self._tree[node]

            self._push_down(node, node_low, node_high)
            middle = (node_low + node_high) // 2
            return search(node * 2, node_low, middle) + search(node * 2 + 1, middle + 1, node_high)

        return search(1, 0, self._size - 1)

    def to_list(self) -> list[int]:
        """The current array, by querying each position. Used by the tests."""
        return [self.query(index, index) for index in range(self._size)]


class FenwickTree:
    """A binary indexed tree: prefix sums with updates, in a third of the code.

    The trick is entirely about binary representation, and it is the neatest use
    of bit manipulation in the project.

    Each slot i stores the sum of a block of values ending at i, and the **length
    of that block is the lowest set bit of i**. So slot 12 (binary 1100, lowest
    set bit 4) stores the sum of positions 9 to 12. Slot 8 (binary 1000, lowest
    set bit 8) stores positions 1 to 8.

    Because of that, a prefix sum up to i is assembled by repeatedly stripping the
    lowest set bit: 13 = 1101 gives slot 13, then 12, then 8, and those three
    blocks tile the range 1 to 13 exactly. The number of steps is the number of
    set bits, which is at most log n.

    Updates go the other way: adding to position i means updating every block that
    contains it, found by repeatedly **adding** the lowest set bit.

    `i & -i` extracts the lowest set bit, which is worth unpacking because it looks
    like nonsense. In two's complement, `-i` is `~i + 1`, which flips every bit and
    adds one. The effect is that every bit above the lowest set bit is inverted and
    everything below stays zero, so the AND leaves exactly the lowest set bit
    standing.

    | Operation | Cost |
    | - | - |
    | build from n values | O(n) |
    | prefix sum | O(log n) |
    | range sum | O(log n), as a difference of two prefixes |
    | add to one position | O(log n) |
    | memory | n + 1 slots, against 4n for a segment tree |

    One implementation detail: indices inside are 1 based, because index 0 has no
    lowest set bit and the loops would never terminate. The public methods take 0
    based indices and convert, so callers never see it.
    """

    def __init__(self, values: Sequence[int] | int = 0) -> None:
        if isinstance(values, int):
            self._size = values
            self._tree = [0] * (self._size + 1)
        else:
            self._size = len(values)
            self._tree = [0] * (self._size + 1)
            # Building in O(n) rather than n separate O(log n) updates: add each
            # value into its slot, then push each slot into its parent once.
            for index, value in enumerate(values, start=1):
                self._tree[index] += value
                parent = index + (index & -index)
                if parent <= self._size:
                    self._tree[parent] += self._tree[index]

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"FenwickTree({self.to_list()!r})"

    def add(self, index: int, amount: int) -> None:
        run(self.add_traced(index, amount))

    def add_traced(self, index: int, amount: int) -> Traced[None]:
        """Add `amount` to position `index`. O(log n)."""
        if not 0 <= index < self._size:
            raise IndexError(f"index {index} is outside an array of {self._size}")

        position = index + 1
        while position <= self._size:
            self._tree[position] += amount
            yield Step(
                "update",
                f"Slot {position} covers a block containing position {index}, so it goes "
                f"up by {amount}. Next comes slot {position + (position & -position)}, "
                "found by adding the lowest set bit.",
                {"slot": position, "amount": amount},
            )
            position += position & -position

        verify_if_checking(self)

    def prefix_sum(self, index: int) -> int:
        return run(self.prefix_sum_traced(index))

    def prefix_sum_traced(self, index: int) -> Traced[int]:
        """Sum of positions 0 to `index` inclusive. O(log n)."""
        if index < 0:
            return 0
        if index >= self._size:
            index = self._size - 1

        total = 0
        position = index + 1

        while position > 0:
            total += self._tree[position]
            yield Step(
                "accumulate",
                f"Slot {position} holds the sum of a block of "
                f"{position & -position} value(s), so it is added, giving {total}. "
                "Stripping the lowest set bit moves to the next block.",
                {"slot": position, "block": position & -position, "running": total},
            )
            position -= position & -position

        return total

    def range_sum(self, low: int, high: int) -> int:
        """Sum from low to high inclusive, as the difference of two prefix sums.

        This subtraction is exactly why a Fenwick tree only handles operations with
        an inverse. It cannot answer a range **minimum**, because there is no way
        to subtract the minimum of one prefix from another. A segment tree can,
        which is the practical difference between the two.
        """
        if low > high:
            return 0
        return self.prefix_sum(high) - self.prefix_sum(low - 1)

    def value_at(self, index: int) -> int:
        return self.range_sum(index, index)

    def set(self, index: int, value: int) -> None:
        """Set a position outright, by adding the difference."""
        self.add(index, value - self.value_at(index))

    def to_list(self) -> list[int]:
        return [self.value_at(index) for index in range(self._size)]

    def check_invariants(self) -> list[Violation]:
        """What can be checked here, which is less than you would hope.

        A Fenwick tree **cannot detect its own corruption**, and understanding why
        is more useful than a check that pretends otherwise.

        The reason is that the structure carries no redundancy. A segment tree
        stores a parent and its two children, and the parent must equal the two
        children combined, so changing any one of them contradicts the others. A
        Fenwick tree stores each value exactly once, spread across slots. Change
        any slot and the result is still a perfectly valid Fenwick tree, just of a
        different array. There is nothing to contradict.

        My first version of this method looked convincing and was worthless: it
        derived the array with `to_list()`, which reads the tree, and then checked
        the tree against it. Corrupting a slot changed both sides equally and no
        violation ever appeared. A self referential check is not a check.

        So this verifies only what is genuinely independent: the array is the
        right length, and no slot describes a block running off the front. To
        check the contents, compare against something built separately, which is
        what the tests do by cross checking every range sum against a segment tree
        over the same values.
        """
        violations: list[Violation] = []

        if len(self._tree) != self._size + 1:
            violations.append(Violation(
                "the internal array has one slot per value plus the unused slot zero",
                f"{self._size} values but {len(self._tree)} slots",
            ))

        for position in range(1, self._size + 1):
            block = position & -position
            if position - block < 0:
                violations.append(Violation(
                    "no slot describes a block running off the front of the array",
                    f"slot {position} claims a block of {block} values ending at it",
                ))

        return violations
