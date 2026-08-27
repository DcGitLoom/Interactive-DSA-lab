"""B-tree: the structure that databases and filesystems are actually built from.

Every tree so far has been binary, and binary is the right shape when the data is
in memory. It is the wrong shape when the data is on a disk, and understanding why
explains this entire structure.

Reading from a spinning disk or an SSD does not happen a byte at a time. The
hardware reads a whole block, typically four or eight kilobytes, and the cost is
dominated by finding the block rather than by its size. Reading one byte and
reading four kilobytes cost almost exactly the same.

Now put a balanced binary tree of a million records on disk. Its height is about
20, so a lookup is 20 block reads, and each read hands you a block containing one
key and two pointers, wasting the other four kilobytes.

A B-tree fills the block instead. Make each node hold, say, 200 keys and 201
children, and a million records fit in a tree of height 3. **Three disk reads
instead of twenty.** The comparisons inside a node are free by comparison, since
the block is already in memory.

That is the whole idea: when the expensive operation is fetching a node, make
nodes as large as a fetch, and the tree becomes shallow.

The rules for a B-tree of order m:

* Every node holds at most m - 1 keys and at most m children.
* Every node except the root holds at least ceil(m/2) - 1 keys. This is what
  stops the tree becoming a sparse mess and is why nodes merge on deletion.
* A node with k keys has exactly k + 1 children, unless it is a leaf.
* Keys within a node are sorted, and the subtree between two keys holds values
  between them.
* **All leaves are at the same depth.** This is the balance condition, and it is
  maintained in an unusual way: the tree grows at the root rather than at the
  leaves. A full node splits, pushing its middle key up, and when the root itself
  splits the tree gains a level. It is the only structure here that grows upwards.

A 2-3 tree is exactly a B-tree of order 3, so it is not implemented separately.
`two_three_tree()` builds one, and the tests treat it as the special case it is.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from dsalab.invariants import Violation, verify_if_checking
from dsalab.tracing import Step, Traced, run


class BTreeNode:
    """A node holding several sorted keys and the children between them."""

    __slots__ = ("keys", "children", "is_leaf")

    def __init__(self, is_leaf: bool = True) -> None:
        self.keys: list[Any] = []
        self.children: list[BTreeNode] = []
        self.is_leaf = is_leaf

    def __repr__(self) -> str:
        return f"BTreeNode({self.keys!r}{'' if self.is_leaf else ', internal'})"

    def find_slot(self, key: Any) -> int:
        """Index of the first key not smaller than `key`.

        Linear here for clarity. A real implementation would binary search inside
        the node, which matters when m is 200, though it changes nothing about the
        number of disk reads, which is what actually costs.
        """
        slot = 0
        while slot < len(self.keys) and self.keys[slot] < key:
            slot += 1
        return slot


class BTree:
    """A B-tree of configurable order.

    | Operation | Cost | In disk reads |
    | - | - | - |
    | search | O(log_m n times m) | O(log_m n) |
    | insert | O(log_m n times m) | O(log_m n) |
    | delete | O(log_m n times m) | O(log_m n) |
    | in order traversal | O(n) | O(n / m) |

    The middle column counts comparisons and the right one counts node fetches.
    They differ by a factor of m, and on disk only the right one matters. That gap
    is the entire reason the structure exists.
    """

    def __init__(self, order: int = 5, keys: Iterable[Any] | None = None) -> None:
        if order < 3:
            raise ValueError("the order must be at least 3, since order 2 is a binary tree")
        self.order = order
        self.root = BTreeNode(is_leaf=True)
        self._size = 0
        self.splits = 0
        self.merges = 0
        for key in keys or ():
            self.insert(key)

    @property
    def max_keys(self) -> int:
        return self.order - 1

    @property
    def min_keys(self) -> int:
        """The fewest keys a non root node may hold, which is ceil(m/2) - 1.

        This lower bound is the half of the definition people forget, and it is
        what guarantees the tree stays dense enough to keep its height at
        log_m(n). Without it, a tree of a million keys could legally be a chain of
        nodes holding one key each.
        """
        return (self.order + 1) // 2 - 1

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        return iter(self.keys())

    def __repr__(self) -> str:
        return f"BTree(order={self.order}, {self.keys()!r})"

    # Searching

    def contains(self, key: Any) -> bool:
        return run(self.contains_traced(key))

    def contains_traced(self, key: Any) -> Traced[bool]:
        """Search: within a node scan the keys, then descend into one child."""
        node = self.root
        depth = 0

        while True:
            slot = node.find_slot(key)
            if slot < len(node.keys) and node.keys[slot] == key:
                yield Step("found", f"{key!r} found in a node at depth {depth}.",
                           {"key": key, "depth": depth, "node": list(node.keys)})
                return True

            if node.is_leaf:
                yield Step("missing", f"Reached a leaf without finding {key!r}.",
                           {"key": key, "depth": depth})
                return False

            yield Step(
                "descend",
                f"{key!r} is not among {node.keys!r}, so follow child {slot}. "
                "On disk this is one block read.",
                {"depth": depth, "child": slot, "node": list(node.keys)},
            )
            node = node.children[slot]
            depth += 1

    def __contains__(self, key: Any) -> bool:
        return self.contains(key)

    def keys(self) -> list[Any]:
        """Every key in sorted order, by an in order walk of the whole tree."""
        collected: list[Any] = []

        def walk(node: BTreeNode) -> None:
            for index, key in enumerate(node.keys):
                if not node.is_leaf:
                    walk(node.children[index])
                collected.append(key)
            if not node.is_leaf:
                walk(node.children[-1])

        walk(self.root)
        return collected

    def height(self) -> int:
        """Levels below the root. Every leaf is at this depth, by the rules."""
        depth = 0
        node = self.root
        while not node.is_leaf:
            node = node.children[0]
            depth += 1
        return depth

    # Inserting

    def insert(self, key: Any) -> bool:
        return run(self.insert_traced(key))

    def insert_traced(self, key: Any) -> Traced[bool]:
        """Insert a key, splitting any node that overflows on the way back up.

        There are two standard schemes for this, and choosing between them was
        the real work of the day:

        1. **Preemptive splitting.** On the way down, split every full node you
           pass, so the leaf you reach is guaranteed to have room. One pass, each
           node touched once, which is ideal for a tree on disk. This is the
           version in CLRS.
        2. **Split on overflow.** Insert into the leaf, and if a node ends up
           with too many keys, split it and push the middle key up, repeating
           while parents overflow.

        I wrote the first one and the tests failed at orders 3, 5 and 13 while
        passing at 4, 6 and 8. The reason is worth writing down, because it is
        not obvious and it is not a coding mistake.

        Preemptive splitting splits a node holding the **maximum** m-1 keys into
        two halves plus a promoted middle. For both halves to satisfy the minimum
        of ceil(m/2)-1, the maximum has to be odd, which means m has to be even.
        At order 5 a full node holds 4 keys, and 4 splits as 2 + promoted + 1,
        leaving a node with one key where the minimum is two.

        Since a 2-3 tree is a B-tree of order 3, supporting odd orders is not
        optional here, so this uses scheme 2. A node that overflows holds m keys,
        which splits as m//2 + promoted + the rest, and both halves satisfy the
        minimum at every order. The cost is that a split can cascade back up the
        tree, so a node may be touched twice.

        The lesson: **the elegant one pass version has a precondition that nobody
        states out loud.** Half the textbook presentations of B-trees quietly
        assume an even order throughout.
        """
        if self.contains(key):
            yield Step("duplicate", f"{key!r} is already in the tree.", {"key": key})
            return False

        promoted = yield from self._insert_into(self.root, key)

        if promoted is not None:
            middle_key, right = promoted
            fresh = BTreeNode(is_leaf=False)
            fresh.keys = [middle_key]
            fresh.children = [self.root, right]
            self.root = fresh
            yield Step(
                "grow",
                f"The root overflowed and split, pushing {middle_key!r} into a brand new "
                "root, so the tree gained a level. B-trees grow upwards, which is how "
                "every leaf stays at the same depth.",
                {"height": self.height(), "promoted": middle_key},
            )

        self._size += 1
        verify_if_checking(self)
        return True

    def _insert_into(self, node: BTreeNode, key: Any) -> Traced[tuple[Any, BTreeNode] | None]:
        """Insert into this subtree, returning any key that has to move up.

        A return of None means the subtree absorbed the key. A return of
        (key, node) means this node split, and the caller must take that key and
        that new right hand node into itself.
        """
        if node.is_leaf:
            node.keys.insert(node.find_slot(key), key)
            yield Step(
                "insert",
                f"{key!r} slotted into a leaf, which now holds {node.keys!r}.",
                {"key": key, "node": list(node.keys)},
            )
        else:
            slot = node.find_slot(key)
            promoted = yield from self._insert_into(node.children[slot], key)
            if promoted is not None:
                middle_key, right = promoted
                node.keys.insert(slot, middle_key)
                node.children.insert(slot + 1, right)

        if len(node.keys) > self.max_keys:
            return (yield from self._split(node))
        return None

    def _split(self, node: BTreeNode) -> Traced[tuple[Any, BTreeNode]]:
        """Split an overflowing node in two, returning its middle key and right half.

        The middle key leaves this node entirely and goes up to the parent, where
        it will sit between the two halves and tell later searches which way to
        go. That is why a B-tree's internal keys are separators rather than data
        in their own right.
        """
        middle = len(node.keys) // 2
        promoted = node.keys[middle]

        right = BTreeNode(is_leaf=node.is_leaf)
        right.keys = node.keys[middle + 1 :]
        node.keys = node.keys[:middle]

        if not node.is_leaf:
            right.children = node.children[middle + 1 :]
            node.children = node.children[: middle + 1]

        self.splits += 1
        yield Step(
            "split",
            f"A node overflowed and split into {node.keys!r} and {right.keys!r}, with "
            f"{promoted!r} pushed up to separate them.",
            {"promoted": promoted, "left": list(node.keys), "right": list(right.keys)},
        )
        return promoted, right

    # Deleting

    def delete(self, key: Any) -> bool:
        return run(self.delete_traced(key))

    def delete_traced(self, key: Any) -> Traced[bool]:
        """Delete a key, repairing any node left below the minimum.

        Deletion is the mirror of insertion. Insertion worries about nodes being
        too full, deletion about them being too empty, and there are two repairs:

        * **Borrow** from a neighbouring sibling that has a key to spare. The
          sibling's nearest key moves up into the parent and the parent's
          separating key comes down into the short node. It is a rotation, and it
          is the cheaper fix: three nodes touched, height unchanged, and it never
          propagates.
        * **Merge** with a sibling when neither neighbour can spare a key. The two
          nodes and the parent's separating key combine into one. That removes a
          key from the parent, which may leave *it* short, so a merge can cascade
          upwards, and if the root ends up empty the tree loses a level.

        Repairs happen on the way back up, for the same reason splits do. Fixing
        on the way down, which is the other standard approach, needs the merged
        size to fit inside a node, and that only works out at even orders.
        """
        if not self.contains(key):
            yield Step("missing", f"{key!r} is not in the tree.", {"key": key})
            return False

        yield from self._delete_from(self.root, key)
        self._size -= 1

        if not self.root.is_leaf and not self.root.keys:
            # A merge took the root's last key. This is the only way a B-tree
            # ever gets shorter, and it mirrors the way it grows.
            self.root = self.root.children[0]
            yield Step(
                "shrink",
                "A merge emptied the root, so the tree lost a level.",
                {"height": self.height()},
            )

        verify_if_checking(self)
        return True

    def _delete_from(self, node: BTreeNode, key: Any) -> Traced[bool]:
        slot = node.find_slot(key)
        here = slot < len(node.keys) and node.keys[slot] == key

        if here and node.is_leaf:
            node.keys.pop(slot)
            yield Step(
                "remove",
                f"Removed {key!r} from a leaf, which now holds {node.keys!r}.",
                {"key": key, "node": list(node.keys)},
            )
            return True

        if here:
            # An internal key is a separator between two subtrees, so it cannot
            # simply be removed. The same trick as the binary search tree: swap
            # in the predecessor, which lives in a leaf, and delete that instead.
            predecessor = self._largest_in(node.children[slot])
            node.keys[slot] = predecessor
            yield Step(
                "replace",
                f"{key!r} sits inside an internal node where it separates two subtrees, "
                f"so it is replaced by its predecessor {predecessor!r} and that leaf key "
                "is deleted instead.",
                {"key": key, "replacement": predecessor},
            )
            yield from self._delete_from(node.children[slot], predecessor)
            yield from self._repair(node, slot)
            return True

        if node.is_leaf:
            return False

        found = yield from self._delete_from(node.children[slot], key)
        if found:
            yield from self._repair(node, slot)
        return found

    @staticmethod
    def _largest_in(node: BTreeNode) -> Any:
        while not node.is_leaf:
            node = node.children[-1]
        return node.keys[-1]

    @staticmethod
    def _smallest_in(node: BTreeNode) -> Any:
        while not node.is_leaf:
            node = node.children[0]
        return node.keys[0]

    def _repair(self, parent: BTreeNode, index: int) -> Traced[None]:
        """Restore the minimum fill of the child at `index`, if it fell short.

        Borrowing is tried before merging, because it touches fewer nodes, never
        changes the height and never cascades. Merging is the fallback.
        """
        child = parent.children[index]
        if len(child.keys) >= self.min_keys:
            return

        if index > 0 and len(parent.children[index - 1].keys) > self.min_keys:
            yield from self._borrow_from_left(parent, index)
        elif (
            index < len(parent.children) - 1
            and len(parent.children[index + 1].keys) > self.min_keys
        ):
            yield from self._borrow_from_right(parent, index)
        elif index > 0:
            yield from self._merge(parent, index - 1)
        else:
            yield from self._merge(parent, index)

    def _borrow_from_left(self, parent: BTreeNode, index: int) -> Traced[None]:
        child = parent.children[index]
        sibling = parent.children[index - 1]

        child.keys.insert(0, parent.keys[index - 1])
        parent.keys[index - 1] = sibling.keys.pop()
        if not sibling.is_leaf:
            child.children.insert(0, sibling.children.pop())

        yield Step(
            "borrow",
            "The node dropped below the minimum, so it borrowed from its left sibling: "
            "the sibling's largest key moved up into the parent and the parent's "
            "separating key came down.",
            {"from": "left", "node": list(child.keys)},
        )

    def _borrow_from_right(self, parent: BTreeNode, index: int) -> Traced[None]:
        child = parent.children[index]
        sibling = parent.children[index + 1]

        child.keys.append(parent.keys[index])
        parent.keys[index] = sibling.keys.pop(0)
        if not sibling.is_leaf:
            child.children.append(sibling.children.pop(0))

        yield Step(
            "borrow",
            "Borrowed from the right sibling instead, rotating a key down through "
            "the parent.",
            {"from": "right", "node": list(child.keys)},
        )

    def _merge(self, parent: BTreeNode, index: int) -> Traced[None]:
        """Merge the two children either side of parent.keys[index] into one."""
        left = parent.children[index]
        right = parent.children[index + 1]
        separator = parent.keys.pop(index)

        left.keys.append(separator)
        left.keys.extend(right.keys)
        left.children.extend(right.children)
        parent.children.pop(index + 1)
        self.merges += 1

        yield Step(
            "merge",
            f"Neither sibling could spare a key, so the two nodes and the separating "
            f"key {separator!r} merged into one holding {left.keys!r}. The parent is "
            "now one key shorter, so this can cascade upwards.",
            {"separator": separator, "merged": list(left.keys)},
        )

    def check_invariants(self) -> list[Violation]:
        """Every B-tree rule, checked by name."""
        violations: list[Violation] = []
        leaf_depths: set[int] = set()

        def walk(node: BTreeNode, depth: int, low: Any, high: Any) -> int:
            if len(node.keys) > self.max_keys:
                violations.append(Violation(
                    f"no node holds more than {self.max_keys} keys",
                    f"a node at depth {depth} holds {len(node.keys)}",
                ))

            if node is not self.root and len(node.keys) < self.min_keys:
                violations.append(Violation(
                    f"every node except the root holds at least {self.min_keys} keys",
                    f"a node at depth {depth} holds only {len(node.keys)}: {node.keys!r}",
                ))

            if node is self.root and not node.is_leaf and not node.keys:
                violations.append(Violation(
                    "the root holds at least one key unless the tree is empty",
                    "the root is internal but empty",
                ))

            if node.keys != sorted(node.keys):
                violations.append(Violation(
                    "keys within a node are sorted",
                    f"a node at depth {depth} holds {node.keys!r}",
                ))

            for key in node.keys:
                if low is not None and key <= low:
                    violations.append(Violation(
                        "every key falls inside the range its position allows",
                        f"{key!r} should be above {low!r}",
                    ))
                if high is not None and key >= high:
                    violations.append(Violation(
                        "every key falls inside the range its position allows",
                        f"{key!r} should be below {high!r}",
                    ))

            if node.is_leaf:
                leaf_depths.add(depth)
                return len(node.keys)

            if len(node.children) != len(node.keys) + 1:
                violations.append(Violation(
                    "a node with k keys has exactly k+1 children",
                    f"a node at depth {depth} has {len(node.keys)} keys and "
                    f"{len(node.children)} children",
                ))
                return len(node.keys)

            total = len(node.keys)
            for index, child in enumerate(node.children):
                child_low = node.keys[index - 1] if index > 0 else low
                child_high = node.keys[index] if index < len(node.keys) else high
                total += walk(child, depth + 1, child_low, child_high)
            return total

        counted = walk(self.root, 0, None, None)

        if len(leaf_depths) > 1:
            violations.append(Violation(
                "every leaf is at the same depth",
                f"leaves were found at depths {sorted(leaf_depths)}",
            ))

        if counted != self._size:
            violations.append(Violation(
                "the recorded size matches the tree",
                f"the tree says {self._size} keys but {counted} were found",
            ))

        return violations


def two_three_tree(keys: Iterable[Any] | None = None) -> BTree:
    """A 2-3 tree, which is exactly a B-tree of order 3.

    Every node holds one or two keys and has two or three children, which is where
    the name comes from. Worth naming separately because textbooks introduce it as
    its own structure, and it is easier to draw by hand, but there is nothing to
    implement: it is this same code with order set to 3.
    """
    return BTree(order=3, keys=keys)
