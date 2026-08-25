"""AVL tree: a search tree that refuses to become a list.

Yesterday's search tree has one flaw and it is fatal in practice: nothing stops
it becoming a chain. Sorted input gives a height of n and every operation drops
to O(n).

An AVL tree fixes this with one extra rule on top of the search tree rule:

    For every node, the heights of its two subtrees differ by at most 1.

That is the balance factor, defined here as height(left) - height(right), and it
must always be -1, 0 or +1. When an insertion or deletion pushes it to -2 or +2,
the tree repairs itself with a rotation before returning.

Why that rule is enough to guarantee a logarithmic height is worth understanding
rather than accepting. Ask the opposite question: what is the *fewest* nodes an
AVL tree of height h can have? Call it N(h). The root has two subtrees, and to be
as sparse as possible one has height h-1 and the other h-2, since anything more
lopsided breaks the rule. So

    N(h) = 1 + N(h-1) + N(h-2)

which is the Fibonacci recurrence. Fibonacci numbers grow exponentially, roughly
as 1.618^h, so the minimum number of nodes for height h grows exponentially in h.
Turned around: **the height grows only logarithmically in the number of nodes**,
and specifically at most about 1.44 times log2(n). Even the worst possible AVL
tree is within 44 percent of perfect.

The cost of the guarantee is that every insertion and deletion may need rotations
and every node has to store its height. Rotations are O(1), and at most one
rotation (or one double rotation) is ever needed after an insertion, so insertion
stays O(log n). Deletion can need up to O(log n) rotations, one at each level on
the way back up.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from dsalab.invariants import Violation, verify_if_checking
from dsalab.tracing import Step, Traced, run


class AVLNode:
    """A search tree node that also remembers the height of its subtree.

    Storing the height is what makes balance checking O(1). Recomputing it on
    demand would be O(n) per node and would defeat the whole purpose, so the
    height is maintained on the way back up from every insertion and deletion.
    """

    __slots__ = ("value", "left", "right", "height")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.left: AVLNode | None = None
        self.right: AVLNode | None = None
        self.height = 0  # a lone node has height 0, matching day 8's convention

    def __repr__(self) -> str:
        return f"AVLNode({self.value!r}, height={self.height})"


def _height(node: AVLNode | None) -> int:
    """Height of a subtree, with an empty one counted as -1.

    That -1 is not arbitrary. It makes a leaf come out as 1 + max(-1, -1) = 0 and
    keeps the balance arithmetic free of special cases for missing children.
    """
    return -1 if node is None else node.height


def _balance_factor(node: AVLNode | None) -> int:
    """Left height minus right height. Legal values are -1, 0 and +1."""
    return 0 if node is None else _height(node.left) - _height(node.right)


class AVLTree:
    """A height balanced binary search tree.

    | Operation | Cost | Guaranteed, not hoped for |
    | - | - | - |
    | search | O(log n) | height is at most 1.44 log2(n) |
    | insert | O(log n) | at most one rotation, done on the way back up |
    | delete | O(log n) | up to log n rotations, one per level |
    | minimum, maximum | O(log n) | |
    | inorder traversal | O(n) | |

    Compared with a red black tree (day 11), AVL keeps a tighter balance, so
    lookups are slightly faster, and pays for it with more rotations during
    modification. AVL suits read heavy workloads; red black suits write heavy
    ones. That is the entire practical difference between them.
    """

    def __init__(self, values: Iterable[Any] | None = None) -> None:
        self.root: AVLNode | None = None
        self._size = 0
        self.rotations = 0  # counted so tests and the app can report the work done
        for value in values or ():
            self.insert(value)

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        return iter(self.inorder())

    def __repr__(self) -> str:
        return f"AVLTree({self.inorder()!r}, height={self.height()})"

    # The four rotations

    def _rotate_right(self, node: AVLNode) -> AVLNode:
        """Fix a left heavy node by lifting its left child above it.

                y                x
               / \\              / \\
              x   C    -->     A   y
             / \\                  / \\
            A   B                B   C

        The subtree B moves from x's right to y's left, and that is the only
        thing that changes ownership. The ordering survives because everything in
        B is larger than x and smaller than y, which is exactly the range that
        y's left subtree must hold. Reading the diagram left to right, the values
        appear as A x B y C both before and after, which is the same inorder
        traversal, and that is the proof that a rotation preserves the search
        tree rule.
        """
        pivot = node.left
        node.left = pivot.right
        pivot.right = node

        # Heights must be recomputed bottom up: the node that moved down first,
        # then the one that moved up, or the second calculation uses stale data.
        node.height = 1 + max(_height(node.left), _height(node.right))
        pivot.height = 1 + max(_height(pivot.left), _height(pivot.right))
        self.rotations += 1
        return pivot

    def _rotate_left(self, node: AVLNode) -> AVLNode:
        """Fix a right heavy node by lifting its right child above it.

        The mirror image of `_rotate_right`, and worth writing out separately
        rather than being clever about it, because a "generic" rotation with a
        direction parameter is harder to read and no shorter.
        """
        pivot = node.right
        node.right = pivot.left
        pivot.left = node

        node.height = 1 + max(_height(node.left), _height(node.right))
        pivot.height = 1 + max(_height(pivot.left), _height(pivot.right))
        self.rotations += 1
        return pivot

    def _rebalance(self, node: AVLNode) -> Traced[AVLNode]:
        """Restore the balance rule at this node, if it has been broken.

        Four cases, and the two double rotations are the ones people forget:

        * **Left left.** Too tall on the left, and the left child is itself left
          heavy. One right rotation fixes it.
        * **Right right.** The mirror. One left rotation.
        * **Left right.** Too tall on the left, but the left child leans right.
          A single right rotation here does nothing useful: it just moves the
          problem to the other side. Rotate the child left first to turn it into
          a left left case, then rotate right.
        * **Right left.** The mirror of that.

        The way to see why the double rotation is needed: a single rotation moves
        the child's *outer* subtree up. In the left right case the offending
        depth is in the child's inner subtree, and a single rotation leaves it
        exactly as deep as it was.
        """
        node.height = 1 + max(_height(node.left), _height(node.right))
        balance = _balance_factor(node)

        if balance > 1:
            if _balance_factor(node.left) < 0:
                yield Step(
                    "rotate",
                    f"{node.value!r} is left heavy but its left child leans right, so the "
                    "child is rotated left first to make it a straight line.",
                    {"at": node.value, "case": "left right", "kind": "double"},
                )
                node.left = self._rotate_left(node.left)
            yield Step(
                "rotate",
                f"{node.value!r} is too tall on the left, so it rotates right and its "
                "left child takes its place.",
                {"at": node.value, "case": "left left", "kind": "single"},
            )
            return self._rotate_right(node)

        if balance < -1:
            if _balance_factor(node.right) > 0:
                yield Step(
                    "rotate",
                    f"{node.value!r} is right heavy but its right child leans left, so the "
                    "child is rotated right first.",
                    {"at": node.value, "case": "right left", "kind": "double"},
                )
                node.right = self._rotate_right(node.right)
            yield Step(
                "rotate",
                f"{node.value!r} is too tall on the right, so it rotates left and its "
                "right child takes its place.",
                {"at": node.value, "case": "right right", "kind": "single"},
            )
            return self._rotate_left(node)

        return node

    # Inserting

    def insert(self, value: Any) -> bool:
        return run(self.insert_traced(value))

    def insert_traced(self, value: Any) -> Traced[bool]:
        """Insert, then repair the balance on the way back up.

        The recursion is doing two jobs. On the way down it finds where the value
        belongs, exactly like the plain search tree. On the way back up it
        recomputes each ancestor's height and rebalances it if needed, which is
        possible only because the return trip visits precisely the nodes whose
        subtrees changed.

        This is the clearest example in the whole project of day 2's point about
        recursion having two halves. The search is the downward half, the repair
        is the upward half, and neither could be written as the other.
        """
        inserted = False

        def descend(node: AVLNode | None) -> Traced[AVLNode]:
            nonlocal inserted

            if node is None:
                inserted = True
                yield Step("insert", f"{value!r} became a new leaf.", {"value": value})
                return AVLNode(value)

            if value < node.value:
                yield Step("descend", f"{value!r} belongs left of {node.value!r}.",
                           {"at": node.value, "direction": "left"})
                node.left = yield from descend(node.left)
            elif value > node.value:
                yield Step("descend", f"{value!r} belongs right of {node.value!r}.",
                           {"at": node.value, "direction": "right"})
                node.right = yield from descend(node.right)
            else:
                yield Step("duplicate", f"{value!r} is already here, nothing changes.",
                           {"value": value})
                return node

            return (yield from self._rebalance(node))

        self.root = yield from descend(self.root)
        if inserted:
            self._size += 1
        verify_if_checking(self)
        return inserted

    # Deleting

    def delete(self, value: Any) -> bool:
        return run(self.delete_traced(value))

    def delete_traced(self, value: Any) -> Traced[bool]:
        """Delete, then repair the balance on the way back up.

        The removal itself is exactly the search tree's three cases from day 9.
        What is new is that deletion can unbalance every node on the path back to
        the root, so unlike insertion it may need a rotation at each level. That
        is still O(log n) rotations, but it is the reason deletion is the more
        expensive operation and why red black trees, which need at most three
        rotations per deletion, are preferred where writes dominate.
        """
        removed = False

        def descend(node: AVLNode | None) -> Traced[AVLNode | None]:
            nonlocal removed

            if node is None:
                yield Step("missing", f"{value!r} is not in the tree.", {"value": value})
                return None

            if value < node.value:
                node.left = yield from descend(node.left)
            elif value > node.value:
                node.right = yield from descend(node.right)
            else:
                removed = True
                if node.left is None:
                    yield Step("remove", f"Removed {node.value!r}, hoisting its right subtree.",
                               {"value": node.value})
                    return node.right
                if node.right is None:
                    yield Step("remove", f"Removed {node.value!r}, hoisting its left subtree.",
                               {"value": node.value})
                    return node.left

                successor = node.right
                while successor.left is not None:
                    successor = successor.left
                yield Step(
                    "replace",
                    f"{node.value!r} has two children, so it takes its successor "
                    f"{successor.value!r} and that successor is removed from below.",
                    {"removed": node.value, "successor": successor.value},
                )
                node.value = successor.value
                # Deleting the successor cannot recurse into another two child
                # case, because the successor has no left child by definition.
                node.right = yield from _delete_minimum(node.right, self)

            return (yield from self._rebalance(node))

        self.root = yield from descend(self.root)
        if removed:
            self._size -= 1
        verify_if_checking(self)
        return removed

    # Reading

    def contains(self, value: Any) -> bool:
        current = self.root
        while current is not None:
            if value == current.value:
                return True
            current = current.left if value < current.value else current.right
        return False

    def __contains__(self, value: Any) -> bool:
        return self.contains(value)

    def minimum(self) -> Any:
        if self.root is None:
            raise ValueError("an empty tree has no minimum")
        node = self.root
        while node.left is not None:
            node = node.left
        return node.value

    def maximum(self) -> Any:
        if self.root is None:
            raise ValueError("an empty tree has no maximum")
        node = self.root
        while node.right is not None:
            node = node.right
        return node.value

    def inorder(self) -> list[Any]:
        """Sorted order. Recursion is safe here because the height is guaranteed
        logarithmic, which is precisely what the plain search tree could not
        promise."""
        values: list[Any] = []

        def walk(node: AVLNode | None) -> None:
            if node is None:
                return
            walk(node.left)
            values.append(node.value)
            walk(node.right)

        walk(self.root)
        return values

    def height(self) -> int:
        return _height(self.root)

    def check_invariants(self) -> list[Violation]:
        """The search tree rules, plus the two that make this an AVL tree."""
        violations: list[Violation] = []

        def walk(node: AVLNode | None, low: Any, high: Any) -> int:
            if node is None:
                return 0

            if low is not None and node.value <= low:
                violations.append(Violation(
                    "search order holds all the way down",
                    f"{node.value!r} should be above {low!r} but is not",
                ))
            if high is not None and node.value >= high:
                violations.append(Violation(
                    "search order holds all the way down",
                    f"{node.value!r} should be below {high!r} but is not",
                ))

            real_height = 1 + max(_height(node.left), _height(node.right))
            if node.height != real_height:
                violations.append(Violation(
                    "every node's stored height is correct",
                    f"{node.value!r} claims height {node.height} but its subtrees say "
                    f"{real_height}",
                ))

            balance = _balance_factor(node)
            if abs(balance) > 1:
                violations.append(Violation(
                    "no node's subtrees differ in height by more than one",
                    f"{node.value!r} has balance factor {balance}, so it is "
                    f"{'left' if balance > 0 else 'right'} heavy by too much",
                ))

            return 1 + walk(node.left, low, node.value) + walk(node.right, node.value, high)

        counted = walk(self.root, None, None)
        if counted != self._size:
            violations.append(Violation(
                "the recorded size matches the tree",
                f"the tree says {self._size} nodes but {counted} were found",
            ))

        return violations


def _delete_minimum(node: AVLNode | None, tree: AVLTree) -> Traced[AVLNode | None]:
    """Remove the smallest node of a subtree, rebalancing on the way back up.

    Split out because the two child deletion case needs it and inlining it inside
    that case made the flow of the deletion very hard to follow.
    """
    if node is None:
        return None
    if node.left is None:
        return node.right
    node.left = yield from _delete_minimum(node.left, tree)
    return (yield from tree._rebalance(node))
