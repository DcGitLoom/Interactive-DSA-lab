"""Binary search tree: ordering turns a tree into a searchable structure.

The rule is one sentence: for every node, everything in its left subtree is
smaller and everything in its right subtree is larger. That rule alone gives you
search, insertion and deletion in time proportional to the height of the tree,
because at each node you can discard one whole side.

The word "height" is doing a great deal of work in that sentence, and it is the
honest weakness of this structure. If the tree is balanced, the height is about
log2(n) and everything is fast. If the values arrive in sorted order, every new
node goes to the right of the last one and the tree degenerates into a linked
list of height n. Then search is O(n) and you have built something slower than an
array with more memory overhead.

Sorted input is not an unusual case. It is one of the most common cases there is:
importing an already ordered file, inserting timestamps, replaying a log. So a
plain search tree is genuinely dangerous in production, and this is the strongest
possible motivation for the AVL tree tomorrow.

There is a test here that measures exactly this: it inserts sorted values, hands
the resulting heights to the complexity detective from day 3, and confirms the
growth is linear rather than logarithmic. Seeing the failure measured makes the
fix worth caring about.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from dsalab.invariants import Violation, verify_if_checking
from dsalab.structures.stack import ArrayStack
from dsalab.tracing import Step, Traced, run


class BSTNode:
    """One node of a binary search tree."""

    __slots__ = ("value", "left", "right")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.left: BSTNode | None = None
        self.right: BSTNode | None = None

    def __repr__(self) -> str:
        return f"BSTNode({self.value!r})"

    @property
    def children(self) -> int:
        return (self.left is not None) + (self.right is not None)


class BinarySearchTree:
    """An ordered binary tree with insert, search, delete and ordered queries.

    | Operation | Balanced | Degenerate | Why |
    | - | - | - | - |
    | search | O(log n) | O(n) | One comparison per level |
    | insert | O(log n) | O(n) | Search, then attach a leaf |
    | delete | O(log n) | O(n) | Search, then at most one extra descent |
    | minimum, maximum | O(log n) | O(n) | Walk hard left or hard right |
    | successor, predecessor | O(log n) | O(n) | One descent, or one ancestor walk |
    | inorder traversal | O(n) | O(n) | Every node is visited once |

    The second column is the one to remember. The costs above are all "proportional
    to the height", and nothing in this structure controls the height.
    """

    def __init__(self, values: Iterable[Any] | None = None) -> None:
        self.root: BSTNode | None = None
        self._size = 0
        for value in values or ():
            self.insert(value)

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        """Inorder, which for a search tree means sorted order."""
        return iter(self.inorder())

    def __repr__(self) -> str:
        return f"BinarySearchTree({self.inorder()!r})"

    # Searching

    def contains(self, value: Any) -> bool:
        return run(self.contains_traced(value))

    def contains_traced(self, value: Any) -> Traced[bool]:
        """Look for a value by discarding half the tree at each step.

        This is binary search made structural. Each comparison rules out an
        entire subtree, which is why the cost is the height and not the size.
        """
        current = self.root
        depth = 0

        while current is not None:
            if value == current.value:
                yield Step(
                    "found",
                    f"{value!r} found at depth {depth}.",
                    {"value": current.value, "depth": depth},
                )
                return True

            going_left = value < current.value
            yield Step(
                "compare",
                f"{value!r} is {'smaller' if going_left else 'larger'} than "
                f"{current.value!r}, so the whole {'right' if going_left else 'left'} "
                "subtree can be ignored.",
                {"at": current.value, "depth": depth, "direction": "left" if going_left else "right"},
            )
            current = current.left if going_left else current.right
            depth += 1

        yield Step("missing", f"Ran out of tree, so {value!r} is not here.", {"value": value})
        return False

    def __contains__(self, value: Any) -> bool:
        return self.contains(value)

    def minimum(self) -> Any:
        """The smallest value, found by walking left until you cannot."""
        if self.root is None:
            raise ValueError("an empty tree has no minimum")
        return self._leftmost(self.root).value

    def maximum(self) -> Any:
        """The largest value, found by walking right until you cannot."""
        if self.root is None:
            raise ValueError("an empty tree has no maximum")
        node = self.root
        while node.right is not None:
            node = node.right
        return node.value

    @staticmethod
    def _leftmost(node: BSTNode) -> BSTNode:
        while node.left is not None:
            node = node.left
        return node

    def floor(self, value: Any) -> Any | None:
        """The largest value that is not greater than `value`, or None.

        Written down because it is the operation a hash table cannot do at all.
        A hash table answers "is this exact key present"; a search tree answers
        "what is nearest below this", and that difference is usually the reason
        to choose a tree over a hash table.
        """
        best = None
        current = self.root
        while current is not None:
            if current.value == value:
                return current.value
            if current.value < value:
                best = current.value  # a candidate, but something larger may still fit
                current = current.right
            else:
                current = current.left
        return best

    def ceiling(self, value: Any) -> Any | None:
        """The smallest value that is not less than `value`, or None."""
        best = None
        current = self.root
        while current is not None:
            if current.value == value:
                return current.value
            if current.value > value:
                best = current.value
                current = current.left
            else:
                current = current.right
        return best

    def range_query(self, low: Any, high: Any) -> list[Any]:
        """Every value between low and high inclusive, in sorted order.

        Costs O(k + h), where k is how many values come back. The saving over
        scanning everything comes from pruning: if the current node is already
        below `low`, its entire left subtree is too small and is never entered.
        A hash table has no way to do this and must look at every key.
        """
        found: list[Any] = []

        def walk(node: BSTNode | None) -> None:
            if node is None:
                return
            if node.value > low:
                walk(node.left)
            if low <= node.value <= high:
                found.append(node.value)
            if node.value < high:
                walk(node.right)

        walk(self.root)
        return found

    # Inserting

    def insert(self, value: Any) -> bool:
        return run(self.insert_traced(value))

    def insert_traced(self, value: Any) -> Traced[bool]:
        """Insert a value. Returns False if it was already present.

        A new value always becomes a leaf. The search for where it belongs is the
        whole cost; the attachment is two lines.

        Duplicates are rejected rather than stored. That is a design decision, not
        the only possible one: some trees keep a count per node, and some allow
        equal values on one particular side. Rejecting keeps the invariant strict
        (left is strictly smaller, right is strictly larger), which makes both the
        deletion logic and the invariant check simpler, and it matches how a set
        behaves.
        """
        if self.root is None:
            self.root = BSTNode(value)
            self._size += 1
            yield Step("insert", f"The tree was empty, so {value!r} became the root.",
                       {"value": value, "depth": 0})
            verify_if_checking(self)
            return True

        current = self.root
        depth = 0
        while True:
            if value == current.value:
                yield Step("duplicate", f"{value!r} is already in the tree, so nothing changes.",
                           {"value": value})
                return False

            going_left = value < current.value
            side = "left" if going_left else "right"
            child = current.left if going_left else current.right

            if child is None:
                node = BSTNode(value)
                if going_left:
                    current.left = node
                else:
                    current.right = node
                self._size += 1
                yield Step(
                    "insert",
                    f"{value!r} attached as the {side} child of {current.value!r} "
                    f"at depth {depth + 1}.",
                    {"value": value, "parent": current.value, "side": side, "depth": depth + 1},
                )
                verify_if_checking(self)
                return True

            yield Step(
                "descend",
                f"{value!r} belongs {side} of {current.value!r}, so carry on that way.",
                {"at": current.value, "direction": side, "depth": depth},
            )
            current = child
            depth += 1

    # Deleting

    def delete(self, value: Any) -> bool:
        return run(self.delete_traced(value))

    def delete_traced(self, value: Any) -> Traced[bool]:
        """Remove a value. Returns False if it was not there.

        Deletion is the only genuinely fiddly operation on a search tree, and it
        has exactly three cases:

        1. **A leaf.** Detach it. Nothing else is affected.
        2. **One child.** The child takes the removed node's place. Everything in
           that subtree is already on the correct side of the parent, so the
           ordering still holds.
        3. **Two children.** This is the interesting one. The node cannot simply
           be removed, because its parent has only one link and there are two
           subtrees to rehome. Instead its value is replaced by its **inorder
           successor**, the smallest value in the right subtree, and that
           successor is then deleted from the right subtree.

        Why the successor works: it is larger than everything in the left subtree
        (it is in the right subtree) and smaller than everything else in the right
        subtree (it is that subtree's minimum). It is therefore the only value
        besides the original that can sit in this position without breaking the
        ordering. The inorder predecessor, the largest value on the left, works
        equally well for the mirror image reason.

        And the recursion terminates: the successor is by definition the leftmost
        node of the right subtree, so it has no left child, so deleting it is case
        1 or case 2. The two child case can never trigger another two child case.
        """
        parent: BSTNode | None = None
        current = self.root

        while current is not None and current.value != value:
            parent = current
            current = current.left if value < current.value else current.right

        if current is None:
            yield Step("missing", f"{value!r} is not in the tree, so there is nothing to remove.",
                       {"value": value})
            return False

        if current.children == 2:
            successor_parent = current
            successor = current.right
            while successor.left is not None:
                successor_parent = successor
                successor = successor.left

            yield Step(
                "replace",
                f"{value!r} has two children, so it takes the value of its inorder "
                f"successor {successor.value!r}, which is then removed from below.",
                {"removed": value, "successor": successor.value},
            )
            current.value = successor.value
            # Now delete the successor, which has at most a right child.
            parent, current = successor_parent, successor

        # One child or none: hoist whichever child exists, or None.
        child = current.left if current.left is not None else current.right

        if parent is None:
            self.root = child
        elif parent.left is current:
            parent.left = child
        else:
            parent.right = child

        self._size -= 1
        yield Step(
            "remove",
            f"Unlinked the node holding {current.value!r}"
            + (f", hoisting its child {child.value!r} into its place." if child else "."),
            {"value": current.value, "size": self._size},
        )
        verify_if_checking(self)
        return True

    # Traversal and shape

    def inorder(self) -> list[Any]:
        """Sorted order, iteratively so a deep tree cannot overflow the stack.

        A degenerate tree of ten thousand nodes has a height of ten thousand, and
        the recursive traversal would hit Python's recursion limit long before
        that. Since a degenerate tree is exactly the failure mode this structure
        has, the iterative version is the right default here.
        """
        values: list[Any] = []
        pending = ArrayStack()
        current = self.root

        while current is not None or not pending.is_empty():
            while current is not None:
                pending.push(current)
                current = current.left
            current = pending.pop()
            values.append(current.value)
            current = current.right

        return values

    def height(self) -> int:
        """Edges on the longest root to leaf path. -1 for an empty tree.

        Computed iteratively for the same reason `inorder` is.
        """
        if self.root is None:
            return -1

        tallest = 0
        pending = ArrayStack([(self.root, 0)])
        while not pending.is_empty():
            node, depth = pending.pop()
            tallest = max(tallest, depth)
            if node.left is not None:
                pending.push((node.left, depth + 1))
            if node.right is not None:
                pending.push((node.right, depth + 1))
        return tallest

    def check_invariants(self) -> list[Violation]:
        """State the search tree rules in code and check them.

        Two rules are checked, and the second is the subtle one:

        1. Every node's value is larger than everything in its left subtree and
           smaller than everything in its right subtree.
        2. The recorded size matches the number of nodes actually present.

        Rule 1 is checked with a running range for each node rather than by
        comparing each node against its immediate children. That distinction
        matters: a tree where every node beats its own children can still be
        broken, because a value can be on the correct side of its parent and the
        wrong side of its grandparent. Comparing only against children is the
        classic wrong implementation of this check.
        """
        violations: list[Violation] = []

        def walk(node: BSTNode | None, low: Any, high: Any) -> int:
            if node is None:
                return 0
            if low is not None and node.value <= low:
                violations.append(
                    Violation(
                        "every value on the right of an ancestor is larger",
                        f"{node.value!r} sits in a subtree that must hold values above "
                        f"{low!r}, but it does not",
                    )
                )
            if high is not None and node.value >= high:
                violations.append(
                    Violation(
                        "every value on the left of an ancestor is smaller",
                        f"{node.value!r} sits in a subtree that must hold values below "
                        f"{high!r}, but it does not",
                    )
                )
            return 1 + walk(node.left, low, node.value) + walk(node.right, node.value, high)

        counted = walk(self.root, None, None)
        if counted != self._size:
            violations.append(
                Violation(
                    "the recorded size matches the tree",
                    f"the tree says it holds {self._size} nodes but {counted} were found",
                )
            )

        return violations
