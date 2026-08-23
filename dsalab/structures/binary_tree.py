"""Binary trees: the shape almost every fast structure is built from.

A binary tree is a node with a value and up to two children. That is all. The
reason it matters so much is the arithmetic underneath: a tree of height h can
hold up to 2^(h+1) - 1 nodes, which turned around says that n nodes can be
arranged in a tree of height about log2(n).

That is the whole promise. A search that can throw away half the remaining
possibilities at each step finishes in about log2(n) steps, so a million items
take twenty comparisons instead of a million. Every fast searching structure in
this project, from the binary search tree onwards, is chasing that log.

The promise has one enormous catch, which day 10 is entirely about: nothing here
forces a tree to be short. Insert values in increasing order into a plain binary
search tree and you get a chain of height n, all the way back to O(n). Balanced
trees exist to make the log2(n) height a guarantee rather than a hope.

Two representations are here:

* `BinaryTree`, made of linked nodes. This is the normal one.
* `ArrayBinaryTree`, where the tree lives in a flat array and the children of
  index i are at 2i+1 and 2i+2. No pointers at all. It wastes space on a sparse
  tree, which is why it is rarely used for general trees, but it is exactly how
  the heap on day 12 works, so it is worth meeting here first.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from dsalab.structures.deque import Deque
from dsalab.structures.stack import ArrayStack
from dsalab.tracing import Step, Traced, run


class TreeNode:
    """One node: a value and up to two children."""

    __slots__ = ("value", "left", "right")

    def __init__(self, value: Any, left: TreeNode | None = None,
                 right: TreeNode | None = None) -> None:
        self.value = value
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"TreeNode({self.value!r})"

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def children(self) -> int:
        return (self.left is not None) + (self.right is not None)


class BinaryTree:
    """A binary tree of linked nodes, with every traversal written twice.

    Each traversal appears in a recursive form, which is short and obvious, and
    an iterative form, which is longer and uses an explicit stack. Both are here
    on purpose: the recursive version is what you should write, and the iterative
    version is what you need when the tree might be deep enough to overflow the
    call stack, which in Python means around a thousand levels.

    Writing them side by side also makes the point that recursion is not magic.
    The iterative versions are doing exactly what the recursive ones do, with the
    call stack replaced by a stack you can see.
    """

    def __init__(self, root: TreeNode | None = None) -> None:
        self.root = root

    # Traversals, recursive

    def preorder(self) -> list[Any]:
        """Node, then left subtree, then right subtree.

        Use it when the parent must be handled before its children: copying a
        tree, writing one out to a file, or evaluating prefix notation. The first
        value is always the root, which is what makes preorder the natural
        serialisation order.
        """
        return run(self.preorder_traced())

    def preorder_traced(self) -> Traced[list[Any]]:
        visited: list[Any] = []

        def walk(node: TreeNode | None) -> Traced[None]:
            if node is None:
                return
            visited.append(node.value)
            yield Step("visit", f"Visited {node.value!r} on the way down.",
                       {"value": node.value, "order": len(visited)})
            yield from walk(node.left)
            yield from walk(node.right)

        yield from walk(self.root)
        return visited

    def inorder(self) -> list[Any]:
        """Left subtree, then node, then right subtree.

        This is the important one. On a binary search tree, inorder produces the
        values in sorted order, which is the single fact that makes search trees
        useful for anything beyond lookup: ranges, ordered iteration, finding the
        next largest value.
        """
        return run(self.inorder_traced())

    def inorder_traced(self) -> Traced[list[Any]]:
        visited: list[Any] = []

        def walk(node: TreeNode | None) -> Traced[None]:
            if node is None:
                return
            yield from walk(node.left)
            visited.append(node.value)
            yield Step("visit", f"Visited {node.value!r} after finishing its left subtree.",
                       {"value": node.value, "order": len(visited)})
            yield from walk(node.right)

        yield from walk(self.root)
        return visited

    def postorder(self) -> list[Any]:
        """Left subtree, then right subtree, then node.

        Use it when children must be handled before their parent: freeing a tree,
        computing a size or height from the bottom up, or evaluating an
        expression tree where the operands must be worked out before the
        operator can be applied.
        """
        return run(self.postorder_traced())

    def postorder_traced(self) -> Traced[list[Any]]:
        visited: list[Any] = []

        def walk(node: TreeNode | None) -> Traced[None]:
            if node is None:
                return
            yield from walk(node.left)
            yield from walk(node.right)
            visited.append(node.value)
            yield Step("visit", f"Visited {node.value!r} after both its subtrees were done.",
                       {"value": node.value, "order": len(visited)})

        yield from walk(self.root)
        return visited

    def level_order(self) -> list[Any]:
        """Row by row, top to bottom, left to right, using a queue.

        The odd one out: the other three are depth first and use a stack, either
        explicitly or through recursion. This one is breadth first and uses a
        queue, and it cannot be written recursively in any natural way. That is
        the clearest demonstration that stack and queue are what decide whether a
        search goes deep or wide.
        """
        return run(self.level_order_traced())

    def level_order_traced(self) -> Traced[list[Any]]:
        visited: list[Any] = []
        if self.root is None:
            return visited

        waiting = Deque([self.root])
        while not waiting.is_empty():
            node = waiting.pop_front()
            visited.append(node.value)
            yield Step("visit", f"Visited {node.value!r}, taken from the front of the queue.",
                       {"value": node.value, "queued": len(waiting)})
            if node.left is not None:
                waiting.push_back(node.left)
            if node.right is not None:
                waiting.push_back(node.right)

        return visited

    def levels(self) -> list[list[Any]]:
        """Level order, but grouped one list per level.

        The trick is to record how many nodes are in the queue before starting a
        level. That count is exactly the width of the current level, because
        every node of the next level is added after them, so the boundary is
        known without storing depths on the nodes.
        """
        rows: list[list[Any]] = []
        if self.root is None:
            return rows

        waiting = Deque([self.root])
        while not waiting.is_empty():
            width = len(waiting)
            row: list[Any] = []
            for _ in range(width):
                node = waiting.pop_front()
                row.append(node.value)
                if node.left is not None:
                    waiting.push_back(node.left)
                if node.right is not None:
                    waiting.push_back(node.right)
            rows.append(row)

        return rows

    # Traversals, iterative

    def preorder_iterative(self) -> list[Any]:
        """Preorder with a visible stack instead of the call stack.

        The right child is pushed **before** the left one, which looks backwards
        until you remember a stack reverses things. Pushing left first would visit
        the right subtree first and produce a mirror image of the correct answer.
        """
        visited: list[Any] = []
        if self.root is None:
            return visited

        pending = ArrayStack([self.root])
        while not pending.is_empty():
            node = pending.pop()
            visited.append(node.value)
            if node.right is not None:
                pending.push(node.right)
            if node.left is not None:
                pending.push(node.left)

        return visited

    def inorder_iterative(self) -> list[Any]:
        """Inorder with an explicit stack.

        The pattern: run as far left as possible, pushing every node passed. When
        the left runs out, the top of the stack is the next value in order, so
        visit it and move once to the right. Repeat until both the stack and the
        current pointer are exhausted.
        """
        visited: list[Any] = []
        pending = ArrayStack()
        current = self.root

        while current is not None or not pending.is_empty():
            while current is not None:
                pending.push(current)
                current = current.left
            current = pending.pop()
            visited.append(current.value)
            current = current.right

        return visited

    def postorder_iterative(self) -> list[Any]:
        """Postorder with one stack, by building the answer backwards.

        Postorder is the awkward one iteratively, because a node has to be
        visited after both of its subtrees, so you cannot tell on arrival whether
        it is ready.

        The trick used here sidesteps that entirely. Run a preorder that takes the
        **right** child before the left, which produces node, right, left, then
        reverse the whole result. Reversing node-right-left gives left-right-node,
        which is postorder exactly.

        The honest alternative uses two stacks or a "last visited" pointer and is
        considerably more code. This version is worth knowing because it turns a
        hard traversal into an easy one plus a reverse.
        """
        visited: list[Any] = []
        if self.root is None:
            return visited

        pending = ArrayStack([self.root])
        while not pending.is_empty():
            node = pending.pop()
            visited.append(node.value)
            if node.left is not None:
                pending.push(node.left)
            if node.right is not None:
                pending.push(node.right)

        return visited[::-1]

    # Measurements

    def height(self) -> int:
        """Edges on the longest path from the root down to a leaf.

        An empty tree is -1 and a single node is 0, so that height counts edges
        rather than nodes. Both conventions exist; this one is used because it
        makes the AVL balance arithmetic on day 10 come out without extra
        adjustments.
        """

        def measure(node: TreeNode | None) -> int:
            if node is None:
                return -1
            return 1 + max(measure(node.left), measure(node.right))

        return measure(self.root)

    def size(self) -> int:
        """How many nodes there are. O(n), because every node must be counted."""

        def count(node: TreeNode | None) -> int:
            return 0 if node is None else 1 + count(node.left) + count(node.right)

        return count(self.root)

    def leaves(self) -> int:
        """How many nodes have no children."""

        def count(node: TreeNode | None) -> int:
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return count(node.left) + count(node.right)

        return count(self.root)

    def internal_nodes(self) -> int:
        """How many nodes have at least one child."""
        return self.size() - self.leaves()

    def is_full(self) -> bool:
        """True when every node has either no children or exactly two.

        A useful fact that falls out of this shape: in a full binary tree the
        number of leaves is always one more than the number of nodes with two
        children. There is a test for it.
        """

        def check(node: TreeNode | None) -> bool:
            if node is None:
                return True
            if node.children == 1:
                return False
            return check(node.left) and check(node.right)

        return check(self.root)

    def is_perfect(self) -> bool:
        """True when every level is completely filled.

        Equivalent to having exactly 2^(h+1) - 1 nodes for height h, which is the
        cheapest way to check it.
        """
        return self.size() == 2 ** (self.height() + 1) - 1

    def is_balanced(self) -> bool:
        """True when no node's two subtrees differ in height by more than one.

        This is the AVL condition, checked here the slow and obvious way. Day 10
        maintains it during insertion instead of checking it afterwards, which is
        the difference between knowing a tree is balanced and making it so.
        """

        def measure(node: TreeNode | None) -> tuple[bool, int]:
            if node is None:
                return True, -1
            left_ok, left_height = measure(node.left)
            right_ok, right_height = measure(node.right)
            balanced = left_ok and right_ok and abs(left_height - right_height) <= 1
            return balanced, 1 + max(left_height, right_height)

        return measure(self.root)[0]

    def mirror(self) -> None:
        """Swap every node's children, turning the tree into its reflection."""

        def flip(node: TreeNode | None) -> None:
            if node is None:
                return
            node.left, node.right = node.right, node.left
            flip(node.left)
            flip(node.right)

        flip(self.root)

    def __iter__(self) -> Iterator[Any]:
        """Iterating a tree means inorder, since that is the useful order."""
        return iter(self.inorder())

    def __len__(self) -> int:
        return self.size()

    def __repr__(self) -> str:
        return f"BinaryTree({self.level_order()!r})"

    # Construction

    @classmethod
    def from_level_order(cls, values: list[Any | None]) -> BinaryTree:
        """Build from a level order list where None marks a missing child.

        This is the format used by most online judges and by the tests here,
        because it describes any shape of tree compactly.
        """
        if not values or values[0] is None:
            return cls()

        root = TreeNode(values[0])
        waiting = Deque([root])
        index = 1

        while not waiting.is_empty() and index < len(values):
            node = waiting.pop_front()

            if index < len(values) and values[index] is not None:
                node.left = TreeNode(values[index])
                waiting.push_back(node.left)
            index += 1

            if index < len(values) and values[index] is not None:
                node.right = TreeNode(values[index])
                waiting.push_back(node.right)
            index += 1

        return cls(root)

    @classmethod
    def from_preorder_and_inorder(cls, preorder: list[Any], inorder: list[Any]) -> BinaryTree:
        """Rebuild the unique tree that produces these two traversals.

        Why this works, and why two traversals are needed rather than one:

        Preorder's first value is always the root. Finding that value in the
        inorder list splits it in two: everything before it is the left subtree,
        everything after is the right. The sizes of those halves then say exactly
        how much of the preorder list belongs to each side, so the same reasoning
        recurses.

        One traversal alone is not enough, because many different trees produce
        the same preorder. Preorder plus inorder pins it down uniquely, as long as
        the values are distinct. Postorder plus inorder also works. Preorder plus
        postorder does **not**, which surprises people: those two cannot tell
        whether a lone child is a left or a right child.

        This implementation costs O(n) using a lookup table of positions in the
        inorder list. Searching that list on each call instead would be O(n^2),
        which is the version most people write first.
        """
        if len(preorder) != len(inorder):
            raise ValueError("the two traversals must describe the same number of nodes")
        if sorted(map(repr, preorder)) != sorted(map(repr, inorder)):
            raise ValueError("the two traversals must contain the same values")
        if len(set(map(repr, inorder))) != len(inorder):
            raise ValueError("rebuilding needs distinct values, otherwise the split is ambiguous")

        position = {value: index for index, value in enumerate(inorder)}
        next_root = 0

        def build(low: int, high: int) -> TreeNode | None:
            nonlocal next_root
            if low > high:
                return None

            value = preorder[next_root]
            next_root += 1
            node = TreeNode(value)

            split = position[value]
            node.left = build(low, split - 1)
            node.right = build(split + 1, high)
            return node

        return cls(build(0, len(inorder) - 1))


class ArrayBinaryTree:
    """A binary tree stored in a flat array, with no pointers at all.

    The layout is the useful part:

        parent of i   = (i - 1) // 2
        left child    = 2i + 1
        right child   = 2i + 2

    No node objects, no pointers, and moving around the tree is arithmetic. It is
    also cache friendly, because a node and its children sit near each other in
    memory rather than wherever the allocator happened to put them.

    The catch is that the array has to have a slot for every position in the tree,
    filled or not. A tree that is one long chain of n nodes needs an array of
    2^n slots, which is unusable. So this representation is only sensible for
    trees that are complete or nearly so, meaning every level is full except
    possibly the last, which fills from the left.

    That is exactly what a binary heap is, which is why heaps are stored this way
    and why this class is here on day 8 rather than being skipped.
    """

    def __init__(self, values: list[Any | None] | None = None) -> None:
        self._slots: list[Any | None] = list(values or [])

    def __len__(self) -> int:
        return sum(1 for slot in self._slots if slot is not None)

    @staticmethod
    def parent_of(index: int) -> int:
        if index == 0:
            raise IndexError("the root has no parent")
        return (index - 1) // 2

    @staticmethod
    def left_of(index: int) -> int:
        return 2 * index + 1

    @staticmethod
    def right_of(index: int) -> int:
        return 2 * index + 2

    def value_at(self, index: int) -> Any:
        if not 0 <= index < len(self._slots):
            return None
        return self._slots[index]

    def to_linked(self) -> BinaryTree:
        """Convert to the pointer form, so the same traversals can be reused."""

        def build(index: int) -> TreeNode | None:
            if index >= len(self._slots) or self._slots[index] is None:
                return None
            return TreeNode(
                self._slots[index],
                build(self.left_of(index)),
                build(self.right_of(index)),
            )

        return BinaryTree(build(0))

    def __repr__(self) -> str:
        return f"ArrayBinaryTree({self._slots!r})"
