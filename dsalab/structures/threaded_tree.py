"""Threaded binary trees: using the wasted pointers.

Start with an observation. A binary tree of n nodes has 2n child pointers, and
exactly n - 1 of them point at something, because every node except the root is
pointed at by exactly one parent. So **n + 1 pointers in every binary tree are
null**. In a typical tree that is more than half of them, sitting there doing
nothing.

A threaded tree puts them to work. Instead of leaving a null left pointer null,
point it at the node that comes *before* this one in inorder. Instead of a null
right pointer, point at the node that comes *after*. Those borrowed pointers are
called threads, and one extra flag per side records whether a pointer is a real
child or a thread.

What that buys is inorder traversal with **no stack and no recursion at all**, in
O(1) extra memory. When you finish a node, if its right pointer is a thread you
follow it straight to the successor. If it is a real child, the successor is the
leftmost node of that subtree.

The trade off is honest and worth stating: every insertion has to maintain the
threads, so writes get more complicated and slightly slower, in exchange for
traversals that allocate nothing. That was a serious win on machines where memory
was scarce, and it still matters in embedded work today. On a modern desktop the
recursive traversal is usually fine, so this is more often studied than deployed.

It is included because the underlying idea generalises: when a structure has
wasted space, look at whether that space can carry useful information.
"""

from __future__ import annotations

from typing import Any


class ThreadedNode:
    """A node whose empty child pointers are reused as inorder links.

    `left_is_thread` and `right_is_thread` say whether the corresponding pointer
    is a genuine child or a borrowed inorder link. Without those flags the two
    are indistinguishable, and following a thread thinking it is a child would
    walk in circles forever.
    """

    __slots__ = ("value", "left", "right", "left_is_thread", "right_is_thread")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.left: ThreadedNode | None = None
        self.right: ThreadedNode | None = None
        self.left_is_thread = True
        self.right_is_thread = True

    def __repr__(self) -> str:
        return f"ThreadedNode({self.value!r})"


class ThreadedBinaryTree:
    """An inorder threaded binary search tree.

    Values are kept in search tree order so that "the next node in inorder" is a
    meaningful thing to want, which is what threads give you cheaply.

    | Operation | Cost | Memory used while running |
    | - | - | - |
    | insert | O(h) | O(1) |
    | contains | O(h) | O(1) |
    | inorder traversal | O(n) | O(1), no stack at all |
    | successor of a node | O(h) worst case, O(1) when a thread is followed | O(1) |

    where h is the height of the tree.
    """

    def __init__(self, values: list[Any] | None = None) -> None:
        self.root: ThreadedNode | None = None
        self._size = 0
        for value in values or ():
            self.insert(value)

    def __len__(self) -> int:
        return self._size

    def insert(self, value: Any) -> None:
        """Insert in search tree order, wiring up the threads as it goes.

        The bookkeeping is the whole difficulty. When a new node is attached as a
        left child, its own right thread must point at its parent, because the
        parent is the next node in inorder. When attached as a right child, its
        left thread points at the parent for the same reason in reverse. The new
        node also inherits whichever thread the parent was using on that side.
        """
        node = ThreadedNode(value)

        if self.root is None:
            self.root = node
            self._size += 1
            return

        current = self.root
        while True:
            if value < current.value:
                if not current.left_is_thread:
                    current = current.left
                    continue
                # Attach on the left. The new node's predecessor is whatever the
                # parent's left thread pointed at, and its successor is the parent.
                node.left = current.left
                node.right = current
                current.left = node
                current.left_is_thread = False
                break

            if value > current.value:
                if not current.right_is_thread:
                    current = current.right
                    continue
                # Attach on the right, mirror image of the case above.
                node.right = current.right
                node.left = current
                current.right = node
                current.right_is_thread = False
                break

            return  # already present, so nothing to do

        self._size += 1

    def contains(self, value: Any) -> bool:
        """Ordinary search tree lookup, taking care not to follow threads."""
        current = self.root
        while current is not None:
            if value == current.value:
                return True
            if value < current.value:
                current = current.left if not current.left_is_thread else None
            else:
                current = current.right if not current.right_is_thread else None
        return False

    def __contains__(self, value: Any) -> bool:
        return self.contains(value)

    def _leftmost(self, node: ThreadedNode) -> ThreadedNode:
        while not node.left_is_thread and node.left is not None:
            node = node.left
        return node

    def successor(self, node: ThreadedNode) -> ThreadedNode | None:
        """The node that comes after this one in inorder.

        Two cases, and the first is the payoff of the whole structure:

        * If the right pointer is a thread, it already points at the successor.
          One step, O(1), no searching.
        * Otherwise the successor is the leftmost node of the right subtree.
        """
        if node.right_is_thread:
            return node.right
        return self._leftmost(node.right)

    def inorder(self) -> list[Any]:
        """Inorder traversal using no stack and no recursion. O(n) time, O(1) memory.

        Compare this with `BinaryTree.inorder_iterative`, which needs a stack that
        can grow to the height of the tree. Here the tree carries its own
        navigation, so the traversal holds exactly one pointer at a time.
        """
        values: list[Any] = []
        if self.root is None:
            return values

        current = self._leftmost(self.root)
        while current is not None:
            values.append(current.value)
            current = self.successor(current)
        return values

    def reverse_inorder(self) -> list[Any]:
        """Inorder backwards, which threads make just as cheap in the other direction."""
        values: list[Any] = []
        if self.root is None:
            return values

        current = self.root
        while not current.right_is_thread and current.right is not None:
            current = current.right

        while current is not None:
            values.append(current.value)
            current = self.predecessor(current)
        return values

    def predecessor(self, node: ThreadedNode) -> ThreadedNode | None:
        """The node that comes before this one in inorder, the mirror of successor."""
        if node.left_is_thread:
            return node.left
        current = node.left
        while not current.right_is_thread and current.right is not None:
            current = current.right
        return current

    def wasted_pointer_count(self) -> int:
        """How many pointers would have been null in an ordinary tree.

        Always n + 1 for a tree of n nodes, and there is a test for it, because
        that count is the entire justification for the structure existing.
        """
        threads = 0
        for node in self._nodes():
            threads += node.left_is_thread + node.right_is_thread
        return threads

    def _nodes(self) -> list[ThreadedNode]:
        collected: list[ThreadedNode] = []
        if self.root is None:
            return collected

        current = self._leftmost(self.root)
        while current is not None:
            collected.append(current)
            current = self.successor(current)
        return collected

    def __repr__(self) -> str:
        return f"ThreadedBinaryTree({self.inorder()!r})"
