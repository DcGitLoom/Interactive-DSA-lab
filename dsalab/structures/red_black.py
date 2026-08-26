"""Red black tree: balance kept with colours instead of heights.

An AVL tree measures. Every node stores its height, and the tree rebalances
whenever two subtrees differ by more than one. That produces a very short tree
and a fair amount of rotating.

A red black tree does something cleverer and stranger. It stores one bit per
node, a colour, and enforces five rules that never mention height at all:

1. Every node is either red or black.
2. The root is black.
3. Every leaf, meaning every empty position, counts as black.
4. A red node never has a red child. (No two reds in a row.)
5. Every path from a given node down to any empty position passes through the
   same number of black nodes. This count is called the black height.

Rules 4 and 5 together are what force the balance, and the argument is short:
rule 5 says every root to leaf path has the same number of black nodes, so the
shortest possible path is all black, and rule 4 says reds cannot be adjacent, so
the longest possible path alternates red and black and is therefore at most twice
the shortest. **No path is more than twice as long as any other**, which bounds
the height at 2 log2(n + 1).

That bound is looser than AVL's 1.44 log2(n), so lookups are slightly slower. In
exchange, repairs are cheaper: an insertion needs at most two rotations and a
deletion at most three, no matter how big the tree is, where AVL deletion can
rotate at every level. That is why red black trees are what you find inside
`std::map`, Java's `TreeMap` and the Linux kernel, all of which are written for
mixed read and write workloads.

This implementation uses a shared sentinel node for every empty position rather
than None. That is not a micro optimisation, it is what makes deletion tractable:
the fixup has to ask about the colour and the parent of a position that no longer
holds anything, and a sentinel can answer while None cannot.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from dsalab.invariants import Violation, verify_if_checking
from dsalab.tracing import Step, Traced, run

RED = "red"
BLACK = "black"


class RBNode:
    """A node with a colour and a parent pointer.

    The parent pointer is needed because the repair procedures walk **upwards**
    from the point of damage, looking at parents, grandparents and uncles. The
    AVL tree got that path for free from the recursion; the red black fixups are
    written iteratively, so the path has to be stored.
    """

    __slots__ = ("value", "colour", "left", "right", "parent")

    def __init__(self, value: Any, colour: str = RED) -> None:
        self.value = value
        self.colour = colour
        self.left: RBNode
        self.right: RBNode
        self.parent: RBNode

    def __repr__(self) -> str:
        return f"RBNode({self.value!r}, {self.colour})"

    @property
    def is_red(self) -> bool:
        return self.colour == RED


class RedBlackTree:
    """A red black balanced binary search tree.

    | Operation | Cost | Rotations |
    | - | - | - |
    | search | O(log n) | none |
    | insert | O(log n) | at most 2 |
    | delete | O(log n) | at most 3 |
    | minimum, maximum | O(log n) | none |
    | inorder traversal | O(n) | none |

    Height is at most 2 log2(n + 1), which is looser than AVL but bought with far
    fewer rotations on modification.
    """

    def __init__(self, values: Iterable[Any] | None = None) -> None:
        # One sentinel shared by every empty position. It is black, which is
        # rule 3 built into the structure rather than special cased everywhere.
        self.nil = RBNode(None, BLACK)
        self.nil.left = self.nil.right = self.nil.parent = self.nil

        self.root = self.nil
        self._size = 0
        self.rotations = 0
        self.recolourings = 0

        for value in values or ():
            self.insert(value)

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        return iter(self.inorder())

    def __repr__(self) -> str:
        return f"RedBlackTree({self.inorder()!r}, height={self.height()})"

    # Rotations, identical in spirit to the AVL ones but maintaining parents

    def _rotate_left(self, node: RBNode) -> None:
        pivot = node.right
        node.right = pivot.left
        if pivot.left is not self.nil:
            pivot.left.parent = node

        pivot.parent = node.parent
        if node.parent is self.nil:
            self.root = pivot
        elif node is node.parent.left:
            node.parent.left = pivot
        else:
            node.parent.right = pivot

        pivot.left = node
        node.parent = pivot
        self.rotations += 1

    def _rotate_right(self, node: RBNode) -> None:
        pivot = node.left
        node.left = pivot.right
        if pivot.right is not self.nil:
            pivot.right.parent = node

        pivot.parent = node.parent
        if node.parent is self.nil:
            self.root = pivot
        elif node is node.parent.right:
            node.parent.right = pivot
        else:
            node.parent.left = pivot

        pivot.right = node
        node.parent = pivot
        self.rotations += 1

    # Inserting

    def insert(self, value: Any) -> bool:
        return run(self.insert_traced(value))

    def insert_traced(self, value: Any) -> Traced[bool]:
        """Insert a value, then repair whatever the insertion broke.

        The new node is coloured **red**, and that choice is the key to the whole
        design. A red node adds nothing to the black height, so rule 5, the
        expensive rule to restore, is never broken by an insertion. The only rule
        that can break is rule 4, no two reds in a row, and that is a purely
        local problem involving the node, its parent and its uncle.

        Colouring the new node black instead would break rule 5 on every path
        through it, which would mean repairing the whole tree.
        """
        parent = self.nil
        current = self.root

        while current is not self.nil:
            parent = current
            if value == current.value:
                yield Step("duplicate", f"{value!r} is already here.", {"value": value})
                return False
            current = current.left if value < current.value else current.right

        node = RBNode(value, RED)
        node.left = node.right = self.nil
        node.parent = parent

        if parent is self.nil:
            self.root = node
        elif value < parent.value:
            parent.left = node
        else:
            parent.right = node

        self._size += 1
        yield Step(
            "insert",
            f"{value!r} inserted as a red leaf. Red is chosen because it leaves "
            "the black height of every path unchanged.",
            {"value": value, "colour": RED},
        )

        yield from self._repair_after_insert(node)
        verify_if_checking(self)
        return True

    def _repair_after_insert(self, node: RBNode) -> Traced[None]:
        """Restore rule 4, no red node with a red child.

        Everything turns on the colour of the **uncle**, the parent's sibling:

        * **Red uncle.** Recolour: parent and uncle become black, grandparent
          becomes red. That fixes this spot without moving a single pointer, but
          the grandparent is now red and may clash with *its* parent, so the
          problem moves two levels up and the loop repeats. This is why an
          insertion can cause O(log n) recolourings while still needing only O(1)
          rotations.

        * **Black uncle.** Recolouring cannot work here, because making the
          parent black would add a black node to this side only and break rule 5.
          A rotation is needed instead, and after it the loop always ends. If the
          new node is on the inside (a left child of a right child, or the
          mirror), one extra rotation straightens it first, exactly like the AVL
          double rotation cases.
        """
        while node.parent.is_red:
            grandparent = node.parent.parent

            if node.parent is grandparent.left:
                uncle = grandparent.right

                if uncle.is_red:
                    node.parent.colour = BLACK
                    uncle.colour = BLACK
                    grandparent.colour = RED
                    self.recolourings += 1
                    yield Step(
                        "recolour",
                        f"The uncle of {node.value!r} is red, so the parent and uncle go "
                        f"black and the grandparent {grandparent.value!r} goes red. No "
                        "pointers moved, but the clash may now be two levels up.",
                        {"at": node.value, "case": "red uncle"},
                    )
                    node = grandparent
                    continue

                if node is node.parent.right:
                    node = node.parent
                    yield Step(
                        "rotate",
                        f"{node.value!r} leans the wrong way, so it is straightened with "
                        "a left rotation first.",
                        {"at": node.value, "case": "inside, black uncle"},
                    )
                    self._rotate_left(node)

                node.parent.colour = BLACK
                grandparent.colour = RED
                yield Step(
                    "rotate",
                    f"The uncle is black, so recolouring alone would break the black "
                    f"height. Rotating {grandparent.value!r} right instead.",
                    {"at": grandparent.value, "case": "outside, black uncle"},
                )
                self._rotate_right(grandparent)

            else:
                # The exact mirror image of the block above.
                uncle = grandparent.left

                if uncle.is_red:
                    node.parent.colour = BLACK
                    uncle.colour = BLACK
                    grandparent.colour = RED
                    self.recolourings += 1
                    yield Step(
                        "recolour",
                        f"The uncle of {node.value!r} is red, so recolour and carry the "
                        "problem two levels up.",
                        {"at": node.value, "case": "red uncle"},
                    )
                    node = grandparent
                    continue

                if node is node.parent.left:
                    node = node.parent
                    yield Step(
                        "rotate",
                        f"{node.value!r} leans the wrong way, straightening it first.",
                        {"at": node.value, "case": "inside, black uncle"},
                    )
                    self._rotate_right(node)

                node.parent.colour = BLACK
                grandparent.colour = RED
                yield Step(
                    "rotate",
                    f"Black uncle, so rotating {grandparent.value!r} left.",
                    {"at": grandparent.value, "case": "outside, black uncle"},
                )
                self._rotate_left(grandparent)

        # Rule 2: the root is always black. Forcing it here is free, because
        # colouring the root black adds one to the black height of every single
        # path, which keeps rule 5 satisfied.
        if self.root.is_red:
            self.root.colour = BLACK
            yield Step("recolour", "The root is always painted black, which is free "
                                   "because it lengthens every path equally.", {})

    # Deleting

    def delete(self, value: Any) -> bool:
        return run(self.delete_traced(value))

    def delete_traced(self, value: Any) -> Traced[bool]:
        """Remove a value and repair the rules.

        The structure of the deletion is the same three cases as an ordinary
        search tree. What is new is the colour bookkeeping:

        **Removing a red node is free.** Reds do not count towards the black
        height and cannot have been holding rule 4 together on their own.

        **Removing a black node breaks rule 5** on every path that went through
        it, and that is what the fixup below repairs.
        """
        node = self.root
        while node is not self.nil and node.value != value:
            node = node.left if value < node.value else node.right

        if node is self.nil:
            yield Step("missing", f"{value!r} is not in the tree.", {"value": value})
            return False

        removed_colour = node.colour
        if node.left is self.nil:
            replacement = node.right
            self._transplant(node, node.right)
        elif node.right is self.nil:
            replacement = node.left
            self._transplant(node, node.left)
        else:
            successor = node.right
            while successor.left is not self.nil:
                successor = successor.left

            removed_colour = successor.colour
            replacement = successor.right

            yield Step(
                "replace",
                f"{value!r} has two children, so its inorder successor "
                f"{successor.value!r} takes its place.",
                {"removed": value, "successor": successor.value},
            )

            if successor.parent is node:
                replacement.parent = successor
            else:
                self._transplant(successor, successor.right)
                successor.right = node.right
                successor.right.parent = successor

            self._transplant(node, successor)
            successor.left = node.left
            successor.left.parent = successor
            # The successor inherits the colour of the node it replaces, so the
            # tree's colouring is unchanged at that position. What actually left
            # the tree, colour wise, is the successor's own original colour.
            successor.colour = node.colour

        self._size -= 1
        yield Step("remove", f"Removed {value!r}, which was {removed_colour}.",
                   {"value": value, "colour": removed_colour})

        if removed_colour == BLACK:
            yield from self._repair_after_delete(replacement)

        verify_if_checking(self)
        return True

    def _transplant(self, target: RBNode, replacement: RBNode) -> None:
        """Put `replacement` into the position currently held by `target`."""
        if target.parent is self.nil:
            self.root = replacement
        elif target is target.parent.left:
            target.parent.left = replacement
        else:
            target.parent.right = replacement
        replacement.parent = target.parent

    def _repair_after_delete(self, node: RBNode) -> Traced[None]:
        """Restore rule 5 after a black node was removed.

        The idea that makes this tractable is to imagine the node now sitting in
        the vacated position as carrying an **extra black**, a debt of one black
        node that its paths are short by. The job is to discharge that debt, and
        there are four ways, depending on the sibling:

        1. **Red sibling.** Not a solution by itself. Rotate to turn it into one
           of the other three cases, which have a black sibling.
        2. **Black sibling with two black children.** Paint the sibling red. Now
           both sides of the parent are short by one, which is consistent, so the
           debt moves up to the parent and the loop repeats. This is the only
           case that iterates, and it is why deletion is O(log n) overall while
           still using at most three rotations.
        3. **Black sibling whose far child is black and near child is red.**
           Rotate the sibling to turn it into case 4.
        4. **Black sibling with a red far child.** One rotation at the parent
           discharges the debt completely, and the loop ends.

        Deletion is harder than insertion because a missing black is a global
        problem, affecting every path through that position, while a red red
        clash is local. Cases 1 and 3 exist purely to funnel every situation into
        cases 2 and 4, which are the ones that actually make progress.
        """
        while node is not self.root and not node.is_red:
            if node is node.parent.left:
                sibling = node.parent.right

                if sibling.is_red:
                    sibling.colour = BLACK
                    node.parent.colour = RED
                    yield Step("rotate", "Case 1: the sibling is red, so rotate to get a "
                                         "black sibling and one of the other cases.",
                               {"case": 1})
                    self._rotate_left(node.parent)
                    sibling = node.parent.right

                if not sibling.left.is_red and not sibling.right.is_red:
                    sibling.colour = RED
                    self.recolourings += 1
                    yield Step("recolour", "Case 2: the sibling's children are both black, "
                                           "so painting the sibling red evens the two sides "
                                           "and moves the debt up to the parent.",
                               {"case": 2})
                    node = node.parent
                    continue

                if not sibling.right.is_red:
                    sibling.left.colour = BLACK
                    sibling.colour = RED
                    yield Step("rotate", "Case 3: the sibling's far child is black, so "
                                         "rotate the sibling to turn this into case 4.",
                               {"case": 3})
                    self._rotate_right(sibling)
                    sibling = node.parent.right

                sibling.colour = node.parent.colour
                node.parent.colour = BLACK
                sibling.right.colour = BLACK
                yield Step("rotate", "Case 4: the sibling has a red far child, so one "
                                     "rotation at the parent clears the debt for good.",
                           {"case": 4})
                self._rotate_left(node.parent)
                node = self.root

            else:
                # Mirror image of everything above.
                sibling = node.parent.left

                if sibling.is_red:
                    sibling.colour = BLACK
                    node.parent.colour = RED
                    yield Step("rotate", "Case 1 mirrored: red sibling, rotate first.",
                               {"case": 1})
                    self._rotate_right(node.parent)
                    sibling = node.parent.left

                if not sibling.right.is_red and not sibling.left.is_red:
                    sibling.colour = RED
                    self.recolourings += 1
                    yield Step("recolour", "Case 2 mirrored: the debt moves up to the parent.",
                               {"case": 2})
                    node = node.parent
                    continue

                if not sibling.left.is_red:
                    sibling.right.colour = BLACK
                    sibling.colour = RED
                    yield Step("rotate", "Case 3 mirrored: rotate the sibling first.",
                               {"case": 3})
                    self._rotate_left(sibling)
                    sibling = node.parent.left

                sibling.colour = node.parent.colour
                node.parent.colour = BLACK
                sibling.left.colour = BLACK
                yield Step("rotate", "Case 4 mirrored: one rotation clears the debt.",
                           {"case": 4})
                self._rotate_right(node.parent)
                node = self.root

        # Whatever position the debt ended on, painting it black discharges it.
        node.colour = BLACK

    # Reading

    def contains(self, value: Any) -> bool:
        node = self.root
        while node is not self.nil:
            if value == node.value:
                return True
            node = node.left if value < node.value else node.right
        return False

    def __contains__(self, value: Any) -> bool:
        return self.contains(value)

    def minimum(self) -> Any:
        if self.root is self.nil:
            raise ValueError("an empty tree has no minimum")
        node = self.root
        while node.left is not self.nil:
            node = node.left
        return node.value

    def maximum(self) -> Any:
        if self.root is self.nil:
            raise ValueError("an empty tree has no maximum")
        node = self.root
        while node.right is not self.nil:
            node = node.right
        return node.value

    def inorder(self) -> list[Any]:
        values: list[Any] = []

        def walk(node: RBNode) -> None:
            if node is self.nil:
                return
            walk(node.left)
            values.append(node.value)
            walk(node.right)

        walk(self.root)
        return values

    def height(self) -> int:
        def measure(node: RBNode) -> int:
            if node is self.nil:
                return -1
            return 1 + max(measure(node.left), measure(node.right))

        return measure(self.root)

    def black_height(self) -> int:
        """Black nodes on any root to leaf path, not counting the root itself.

        Rule 5 says every path gives the same answer, so walking down the left
        spine is enough. If the rules are broken this will disagree with other
        paths, which is exactly what the invariant check tests for.
        """
        count = 0
        node = self.root
        while node is not self.nil:
            if not node.is_red:
                count += 1
            node = node.left
        return count

    def check_invariants(self) -> list[Violation]:
        """All five rules, checked by name so a failure says which one broke.

        This is what makes the app's rule display possible: each rule is a
        separate named check rather than one big "is it valid" boolean.
        """
        violations: list[Violation] = []

        if self.root is not self.nil and self.root.is_red:
            violations.append(Violation("rule 2: the root is black",
                                        f"the root {self.root.value!r} is red"))

        if self.nil.is_red:
            violations.append(Violation("rule 3: every empty position counts as black",
                                        "the sentinel has been painted red"))

        def walk(node: RBNode, low: Any, high: Any) -> tuple[int, int]:
            """Returns (node count, black height) for this subtree."""
            if node is self.nil:
                return 0, 1

            if node.colour not in (RED, BLACK):
                violations.append(Violation("rule 1: every node is red or black",
                                            f"{node.value!r} is {node.colour!r}"))

            if low is not None and node.value <= low:
                violations.append(Violation("search order holds all the way down",
                                            f"{node.value!r} should be above {low!r}"))
            if high is not None and node.value >= high:
                violations.append(Violation("search order holds all the way down",
                                            f"{node.value!r} should be below {high!r}"))

            if node.is_red and (node.left.is_red or node.right.is_red):
                violations.append(Violation(
                    "rule 4: a red node never has a red child",
                    f"red node {node.value!r} has a red child",
                ))

            for child, side in ((node.left, "left"), (node.right, "right")):
                if child is not self.nil and child.parent is not node:
                    violations.append(Violation(
                        "every node's parent pointer is correct",
                        f"the {side} child of {node.value!r} points elsewhere",
                    ))

            left_count, left_black = walk(node.left, low, node.value)
            right_count, right_black = walk(node.right, node.value, high)

            if left_black != right_black:
                violations.append(Violation(
                    "rule 5: every path has the same number of black nodes",
                    f"below {node.value!r} the left side has {left_black} black nodes "
                    f"and the right side has {right_black}",
                ))

            return 1 + left_count + right_count, left_black + (0 if node.is_red else 1)

        counted, _ = walk(self.root, None, None)
        if counted != self._size:
            violations.append(Violation(
                "the recorded size matches the tree",
                f"the tree says {self._size} nodes but {counted} were found",
            ))

        return violations
