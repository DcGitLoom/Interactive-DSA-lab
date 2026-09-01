"""Union find: keeping track of which things are connected.

The problem: you have n items and a stream of statements like "a and b are in the
same group". You need to answer "are x and y in the same group" at any point.

This is not as narrow as it sounds. It is Kruskal's algorithm asking whether an
edge would create a cycle, it is detecting cycles in an undirected graph, it is
percolation, image segmentation, and it is how a compiler tracks which type
variables have been unified.

The obvious approach is to keep a list per group and merge lists when groups
join, which makes merging O(n). Union find instead keeps each group as a **tree**
where every node points at its parent and the root names the group. Merging is
then a single pointer assignment.

Two optimisations turn that from good to essentially free, and both are one line:

* **Union by rank or size**: when merging, hang the smaller tree under the larger
  one. Without it, merging in the wrong order builds a chain of length n and every
  lookup walks it.
* **Path compression**: after finding a root, point every node passed directly at
  it, so the next lookup is one hop.

Together they give an amortised cost of O(alpha(n)) per operation, where alpha is
the inverse Ackermann function from day 2. **That value is at most 4 for any n
that fits in the universe**, so in practice each operation is constant time. It is
one of the most striking results in the subject: the analysis is famously hard and
the code is fifteen lines.
"""

from __future__ import annotations

from dsalab.invariants import Violation, verify_if_checking
from dsalab.tracing import Step, Traced, run


class UnionFind:
    """Disjoint sets with union by size and path compression.

    | Operation | Cost |
    | - | - |
    | find the group of an item | O(alpha(n)) amortised, so effectively O(1) |
    | union two groups | O(alpha(n)) amortised |
    | connected | O(alpha(n)) amortised |
    | size of a group | O(alpha(n)) amortised |
    | memory | 2n integers |

    Union by **size** is used rather than by rank. They give the same asymptotic
    guarantee, and size is more useful: it answers "how big is this group" for
    free, which is a question people actually ask, while rank is an internal
    number that means nothing outside the algorithm.
    """

    def __init__(self, count: int) -> None:
        if count < 0:
            raise ValueError("cannot have a negative number of items")

        # Every item starts as its own parent, meaning n groups of one.
        self._parent = list(range(count))
        self._size = [1] * count
        self._groups = count
        self.path_hops = 0  # counted so the tests can show compression working

    def __len__(self) -> int:
        """How many items there are, not how many groups."""
        return len(self._parent)

    @property
    def groups(self) -> int:
        """How many separate groups remain.

        Maintained as a counter rather than recomputed, because counting distinct
        roots is O(n) and this is a question asked constantly, for example by
        Kruskal's algorithm to know when the spanning tree is complete.
        """
        return self._groups

    def find(self, item: int) -> int:
        return run(self.find_traced(item))

    def find_traced(self, item: int) -> Traced[int]:
        """The root naming this item's group, compressing the path on the way.

        Written as a loop with a second pass rather than recursively. The
        recursive version is shorter and elegant, but it is O(depth) stack frames,
        and before compression has done its work the depth can be large. This is
        the same reasoning as day 9 using an iterative traversal for a structure
        whose failure mode is depth.
        """
        if not 0 <= item < len(self._parent):
            raise IndexError(f"{item} is not one of the {len(self._parent)} items")

        root = item
        while self._parent[root] != root:
            root = self._parent[root]
            self.path_hops += 1

        # Second pass: point everything on the path straight at the root, so this
        # walk never has to happen again.
        current = item
        compressed = 0
        while self._parent[current] != root:
            parent = self._parent[current]
            self._parent[current] = root
            current = parent
            compressed += 1

        if compressed:
            yield Step(
                "compress",
                f"Found the root {root} for {item}, and pointed {compressed} node(s) on the "
                "path straight at it, so the next lookup is one hop.",
                {"item": item, "root": root, "flattened": compressed},
            )
        else:
            yield Step(
                "find",
                f"{item} belongs to the group named by {root}.",
                {"item": item, "root": root},
            )

        return root

    def union(self, first: int, second: int) -> bool:
        return run(self.union_traced(first, second))

    def union_traced(self, first: int, second: int) -> Traced[bool]:
        """Merge two groups. Returns False if they were already the same group.

        That return value is what makes Kruskal's algorithm work on day 18: an
        edge is safe to add exactly when this returns True, because a False means
        both ends were already connected and the edge would close a cycle.
        """
        left = yield from self.find_traced(first)
        right = yield from self.find_traced(second)

        if left == right:
            yield Step(
                "already",
                f"{first} and {second} are already in the same group, so nothing changes. "
                "In Kruskal's algorithm this is exactly the test for a cycle.",
                {"first": first, "second": second},
            )
            return False

        # Hang the smaller tree under the larger one. Skipping this is the
        # difference between constant time and a linked list.
        if self._size[left] < self._size[right]:
            left, right = right, left

        self._parent[right] = left
        self._size[left] += self._size[right]
        self._groups -= 1

        yield Step(
            "union",
            f"Merged the group of {second} into the group of {first}. The smaller tree "
            f"hangs under the larger, so the result is {self._size[left]} item(s) deep by "
            f"at most log of that. {self._groups} group(s) remain.",
            {"root": left, "absorbed": right, "size": self._size[left],
             "groups": self._groups},
        )
        verify_if_checking(self)
        return True

    def connected(self, first: int, second: int) -> bool:
        """Whether two items are in the same group."""
        return self.find(first) == self.find(second)

    def group_size(self, item: int) -> int:
        """How many items are in this item's group."""
        return self._size[self.find(item)]

    def members(self) -> dict[int, list[int]]:
        """Every group, keyed by its root. O(n), for display and for tests."""
        collected: dict[int, list[int]] = {}
        for item in range(len(self._parent)):
            collected.setdefault(self.find(item), []).append(item)
        return collected

    def max_depth(self) -> int:
        """The deepest chain of parent pointers, before any compression.

        Only used by the tests, to show that union by size keeps the trees flat.
        Not something a caller would need.
        """
        deepest = 0
        for item in range(len(self._parent)):
            depth = 0
            current = item
            while self._parent[current] != current:
                current = self._parent[current]
                depth += 1
            deepest = max(deepest, depth)
        return deepest

    def __repr__(self) -> str:
        return f"UnionFind({len(self._parent)} items in {self._groups} group(s))"

    def check_invariants(self) -> list[Violation]:
        violations: list[Violation] = []
        roots = set()

        for item in range(len(self._parent)):
            # Follow the parents, watching for a cycle that is not a self loop.
            seen = set()
            current = item
            while self._parent[current] != current:
                if current in seen:
                    violations.append(Violation(
                        "the parent pointers form trees with no cycles",
                        f"following the parents from {item} goes round in a circle",
                    ))
                    break
                seen.add(current)
                current = self._parent[current]
            else:
                roots.add(current)

        if len(roots) != self._groups:
            violations.append(Violation(
                "the group count matches the number of roots",
                f"the structure says {self._groups} groups but {len(roots)} roots were found",
            ))

        for root in roots:
            actual = sum(1 for item in range(len(self._parent)) if self.find(item) == root)
            if self._size[root] != actual:
                violations.append(Violation(
                    "each root records the true size of its group",
                    f"root {root} claims {self._size[root]} members but has {actual}",
                ))

        return violations


class UnionFindWithoutOptimisations:
    """The naive version, kept deliberately, so the optimisations can be measured.

    No union by size, no path compression: just point one root at the other. It is
    correct and it is quadratic in the worst case, which the tests demonstrate by
    building a chain and measuring the depth against the optimised version on the
    same input.

    Keeping a bad implementation around to measure against is worth doing. It is
    much more convincing than a claim in a comment, and it makes the benchmark
    honest.
    """

    def __init__(self, count: int) -> None:
        self._parent = list(range(count))
        self.path_hops = 0

    def find(self, item: int) -> int:
        while self._parent[item] != item:
            item = self._parent[item]
            self.path_hops += 1
        return item

    def union(self, first: int, second: int) -> bool:
        left, right = self.find(first), self.find(second)
        if left == right:
            return False
        self._parent[right] = left  # no thought about which tree is bigger
        return True

    def connected(self, first: int, second: int) -> bool:
        return self.find(first) == self.find(second)

    def max_depth(self) -> int:
        deepest = 0
        for item in range(len(self._parent)):
            depth = 0
            current = item
            while self._parent[current] != current:
                current = self._parent[current]
                depth += 1
            deepest = max(deepest, depth)
        return deepest
