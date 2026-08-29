"""LRU cache: the classic example of two structures solving one problem together.

A cache with a size limit needs to decide what to throw away when it is full.
Least recently used is the usual answer: evict whatever has gone longest without
being touched, on the theory that what you used recently you will use again.

The requirement is that **both** operations are O(1):

* `get(key)`: return the value and mark it as most recently used.
* `put(key, value)`: store it, and if that exceeds the limit, evict the least
  recently used entry.

Neither structure can do this alone, and seeing why is the whole lesson:

* A **hash map** finds a key in O(1) but knows nothing about recency ordering.
* A **doubly linked list** maintains a recency order in O(1) at both ends, but
  finding a particular key in it is O(n).

Put them together and each covers the other's gap. The list holds the entries in
recency order, most recent at the front. The hash map maps each key to **the list
node itself**, not to the value. So a lookup finds the node in O(1) and, because
the list is doubly linked, that node can be unlinked and moved to the front in
O(1) as well.

That last point is exactly why day 5 made a fuss about the doubly linked list
being able to unlink a node you already hold. This is the structure that pays for
that extra pointer.
"""

from __future__ import annotations

from typing import Any

from dsalab.invariants import Violation, verify_if_checking
from dsalab.structures.linked_list import DoublyLinkedList, DoublyNode
from dsalab.tracing import Step, Traced, run


class LRUCache:
    """A fixed size cache that evicts the least recently used entry.

    | Operation | Cost |
    | - | - |
    | get | O(1) |
    | put | O(1) |
    | eviction | O(1) |
    | memory | O(capacity) |

    Statistics are kept for `hits`, `misses` and `evictions`, because the number
    that actually matters for a cache is the hit rate, and a cache you cannot
    measure is a cache you cannot tune.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("a cache needs room for at least one entry")

        self.capacity = capacity
        self._order = DoublyLinkedList()  # most recently used at the front
        self._nodes: dict[Any, DoublyNode] = {}
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, key: Any) -> bool:
        """Membership testing deliberately does not count as a use.

        Asking whether something is cached is not the same as using it, so this
        does not update the recency order or the hit counters. Getting that wrong
        makes the statistics meaningless, because every containment check would
        look like a hit.
        """
        return key in self._nodes

    @property
    def hit_rate(self) -> float:
        """Hits divided by total lookups, or 0 when nothing has been looked up."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def get(self, key: Any, default: Any = None) -> Any:
        return run(self.get_traced(key, default))

    def get_traced(self, key: Any, default: Any = None) -> Traced[Any]:
        """Fetch a value and mark it as most recently used. O(1)."""
        node = self._nodes.get(key)

        if node is None:
            self.misses += 1
            yield Step(
                "miss",
                f"{key!r} is not cached. Hit rate is now {self.hit_rate:.0%}.",
                {"key": key, "hits": self.hits, "misses": self.misses},
            )
            return default

        self.hits += 1
        self._order.move_to_front(node)
        yield Step(
            "hit",
            f"{key!r} found and moved to the front, so it is now the most recently "
            f"used. Hit rate is {self.hit_rate:.0%}.",
            {"key": key, "order": self._keys_in_order()},
        )
        return node.value[1]

    def put(self, key: Any, value: Any) -> None:
        run(self.put_traced(key, value))

    def put_traced(self, key: Any, value: Any) -> Traced[None]:
        """Store a value, evicting the least recently used entry if full. O(1)."""
        existing = self._nodes.get(key)

        if existing is not None:
            existing.value = (key, value)
            self._order.move_to_front(existing)
            yield Step(
                "update",
                f"{key!r} was already cached, so its value was replaced and it moved "
                "to the front.",
                {"key": key, "order": self._keys_in_order()},
            )
            verify_if_checking(self)
            return

        if len(self._nodes) >= self.capacity:
            # The tail is the least recently used entry, and reaching it is O(1)
            # only because the list is doubly linked and keeps a tail pointer.
            victim_node = self._order.tail
            victim_key = victim_node.value[0]
            self._order.unlink(victim_node)
            del self._nodes[victim_key]
            self.evictions += 1
            yield Step(
                "evict",
                f"The cache was full, so {victim_key!r} was evicted for having gone "
                "longest without being used.",
                {"evicted": victim_key, "evictions": self.evictions},
            )

        self._nodes[key] = self._order.push_front((key, value))
        yield Step(
            "insert",
            f"{key!r} cached at the front. The cache now holds {len(self._nodes)} of "
            f"{self.capacity} entries.",
            {"key": key, "order": self._keys_in_order()},
        )
        verify_if_checking(self)

    def delete(self, key: Any) -> bool:
        """Remove an entry outright. Returns whether it was there."""
        node = self._nodes.pop(key, None)
        if node is None:
            return False
        self._order.unlink(node)
        verify_if_checking(self)
        return True

    def clear(self) -> None:
        """Empty the cache, keeping the statistics, which is usually what you want."""
        self._order = DoublyLinkedList()
        self._nodes = {}

    def _keys_in_order(self) -> list[Any]:
        """Keys from most to least recently used."""
        return [key for key, _ in self._order]

    def keys_by_recency(self) -> list[Any]:
        """Most recently used first, least recently used last.

        This is what the visualiser draws, and it is also how the tests check that
        the recency order is what it claims to be rather than merely plausible.
        """
        return self._keys_in_order()

    def __repr__(self) -> str:
        return f"LRUCache({len(self._nodes)}/{self.capacity}, {self._keys_in_order()!r})"

    def check_invariants(self) -> list[Violation]:
        """The rules that keep the two structures agreeing with each other.

        A cache built from two structures has one characteristic failure: they
        drift apart. The map still holds a key whose node was unlinked, or the
        list holds a node the map has forgotten. Either way lookups keep working
        for a while and then quietly return stale values or leak memory.
        """
        violations: list[Violation] = []
        ordered = self._keys_in_order()

        if len(ordered) != len(self._nodes):
            violations.append(Violation(
                "the list and the map hold the same entries",
                f"the list has {len(ordered)} nodes but the map has {len(self._nodes)} keys",
            ))

        if len(self._nodes) > self.capacity:
            violations.append(Violation(
                "the cache never holds more than its capacity",
                f"{len(self._nodes)} entries with a capacity of {self.capacity}",
            ))

        for key in ordered:
            if key not in self._nodes:
                violations.append(Violation(
                    "every node in the list is findable in the map",
                    f"{key!r} is in the recency list but not in the map",
                ))

        for key, node in self._nodes.items():
            if node.value[0] != key:
                violations.append(Violation(
                    "every map entry points at the node holding that key",
                    f"the map sends {key!r} to a node holding {node.value[0]!r}",
                ))

        if len(set(map(repr, ordered))) != len(ordered):
            violations.append(Violation(
                "no key appears twice in the recency list",
                "a key was found more than once, which means a node was not unlinked",
            ))

        return violations
