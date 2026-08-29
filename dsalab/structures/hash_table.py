"""Hash tables: turning a key straight into an address.

Every search structure so far has *compared* its way to an answer. A hash table
does not compare at all. It computes an array index directly from the key:

    index = hash(key) % number_of_buckets

One arithmetic operation, then one array access. Not O(log n) comparisons.
**No comparisons.** That is why a hash table beats every tree on plain lookup, and
it is worth being clear that O(1) here is average, not guaranteed. Two keys can
land in the same bucket, and everything interesting about this structure is about
what happens then.

Two families of answers, and both are implemented here because the trade between
them is genuinely close:

* **Separate chaining.** Each bucket holds a small list of everything that landed
  there. Simple, tolerant of a full table, and deletion is trivial. Costs a
  pointer per entry and scatters memory.
* **Open addressing.** Everything lives in the array itself. A collision sends the
  key to another slot by a fixed probing rule. No extra allocation, excellent
  cache behaviour, and deletion becomes surprisingly awkward.

What a hash table gives up compared with a tree is **order**. There is no
minimum, no floor, no ceiling, no range query and no sorted iteration, because
hashing deliberately scatters keys. Day 9's `floor` and `range_query` cannot be
written here at any price short of looking at every key.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from dsalab.invariants import Violation, verify_if_checking
from dsalab.tracing import Step, Traced, run

# Grow when the table is this full. The reasoning is in docs/14-hashing.md, and
# it is the single most argued about number in this project.
DEFAULT_MAX_LOAD = 0.75

# Open addressing degrades much faster than chaining as the table fills, so it
# gets a lower threshold. Explained in the OpenAddressingTable docstring.
OPEN_ADDRESSING_MAX_LOAD = 0.5

INITIAL_BUCKETS = 8


class _Missing:
    """A sentinel for "no value", so that None can be stored as a real value."""

    def __repr__(self) -> str:
        return "<missing>"


MISSING = _Missing()


class ChainedHashTable:
    """A hash table where each bucket holds a list of the entries that landed there.

    | Operation | Average | Worst case |
    | - | - | - |
    | get, put, delete | O(1) | O(n), when every key collides |
    | iterate everything | O(n + buckets) | same |
    | memory | O(n + buckets) | plus one list per occupied bucket |

    The worst case is not hypothetical. It is what an attacker produces on purpose
    by sending keys chosen to collide, which turns every request into a linear
    scan. That attack is why Python randomises its string hashing per process, and
    it is worth knowing about before you build a web service on top of a hash map.
    """

    def __init__(self, buckets: int = INITIAL_BUCKETS, max_load: float = DEFAULT_MAX_LOAD) -> None:
        if buckets < 1:
            raise ValueError("a hash table needs at least one bucket")
        if not 0 < max_load <= 1:
            raise ValueError("the load factor threshold must be between 0 and 1")

        self._buckets: list[list[tuple[Any, Any]]] = [[] for _ in range(buckets)]
        self._size = 0
        self._max_load = max_load
        self.collisions = 0
        self.resizes = 0

    def __len__(self) -> int:
        return self._size

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)

    @property
    def load_factor(self) -> float:
        """Entries divided by buckets. This is the number that decides everything.

        For chaining it is the **average chain length**, so a load factor of 0.75
        means the average lookup walks about three quarters of one entry. It can
        legitimately exceed 1: a table with twice as many entries as buckets still
        works, it just gets slower in proportion.
        """
        return self._size / len(self._buckets)

    def _index(self, key: Any) -> int:
        """Which bucket a key belongs in.

        Python's `hash` can be negative, and `%` on a negative number in Python
        already returns a non negative result, so no extra masking is needed. In
        C or Java it would be, and forgetting it there gives a negative index.
        """
        return hash(key) % len(self._buckets)

    def put(self, key: Any, value: Any) -> None:
        run(self.put_traced(key, value))

    def put_traced(self, key: Any, value: Any) -> Traced[None]:
        """Insert or update. O(1) average."""
        index = self._index(key)
        bucket = self._buckets[index]

        for position, (existing_key, _) in enumerate(bucket):
            if existing_key == key:
                bucket[position] = (key, value)
                yield Step(
                    "update",
                    f"{key!r} was already in bucket {index}, so its value was replaced.",
                    {"bucket": index, "key": key},
                )
                return

        if bucket:
            self.collisions += 1
            yield Step(
                "collision",
                f"Bucket {index} already holds {len(bucket)} entr(ies), so {key!r} joins "
                "the chain there. This is a collision, not an error.",
                {"bucket": index, "key": key, "chain_length": len(bucket) + 1},
            )

        bucket.append((key, value))
        self._size += 1
        yield Step(
            "insert",
            f"{key!r} stored in bucket {index}. Load factor is now "
            f"{self.load_factor:.2f}.",
            {"bucket": index, "key": key, "load": self.load_factor},
        )

        if self.load_factor > self._max_load:
            yield from self._grow()

        verify_if_checking(self)

    def _grow(self) -> Traced[None]:
        """Double the bucket count and rehash everything. O(n).

        Every key has to be rehashed, not merely copied, because the bucket index
        depends on the number of buckets. This is the same amortised argument as
        the dynamic array on day 4: an occasional O(n) rebuild, paid for by the
        many O(1) insertions between them.
        """
        old = self._buckets
        self._buckets = [[] for _ in range(len(old) * 2)]
        moved = self._size
        self._size = 0
        self.resizes += 1

        for bucket in old:
            for key, value in bucket:
                index = self._index(key)
                self._buckets[index].append((key, value))
                self._size += 1

        yield Step(
            "resize",
            f"Load factor passed {self._max_load}, so the table doubled to "
            f"{len(self._buckets)} buckets and all {moved} keys were rehashed. "
            "Rehashing is required, not optional, because the index depends on the "
            "bucket count.",
            {"buckets": len(self._buckets), "rehashed": moved},
        )

    def get(self, key: Any, default: Any = MISSING) -> Any:
        return run(self.get_traced(key, default))

    def get_traced(self, key: Any, default: Any = MISSING) -> Traced[Any]:
        """Look up a key. O(1) average, O(chain length) in a bad bucket."""
        index = self._index(key)
        bucket = self._buckets[index]

        for position, (existing_key, value) in enumerate(bucket):
            yield Step(
                "probe",
                f"Checking entry {position} of bucket {index}: is it {key!r}? "
                f"{'Yes.' if existing_key == key else 'No, keep walking the chain.'}",
                {"bucket": index, "position": position, "match": existing_key == key},
            )
            if existing_key == key:
                return value

        if default is MISSING:
            raise KeyError(key)
        return default

    def delete(self, key: Any) -> bool:
        """Remove a key. Returns whether it was there. Trivial with chaining."""
        index = self._index(key)
        bucket = self._buckets[index]

        for position, (existing_key, _) in enumerate(bucket):
            if existing_key == key:
                bucket.pop(position)
                self._size -= 1
                verify_if_checking(self)
                return True
        return False

    def __contains__(self, key: Any) -> bool:
        return self.get(key, None) is not None or any(
            existing == key for existing, _ in self._buckets[self._index(key)]
        )

    def __getitem__(self, key: Any) -> Any:
        return self.get(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        self.put(key, value)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.keys())

    def keys(self) -> list[Any]:
        """Every key, in bucket order, which is not a meaningful order.

        Worth stating plainly: the sequence looks arbitrary and it changes when
        the table resizes. Relying on it is a bug waiting to happen, and it is
        exactly what a tree would give you properly.
        """
        return [key for bucket in self._buckets for key, _ in bucket]

    def items(self) -> list[tuple[Any, Any]]:
        return [entry for bucket in self._buckets for entry in bucket]

    def chain_lengths(self) -> list[int]:
        """How long each bucket's chain is, for the visualiser and the tests."""
        return [len(bucket) for bucket in self._buckets]

    def longest_chain(self) -> int:
        return max(self.chain_lengths(), default=0)

    def __repr__(self) -> str:
        return f"ChainedHashTable({self._size} entries in {len(self._buckets)} buckets)"

    def check_invariants(self) -> list[Violation]:
        violations: list[Violation] = []
        counted = 0

        for index, bucket in enumerate(self._buckets):
            counted += len(bucket)
            for key, _ in bucket:
                if self._index(key) != index:
                    violations.append(Violation(
                        "every key sits in the bucket its hash chooses",
                        f"{key!r} is in bucket {index} but hashes to {self._index(key)}",
                    ))
            seen = [key for key, _ in bucket]
            if len(seen) != len(set(map(repr, seen))):
                violations.append(Violation(
                    "no key appears twice",
                    f"bucket {index} holds a duplicate key",
                ))

        if counted != self._size:
            violations.append(Violation(
                "the recorded size matches the contents",
                f"the table says {self._size} entries but {counted} were found",
            ))

        return violations


class OpenAddressingTable:
    """A hash table that stores everything in the array itself.

    When a slot is taken, a **probe sequence** decides where to look next. Three
    are implemented, because the differences are real and easy to demonstrate:

    * **Linear probing**: try index+1, index+2, and so on. The best cache
      behaviour of the three, since it walks straight through memory. Its weakness
      is **primary clustering**: any run of occupied slots grows at both ends,
      because every key landing anywhere in the run has to walk past all of it,
      and long runs then attract more keys. Clusters merge into longer clusters.
    * **Quadratic probing**: try index+1, index+4, index+9. Jumping further breaks
      up primary clusters, at the cost of worse cache behaviour and of **secondary
      clustering**: keys hashing to the same starting slot still follow the exact
      same path. It also needs care to visit every slot, which is why the table
      size is kept a power of two and the probe uses the triangular numbers, a
      combination that provably reaches every slot.
    * **Double hashing**: the step size itself comes from a second hash of the
      key, so two keys sharing a start almost never share a path. Best distribution
      of the three, worst cache behaviour, and two hash computations per lookup.

    The load factor threshold is 0.5 here, against 0.75 for chaining, and that is
    not timidity. With chaining, a load factor of 1 means an average chain of one.
    With open addressing, the expected number of probes for a failed linear probe
    lookup is roughly (1 + 1/(1-load)^2)/2, which is 2.5 probes at load 0.5, 8.5
    at 0.75, and 50.5 at 0.9. It does not degrade gracefully, it falls off a cliff,
    and there is a test that measures exactly this curve.
    """

    LINEAR = "linear"
    QUADRATIC = "quadratic"
    DOUBLE = "double"

    # A slot that once held a key that has since been deleted. See `delete`.
    TOMBSTONE = object()
    EMPTY = object()

    def __init__(
        self,
        buckets: int = INITIAL_BUCKETS,
        probing: str = LINEAR,
        max_load: float = OPEN_ADDRESSING_MAX_LOAD,
    ) -> None:
        if probing not in (self.LINEAR, self.QUADRATIC, self.DOUBLE):
            raise ValueError(f"unknown probing strategy {probing!r}")
        if not 0 < max_load < 1:
            raise ValueError("open addressing needs a load factor threshold below 1")

        # A power of two size is what makes the quadratic probe sequence reach
        # every slot, so it is enforced rather than assumed.
        size = 1
        while size < buckets:
            size *= 2

        self._keys: list[Any] = [self.EMPTY] * size
        self._values: list[Any] = [None] * size
        self._size = 0
        self._tombstones = 0
        self._probing = probing
        self._max_load = max_load
        self.probes = 0
        self.resizes = 0

    def __len__(self) -> int:
        return self._size

    @property
    def bucket_count(self) -> int:
        return len(self._keys)

    @property
    def load_factor(self) -> float:
        """Occupied and tombstoned slots divided by total slots.

        Tombstones count, and that matters: they still have to be probed past, so
        a table full of them is slow even though it holds nothing. Ignoring them
        here is a classic way to build a table that mysteriously grinds to a halt
        under a heavy delete workload.
        """
        return (self._size + self._tombstones) / len(self._keys)

    def _probe_sequence(self, key: Any) -> Iterator[int]:
        """Every slot this key will try, in order."""
        size = len(self._keys)
        start = hash(key) % size

        if self._probing == self.LINEAR:
            for step in range(size):
                yield (start + step) % size

        elif self._probing == self.QUADRATIC:
            # Triangular numbers, step*(step+1)/2, are guaranteed to visit every
            # slot of a power of two sized table. The more obvious step^2 does
            # not, and can loop forever on a table that still has free slots.
            for step in range(size):
                yield (start + step * (step + 1) // 2) % size

        else:
            # The second hash must never be 0, or the step is zero and the probe
            # sits on one slot forever. Forcing it odd also guarantees it is
            # coprime with a power of two size, so the sequence reaches every slot.
            step_size = (hash(str(key)) % size) | 1
            for step in range(size):
                yield (start + step * step_size) % size

    def put(self, key: Any, value: Any) -> None:
        run(self.put_traced(key, value))

    def put_traced(self, key: Any, value: Any) -> Traced[None]:
        """Insert or update. O(1) average while the table is not too full."""
        if self.load_factor >= self._max_load:
            yield from self._rebuild()

        first_tombstone = None

        for attempt, index in enumerate(self._probe_sequence(key)):
            self.probes += 1
            slot = self._keys[index]

            if slot is self.EMPTY:
                # A tombstone seen earlier is a better home than this empty slot,
                # because reusing it keeps probe sequences shorter.
                target = first_tombstone if first_tombstone is not None else index
                if first_tombstone is not None:
                    self._tombstones -= 1
                self._keys[target] = key
                self._values[target] = value
                self._size += 1
                yield Step(
                    "insert",
                    f"{key!r} stored in slot {target} after {attempt + 1} probe(s).",
                    {"slot": target, "probes": attempt + 1, "key": key},
                )
                verify_if_checking(self)
                return

            if slot is self.TOMBSTONE:
                if first_tombstone is None:
                    first_tombstone = index
                yield Step(
                    "tombstone",
                    f"Slot {index} is a tombstone, left by a deletion. The search must "
                    "carry on past it, but this slot can be reused.",
                    {"slot": index},
                )
                continue

            if slot == key:
                self._values[index] = value
                yield Step("update", f"{key!r} found in slot {index}, value replaced.",
                           {"slot": index, "key": key})
                return

            yield Step(
                "collision",
                f"Slot {index} is taken by {slot!r}, so the probe moves on.",
                {"slot": index, "occupant": slot, "probe": attempt + 1},
            )

        raise RuntimeError("the probe sequence failed to find a free slot, which should "
                           "be impossible while the load factor is below 1")

    def _rebuild(self) -> Traced[None]:
        """Reinsert everything into a fresh table, dropping the tombstones.

        Whether the new table is bigger is a decision worth making carefully,
        because the load factor counts tombstones and there are two very different
        reasons it can be high:

        * **Genuinely full.** Real entries have filled the table, so it must grow.
        * **Full of graves.** A table that has been written and deleted repeatedly
          can hold five entries and eleven tombstones. Doubling here would be
          absurd: the table is nearly empty, it is just clogged.

        The test used here is simply whether there are at least as many tombstones
        as live entries. If there are, the table is clogged rather than full, and
        it is rebuilt at the same size, which sweeps the graves away and costs no
        extra memory. If there are not, the pressure is real and the table doubles.

        My first attempt compared live entries against half the load threshold,
        which is a fussier rule that happens to sit exactly on the boundary in the
        common case of deleting n entries and inserting n more, so it doubled
        anyway. Counting graves against entries says what is actually meant.

        I found this by writing a test that deleted everything and inserted the
        same number again, and watching a table holding five entries grow to
        thirty two slots. The first version simply doubled every time, which is
        the obvious implementation and quietly wastes memory in exactly the
        workload open addressing is worst at.
        """
        current = len(self._keys)
        clogged = self._tombstones >= self._size
        new_size = current if clogged else current * 2

        old_keys, old_values = self._keys, self._values
        self._keys = [self.EMPTY] * new_size
        self._values = [None] * new_size
        moved = self._size
        self._size = 0
        buried = self._tombstones
        self._tombstones = 0
        self.resizes += 1

        for key, value in zip(old_keys, old_values, strict=True):
            if key is not self.EMPTY and key is not self.TOMBSTONE:
                run(self.put_traced(key, value))

        yield Step(
            "resize",
            (
                f"The table was clogged with {buried} tombstone(s) rather than full, so it "
                f"was rebuilt at the same {len(self._keys)} slots, sweeping them away."
                if clogged
                else f"Grew to {len(self._keys)} slots, rehashing {moved} keys and clearing "
                f"{buried} tombstone(s)."
            ),
            {
                "slots": len(self._keys),
                "rehashed": moved,
                "tombstones_cleared": buried,
                "grew": not clogged,
            },
        )

    def get(self, key: Any, default: Any = MISSING) -> Any:
        return run(self.get_traced(key, default))

    def get_traced(self, key: Any, default: Any = MISSING) -> Traced[Any]:
        """Look up a key by walking its probe sequence until a free slot appears."""
        for attempt, index in enumerate(self._probe_sequence(key)):
            self.probes += 1
            slot = self._keys[index]

            if slot is self.EMPTY:
                yield Step(
                    "missing",
                    f"Slot {index} was never used, so {key!r} cannot be anywhere further "
                    "along its probe sequence.",
                    {"slot": index, "probes": attempt + 1},
                )
                break

            if slot is not self.TOMBSTONE and slot == key:
                yield Step("found", f"{key!r} found in slot {index} after {attempt + 1} "
                                    "probe(s).", {"slot": index, "probes": attempt + 1})
                return self._values[index]

            yield Step(
                "probe",
                f"Slot {index} holds {'a tombstone' if slot is self.TOMBSTONE else repr(slot)}, "
                "so keep probing.",
                {"slot": index, "probe": attempt + 1},
            )

        if default is MISSING:
            raise KeyError(key)
        return default

    def delete(self, key: Any) -> bool:
        """Remove a key, leaving a tombstone behind.

        This is where open addressing gets awkward, and the reason is worth
        understanding because it is not obvious.

        Suppose A and B both hash to slot 3. A takes slot 3, B probes on to slot 4.
        Now delete A and simply blank slot 3. A later search for B starts at slot
        3, finds it empty, and concludes B is not in the table. **B is still
        there and has become unreachable.**

        The fix is a tombstone: a marker meaning "empty now, but keep searching".
        Lookups walk past it, insertions may reuse it. The price is that tombstones
        accumulate and lengthen every probe sequence that crosses them, so they
        count towards the load factor and are only truly cleared by a resize.

        Chaining has none of this. You remove the entry from its list and you are
        finished. That simplicity is a real argument in chaining's favour.
        """
        for index in self._probe_sequence(key):
            slot = self._keys[index]
            if slot is self.EMPTY:
                return False
            if slot is not self.TOMBSTONE and slot == key:
                self._keys[index] = self.TOMBSTONE
                self._values[index] = None
                self._size -= 1
                self._tombstones += 1
                verify_if_checking(self)
                return True
        return False

    def __contains__(self, key: Any) -> bool:
        try:
            self.get(key)
            return True
        except KeyError:
            return False

    def __getitem__(self, key: Any) -> Any:
        return self.get(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        self.put(key, value)

    def keys(self) -> list[Any]:
        return [key for key in self._keys if key is not self.EMPTY and key is not self.TOMBSTONE]

    def items(self) -> list[tuple[Any, Any]]:
        return [
            (key, value)
            for key, value in zip(self._keys, self._values, strict=True)
            if key is not self.EMPTY and key is not self.TOMBSTONE
        ]

    def __repr__(self) -> str:
        return (
            f"OpenAddressingTable({self._size} entries, {self._tombstones} tombstones, "
            f"{len(self._keys)} slots, {self._probing} probing)"
        )

    def check_invariants(self) -> list[Violation]:
        violations: list[Violation] = []
        real = sum(
            1 for key in self._keys if key is not self.EMPTY and key is not self.TOMBSTONE
        )
        graves = sum(1 for key in self._keys if key is self.TOMBSTONE)

        if real != self._size:
            violations.append(Violation(
                "the recorded size matches the slots in use",
                f"the table says {self._size} but {real} slots hold keys",
            ))
        if graves != self._tombstones:
            violations.append(Violation(
                "the tombstone count is correct",
                f"the table says {self._tombstones} but {graves} were found",
            ))
        if self._size + self._tombstones > len(self._keys):
            violations.append(Violation(
                "the table is never over full",
                "entries plus tombstones exceed the number of slots",
            ))

        for key in self.keys():
            reachable = False
            for index in self._probe_sequence(key):
                if self._keys[index] is self.EMPTY:
                    break
                if self._keys[index] is not self.TOMBSTONE and self._keys[index] == key:
                    reachable = True
                    break
            if not reachable:
                violations.append(Violation(
                    "every stored key is still findable along its probe sequence",
                    f"{key!r} is in the table but a lookup would never reach it, which is "
                    "what happens when a deletion blanks a slot instead of marking it",
                ))

        return violations
