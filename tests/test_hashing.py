"""Tests for both hash tables and for the LRU cache.

The most valuable test in this file is the one that deliberately creates the
unreachable key bug: delete a key whose slot other keys probed past, and check
that they can still be found. That is the bug tombstones exist to prevent, and it
is invisible unless you go looking for it.
"""

import random

import pytest

from dsalab.invariants import checked, verify
from dsalab.structures.hash_table import (
    ChainedHashTable,
    OpenAddressingTable,
)
from dsalab.structures.lru_cache import LRUCache
from dsalab.tracing import count_kinds, record

PROBING = [OpenAddressingTable.LINEAR, OpenAddressingTable.QUADRATIC, OpenAddressingTable.DOUBLE]


class Colliding:
    """A key type whose hash is deliberately terrible, for testing the worst case.

    Every instance hashes to the same bucket, which turns any hash table into a
    linear scan. This is what a hash flooding attack does on purpose.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __hash__(self) -> int:
        return 42

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Colliding) and other.name == self.name

    def __repr__(self) -> str:
        return f"Colliding({self.name!r})"


class TestChainedHashTable:
    def test_storing_and_reading_back(self):
        table = ChainedHashTable()
        table["one"] = 1
        table["two"] = 2

        assert table["one"] == 1
        assert table["two"] == 2
        assert len(table) == 2

    def test_a_missing_key_raises_unless_a_default_is_given(self):
        table = ChainedHashTable()

        with pytest.raises(KeyError):
            table["nobody"]
        assert table.get("nobody", "fallback") == "fallback"

    def test_storing_the_same_key_twice_replaces_rather_than_duplicates(self):
        table = ChainedHashTable()
        table["key"] = 1
        table["key"] = 2

        assert table["key"] == 2
        assert len(table) == 1

    def test_deleting(self):
        table = ChainedHashTable()
        table["key"] = 1

        assert table.delete("key") is True
        assert table.delete("key") is False
        assert len(table) == 0

    def test_none_can_be_stored_as_a_real_value(self):
        # A table that uses None to mean "absent" cannot tell a stored None from a
        # missing key. The sentinel exists so this works.
        table = ChainedHashTable()
        table["key"] = None

        assert table["key"] is None
        assert table.get("key", "default") is None

    def test_it_agrees_with_a_plain_dictionary_over_many_operations(self):
        rng = random.Random(20260829)
        table = ChainedHashTable()
        mirror: dict[int, int] = {}

        with checked():
            for _ in range(2000):
                key = rng.randint(0, 300)
                if rng.random() < 0.7:
                    value = rng.randint(0, 1000)
                    table[key] = value
                    mirror[key] = value
                else:
                    assert table.delete(key) == (mirror.pop(key, None) is not None)

        assert len(table) == len(mirror)
        assert sorted(table.items()) == sorted(mirror.items())

    def test_it_grows_when_the_load_factor_passes_the_threshold(self):
        table = ChainedHashTable(buckets=8, max_load=0.75)
        for value in range(7):
            table[value] = value

        assert table.bucket_count > 8
        assert table.resizes >= 1

    def test_growing_rehashes_rather_than_copying(self):
        # Every key's bucket depends on the bucket count, so a resize that copied
        # buckets across would put keys in the wrong place. The invariant check
        # verifies each key is in the bucket its hash chooses.
        table = ChainedHashTable(buckets=4)
        for value in range(200):
            table[value] = value * 2

        verify(table)
        assert all(table[value] == value * 2 for value in range(200))

    def test_the_load_factor_stays_near_the_threshold(self):
        table = ChainedHashTable(max_load=0.75)
        for value in range(1000):
            table[value] = value

        assert table.load_factor <= 0.75

    def test_chains_stay_short_with_a_sensible_load_factor(self):
        table = ChainedHashTable()
        for value in range(2000):
            table[value] = value

        assert table.longest_chain() <= 6, (
            f"longest chain was {table.longest_chain()}, which suggests the keys are "
            "not spreading out"
        )

    def test_the_worst_case_really_is_a_linear_scan(self):
        # Every key in one bucket. The table still works and is still correct, it
        # is just no better than a list, which is exactly what an attacker wants.
        table = ChainedHashTable()
        keys = [Colliding(f"k{index}") for index in range(50)]
        for index, key in enumerate(keys):
            table[key] = index

        assert table.longest_chain() == 50
        assert all(table[key] == index for index, key in enumerate(keys))

        _, steps = record(table.get_traced(keys[-1]))
        assert count_kinds(steps)["probe"] == 50, "the last key needs a full scan"

    def test_a_hand_corrupted_table_is_caught(self):
        table = ChainedHashTable()
        table["key"] = 1
        table._size = 99

        assert any("size" in violation.rule for violation in table.check_invariants())


@pytest.mark.parametrize("probing", PROBING)
class TestOpenAddressing:
    def test_storing_and_reading_back(self, probing):
        table = OpenAddressingTable(probing=probing)
        table["one"] = 1
        table["two"] = 2

        assert table["one"] == 1
        assert table["two"] == 2

    def test_a_missing_key(self, probing):
        table = OpenAddressingTable(probing=probing)

        with pytest.raises(KeyError):
            table["nobody"]
        assert table.get("nobody", "fallback") == "fallback"

    def test_it_agrees_with_a_plain_dictionary(self, probing):
        rng = random.Random(20260829)
        table = OpenAddressingTable(probing=probing)
        mirror: dict[int, int] = {}

        with checked():
            for _ in range(1500):
                key = rng.randint(0, 200)
                if rng.random() < 0.7:
                    value = rng.randint(0, 1000)
                    table[key] = value
                    mirror[key] = value
                else:
                    table.delete(key)
                    mirror.pop(key, None)

        assert len(table) == len(mirror)
        assert sorted(table.items()) == sorted(mirror.items())

    def test_deleting_leaves_a_tombstone_so_later_keys_stay_reachable(self, probing):
        # The bug this prevents: A and B collide, A takes the slot and B probes on.
        # Delete A by blanking its slot and a search for B stops at the empty slot
        # and reports it missing, even though B is right there.
        table = OpenAddressingTable(buckets=16, probing=probing)
        keys = [Colliding(f"k{index}") for index in range(6)]

        for index, key in enumerate(keys):
            table[key] = index

        table.delete(keys[0])

        for index, key in enumerate(keys[1:], start=1):
            assert table[key] == index, f"{key!r} became unreachable after a deletion"

        verify(table)

    def test_the_probe_sequence_reaches_every_slot(self, probing):
        # A probe sequence that misses slots can report a table as full when it is
        # not. Quadratic probing with plain squares has this problem, which is why
        # the triangular numbers are used instead.
        table = OpenAddressingTable(buckets=16, probing=probing, max_load=0.99)
        visited = set(table._probe_sequence("anything"))

        assert len(visited) == table.bucket_count

    def test_a_table_full_of_tombstones_still_works(self, probing):
        table = OpenAddressingTable(buckets=8, probing=probing)

        for round_number in range(50):
            table[f"key{round_number}"] = round_number
            table.delete(f"key{round_number}")

        assert len(table) == 0
        table["fresh"] = 1
        assert table["fresh"] == 1
        verify(table)

    def test_tombstones_are_reused_rather_than_left_forever(self, probing):
        table = OpenAddressingTable(buckets=16, probing=probing)
        for value in range(5):
            table[value] = value
        for value in range(5):
            table.delete(value)

        before = table.bucket_count
        for value in range(100, 105):
            table[value] = value

        assert table.bucket_count == before, "reusing tombstones should avoid a resize here"

    def test_it_grows_before_getting_too_full(self, probing):
        table = OpenAddressingTable(buckets=8, probing=probing, max_load=0.5)
        for value in range(100):
            table[value] = value

        assert table.load_factor <= 0.5
        assert table.resizes > 0

    def test_none_can_be_stored(self, probing):
        table = OpenAddressingTable(probing=probing)
        table["key"] = None

        assert table["key"] is None


class TestProbingStrategiesCompared:
    def test_an_unknown_strategy_is_rejected(self):
        with pytest.raises(ValueError):
            OpenAddressingTable(probing="magic")

    def test_probing_cost_climbs_steeply_as_the_table_fills(self):
        # The measurement behind the 0.5 threshold for open addressing. The
        # expected probes for a failed linear probe lookup is roughly
        # (1 + 1/(1-load)^2)/2, which is a cliff rather than a slope.
        def probes_at_load(load: float) -> float:
            size = 1024
            table = OpenAddressingTable(buckets=size, max_load=0.999)
            for value in range(int(size * load)):
                table[value] = value

            table.probes = 0
            for missing in range(100000, 100200):
                table.get(missing, None)
            return table.probes / 200

        gentle = probes_at_load(0.5)
        heavy = probes_at_load(0.9)

        assert heavy > 3 * gentle, (
            f"at load 0.5 a failed lookup took {gentle:.1f} probes and at 0.9 it took "
            f"{heavy:.1f}, which should be far worse"
        )

    def test_linear_probing_clusters_more_than_double_hashing(self):
        # Primary clustering: with linear probing, any run of occupied slots grows
        # at both ends, because every key landing in the run walks past all of it.
        # Double hashing gives each key its own step size, so runs do not form.
        def total_probes(strategy: str) -> int:
            table = OpenAddressingTable(buckets=2048, probing=strategy, max_load=0.9)
            for value in range(1400):
                table[value * 3] = value
            table.probes = 0
            for value in range(1400):
                table.get(value * 3)
            return table.probes

        linear = total_probes(OpenAddressingTable.LINEAR)
        double = total_probes(OpenAddressingTable.DOUBLE)

        assert double <= linear, (
            f"double hashing took {double} probes and linear took {linear}; double "
            "hashing should spread keys at least as well"
        )


class TestLRUCache:
    def test_it_stores_and_returns_values(self):
        cache = LRUCache(capacity=2)
        cache.put("a", 1)

        assert cache.get("a") == 1
        assert cache.get("missing") is None

    def test_a_capacity_below_one_is_rejected(self):
        with pytest.raises(ValueError):
            LRUCache(capacity=0)

    def test_it_evicts_the_least_recently_used_entry(self):
        with checked():
            cache = LRUCache(capacity=2)
            cache.put("a", 1)
            cache.put("b", 2)
            cache.put("c", 3)  # "a" is the oldest, so it goes

        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_reading_an_entry_protects_it_from_the_next_eviction(self):
        # The behaviour that makes it an LRU cache rather than a queue.
        with checked():
            cache = LRUCache(capacity=2)
            cache.put("a", 1)
            cache.put("b", 2)
            cache.get("a")      # "a" is now the most recently used
            cache.put("c", 3)   # so "b" should be evicted instead

        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_updating_an_existing_key_also_counts_as_using_it(self):
        with checked():
            cache = LRUCache(capacity=2)
            cache.put("a", 1)
            cache.put("b", 2)
            cache.put("a", 99)
            cache.put("c", 3)

        assert cache.get("a") == 99
        assert cache.get("b") is None

    def test_membership_testing_does_not_count_as_a_use(self):
        # Asking whether something is cached is not using it. If it counted, the
        # hit rate would be meaningless and the recency order would be wrong.
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)

        assert "a" in cache
        cache.put("c", 3)

        assert "a" not in cache, "a containment check should not have protected 'a'"
        assert cache.hits == 0 and cache.misses == 0

    def test_the_recency_order_is_what_it_claims_to_be(self):
        cache = LRUCache(capacity=3)
        for key in "abc":
            cache.put(key, key)

        assert cache.keys_by_recency() == ["c", "b", "a"]
        cache.get("a")
        assert cache.keys_by_recency() == ["a", "c", "b"]

    def test_statistics_are_kept(self):
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.get("a")
        cache.get("b")
        cache.put("b", 2)
        cache.put("c", 3)

        assert cache.hits == 1
        assert cache.misses == 1
        assert cache.evictions == 1
        assert cache.hit_rate == 0.5

    def test_the_hit_rate_of_a_cache_nobody_has_read_is_zero(self):
        assert LRUCache(capacity=2).hit_rate == 0.0

    def test_a_capacity_of_one_works(self):
        with checked():
            cache = LRUCache(capacity=1)
            cache.put("a", 1)
            cache.put("b", 2)

        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_deleting_an_entry(self):
        with checked():
            cache = LRUCache(capacity=2)
            cache.put("a", 1)

            assert cache.delete("a") is True
            assert cache.delete("a") is False
            assert len(cache) == 0

    def test_clearing_keeps_the_statistics(self):
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.get("a")
        cache.clear()

        assert len(cache) == 0
        assert cache.hits == 1

    def test_the_map_and_the_list_never_drift_apart(self):
        # The characteristic failure of a structure built from two others. Every
        # operation is checked, and the contents are compared against a plain
        # dictionary with a hand written eviction rule.
        rng = random.Random(20260829)
        capacity = 8
        cache = LRUCache(capacity=capacity)
        recency: list[str] = []
        values: dict[str, int] = {}

        with checked():
            for step in range(2000):
                key = f"k{rng.randint(0, 20)}"
                if rng.random() < 0.6:
                    value = rng.randint(0, 100)
                    cache.put(key, value)
                    if key in recency:
                        recency.remove(key)
                    elif len(recency) >= capacity:
                        values.pop(recency.pop())
                    recency.insert(0, key)
                    values[key] = value
                else:
                    result = cache.get(key)
                    assert result == values.get(key)
                    if key in recency:
                        recency.remove(key)
                        recency.insert(0, key)

                assert cache.keys_by_recency() == recency

        assert len(cache) == len(values)

    def test_it_never_exceeds_its_capacity(self):
        with checked():
            cache = LRUCache(capacity=5)
            for value in range(500):
                cache.put(value, value)

        assert len(cache) == 5
        assert cache.evictions == 495

    def test_a_hand_corrupted_cache_is_caught(self):
        cache = LRUCache(capacity=3)
        cache.put("a", 1)
        cache._nodes["ghost"] = cache._nodes["a"]

        violations = cache.check_invariants()
        assert violations

    def test_every_operation_produces_a_readable_step(self):
        cache = LRUCache(capacity=1)
        _, steps = record(cache.put_traced("a", 1))
        _, more = record(cache.put_traced("b", 2))

        assert all(step.note.endswith(".") for step in steps + more)
        assert count_kinds(more)["evict"] == 1
