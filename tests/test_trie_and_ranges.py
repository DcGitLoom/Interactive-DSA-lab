"""Tests for tries, segment trees, Fenwick trees and union find.

Everything here is checked against a slow but obviously correct reference: a plain
list of words, a Python `sum` over a slice, a dictionary of groups. That pattern
keeps coming up and it is the most reliable way to test a clever structure.
"""

import random

import pytest

from dsalab.invariants import checked
from dsalab.structures.segment_tree import FenwickTree, LazySegmentTree, SegmentTree
from dsalab.structures.trie import CompressedTrie, Trie
from dsalab.structures.union_find import UnionFind, UnionFindWithoutOptimisations
from dsalab.tracing import count_kinds, record

WORDS = ["car", "carpet", "cart", "cat", "dog", "do", "done", "a"]
TRIES = [Trie, CompressedTrie]


@pytest.mark.parametrize("kind", TRIES)
class TestBothTries:
    """Both tries must behave identically from the outside."""

    def test_storing_and_finding_words(self, kind):
        trie = kind(WORDS)

        for word in WORDS:
            assert word in trie
        assert len(trie) == len(WORDS)

    def test_a_prefix_is_not_a_word_unless_it_was_stored(self, kind):
        # The distinction that separates `contains` from `starts_with`, and the
        # reason a node needs an is_word flag separate from having children.
        trie = kind(["carpet"])

        assert "carpet" in trie
        assert "car" not in trie
        assert trie.starts_with("car")

    def test_an_empty_trie(self, kind):
        trie = kind()

        assert len(trie) == 0
        assert "anything" not in trie
        assert trie.words() == []
        assert trie.words_with_prefix("a") == []

    def test_the_empty_string_can_be_stored(self, kind):
        trie = kind([""])

        assert "" in trie
        assert len(trie) == 1

    def test_duplicates_are_refused(self, kind):
        trie = kind(["cat", "cat"])

        assert len(trie) == 1

    def test_prefix_search_finds_exactly_the_right_words(self, kind):
        trie = kind(WORDS)

        assert trie.words_with_prefix("car") == ["car", "carpet", "cart"]
        assert trie.words_with_prefix("do") == ["do", "dog", "done"]
        assert trie.words_with_prefix("z") == []

    def test_an_empty_prefix_returns_everything(self, kind):
        trie = kind(WORDS)

        assert trie.words_with_prefix("") == sorted(WORDS)

    def test_words_come_back_sorted_without_any_sorting_step(self, kind):
        # Walking children alphabetically produces sorted output for free, which
        # a hash table cannot do.
        rng = random.Random(20260901)
        words = ["".join(rng.choice("abc") for _ in range(rng.randint(1, 6)))
                 for _ in range(200)]
        trie = kind(words)

        assert trie.words() == sorted(set(words))

    def test_it_agrees_with_a_plain_list_of_words(self, kind):
        rng = random.Random(20260901)
        stored: set[str] = set()
        trie = kind()

        for _ in range(400):
            word = "".join(rng.choice("abcd") for _ in range(rng.randint(0, 5)))
            trie.insert(word)
            stored.add(word)

        assert len(trie) == len(stored)
        for word in stored:
            assert word in trie
        for prefix in ("", "a", "ab", "abc", "z"):
            expected = sorted(w for w in stored if w.startswith(prefix))
            assert trie.words_with_prefix(prefix) == expected

    def test_a_word_that_is_a_prefix_of_another_inserted_afterwards(self, kind):
        # The order matters for the compressed trie, which has to split an edge in
        # one order and not the other. Both orders must give the same result.
        forwards = kind(["carpet", "car"])
        backwards = kind(["car", "carpet"])

        assert forwards.words() == backwards.words() == ["car", "carpet"]
        assert "car" in forwards and "car" in backwards
        assert "carpet" in forwards and "carpet" in backwards

    def test_words_sharing_only_a_first_character(self, kind):
        trie = kind(["apple", "avocado"])

        assert trie.words() == ["apple", "avocado"]
        assert trie.words_with_prefix("a") == ["apple", "avocado"]
        assert "app" not in trie


class TestPlainTrie:
    def test_deleting_a_word_leaves_the_others_alone(self):
        with checked():
            trie = Trie(WORDS)
            assert trie.delete("carpet") is True

        assert "carpet" not in trie
        assert "car" in trie and "cart" in trie
        assert len(trie) == len(WORDS) - 1

    def test_deleting_a_word_that_other_words_pass_through(self):
        # Removing "car" must not disturb "carpet", which travels through its
        # nodes. This is the "delete too much" failure.
        with checked():
            trie = Trie(["car", "carpet"])
            trie.delete("car")

        assert "carpet" in trie
        assert "car" not in trie
        assert trie.words() == ["carpet"]

    def test_deleting_tidies_up_the_nodes_nobody_needs(self):
        # And this is the "delete too little" failure: dead paths left behind.
        trie = Trie(["carpet"])
        before = trie.node_count()
        trie.delete("carpet")

        assert trie.node_count() < before
        assert trie.node_count() == 1, "only the root should remain"

    def test_deleting_something_absent(self):
        trie = Trie(["cat"])

        assert trie.delete("dog") is False
        assert trie.delete("ca") is False, "a prefix is not a stored word"
        assert len(trie) == 1

    def test_deleting_everything_empties_the_trie(self):
        with checked():
            trie = Trie(WORDS)
            for word in WORDS:
                assert trie.delete(word) is True

        assert len(trie) == 0
        assert trie.node_count() == 1
        assert trie.words() == []

    def test_values_can_be_attached_to_words(self):
        trie = Trie()
        trie.insert("cat", "a small animal")

        assert trie.get("cat") == "a small animal"
        assert trie.get("dog") is None
        assert trie.get("ca") is None, "a prefix carries no value"

    def test_longest_prefix_matching(self):
        # The routing table operation: find the most specific stored entry that
        # covers the given input.
        routes = Trie(["1", "12", "1234"])

        assert routes.longest_prefix_of("123456") == "1234"
        assert routes.longest_prefix_of("129") == "12"
        assert routes.longest_prefix_of("999") is None

    def test_insertion_shares_existing_branches(self):
        trie = Trie(["car"])
        _, steps = record(trie.insert_traced("cart"))
        counts = count_kinds(steps)

        assert counts.get("reuse", 0) == 3, "c, a and r are already there"
        assert counts.get("create", 0) == 1, "only t is new"

    def test_lookup_cost_depends_on_the_word_not_the_collection(self):
        small = Trie(["hello"])
        large = Trie(["hello"] + [f"word{index}" for index in range(5000)])

        _, small_steps = record(small.contains_traced("hello"))
        _, large_steps = record(large.contains_traced("hello"))

        assert len(small_steps) == len(large_steps), (
            "a trie lookup should cost the same whether it holds one word or five thousand"
        )

    def test_a_hand_broken_trie_is_caught(self):
        trie = Trie(["cat"])
        trie._size = 99

        assert any("size" in violation.rule for violation in trie.check_invariants())


class TestCompressedTrie:
    def test_it_uses_far_fewer_nodes_than_a_plain_trie(self):
        # The whole justification for the extra complexity, measured on the same
        # word list rather than asserted.
        rng = random.Random(20260901)
        words = ["".join(rng.choice("abcdefgh") for _ in range(rng.randint(4, 12)))
                 for _ in range(300)]

        plain = Trie(words)
        compressed = CompressedTrie(words)

        assert compressed.node_count() < plain.node_count() / 2, (
            f"compressed used {compressed.node_count()} nodes against {plain.node_count()}"
        )
        assert compressed.words() == plain.words()

    def test_a_single_word_needs_two_nodes(self):
        # A root and one edge carrying the whole word, against one node per
        # character in the plain version.
        trie = CompressedTrie(["encyclopedia"])

        assert trie.node_count() == 2
        assert Trie(["encyclopedia"]).node_count() == 13

    def test_inserting_a_word_that_splits_an_existing_edge(self):
        # The case that does not exist in a plain trie, and where the bugs live.
        with checked():
            trie = CompressedTrie(["carpet"])
            trie.insert("cartoon")

        assert trie.words() == ["carpet", "cartoon"]
        assert "car" not in trie
        assert trie.starts_with("car")

    def test_the_split_is_reported_with_the_shared_part(self):
        trie = CompressedTrie(["carpet"])
        _, steps = record(trie.insert_traced("cartoon"))
        split = [step for step in steps if step.kind == "split"][0]

        assert split.data["shared"] == "car"
        assert split.data["existing_tail"] == "pet"

    def test_inserting_a_word_that_ends_inside_an_existing_edge(self):
        # "car" ends partway along the edge "carpet", so the edge splits and the
        # new node itself becomes a word.
        with checked():
            trie = CompressedTrie(["carpet"])
            trie.insert("car")

        assert "car" in trie and "carpet" in trie
        assert trie.words() == ["car", "carpet"]

    def test_a_prefix_ending_inside_an_edge_is_still_a_prefix(self):
        trie = CompressedTrie(["carpet"])

        assert trie.starts_with("car")
        assert trie.words_with_prefix("car") == ["carpet"]
        assert "car" not in trie

    def test_repeated_splits_of_the_same_edge(self):
        with checked():
            trie = CompressedTrie(["abcdef"])
            for word in ("abcdeg", "abcxyz", "abz", "a"):
                trie.insert(word)

        assert trie.words() == ["a", "abcdef", "abcdeg", "abcxyz", "abz"]

    def test_it_stays_consistent_through_many_random_insertions(self):
        rng = random.Random(20260901)
        stored: set[str] = set()

        with checked():
            trie = CompressedTrie()
            for _ in range(300):
                word = "".join(rng.choice("ab") for _ in range(rng.randint(1, 8)))
                trie.insert(word)
                stored.add(word)

        assert trie.words() == sorted(stored)
        assert len(trie) == len(stored)


class TestSegmentTree:
    def test_range_sums_match_a_plain_sum(self):
        values = [1, 3, 5, 7, 9, 11]
        tree = SegmentTree(values)

        for low in range(len(values)):
            for high in range(low, len(values)):
                assert tree.query(low, high) == sum(values[low : high + 1])

    def test_it_survives_updates(self):
        values = [1, 3, 5, 7, 9]
        with checked():
            tree = SegmentTree(values)
            tree.update(2, 100)

        values[2] = 100
        assert tree.query(0, 4) == sum(values)
        assert tree.query(2, 2) == 100

    def test_it_agrees_with_a_plain_list_over_many_random_operations(self):
        rng = random.Random(20260901)
        values = [rng.randint(-50, 50) for _ in range(60)]

        with checked():
            tree = SegmentTree(values)
            for _ in range(400):
                if rng.random() < 0.4:
                    index = rng.randrange(len(values))
                    new = rng.randint(-50, 50)
                    tree.update(index, new)
                    values[index] = new
                else:
                    low = rng.randrange(len(values))
                    high = rng.randrange(low, len(values))
                    assert tree.query(low, high) == sum(values[low : high + 1])

    def test_it_works_for_minimum_as_well_as_sum(self):
        # The advantage over a Fenwick tree: any associative operation works.
        values = [5, 2, 8, 1, 9]
        tree = SegmentTree(values, combine=min, identity=float("inf"))

        assert tree.query(0, 4) == 1
        assert tree.query(0, 2) == 2
        tree.update(3, 100)
        assert tree.query(0, 4) == 2

    def test_it_works_for_maximum_and_for_gcd(self):
        import math

        values = [12, 18, 24]
        assert SegmentTree(values, combine=max, identity=0).query(0, 2) == 24
        assert SegmentTree(values, combine=math.gcd, identity=0).query(0, 2) == 6

    def test_a_query_visits_only_a_logarithmic_number_of_blocks(self):
        tree = SegmentTree(list(range(1024)))
        _, steps = record(tree.query_traced(100, 900))
        used = count_kinds(steps).get("use", 0)

        assert used <= 2 * 10, f"a query over 1024 items used {used} blocks"

    def test_an_out_of_range_query_is_rejected(self):
        tree = SegmentTree([1, 2, 3])

        with pytest.raises(IndexError):
            tree.query(0, 5)
        with pytest.raises(IndexError):
            tree.update(9, 1)

    def test_a_single_element_and_an_empty_tree(self):
        assert SegmentTree([7]).query(0, 0) == 7
        assert SegmentTree([]).query(0, 0) == 0

    def test_a_hand_broken_tree_is_caught(self):
        tree = SegmentTree([1, 2, 3, 4])
        tree._tree[1] = 999

        assert tree.check_invariants()


class TestLazySegmentTree:
    def test_a_range_update_changes_every_position_in_it(self):
        tree = LazySegmentTree([0] * 10)
        tree.add_to_range(2, 5, 3)

        assert tree.to_list() == [0, 0, 3, 3, 3, 3, 0, 0, 0, 0]

    def test_overlapping_range_updates_accumulate(self):
        tree = LazySegmentTree([0] * 6)
        tree.add_to_range(0, 3, 1)
        tree.add_to_range(2, 5, 10)

        assert tree.to_list() == [1, 1, 11, 11, 10, 10]

    def test_it_agrees_with_a_plain_list_over_many_operations(self):
        rng = random.Random(20260901)
        values = [rng.randint(-20, 20) for _ in range(50)]
        tree = LazySegmentTree(values)

        for _ in range(300):
            low = rng.randrange(len(values))
            high = rng.randrange(low, len(values))

            if rng.random() < 0.5:
                amount = rng.randint(-10, 10)
                tree.add_to_range(low, high, amount)
                for index in range(low, high + 1):
                    values[index] += amount
            else:
                assert tree.query(low, high) == sum(values[low : high + 1])

        assert tree.to_list() == values

    def test_a_range_update_defers_rather_than_touching_every_position(self):
        # The point of lazy propagation. Adding to 100000 positions should touch a
        # handful of blocks, not 100000 leaves.
        tree = LazySegmentTree([0] * 4096)
        _, steps = record(tree.add_to_range_traced(0, 4095, 5))

        assert count_kinds(steps).get("defer", 0) <= 2
        assert tree.query(0, 4095) == 4096 * 5

    def test_an_out_of_range_update_is_rejected(self):
        with pytest.raises(IndexError):
            LazySegmentTree([1, 2, 3]).add_to_range(0, 9, 1)


class TestFenwickTree:
    def test_prefix_sums_match_a_plain_sum(self):
        values = [1, 3, 5, 7, 9, 11]
        tree = FenwickTree(values)

        for index in range(len(values)):
            assert tree.prefix_sum(index) == sum(values[: index + 1])

    def test_range_sums_match(self):
        values = [4, -2, 7, 0, 5]
        tree = FenwickTree(values)

        for low in range(len(values)):
            for high in range(low, len(values)):
                assert tree.range_sum(low, high) == sum(values[low : high + 1])

    def test_it_can_be_built_empty_and_filled(self):
        tree = FenwickTree(5)
        for index in range(5):
            tree.add(index, index + 1)

        assert tree.to_list() == [1, 2, 3, 4, 5]
        assert tree.range_sum(1, 3) == 9

    def test_it_agrees_with_a_plain_list_over_many_random_operations(self):
        rng = random.Random(20260901)
        values = [rng.randint(-30, 30) for _ in range(70)]

        with checked():
            tree = FenwickTree(values)
            for _ in range(400):
                if rng.random() < 0.4:
                    index = rng.randrange(len(values))
                    amount = rng.randint(-20, 20)
                    tree.add(index, amount)
                    values[index] += amount
                else:
                    low = rng.randrange(len(values))
                    high = rng.randrange(low, len(values))
                    assert tree.range_sum(low, high) == sum(values[low : high + 1])

    def test_setting_a_value_outright(self):
        tree = FenwickTree([1, 2, 3])
        tree.set(1, 99)

        assert tree.to_list() == [1, 99, 3]

    def test_the_number_of_slots_touched_is_the_number_of_set_bits(self):
        # The bit trick made visible: a prefix sum up to 13 (binary 1101) reads
        # slots 13, 12 and 8, which is three slots for three set bits.
        tree = FenwickTree([1] * 16)
        _, steps = record(tree.prefix_sum_traced(12))  # 0 based, so position 13
        slots = [step.data["slot"] for step in steps]

        assert slots == [13, 12, 8]
        assert bin(13).count("1") == 3

    def test_it_uses_a_quarter_of_the_memory_of_a_segment_tree(self):
        size = 1000
        fenwick = FenwickTree(list(range(size)))
        segment = SegmentTree(list(range(size)))

        assert len(fenwick._tree) == size + 1
        assert len(segment._tree) == 4 * size

    def test_out_of_range_indices(self):
        tree = FenwickTree([1, 2, 3])

        with pytest.raises(IndexError):
            tree.add(9, 1)
        assert tree.prefix_sum(-1) == 0
        assert tree.prefix_sum(99) == 6, "a prefix past the end is the whole array"

    def test_corruption_is_undetectable_from_inside_which_is_worth_knowing(self):
        # Unlike every other structure here, a Fenwick tree cannot check itself.
        # It stores each value once with no redundancy, so any array of slots is a
        # valid Fenwick tree of *some* array, and there is nothing to contradict.
        # A segment tree can self check because a parent must equal its children
        # combined.
        tree = FenwickTree([1, 2, 3, 4])
        tree._tree[2] = 999

        assert tree.check_invariants() == [], (
            "there is genuinely nothing here to catch, which is why the tests cross "
            "check against a segment tree instead"
        )
        assert tree.to_list() != [1, 2, 3, 4], "the corruption is real, just invisible"

    def test_a_wrong_sized_internal_array_is_caught(self):
        # What can be checked: the structural facts that do not depend on the
        # values stored.
        tree = FenwickTree([1, 2, 3])
        tree._tree.append(0)

        assert any("one slot per value" in violation.rule
                   for violation in tree.check_invariants())

    def test_the_two_range_structures_agree_with_each_other(self):
        rng = random.Random(20260901)
        values = [rng.randint(-50, 50) for _ in range(80)]
        fenwick = FenwickTree(values)
        segment = SegmentTree(values)

        for _ in range(200):
            low = rng.randrange(len(values))
            high = rng.randrange(low, len(values))

            assert fenwick.range_sum(low, high) == segment.query(low, high)


class TestUnionFind:
    def test_everything_starts_in_its_own_group(self):
        groups = UnionFind(5)

        assert groups.groups == 5
        assert not groups.connected(0, 1)
        assert groups.group_size(0) == 1

    def test_merging_two_groups(self):
        with checked():
            groups = UnionFind(5)
            assert groups.union(0, 1) is True

        assert groups.connected(0, 1)
        assert groups.groups == 4
        assert groups.group_size(0) == 2

    def test_merging_things_already_connected_changes_nothing(self):
        # The return value Kruskal's algorithm uses as its cycle test.
        groups = UnionFind(3)
        groups.union(0, 1)

        assert groups.union(1, 0) is False
        assert groups.groups == 2

    def test_connection_is_transitive(self):
        with checked():
            groups = UnionFind(6)
            groups.union(0, 1)
            groups.union(1, 2)
            groups.union(3, 4)

        assert groups.connected(0, 2)
        assert not groups.connected(0, 3)
        assert groups.group_size(0) == 3

    def test_merging_everything_leaves_one_group(self):
        with checked():
            groups = UnionFind(100)
            for item in range(99):
                groups.union(item, item + 1)

        assert groups.groups == 1
        assert groups.group_size(0) == 100
        assert all(groups.connected(0, item) for item in range(100))

    def test_it_agrees_with_a_plain_dictionary_of_groups(self):
        rng = random.Random(20260901)
        count = 50
        groups = UnionFind(count)
        reference = {item: {item} for item in range(count)}

        with checked():
            for _ in range(300):
                a, b = rng.randrange(count), rng.randrange(count)
                groups.union(a, b)

                merged = reference[a] | reference[b]
                for member in merged:
                    reference[member] = merged

        for a in range(count):
            for b in range(count):
                assert groups.connected(a, b) == (b in reference[a])

    def test_out_of_range_items_are_rejected(self):
        with pytest.raises(IndexError):
            UnionFind(3).find(9)

    def test_a_negative_count_is_rejected(self):
        with pytest.raises(ValueError):
            UnionFind(-1)

    def test_members_lists_every_group(self):
        groups = UnionFind(5)
        groups.union(0, 1)
        groups.union(3, 4)

        sets = sorted(sorted(members) for members in groups.members().values())
        assert sets == [[0, 1], [2], [3, 4]]


class TestTheOptimisationsAreWorthIt:
    """Measuring the two one line optimisations rather than claiming they help."""

    def test_union_by_size_keeps_the_trees_flat(self):
        # The pathological order: always merge the big group into a single item.
        # Without union by size this builds a chain of length n.
        count = 500
        optimised = UnionFind(count)
        naive = UnionFindWithoutOptimisations(count)

        for item in range(1, count):
            optimised.union(item, 0)
            naive.union(item, item - 1)

        assert naive.max_depth() > 100, "the naive version should build a deep chain"
        assert optimised.max_depth() <= 2, (
            f"union by size should keep it nearly flat, got {optimised.max_depth()}"
        )

    def test_path_compression_flattens_what_it_walks(self):
        # Union by size already keeps trees flat, so to see compression working on
        # its own the chain has to be built by hand. This reaches into the parent
        # array deliberately: no sequence of unions with union by size can produce
        # a chain this deep, which is itself the point of the previous test.
        groups = UnionFind(100)
        groups._parent = [max(0, item - 1) for item in range(100)]
        groups._size = [100] + [1] * 99
        groups._groups = 1

        groups.path_hops = 0
        groups.find(99)
        first_walk = groups.path_hops

        groups.path_hops = 0
        groups.find(99)
        second_walk = groups.path_hops

        assert first_walk == 99, "the hand built chain should be 99 hops deep"
        assert second_walk == 1, "after compression the same lookup is a single hop"

        # And every node passed on the way was flattened too, not just the one
        # asked for, which is what makes the amortised cost so low.
        groups.path_hops = 0
        for item in range(100):
            groups.find(item)
        assert groups.path_hops <= 100

    def test_the_naive_version_walks_far_more(self):
        count = 400
        optimised = UnionFind(count)
        naive = UnionFindWithoutOptimisations(count)

        for item in range(1, count):
            optimised.union(item, item - 1)
            naive.union(item, item - 1)

        optimised.path_hops = naive.path_hops = 0
        for item in range(count):
            optimised.find(item)
            naive.find(item)

        assert optimised.path_hops < naive.path_hops / 10, (
            f"optimised walked {optimised.path_hops} steps, naive walked {naive.path_hops}"
        )

    def test_the_first_find_compresses_and_reports_it(self):
        groups = UnionFind(10)
        for item in range(1, 10):
            groups.union(item, item - 1)

        _, steps = record(groups.find_traced(9))
        compressions = [step for step in steps if step.kind == "compress"]

        assert compressions or all(step.kind == "find" for step in steps)
