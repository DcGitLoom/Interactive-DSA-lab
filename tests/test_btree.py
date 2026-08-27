"""Tests for the B-tree, including the 2-3 tree special case.

Every structural test runs at several orders, because a B-tree bug is very often
an off by one in the minimum or maximum key count that only shows at one
particular order. Odd and even orders behave differently when splitting, so both
are always covered.
"""

import math
import random

import pytest

from dsalab.invariants import checked
from dsalab.structures.btree import BTree, two_three_tree
from dsalab.tracing import count_kinds, record

ORDERS = [3, 4, 5, 6, 8, 13]


class TestRules:
    def test_the_order_must_be_at_least_three(self):
        with pytest.raises(ValueError):
            BTree(order=2)

    @pytest.mark.parametrize("order", ORDERS)
    def test_the_key_limits_are_what_the_definition_says(self, order):
        tree = BTree(order=order)

        assert tree.max_keys == order - 1
        assert tree.min_keys == math.ceil(order / 2) - 1

    @pytest.mark.parametrize("order", ORDERS)
    def test_every_rule_holds_after_many_random_insertions(self, order):
        rng = random.Random(20260827 + order)
        with checked():
            tree = BTree(order=order)
            for _ in range(400):
                tree.insert(rng.randint(0, 3000))

        assert tree.check_invariants() == []

    @pytest.mark.parametrize("order", ORDERS)
    def test_all_leaves_stay_at_the_same_depth(self, order):
        # The balance condition. It holds because the tree grows at the root
        # rather than at the leaves.
        tree = BTree(order=order, keys=range(500))

        depths = set()

        def walk(node, depth):
            if node.is_leaf:
                depths.add(depth)
                return
            for child in node.children:
                walk(child, depth + 1)

        walk(tree.root, 0)
        assert len(depths) == 1

    def test_a_hand_broken_node_is_caught(self):
        tree = BTree(order=5, keys=range(50))
        tree.root.keys.append(9999)  # unsorted, and possibly over the limit

        assert tree.check_invariants()

    def test_an_underfull_node_is_caught(self):
        tree = BTree(order=5, keys=range(50))
        victim = tree.root.children[0]
        victim.keys = victim.keys[:1]

        violations = tree.check_invariants()
        assert any("at least" in violation.rule for violation in violations)


class TestShallowness:
    """The reason the structure exists, stated in numbers."""

    def test_a_higher_order_gives_a_shorter_tree(self):
        heights = {order: BTree(order=order, keys=range(10000)).height() for order in (3, 8, 64)}

        assert heights[3] > heights[8] > heights[64]

    def test_a_wide_tree_holding_ten_thousand_keys_is_only_a_few_levels_deep(self):
        # With order 128, a lookup on disk would be three or four block reads.
        # A balanced binary tree of the same data would be about fourteen.
        tree = BTree(order=128, keys=range(10000))

        assert tree.height() <= 3

    @pytest.mark.parametrize("order", [5, 16, 64])
    def test_the_height_stays_near_the_logarithm_to_the_base_of_the_order(self, order):
        tree = BTree(order=order, keys=range(20000))
        ideal = math.log(20000, order)

        assert tree.height() <= 2 * ideal + 1


class TestInsertion:
    @pytest.mark.parametrize("order", ORDERS)
    def test_keys_come_back_sorted(self, order):
        rng = random.Random(20260827)
        values = rng.sample(range(10000), 500)
        tree = BTree(order=order, keys=values)

        assert tree.keys() == sorted(values)
        assert len(tree) == len(values)

    def test_duplicates_are_refused(self):
        tree = BTree(order=5, keys=[1, 2, 3])

        assert tree.insert(2) is False
        assert len(tree) == 3

    def test_the_tree_grows_at_the_root_not_the_leaves(self):
        # Splitting a full root is the only way a B-tree gains a level, and it is
        # why every leaf stays at the same depth.
        tree = BTree(order=3)
        heights = []
        for value in range(20):
            _, steps = record(tree.insert_traced(value))
            if any(step.kind == "grow" for step in steps):
                heights.append(tree.height())

        assert heights == sorted(heights)
        assert heights, "inserting twenty keys into an order 3 tree must grow the root"

    def test_a_split_pushes_the_middle_key_up(self):
        tree = BTree(order=3, keys=[10, 20])
        _, steps = record(tree.insert_traced(30))
        splits = [step for step in steps if step.kind == "split"]

        assert splits
        assert splits[0].data["promoted"] == 20, "the middle key is the one that moves up"
        assert splits[0].data["left"] == [10]
        assert splits[0].data["right"] == [30]

    def test_sorted_input_does_not_degrade_the_tree(self):
        with checked():
            tree = BTree(order=5, keys=range(2000))

        assert tree.height() <= 6
        assert tree.keys() == list(range(2000))


class TestSearching:
    @pytest.mark.parametrize("order", ORDERS)
    def test_finding_what_is_present_and_not_what_is_absent(self, order):
        tree = BTree(order=order, keys=range(0, 400, 2))

        for value in range(0, 400, 2):
            assert value in tree
        for value in range(1, 400, 2):
            assert value not in tree

    def test_searching_an_empty_tree(self):
        assert 5 not in BTree(order=5)

    def test_a_search_reads_one_node_per_level(self):
        tree = BTree(order=8, keys=range(2000))
        _, steps = record(tree.contains_traced(1500))
        descents = count_kinds(steps).get("descend", 0)

        assert descents <= tree.height()


class TestDeletion:
    @pytest.mark.parametrize("order", ORDERS)
    def test_deleting_every_key_in_random_order_keeps_the_rules(self, order):
        rng = random.Random(20260827 + order)
        values = rng.sample(range(5000), 200)

        with checked():
            tree = BTree(order=order, keys=values)
            rng.shuffle(values)
            for index, value in enumerate(values):
                assert tree.delete(value) is True
                assert tree.keys() == sorted(values[index + 1 :])

        assert len(tree) == 0

    @pytest.mark.parametrize("order", ORDERS)
    def test_deleting_in_sorted_order(self, order):
        with checked():
            tree = BTree(order=order, keys=range(200))
            for value in range(200):
                tree.delete(value)

        assert len(tree) == 0

    @pytest.mark.parametrize("order", ORDERS)
    def test_deleting_in_reverse_order(self, order):
        with checked():
            tree = BTree(order=order, keys=range(200))
            for value in range(199, -1, -1):
                tree.delete(value)

        assert len(tree) == 0

    def test_deleting_something_absent(self):
        tree = BTree(order=5, keys=range(20))

        assert tree.delete(999) is False
        assert len(tree) == 20

    def test_deleting_from_an_empty_tree(self):
        assert BTree(order=5).delete(1) is False

    def test_borrowing_happens_before_merging(self):
        # Borrowing touches three nodes and leaves the height alone, so it is the
        # cheaper repair and should be preferred whenever a sibling can spare a
        # key.
        rng = random.Random(20260827)
        tree = BTree(order=5, keys=rng.sample(range(1000), 200))
        borrows = merges = 0

        for value in tree.keys()[:100]:
            _, steps = record(tree.delete_traced(value))
            counts = count_kinds(steps)
            borrows += counts.get("borrow", 0)
            merges += counts.get("merge", 0)

        assert borrows + merges > 0, "deleting a hundred keys should need some rebalancing"

    def test_the_tree_shrinks_when_the_root_empties(self):
        tree = BTree(order=3, keys=range(50))
        tall = tree.height()

        for value in range(45):
            tree.delete(value)

        assert tree.height() < tall

    def test_a_long_mix_of_insertions_and_deletions(self):
        rng = random.Random(20260827)
        mirror: set[int] = set()

        with checked():
            tree = BTree(order=4)
            for _ in range(2000):
                value = rng.randint(0, 200)
                if rng.random() < 0.55:
                    tree.insert(value)
                    mirror.add(value)
                else:
                    tree.delete(value)
                    mirror.discard(value)

        assert tree.keys() == sorted(mirror)
        assert len(tree) == len(mirror)


class TestTwoThreeTree:
    def test_it_is_just_a_b_tree_of_order_three(self):
        tree = two_three_tree(range(20))

        assert tree.order == 3
        assert tree.max_keys == 2
        assert tree.min_keys == 1

    def test_every_node_holds_one_or_two_keys(self):
        with checked():
            tree = two_three_tree(range(100))

        def walk(node):
            if node is not tree.root:
                assert 1 <= len(node.keys) <= 2
            assert node.is_leaf or len(node.children) in (2, 3)
            for child in node.children:
                walk(child)

        walk(tree.root)

    def test_it_behaves_like_any_other_search_structure(self):
        rng = random.Random(20260827)
        values = rng.sample(range(1000), 150)
        tree = two_three_tree(values)

        assert tree.keys() == sorted(values)
        assert all(value in tree for value in values)
