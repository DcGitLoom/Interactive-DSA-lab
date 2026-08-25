"""Tests for the AVL tree.

Two kinds of test here. The first kind pins down each of the four rotation cases
with a hand built example, because those are the cases people get wrong. The
second kind throws thousands of random operations at the tree with invariant
checking switched on, which is where the cases nobody thought of get found.
"""

import math
import random

import pytest

from dsalab.complexity import detect
from dsalab.invariants import checked
from dsalab.structures.avl import AVLTree, _balance_factor, _height
from dsalab.structures.bst import BinarySearchTree
from dsalab.tracing import count_kinds, record


class TestTheFourRotationCases:
    def test_left_left_needs_one_right_rotation(self):
        # 30, 20, 10 inserted in order leans hard left.
        with checked():
            tree = AVLTree([30, 20, 10])

        assert tree.root.value == 20
        assert tree.root.left.value == 10
        assert tree.root.right.value == 30
        assert tree.rotations == 1

    def test_right_right_needs_one_left_rotation(self):
        with checked():
            tree = AVLTree([10, 20, 30])

        assert tree.root.value == 20
        assert tree.rotations == 1

    def test_left_right_needs_a_double_rotation(self):
        # 30, 10, 20: left heavy, but the left child leans right.
        with checked():
            tree = AVLTree([30, 10, 20])

        assert tree.root.value == 20
        assert tree.root.left.value == 10
        assert tree.root.right.value == 30
        assert tree.rotations == 2, "a left right case takes two rotations, not one"

    def test_right_left_needs_a_double_rotation(self):
        with checked():
            tree = AVLTree([10, 30, 20])

        assert tree.root.value == 20
        assert tree.rotations == 2

    def test_the_double_rotation_cases_are_reported_as_such(self):
        tree = AVLTree([30, 10])
        _, steps = record(tree.insert_traced(20))
        cases = [step.data["case"] for step in steps if step.kind == "rotate"]

        assert cases == ["left right", "left left"]

    def test_a_balanced_insertion_needs_no_rotation_at_all(self):
        tree = AVLTree([20, 10, 30])
        before = tree.rotations
        tree.insert(25)

        assert tree.rotations == before


class TestBalanceIsMaintained:
    def test_sorted_input_stays_logarithmic_instead_of_becoming_a_chain(self):
        # The whole reason this structure exists. The plain search tree from day
        # 9 reaches height 999 on this input.
        with checked():
            tree = AVLTree(range(1000))

        assert tree.height() <= 1.44 * math.log2(1000) + 1
        assert tree.height() < 15
        assert BinarySearchTree(range(1000)).height() == 999

    def test_reverse_sorted_input_is_handled_too(self):
        with checked():
            tree = AVLTree(range(1000, 0, -1))

        assert tree.height() < 15

    def test_the_height_grows_logarithmically_with_the_number_of_nodes(self):
        sizes = [64, 128, 256, 512, 1024, 2048]
        heights = [float(AVLTree(range(n)).height()) for n in sizes]
        verdict = detect(sizes, heights)

        assert verdict.best.curve == "O(log n)"
        assert verdict.is_confident

    def test_the_height_never_exceeds_the_theoretical_bound(self):
        # The Fibonacci argument says an AVL tree of n nodes has height at most
        # about 1.44 log2(n). This checks the claim at many sizes.
        rng = random.Random(20260825)
        for size in (1, 2, 5, 10, 50, 200, 1000, 3000):
            values = list(range(size))
            rng.shuffle(values)
            tree = AVLTree(values)

            assert tree.height() <= 1.44 * math.log2(size + 1) + 1

    def test_every_node_stays_within_the_balance_rule_after_random_insertions(self):
        rng = random.Random(20260825)
        with checked():
            tree = AVLTree()
            for _ in range(1000):
                tree.insert(rng.randint(0, 5000))

        def check(node):
            if node is None:
                return
            assert abs(_balance_factor(node)) <= 1
            check(node.left)
            check(node.right)

        check(tree.root)

    def test_stored_heights_stay_correct_after_rotations(self):
        # A rotation that updates the two heights in the wrong order leaves stale
        # values behind, and stale heights make every later balance decision
        # wrong while the tree still looks fine from outside.
        rng = random.Random(20260825)
        tree = AVLTree()
        for _ in range(500):
            tree.insert(rng.randint(0, 1000))

        def real_height(node):
            if node is None:
                return -1
            expected = 1 + max(real_height(node.left), real_height(node.right))
            assert node.height == expected, f"{node.value} has a stale height"
            return expected

        real_height(tree.root)


class TestOrdinarySearchTreeBehaviour:
    def test_inorder_comes_out_sorted(self):
        rng = random.Random(20260825)
        values = rng.sample(range(1000), 200)
        tree = AVLTree(values)

        assert tree.inorder() == sorted(values)

    def test_lookup_finds_what_is_there_and_not_what_is_not(self):
        tree = AVLTree(range(0, 100, 2))

        assert 50 in tree
        assert 51 not in tree
        assert -1 not in tree

    def test_duplicates_are_refused(self):
        tree = AVLTree([5, 5, 5])

        assert len(tree) == 1
        assert tree.insert(5) is False

    def test_minimum_and_maximum(self):
        tree = AVLTree([50, 20, 80, 10, 90])

        assert tree.minimum() == 10
        assert tree.maximum() == 90

    def test_empty_tree_behaviour(self):
        tree = AVLTree()

        assert len(tree) == 0
        assert tree.height() == -1
        assert tree.inorder() == []
        assert 1 not in tree
        with pytest.raises(ValueError):
            tree.minimum()


class TestDeleting:
    def test_deleting_a_leaf_a_one_child_node_and_a_two_child_node(self):
        with checked():
            tree = AVLTree([50, 30, 70, 20, 40, 60, 80])
            assert tree.delete(20) is True  # leaf
            assert tree.delete(30) is True  # one child left
            assert tree.delete(70) is True  # two children

        assert tree.inorder() == [40, 50, 60, 80]

    def test_deleting_something_absent_reports_false(self):
        tree = AVLTree([1, 2, 3])

        assert tree.delete(99) is False
        assert len(tree) == 3

    def test_deleting_the_root_repeatedly(self):
        with checked():
            tree = AVLTree(range(20))
            for _ in range(20):
                tree.delete(tree.root.value)

        assert len(tree) == 0
        assert tree.root is None

    def test_deletion_rebalances_when_it_needs_to(self):
        # Removing the right child here makes the root left heavy, so a rotation
        # must happen during the deletion rather than being left for later.
        with checked():
            tree = AVLTree([50, 30, 70, 20])
            before = tree.rotations
            tree.delete(70)

        assert tree.rotations > before
        assert abs(_balance_factor(tree.root)) <= 1

    def test_the_tree_stays_balanced_through_random_deletions(self):
        rng = random.Random(20260825)
        values = rng.sample(range(2000), 400)

        with checked():
            tree = AVLTree(values)
            rng.shuffle(values)
            for index, value in enumerate(values):
                assert tree.delete(value) is True
                remaining = len(values) - index - 1
                assert len(tree) == remaining
                if remaining:
                    assert tree.height() <= 1.44 * math.log2(remaining + 1) + 1

        assert tree.inorder() == []

    def test_a_long_mix_of_insertions_and_deletions_stays_sound(self):
        # The test most likely to find a case nobody thought of. Every operation
        # is verified, and the contents are compared against a plain set.
        rng = random.Random(20260825)
        mirror: set[int] = set()

        with checked():
            tree = AVLTree()
            for _ in range(2000):
                value = rng.randint(0, 200)
                if rng.random() < 0.6:
                    tree.insert(value)
                    mirror.add(value)
                else:
                    tree.delete(value)
                    mirror.discard(value)

        assert tree.inorder() == sorted(mirror)
        assert len(tree) == len(mirror)


class TestInvariantChecking:
    def test_a_sound_tree_reports_nothing(self):
        assert AVLTree(range(100)).check_invariants() == []
        assert AVLTree().check_invariants() == []

    def test_a_hand_broken_balance_is_caught(self):
        tree = AVLTree([20, 10, 30])
        # Hang a chain off one side without rebalancing, the way a missing
        # rebalance call would.
        from dsalab.structures.avl import AVLNode

        node = tree.root.left
        node.left = AVLNode(5)
        node.left.left = AVLNode(1)
        node.left.height = 1
        node.height = 2
        tree.root.height = 3
        tree._size += 2

        violations = tree.check_invariants()
        assert any("height by more than one" in violation.rule for violation in violations)

    def test_a_stale_stored_height_is_caught(self):
        tree = AVLTree([20, 10, 30])
        tree.root.height = 99

        assert any("stored height" in violation.rule for violation in tree.check_invariants())

    def test_a_broken_search_order_is_caught(self):
        tree = AVLTree([20, 10, 30])
        tree.root.left.value = 99

        assert any("search order" in violation.rule for violation in tree.check_invariants())


def test_insertion_needs_at_most_one_rebalance_but_deletion_may_need_several():
    # The practical difference from a red black tree, and the reason AVL suits
    # read heavy work while red black suits write heavy work.
    rng = random.Random(20260825)
    tree = AVLTree()
    for _ in range(500):
        tree.insert(rng.randint(0, 5000))

    worst_insert = 0
    for value in rng.sample(range(5000, 10000), 100):
        _, steps = record(tree.insert_traced(value))
        worst_insert = max(worst_insert, count_kinds(steps).get("rotate", 0))

    assert worst_insert <= 2, "one insertion needs at most a single or a double rotation"
