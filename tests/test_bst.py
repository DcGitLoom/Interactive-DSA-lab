"""Tests for the binary search tree and the invariant framework.

Most of these run inside `checked()`, so the tree verifies its own rules after
every single mutation. That turns a test of the answers into a test of the
structure: corruption is caught at the operation that caused it rather than
several operations later somewhere unrelated.
"""

import random

import pytest

from dsalab.complexity import detect
from dsalab.invariants import InvariantError, Violation, checked, is_checking, verify
from dsalab.structures.bst import BinarySearchTree, BSTNode
from dsalab.tracing import count_kinds, record


class TestInvariantFramework:
    def test_checking_is_off_by_default(self):
        assert is_checking() is False

    def test_checking_is_on_inside_the_block_and_off_afterwards(self):
        with checked():
            assert is_checking() is True
        assert is_checking() is False

    def test_it_is_switched_back_off_even_when_something_raises(self):
        with pytest.raises(RuntimeError):
            with checked():
                raise RuntimeError("something went wrong")

        assert is_checking() is False, "a failing test must not leave checking on for the rest"

    def test_nested_blocks_are_safe(self):
        with checked():
            with checked():
                assert is_checking() is True
            assert is_checking() is True
        assert is_checking() is False

    def test_verify_raises_with_every_broken_rule_listed(self):
        class Broken:
            def check_invariants(self):
                return [Violation("first rule", "it is broken"), Violation("second", "also broken")]

        with pytest.raises(InvariantError) as caught:
            verify(Broken())

        assert len(caught.value.violations) == 2
        assert "first rule" in str(caught.value)
        assert "also broken" in str(caught.value)

    def test_a_sound_structure_verifies_silently(self):
        verify(BinarySearchTree([5, 3, 8]))


class TestSearching:
    def test_finding_values_that_are_and_are_not_present(self):
        tree = BinarySearchTree([50, 30, 70, 20, 40, 60, 80])

        for value in [50, 30, 70, 20, 40, 60, 80]:
            assert value in tree
        for value in [10, 35, 55, 90]:
            assert value not in tree

    def test_searching_an_empty_tree(self):
        assert 5 not in BinarySearchTree()

    def test_a_search_visits_at_most_one_node_per_level(self):
        tree = BinarySearchTree([50, 30, 70, 20, 40, 60, 80])
        _, steps = record(tree.contains_traced(20))

        assert len(steps) == 3, "50, then 30, then found at 20"

    def test_minimum_and_maximum(self):
        tree = BinarySearchTree([50, 30, 70, 20, 80])

        assert tree.minimum() == 20
        assert tree.maximum() == 80

    def test_minimum_and_maximum_of_an_empty_tree_raise(self):
        with pytest.raises(ValueError):
            BinarySearchTree().minimum()
        with pytest.raises(ValueError):
            BinarySearchTree().maximum()


class TestOrderedQueries:
    """The operations a hash table cannot do at all, which is why trees exist."""

    def test_floor_and_ceiling_on_values_that_are_present(self):
        tree = BinarySearchTree([10, 20, 30, 40])

        assert tree.floor(20) == 20
        assert tree.ceiling(20) == 20

    def test_floor_and_ceiling_on_values_that_are_not(self):
        tree = BinarySearchTree([10, 20, 30, 40])

        assert tree.floor(25) == 20
        assert tree.ceiling(25) == 30

    def test_floor_below_everything_and_ceiling_above_everything_are_none(self):
        tree = BinarySearchTree([10, 20, 30])

        assert tree.floor(5) is None
        assert tree.ceiling(35) is None

    def test_floor_and_ceiling_agree_with_brute_force_on_random_trees(self):
        rng = random.Random(20260824)
        for _ in range(50):
            values = rng.sample(range(200), rng.randint(1, 40))
            tree = BinarySearchTree(values)

            for query in range(-5, 205, 7):
                below = [value for value in values if value <= query]
                above = [value for value in values if value >= query]

                assert tree.floor(query) == (max(below) if below else None)
                assert tree.ceiling(query) == (min(above) if above else None)

    def test_range_queries_return_sorted_values_within_the_range(self):
        tree = BinarySearchTree([50, 30, 70, 20, 40, 60, 80])

        assert tree.range_query(30, 60) == [30, 40, 50, 60]
        assert tree.range_query(0, 100) == [20, 30, 40, 50, 60, 70, 80]
        assert tree.range_query(41, 49) == []

    def test_range_queries_agree_with_brute_force(self):
        rng = random.Random(20260824)
        for _ in range(50):
            values = rng.sample(range(200), rng.randint(1, 40))
            tree = BinarySearchTree(values)
            low, high = sorted(rng.sample(range(200), 2))

            expected = sorted(value for value in values if low <= value <= high)
            assert tree.range_query(low, high) == expected


class TestInserting:
    def test_inorder_comes_out_sorted(self):
        with checked():
            tree = BinarySearchTree([50, 30, 70, 20, 40, 60, 80])

        assert tree.inorder() == [20, 30, 40, 50, 60, 70, 80]
        assert list(tree) == sorted([50, 30, 70, 20, 40, 60, 80])

    def test_duplicates_are_refused_and_reported(self):
        tree = BinarySearchTree([5])

        assert tree.insert(5) is False
        assert len(tree) == 1

    def test_size_tracks_successful_insertions_only(self):
        with checked():
            tree = BinarySearchTree()
            for value in [5, 3, 8, 3, 5]:
                tree.insert(value)

        assert len(tree) == 3

    def test_a_new_value_always_becomes_a_leaf(self):
        tree = BinarySearchTree([50, 30, 70])
        _, steps = record(tree.insert_traced(35))
        placement = [step for step in steps if step.kind == "insert"][0]

        assert placement.data["parent"] == 30
        assert placement.data["side"] == "right"

    def test_the_tree_stays_valid_through_many_random_insertions(self):
        rng = random.Random(20260824)
        with checked():
            tree = BinarySearchTree()
            for _ in range(300):
                tree.insert(rng.randint(0, 1000))

        assert tree.inorder() == sorted(set(tree.inorder()))


class TestDeleting:
    def test_deleting_a_leaf(self):
        with checked():
            tree = BinarySearchTree([50, 30, 70])
            assert tree.delete(30) is True

        assert tree.inorder() == [50, 70]

    def test_deleting_a_node_with_one_child(self):
        with checked():
            tree = BinarySearchTree([50, 30, 70, 20])
            assert tree.delete(30) is True

        assert tree.inorder() == [20, 50, 70]

    def test_deleting_a_node_with_two_children(self):
        with checked():
            tree = BinarySearchTree([50, 30, 70, 60, 80])
            assert tree.delete(70) is True

        assert tree.inorder() == [30, 50, 60, 80]

    def test_the_two_child_case_promotes_the_inorder_successor(self):
        # Deleting 50, whose right subtree is 70 with a left child 60. The
        # successor is the leftmost node of that subtree, so 60 and not 70.
        # Picking the right child directly instead of its leftmost descendant is
        # the usual wrong version of this, and it only breaks when the right
        # child has a left child of its own.
        tree = BinarySearchTree([50, 30, 70, 60, 80])
        _, steps = record(tree.delete_traced(50))
        replacement = [step for step in steps if step.kind == "replace"][0]

        assert replacement.data["successor"] == 60, "the successor is the right subtree's minimum"

    def test_the_successor_of_a_node_whose_right_child_has_no_left_child(self):
        # The other shape: here the right child itself is the minimum.
        tree = BinarySearchTree([50, 30, 70, 60, 80])
        _, steps = record(tree.delete_traced(70))
        replacement = [step for step in steps if step.kind == "replace"][0]

        assert replacement.data["successor"] == 80

    def test_deleting_the_root(self):
        with checked():
            tree = BinarySearchTree([50, 30, 70, 20, 40, 60, 80])
            tree.delete(50)

        assert tree.inorder() == [20, 30, 40, 60, 70, 80]
        assert tree.root.value == 60

    def test_deleting_the_only_node_empties_the_tree(self):
        with checked():
            tree = BinarySearchTree([5])
            tree.delete(5)

        assert tree.root is None
        assert len(tree) == 0
        assert tree.inorder() == []

    def test_deleting_something_absent_changes_nothing(self):
        tree = BinarySearchTree([5, 3, 8])

        assert tree.delete(99) is False
        assert len(tree) == 3

    def test_deleting_from_an_empty_tree(self):
        assert BinarySearchTree().delete(1) is False

    def test_deleting_every_value_in_a_random_order_leaves_a_valid_tree_throughout(self):
        # The strongest test in this file. Every intermediate state is checked,
        # so a deletion that corrupts the tree is caught at that deletion rather
        # than showing up as a wrong answer much later.
        rng = random.Random(20260824)
        for _ in range(30):
            values = rng.sample(range(500), 40)
            with checked():
                tree = BinarySearchTree(values)
                rng.shuffle(values)

                for index, value in enumerate(values):
                    assert tree.delete(value) is True
                    remaining = sorted(values[index + 1 :])
                    assert tree.inorder() == remaining
                    assert len(tree) == len(remaining)

    def test_deleting_and_reinserting_keeps_the_tree_sound(self):
        with checked():
            tree = BinarySearchTree([50, 30, 70, 20, 40])
            tree.delete(30)
            tree.insert(35)
            tree.delete(50)
            tree.insert(45)

        assert tree.inorder() == [20, 35, 40, 45, 70]


class TestInvariantChecking:
    def test_a_hand_corrupted_tree_is_caught(self):
        tree = BinarySearchTree([50, 30, 70])
        tree.root.left.value = 99  # 99 is not smaller than 50, so the rule breaks

        violations = tree.check_invariants()
        assert violations
        assert "smaller" in violations[0].rule

    def test_a_violation_two_levels_up_is_caught(self):
        # The subtle case. 60 is correctly larger than its parent 40, but it sits
        # in the left subtree of 50, where everything must be below 50. Checking
        # each node only against its immediate children would miss this entirely,
        # which is the classic wrong way to write this check.
        tree = BinarySearchTree([50, 30, 70])
        tree.root.left.right = BSTNode(60)
        tree.root.left.value = 40
        tree._size += 1

        violations = tree.check_invariants()
        assert any("smaller" in violation.rule for violation in violations)

    def test_a_wrong_size_is_caught(self):
        tree = BinarySearchTree([1, 2, 3])
        tree._size = 99

        assert any("size" in violation.rule for violation in tree.check_invariants())

    def test_a_sound_tree_reports_nothing(self):
        assert BinarySearchTree([50, 30, 70, 20, 40]).check_invariants() == []
        assert BinarySearchTree().check_invariants() == []


class TestTheDegenerateCase:
    """The honest weakness, measured rather than described."""

    def test_sorted_input_produces_a_chain(self):
        tree = BinarySearchTree(range(20))

        assert tree.height() == 19, "every node ended up to the right of the last"
        assert len(tree) == 20

    def test_reverse_sorted_input_is_just_as_bad(self):
        tree = BinarySearchTree(range(20, 0, -1))

        assert tree.height() == 19

    def test_the_height_of_a_sorted_insertion_grows_linearly(self):
        # Handing the measurements to the complexity detective from day 3 turns
        # "sorted input is bad" from a warning into a measurement. This is the
        # argument for the AVL tree tomorrow.
        sizes = [50, 100, 200, 400, 800]
        heights = [float(BinarySearchTree(range(n)).height()) for n in sizes]
        verdict = detect(sizes, heights)

        assert verdict.best.curve == "O(n)"
        assert verdict.is_confident

    def test_random_input_stays_close_to_logarithmic(self):
        # The good case, for contrast. Random insertion order gives an expected
        # height of about 4.3 times log2(n), which is a constant factor away from
        # the ideal and therefore still O(log n).
        rng = random.Random(20260824)
        sizes = [64, 128, 256, 512, 1024, 2048]
        heights = []
        for n in sizes:
            values = list(range(n))
            rng.shuffle(values)
            heights.append(float(BinarySearchTree(values).height()))

        verdict = detect(sizes, heights)
        assert verdict.best.curve in {"O(log n)", "O(n log n)"}, (
            f"random insertion should stay near logarithmic, got {verdict.best.curve}"
        )

    def test_a_degenerate_tree_can_still_be_traversed_without_a_stack_overflow(self):
        # 5000 nodes deep is well past Python's recursion limit, so a recursive
        # traversal would crash here. This is why inorder and height are written
        # iteratively.
        tree = BinarySearchTree(range(5000))

        assert tree.inorder() == list(range(5000))
        assert tree.height() == 4999


def test_every_step_carries_a_readable_note():
    tree = BinarySearchTree([50, 30, 70])
    _, steps = record(tree.insert_traced(40))

    assert all(step.note.endswith(".") for step in steps)
    assert count_kinds(steps)["insert"] == 1
