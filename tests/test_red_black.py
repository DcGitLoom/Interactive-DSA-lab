"""Tests for the red black tree.

This is the hardest structure in the project to get right, and it is also the one
where a bug is least visible: a tree can hold every value in the correct order
while its colours are nonsense, and it will keep answering lookups correctly right
up until the balance quietly degrades to a chain.

So almost every test here runs with invariant checking on, which verifies all five
rules plus the search order plus the parent pointers after every single operation.
"""

import math
import random

import pytest

from dsalab.complexity import detect
from dsalab.invariants import checked
from dsalab.structures.red_black import BLACK, RED, RedBlackTree
from dsalab.tracing import count_kinds, record


class TestTheFiveRules:
    def test_the_root_is_always_black(self):
        tree = RedBlackTree([10])
        assert tree.root.colour == BLACK

        with checked():
            tree = RedBlackTree(range(50))
        assert tree.root.colour == BLACK

    def test_a_new_node_arrives_red(self):
        tree = RedBlackTree([10])
        _, steps = record(tree.insert_traced(20))
        arrival = [step for step in steps if step.kind == "insert"][0]

        assert arrival.data["colour"] == RED

    def test_no_red_node_ever_has_a_red_child(self):
        rng = random.Random(20260826)
        tree = RedBlackTree()
        for _ in range(500):
            tree.insert(rng.randint(0, 2000))

        def check(node):
            if node is tree.nil:
                return
            if node.is_red:
                assert not node.left.is_red and not node.right.is_red
            check(node.left)
            check(node.right)

        check(tree.root)

    def test_every_path_has_the_same_number_of_black_nodes(self):
        rng = random.Random(20260826)
        tree = RedBlackTree()
        for _ in range(300):
            tree.insert(rng.randint(0, 2000))

        depths = set()

        def walk(node, blacks):
            if node is tree.nil:
                depths.add(blacks + 1)
                return
            walk(node.left, blacks + (0 if node.is_red else 1))
            walk(node.right, blacks + (0 if node.is_red else 1))

        walk(tree.root, 0)
        assert len(depths) == 1, f"paths disagree on black height: {sorted(depths)}"

    def test_the_invariant_check_names_the_rule_that_broke(self):
        tree = RedBlackTree(range(10))
        tree.root.colour = RED

        violations = tree.check_invariants()
        assert any("rule 2" in violation.rule for violation in violations)

    def test_a_hand_made_red_red_clash_is_caught(self):
        tree = RedBlackTree(range(10))
        node = tree.root
        while not node.left.is_red and node.left is not tree.nil:
            node = node.left
        node.colour = RED
        node.left.colour = RED

        assert any("rule 4" in violation.rule for violation in tree.check_invariants())

    def test_a_broken_black_height_is_caught(self):
        tree = RedBlackTree(range(20))
        # Repainting one black node red removes a black from every path through
        # it, which breaks rule 5 without touching any value or pointer.
        node = tree.root.left
        if not node.is_red:
            node.colour = RED
            assert any("rule 5" in violation.rule or "rule 4" in violation.rule
                       for violation in tree.check_invariants())

    def test_a_broken_parent_pointer_is_caught(self):
        tree = RedBlackTree(range(10))
        tree.root.left.parent = tree.root.right

        assert any("parent pointer" in violation.rule for violation in tree.check_invariants())


class TestBalance:
    def test_sorted_input_stays_short(self):
        with checked():
            tree = RedBlackTree(range(1000))

        assert tree.height() <= 2 * math.log2(1001)

    def test_the_height_bound_holds_at_many_sizes(self):
        rng = random.Random(20260826)
        for size in (1, 2, 5, 20, 100, 500, 2000):
            values = list(range(size))
            rng.shuffle(values)
            tree = RedBlackTree(values)

            assert tree.height() <= 2 * math.log2(size + 1)

    def test_the_height_grows_logarithmically(self):
        sizes = [64, 128, 256, 512, 1024, 2048]
        heights = [float(RedBlackTree(range(n)).height()) for n in sizes]
        verdict = detect(sizes, heights)

        assert verdict.best.curve == "O(log n)"

    def test_it_is_taller_than_an_avl_tree_on_the_same_input(self):
        # The trade being made: a looser height bound in exchange for cheaper
        # repairs. Sorted input is where the difference shows most clearly.
        from dsalab.structures.avl import AVLTree

        red_black = RedBlackTree(range(2000))
        avl = AVLTree(range(2000))

        assert red_black.height() >= avl.height()


class TestRotationCounts:
    def test_an_insertion_never_needs_more_than_two_rotations(self):
        # The headline promise of the structure, and the reason it is preferred
        # over AVL for write heavy work.
        rng = random.Random(20260826)
        tree = RedBlackTree()
        for _ in range(200):
            tree.insert(rng.randint(0, 10000))

        for value in rng.sample(range(20000, 40000), 300):
            _, steps = record(tree.insert_traced(value))
            assert count_kinds(steps).get("rotate", 0) <= 2

    def test_a_deletion_never_needs_more_than_three_rotations(self):
        rng = random.Random(20260826)
        values = rng.sample(range(10000), 500)
        tree = RedBlackTree(values)

        rng.shuffle(values)
        for value in values[:300]:
            _, steps = record(tree.delete_traced(value))
            assert count_kinds(steps).get("rotate", 0) <= 3

    def test_one_insertion_can_recolour_many_times_but_never_rotate_many_times(self):
        # The part people miss. The red uncle case repairs things without moving
        # a single pointer and pushes the problem two levels up, so a single
        # insertion can recolour O(log n) times. Rotations stay capped at two,
        # because a rotation always ends the loop.
        #
        # Worth noting what this test does not claim: on sorted input almost
        # every insertion hits the black uncle case and rotates, so the totals
        # come out roughly equal. The asymmetry is per operation, not in total,
        # which is why this counts the worst single insertion instead.
        rng = random.Random(20260826)
        tree = RedBlackTree()
        worst_recolouring = worst_rotation = 0

        for _ in range(2000):
            _, steps = record(tree.insert_traced(rng.randint(0, 100000)))
            counts = count_kinds(steps)
            worst_recolouring = max(worst_recolouring, counts.get("recolour", 0))
            worst_rotation = max(worst_rotation, counts.get("rotate", 0))

        assert worst_rotation <= 2
        assert worst_recolouring > worst_rotation, (
            "a single insertion should be able to recolour more times than it can rotate"
        )


class TestOrdinaryBehaviour:
    def test_inorder_comes_out_sorted(self):
        rng = random.Random(20260826)
        values = rng.sample(range(5000), 400)

        with checked():
            tree = RedBlackTree(values)

        assert tree.inorder() == sorted(values)

    def test_lookup(self):
        tree = RedBlackTree(range(0, 200, 2))

        assert 100 in tree
        assert 101 not in tree

    def test_duplicates_are_refused(self):
        tree = RedBlackTree([5, 5, 5])

        assert len(tree) == 1

    def test_minimum_and_maximum(self):
        tree = RedBlackTree([50, 20, 80, 10, 90])

        assert tree.minimum() == 10
        assert tree.maximum() == 90

    def test_an_empty_tree(self):
        tree = RedBlackTree()

        assert len(tree) == 0
        assert tree.inorder() == []
        assert tree.height() == -1
        assert 1 not in tree
        assert tree.check_invariants() == []
        with pytest.raises(ValueError):
            tree.minimum()


class TestDeleting:
    def test_deleting_leaves_one_child_and_two_child_nodes(self):
        with checked():
            tree = RedBlackTree([50, 30, 70, 20, 40, 60, 80])
            assert tree.delete(20) is True
            assert tree.delete(30) is True
            assert tree.delete(50) is True

        assert tree.inorder() == [40, 60, 70, 80]

    def test_deleting_something_absent(self):
        tree = RedBlackTree([1, 2, 3])

        assert tree.delete(99) is False
        assert len(tree) == 3

    def test_deleting_the_only_node(self):
        with checked():
            tree = RedBlackTree([5])
            tree.delete(5)

        assert len(tree) == 0
        assert tree.root is tree.nil

    def test_deleting_every_value_in_random_order_keeps_all_five_rules(self):
        rng = random.Random(20260826)
        values = rng.sample(range(5000), 300)

        with checked():
            tree = RedBlackTree(values)
            rng.shuffle(values)
            for index, value in enumerate(values):
                assert tree.delete(value) is True
                assert tree.inorder() == sorted(values[index + 1 :])

        assert len(tree) == 0

    def test_a_long_mix_of_insertions_and_deletions_against_a_plain_set(self):
        rng = random.Random(20260826)
        mirror: set[int] = set()

        with checked():
            tree = RedBlackTree()
            for _ in range(2000):
                value = rng.randint(0, 150)
                if rng.random() < 0.55:
                    tree.insert(value)
                    mirror.add(value)
                else:
                    tree.delete(value)
                    mirror.discard(value)

        assert tree.inorder() == sorted(mirror)
        assert len(tree) == len(mirror)

    def test_deleting_in_sorted_order_which_is_the_worst_case_for_the_fixup(self):
        with checked():
            tree = RedBlackTree(range(200))
            for value in range(200):
                tree.delete(value)

        assert len(tree) == 0

    def test_deleting_in_reverse_sorted_order(self):
        with checked():
            tree = RedBlackTree(range(200))
            for value in range(199, -1, -1):
                tree.delete(value)

        assert len(tree) == 0
