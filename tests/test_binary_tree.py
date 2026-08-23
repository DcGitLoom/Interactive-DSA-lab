"""Tests for binary trees, their traversals, and threaded trees.

The recurring check here is that the recursive and iterative versions of each
traversal agree. They are two implementations of one definition, so any
disagreement means one of them is wrong, and random trees are thrown at them to
find cases neither was written with in mind.
"""

import random

import pytest

from dsalab.structures.binary_tree import ArrayBinaryTree, BinaryTree, TreeNode
from dsalab.structures.threaded_tree import ThreadedBinaryTree
from dsalab.tracing import record

#         1
#       /   \
#      2     3
#     / \     \
#    4   5     6
SAMPLE = [1, 2, 3, 4, 5, None, 6]


def random_tree(nodes: int, rng: random.Random) -> BinaryTree:
    """Build a random shaped tree with distinct values, for cross checking."""
    values = list(range(nodes))
    rng.shuffle(values)
    if not values:
        return BinaryTree()

    root = TreeNode(values[0])
    placed = [root]
    for value in values[1:]:
        parent = rng.choice(placed)
        while parent.left is not None and parent.right is not None:
            parent = rng.choice(placed)
        node = TreeNode(value)
        if parent.left is None and (parent.right is not None or rng.random() < 0.5):
            parent.left = node
        else:
            parent.right = node
        placed.append(node)
    return BinaryTree(root)


class TestBuilding:
    def test_a_tree_can_be_built_from_a_level_order_list(self):
        tree = BinaryTree.from_level_order(SAMPLE)

        assert tree.root.value == 1
        assert tree.root.left.value == 2
        assert tree.root.right.value == 3
        assert tree.root.right.left is None
        assert tree.root.right.right.value == 6

    def test_an_empty_list_builds_an_empty_tree(self):
        assert BinaryTree.from_level_order([]).root is None
        assert BinaryTree.from_level_order([None]).root is None


class TestTraversals:
    def test_the_four_traversals_of_the_sample_tree(self):
        tree = BinaryTree.from_level_order(SAMPLE)

        assert tree.preorder() == [1, 2, 4, 5, 3, 6]
        assert tree.inorder() == [4, 2, 5, 1, 3, 6]
        assert tree.postorder() == [4, 5, 2, 6, 3, 1]
        assert tree.level_order() == [1, 2, 3, 4, 5, 6]

    def test_every_traversal_of_an_empty_tree_is_empty(self):
        tree = BinaryTree()

        assert tree.preorder() == []
        assert tree.inorder() == []
        assert tree.postorder() == []
        assert tree.level_order() == []
        assert tree.preorder_iterative() == []
        assert tree.inorder_iterative() == []
        assert tree.postorder_iterative() == []

    def test_every_traversal_of_a_single_node_is_that_node(self):
        tree = BinaryTree(TreeNode(7))

        assert tree.preorder() == tree.inorder() == tree.postorder() == [7]
        assert tree.level_order() == [7]

    def test_preorder_starts_at_the_root_and_postorder_ends_there(self):
        tree = BinaryTree.from_level_order(SAMPLE)

        assert tree.preorder()[0] == 1
        assert tree.postorder()[-1] == 1

    def test_inorder_of_a_search_tree_comes_out_sorted(self):
        # The single most useful fact about inorder, and what makes search trees
        # more than just fast lookup.
        values = [50, 30, 70, 20, 40, 60, 80]
        root = TreeNode(50)
        root.left = TreeNode(30, TreeNode(20), TreeNode(40))
        root.right = TreeNode(70, TreeNode(60), TreeNode(80))

        assert BinaryTree(root).inorder() == sorted(values)

    def test_levels_are_grouped_correctly(self):
        assert BinaryTree.from_level_order(SAMPLE).levels() == [[1], [2, 3], [4, 5, 6]]

    def test_levels_of_an_empty_tree(self):
        assert BinaryTree().levels() == []

    @pytest.mark.parametrize("size", [1, 2, 3, 7, 20, 50])
    def test_recursive_and_iterative_traversals_always_agree(self, size):
        rng = random.Random(20260823 + size)
        for _ in range(20):
            tree = random_tree(size, rng)

            assert tree.preorder() == tree.preorder_iterative()
            assert tree.inorder() == tree.inorder_iterative()
            assert tree.postorder() == tree.postorder_iterative()

    def test_every_traversal_visits_every_node_exactly_once(self):
        tree = BinaryTree.from_level_order(SAMPLE)

        for traversal in (tree.preorder, tree.inorder, tree.postorder, tree.level_order):
            visited = traversal()
            assert sorted(visited) == [1, 2, 3, 4, 5, 6]
            assert len(visited) == len(set(visited))

    def test_traversals_report_one_step_per_node(self):
        tree = BinaryTree.from_level_order(SAMPLE)
        _, steps = record(tree.inorder_traced())

        assert len(steps) == 6

    def test_a_left_leaning_chain_still_traverses_correctly(self):
        # A degenerate tree is really a linked list, and it is where iterative
        # traversals with a stack are most likely to be wrong.
        root = TreeNode(1)
        node = root
        for value in range(2, 10):
            node.left = TreeNode(value)
            node = node.left
        tree = BinaryTree(root)

        assert tree.inorder() == list(range(9, 0, -1))
        assert tree.inorder_iterative() == list(range(9, 0, -1))
        assert tree.height() == 8

    def test_a_right_leaning_chain_still_traverses_correctly(self):
        root = TreeNode(1)
        node = root
        for value in range(2, 10):
            node.right = TreeNode(value)
            node = node.right
        tree = BinaryTree(root)

        assert tree.inorder() == list(range(1, 10))
        assert tree.postorder_iterative() == list(range(9, 0, -1))


class TestMeasurements:
    def test_height_counts_edges_so_an_empty_tree_is_minus_one(self):
        assert BinaryTree().height() == -1
        assert BinaryTree(TreeNode(1)).height() == 0
        assert BinaryTree.from_level_order(SAMPLE).height() == 2

    def test_size_and_leaf_counts(self):
        tree = BinaryTree.from_level_order(SAMPLE)

        assert tree.size() == 6 == len(tree)
        assert tree.leaves() == 3
        assert tree.internal_nodes() == 3

    def test_a_full_tree_is_recognised(self):
        # Every node has zero or two children.
        assert BinaryTree.from_level_order([1, 2, 3]).is_full()
        assert BinaryTree.from_level_order([1, 2, 3, 4, 5]).is_full()

    def test_a_tree_with_a_lone_child_is_not_full(self):
        assert not BinaryTree.from_level_order(SAMPLE).is_full()

    def test_in_a_full_tree_leaves_are_one_more_than_two_child_nodes(self):
        # A property worth knowing, and a good check that is_full is honest.
        rng = random.Random(20260823)
        for _ in range(50):
            tree = random_tree(rng.randint(1, 30), rng)
            if not tree.is_full():
                continue
            two_child_nodes = sum(
                1 for value in tree.preorder() if _node_with(tree.root, value).children == 2
            )
            assert tree.leaves() == two_child_nodes + 1

    def test_a_perfect_tree_is_recognised(self):
        assert BinaryTree.from_level_order([1, 2, 3]).is_perfect()
        assert BinaryTree.from_level_order([1, 2, 3, 4, 5, 6, 7]).is_perfect()
        assert not BinaryTree.from_level_order([1, 2, 3, 4]).is_perfect()

    def test_balance_checking(self):
        assert BinaryTree.from_level_order([1, 2, 3]).is_balanced()
        assert BinaryTree.from_level_order(SAMPLE).is_balanced()

        chain = TreeNode(1, TreeNode(2, TreeNode(3, TreeNode(4))))
        assert not BinaryTree(chain).is_balanced()

    def test_an_empty_tree_is_balanced_and_perfect(self):
        assert BinaryTree().is_balanced()
        assert BinaryTree().is_perfect()

    def test_mirroring_reverses_the_inorder_traversal(self):
        tree = BinaryTree.from_level_order(SAMPLE)
        before = tree.inorder()
        tree.mirror()

        assert tree.inorder() == before[::-1]

    def test_mirroring_twice_restores_the_original(self):
        tree = BinaryTree.from_level_order(SAMPLE)
        before = tree.level_order()
        tree.mirror()
        tree.mirror()

        assert tree.level_order() == before


def _node_with(node, value):
    """Find the node holding a value, for tests that need the node itself."""
    if node is None:
        return None
    if node.value == value:
        return node
    return _node_with(node.left, value) or _node_with(node.right, value)


class TestRebuildingFromTraversals:
    def test_preorder_and_inorder_rebuild_the_original_tree(self):
        tree = BinaryTree.from_level_order(SAMPLE)
        rebuilt = BinaryTree.from_preorder_and_inorder(tree.preorder(), tree.inorder())

        assert rebuilt.preorder() == tree.preorder()
        assert rebuilt.inorder() == tree.inorder()
        assert rebuilt.level_order() == tree.level_order()

    def test_it_rebuilds_random_trees_exactly(self):
        rng = random.Random(20260823)
        for _ in range(100):
            tree = random_tree(rng.randint(1, 40), rng)
            rebuilt = BinaryTree.from_preorder_and_inorder(tree.preorder(), tree.inorder())

            assert rebuilt.level_order() == tree.level_order()

    def test_rebuilding_an_empty_tree(self):
        assert BinaryTree.from_preorder_and_inorder([], []).root is None

    def test_mismatched_traversals_are_rejected(self):
        with pytest.raises(ValueError):
            BinaryTree.from_preorder_and_inorder([1, 2], [1])
        with pytest.raises(ValueError):
            BinaryTree.from_preorder_and_inorder([1, 2], [1, 3])

    def test_duplicate_values_are_rejected_rather_than_guessed_at(self):
        # With repeated values the split point is ambiguous, so more than one
        # tree fits. Refusing is better than silently picking one.
        with pytest.raises(ValueError, match="distinct"):
            BinaryTree.from_preorder_and_inorder([1, 1], [1, 1])


class TestArrayBinaryTree:
    def test_the_index_arithmetic(self):
        assert ArrayBinaryTree.left_of(0) == 1
        assert ArrayBinaryTree.right_of(0) == 2
        assert ArrayBinaryTree.parent_of(1) == 0
        assert ArrayBinaryTree.parent_of(2) == 0
        assert ArrayBinaryTree.parent_of(6) == 2

    def test_the_root_has_no_parent(self):
        with pytest.raises(IndexError):
            ArrayBinaryTree.parent_of(0)

    def test_children_and_parents_are_inverses_of_each_other(self):
        for index in range(1, 100):
            parent = ArrayBinaryTree.parent_of(index)
            assert index in (ArrayBinaryTree.left_of(parent), ArrayBinaryTree.right_of(parent))

    def test_it_converts_to_the_pointer_form_with_the_same_traversals(self):
        array_tree = ArrayBinaryTree([1, 2, 3, 4, 5, None, 6])
        linked = array_tree.to_linked()

        assert linked.level_order() == [1, 2, 3, 4, 5, 6]
        assert linked.inorder() == [4, 2, 5, 1, 3, 6]

    def test_length_counts_only_filled_slots(self):
        assert len(ArrayBinaryTree([1, 2, None, 4])) == 3


class TestThreadedBinaryTree:
    def test_inorder_comes_out_sorted_without_any_stack(self):
        tree = ThreadedBinaryTree([50, 30, 70, 20, 40, 60, 80])

        assert tree.inorder() == [20, 30, 40, 50, 60, 70, 80]

    def test_it_matches_an_ordinary_traversal_on_random_input(self):
        rng = random.Random(20260823)
        for _ in range(50):
            values = rng.sample(range(1000), rng.randint(1, 60))
            tree = ThreadedBinaryTree(values)

            assert tree.inorder() == sorted(values)

    def test_reverse_inorder_walks_the_other_way(self):
        tree = ThreadedBinaryTree([50, 30, 70, 20, 40])

        assert tree.reverse_inorder() == [70, 50, 40, 30, 20]

    def test_an_empty_tree_traverses_to_nothing(self):
        assert ThreadedBinaryTree().inorder() == []
        assert ThreadedBinaryTree().reverse_inorder() == []

    def test_a_single_node(self):
        tree = ThreadedBinaryTree([5])

        assert tree.inorder() == [5]
        assert len(tree) == 1

    def test_lookup_does_not_wander_off_along_a_thread(self):
        # The characteristic threaded tree bug: following a thread as though it
        # were a child, which loops forever or reports values that are not there.
        tree = ThreadedBinaryTree([50, 30, 70, 20, 40, 60, 80])

        for value in [50, 30, 70, 20, 40, 60, 80]:
            assert value in tree
        for missing in [10, 35, 55, 90, 100]:
            assert missing not in tree

    def test_duplicates_are_ignored_rather_than_inserted_twice(self):
        tree = ThreadedBinaryTree([5, 5, 5])

        assert len(tree) == 1
        assert tree.inorder() == [5]

    def test_a_sorted_insertion_order_becomes_a_chain_and_still_works(self):
        tree = ThreadedBinaryTree(list(range(20)))

        assert tree.inorder() == list(range(20))

    def test_there_are_always_n_plus_one_threads(self):
        # This count is the entire justification for the structure: those are the
        # pointers an ordinary binary tree leaves null.
        for size in (1, 2, 5, 17, 40):
            rng = random.Random(size)
            tree = ThreadedBinaryTree(rng.sample(range(1000), size))

            assert tree.wasted_pointer_count() == size + 1

    def test_successor_walks_the_tree_in_sorted_order(self):
        tree = ThreadedBinaryTree([50, 30, 70, 20, 40, 60, 80])
        node = tree._leftmost(tree.root)
        walked = []

        while node is not None:
            walked.append(node.value)
            node = tree.successor(node)

        assert walked == sorted([50, 30, 70, 20, 40, 60, 80])
