"""Tests for greedy, dynamic programming, backtracking and branch and bound.

The theme of this file is that **an algorithm being fast is not the same as it
being right**, so nearly everything here is checked against brute force. Where
greedy is optimal there is a proof sketch in the code and a test comparing it with
an exhaustive search; where it is not, there is a test that finds a failing input.
"""

import itertools
import random

import pytest

from dsalab.algorithms.backtracking import (
    graph_colouring,
    hamiltonian_cycle,
    n_queens,
    n_queens_traced,
    permutations,
    rat_in_a_maze,
    solve_sudoku,
    solve_sudoku_traced,
    subsets,
)
from dsalab.algorithms.branch_and_bound import (
    knapsack_branch_and_bound,
    knapsack_branch_and_bound_traced,
    spanning_tree_tour_bound,
    travelling_salesman,
    travelling_salesman_brute_force,
    travelling_salesman_traced,
)
from dsalab.algorithms.dp import (
    coin_change,
    edit_distance,
    fibonacci_bottom_up,
    knapsack_01,
    longest_common_subsequence,
    longest_increasing_subsequence,
    matrix_chain_order,
    subset_sum,
)
from dsalab.algorithms.greedy import (
    Activity,
    Item,
    activity_selection,
    activity_selection_traced,
    fixed_width_length,
    fractional_knapsack,
    greedy_coin_change,
    huffman_coding,
    huffman_decode,
    huffman_encoded_length,
)
from dsalab.algorithms.greedy_vs_dp import (
    find_activity_selection_counterexample,
    find_coin_change_counterexample,
    find_coin_change_failure,
    find_knapsack_counterexample,
)
from dsalab.structures.graph import Graph
from dsalab.tracing import count_kinds, record


class TestActivitySelection:
    def test_a_worked_example(self):
        activities = [
            Activity("a", 1, 4), Activity("b", 3, 5), Activity("c", 0, 6),
            Activity("d", 5, 7), Activity("e", 8, 9), Activity("f", 5, 9),
        ]

        chosen = activity_selection(activities)
        assert [item.name for item in chosen] == ["a", "d", "e"]

    def test_it_matches_an_exhaustive_search(self):
        # Greedy is provably optimal here, so it must always tie with brute force.
        rng = random.Random(20260903)

        for _ in range(200):
            count = rng.randint(1, 8)
            activities = []
            for index in range(count):
                start = rng.randint(0, 15)
                activities.append(Activity(f"a{index}", start, start + rng.randint(1, 6)))

            def compatible(chosen):
                ordered = sorted(chosen, key=lambda item: item.start)
                return all(
                    ordered[i].finish <= ordered[i + 1].start for i in range(len(ordered) - 1)
                )

            best = 0
            for size in range(len(activities), 0, -1):
                if any(compatible(combo) for combo in itertools.combinations(activities, size)):
                    best = size
                    break

            assert len(activity_selection(activities)) == best

    def test_nothing_to_schedule(self):
        assert activity_selection([]) == []

    def test_activities_that_all_overlap_give_one(self):
        activities = [Activity(f"a{i}", 0, 10) for i in range(5)]

        assert len(activity_selection(activities)) == 1

    def test_it_skips_what_it_cannot_fit(self):
        activities = [Activity("short", 0, 2), Activity("clash", 1, 3), Activity("late", 3, 4)]
        _, steps = record(activity_selection_traced(activities))

        assert count_kinds(steps)["skip"] == 1
        assert count_kinds(steps)["take"] == 2


class TestFractionalKnapsack:
    def test_it_fills_the_bag_with_the_densest_items(self):
        items = [Item("gold", 10, 60), Item("silver", 20, 100), Item("bronze", 30, 120)]
        total, taken = fractional_knapsack(items, 50)

        assert total == pytest.approx(240)
        assert taken["gold"] == 1.0 and taken["silver"] == 1.0
        assert taken["bronze"] == pytest.approx(2 / 3)

    def test_an_empty_bag_holds_nothing(self):
        assert fractional_knapsack([Item("a", 1, 1)], 0) == (0.0, {})

    def test_a_negative_capacity_is_rejected(self):
        with pytest.raises(ValueError):
            fractional_knapsack([], -1)

    def test_it_is_never_worse_than_the_whole_item_version(self):
        # Allowing fractions can only help, which is exactly why the fractional
        # answer is a valid upper bound for branch and bound.
        rng = random.Random(20260903)

        for _ in range(200):
            count = rng.randint(1, 6)
            capacity = rng.randint(1, 15)
            items = [
                Item(f"i{index}", float(rng.randint(1, 10)), float(rng.randint(1, 20)))
                for index in range(count)
            ]

            fractional, _ = fractional_knapsack(items, capacity)
            whole, _ = knapsack_01(items, capacity)

            assert fractional >= whole - 1e-9


class TestHuffman:
    def test_rarer_symbols_get_longer_codes(self):
        codes, _ = huffman_coding("aaaaaaaabbbbccd")

        assert len(codes["a"]) <= len(codes["b"]) <= len(codes["c"])
        assert len(codes["a"]) < len(codes["d"])

    def test_no_code_is_a_prefix_of_another(self):
        # The property that makes decoding possible with no separators.
        rng = random.Random(20260903)

        for _ in range(100):
            text = "".join(rng.choice("abcdefg") for _ in range(rng.randint(2, 100)))
            codes, _ = huffman_coding(text)

            for first in codes.values():
                for second in codes.values():
                    if first is not second:
                        assert not second.startswith(first), (
                            f"{second!r} starts with {first!r}, so decoding is ambiguous"
                        )

    def test_encoding_then_decoding_returns_the_original(self):
        rng = random.Random(20260903)

        for _ in range(100):
            text = "".join(rng.choice("abcde ") for _ in range(rng.randint(1, 80)))
            codes, root = huffman_coding(text)
            bits = "".join(codes[character] for character in text)

            assert huffman_decode(bits, root) == text

    def test_it_beats_fixed_width_codes_on_uneven_text(self):
        text = "a" * 100 + "b" * 20 + "c" * 5 + "d"

        assert huffman_encoded_length(text) < fixed_width_length(text)

    def test_evenly_distributed_text_gains_nothing(self):
        # Huffman wins on unevenness. With four equally common symbols, two bits
        # each is already optimal and there is nothing to gain.
        text = "abcd" * 25

        assert huffman_encoded_length(text) == fixed_width_length(text)

    def test_a_single_repeated_symbol_still_needs_one_bit(self):
        codes, root = huffman_coding("aaaa")

        assert codes == {"a": "0"}
        assert huffman_decode("0000", root) == "aaaa"

    def test_empty_text(self):
        assert huffman_coding("") == ({}, None)

    def test_the_result_is_deterministic(self):
        # Ties between equally frequent symbols must break the same way every run,
        # or the codes change between runs while staying equally optimal, and
        # nothing can be tested.
        text = "abcdabcd"

        assert huffman_coding(text)[0] == huffman_coding(text)[0]


class TestDynamicProgramming:
    def test_longest_increasing_subsequence(self):
        # [2, 3, 7, 18] and [2, 5, 7, 101] are both length four and both correct,
        # so the test checks the length rather than picking a favourite. Same
        # principle as the topological sort tests on day 18: where the answer is
        # valid rather than unique, test the property.
        assert len(longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18])) == 4
        assert longest_increasing_subsequence([]) == []
        assert longest_increasing_subsequence([5]) == [5]
        assert len(longest_increasing_subsequence([5, 4, 3, 2, 1])) == 1
        assert longest_increasing_subsequence([1, 2, 3]) == [1, 2, 3]

    def test_the_subsequence_really_is_increasing_and_in_order(self):
        rng = random.Random(20260903)

        for _ in range(200):
            values = [rng.randint(0, 50) for _ in range(rng.randint(0, 25))]
            result = longest_increasing_subsequence(values)

            assert all(result[i] < result[i + 1] for i in range(len(result) - 1))

            # It must appear in the original order, not merely be a sorted subset.
            position = -1
            for value in result:
                position = values.index(value, position + 1)

    def test_its_length_matches_an_exhaustive_search(self):
        rng = random.Random(20260903)

        for _ in range(50):
            values = [rng.randint(0, 20) for _ in range(rng.randint(1, 12))]

            best = 0
            for size in range(len(values), 0, -1):
                for combo in itertools.combinations(values, size):
                    if all(combo[i] < combo[i + 1] for i in range(len(combo) - 1)):
                        best = max(best, size)
                if best:
                    break

            assert len(longest_increasing_subsequence(values)) == best

    def test_knapsack_matches_an_exhaustive_search(self):
        rng = random.Random(20260903)

        for _ in range(200):
            count = rng.randint(1, 8)
            capacity = rng.randint(1, 15)
            items = [
                Item(f"i{index}", float(rng.randint(1, 8)), float(rng.randint(1, 20)))
                for index in range(count)
            ]

            best = 0.0
            for size in range(count + 1):
                for combo in itertools.combinations(items, size):
                    if sum(item.weight for item in combo) <= capacity:
                        best = max(best, sum(item.value for item in combo))

            value, _ = knapsack_01(items, capacity)
            assert value == pytest.approx(best)

    def test_the_knapsack_reports_which_items_it_took(self):
        items = [Item("a", 3, 4), Item("b", 4, 5), Item("c", 2, 3)]
        value, chosen = knapsack_01(items, 6)

        taken = [item for item in items if item.name in chosen]
        assert sum(item.weight for item in taken) <= 6
        assert sum(item.value for item in taken) == pytest.approx(value)

    def test_coin_change_finds_the_fewest_coins(self):
        assert sorted(coin_change([1, 3, 4], 6)) == [3, 3]
        assert coin_change([2], 3) is None
        assert coin_change([1, 2, 5], 11) == [5, 5, 1]
        assert coin_change([1, 2, 5], 0) == []

    def test_coin_change_matches_an_exhaustive_search(self):
        rng = random.Random(20260903)

        for _ in range(100):
            coins = sorted(rng.sample(range(1, 12), rng.randint(1, 4)))
            amount = rng.randint(0, 25)

            best = None
            for size in range(0, amount + 1):
                if any(sum(combo) == amount
                       for combo in itertools.combinations_with_replacement(coins, size)):
                    best = size
                    break

            result = coin_change(coins, amount)
            assert (len(result) if result is not None else None) == best

    def test_matrix_chain_finds_the_cheap_bracketing(self):
        # 10x100, 100x5, 5x50. One bracketing costs 7500 and the other 75000.
        cost, bracketing = matrix_chain_order([10, 100, 5, 50])

        assert cost == 7500
        assert bracketing == "((M0M1)M2)"

    def test_matrix_chain_matches_an_exhaustive_search(self):
        rng = random.Random(20260903)

        cache: dict = {}

        def brute(dimensions, start, end):
            if start == end:
                return 0
            key = (tuple(dimensions), start, end)
            if key in cache:
                return cache[key]
            best = min(
                brute(dimensions, start, split) + brute(dimensions, split + 1, end)
                + dimensions[start] * dimensions[split + 1] * dimensions[end + 1]
                for split in range(start, end)
            )
            cache[key] = best
            return best

        for _ in range(50):
            dimensions = [rng.randint(1, 30) for _ in range(rng.randint(2, 7))]
            cost, _ = matrix_chain_order(dimensions)

            assert cost == brute(dimensions, 0, len(dimensions) - 2)

    def test_a_single_matrix_costs_nothing(self):
        assert matrix_chain_order([5, 10]) == (0, "M0")

    def test_longest_common_subsequence(self):
        assert longest_common_subsequence("ABCBDAB", "BDCABA") in {"BCBA", "BDAB", "BCAB"}
        assert longest_common_subsequence("abc", "abc") == "abc"
        assert longest_common_subsequence("abc", "xyz") == ""
        assert longest_common_subsequence("", "abc") == ""

    def test_the_common_subsequence_really_appears_in_both(self):
        rng = random.Random(20260903)

        for _ in range(200):
            first = "".join(rng.choice("abc") for _ in range(rng.randint(0, 15)))
            second = "".join(rng.choice("abc") for _ in range(rng.randint(0, 15)))
            result = longest_common_subsequence(first, second)

            for text in (first, second):
                position = -1
                for character in result:
                    position = text.index(character, position + 1)

    def test_edit_distance(self):
        assert edit_distance("kitten", "sitting") == 3
        assert edit_distance("", "abc") == 3
        assert edit_distance("abc", "abc") == 0
        assert edit_distance("abc", "") == 3

    def test_edit_distance_is_symmetric_and_obeys_the_triangle_inequality(self):
        # Real properties of a distance, and a good check that the table is right.
        rng = random.Random(20260903)
        words = ["".join(rng.choice("abc") for _ in range(rng.randint(0, 8)))
                 for _ in range(20)]

        for first in words:
            for second in words:
                assert edit_distance(first, second) == edit_distance(second, first)
                for third in words:
                    assert edit_distance(first, third) <= (
                        edit_distance(first, second) + edit_distance(second, third)
                    )

    def test_subset_sum(self):
        assert subset_sum([3, 34, 4, 12, 5, 2], 9) is not None
        assert sum(subset_sum([3, 34, 4, 12, 5, 2], 9)) == 9
        assert subset_sum([1, 2], 7) is None
        assert subset_sum([], 0) == []

    def test_fibonacci_bottom_up_matches_the_recursive_versions(self):
        from dsalab.algorithms.recursion import fibonacci_memoised
        from dsalab.tracing import run

        for n in range(20):
            assert fibonacci_bottom_up(n) == run(fibonacci_memoised(n))

    def test_fibonacci_bottom_up_handles_large_input_that_would_overflow_the_stack(self):
        # The payoff of removing the recursion: no stack at all, so 10000 is fine.
        assert fibonacci_bottom_up(10000) % 1000 == 875


class TestBacktracking:
    @pytest.mark.parametrize(
        "size,expected", [(1, 1), (2, 0), (3, 0), (4, 2), (5, 10), (6, 4), (7, 40), (8, 92)]
    )
    def test_the_n_queens_solution_counts_are_the_known_ones(self, size, expected):
        assert len(n_queens(size)) == expected

    def test_every_queens_solution_is_actually_legal(self):
        for solution in n_queens(6):
            for row, column in enumerate(solution):
                for other_row, other_column in enumerate(solution):
                    if row == other_row:
                        continue
                    assert column != other_column, "two queens share a column"
                    assert abs(row - other_row) != abs(column - other_column), "shared diagonal"

    def test_asking_for_only_the_first_solution_stops_early(self):
        _, all_steps = record(n_queens_traced(6))
        _, first_steps = record(n_queens_traced(6, first_only=True))

        assert len(first_steps) < len(all_steps)
        assert len(n_queens(6, first_only=True)) == 1

    def test_the_search_backtracks(self):
        _, steps = record(n_queens_traced(4))

        assert count_kinds(steps)["backtrack"] > 0
        assert count_kinds(steps)["reject"] > 0, "pruning is what makes this feasible"

    def test_a_negative_board_is_rejected(self):
        with pytest.raises(ValueError):
            n_queens(-1)

    def test_sudoku_solves_a_real_puzzle(self):
        puzzle = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ]
        solved = solve_sudoku(puzzle)

        assert solved is not None
        for row in solved:
            assert sorted(row) == list(range(1, 10))
        for column in range(9):
            assert sorted(solved[row][column] for row in range(9)) == list(range(1, 10))
        for box_row in (0, 3, 6):
            for box_column in (0, 3, 6):
                box = [solved[r][c] for r in range(box_row, box_row + 3)
                       for c in range(box_column, box_column + 3)]
                assert sorted(box) == list(range(1, 10))

    def test_sudoku_keeps_the_given_numbers(self):
        puzzle = [[0] * 9 for _ in range(9)]
        puzzle[0][0] = 5
        solved = solve_sudoku(puzzle)

        assert solved[0][0] == 5

    def test_a_board_that_already_breaks_the_rules_is_rejected_immediately(self):
        # Two ones in one row. Worth having because backtracking only checks new
        # placements, so without an upfront check the solver fills in around the
        # contradiction and can search for minutes before starving. Rejecting a
        # self contradicting puzzle outright is both correct and fast.
        puzzle = [[0] * 9 for _ in range(9)]
        puzzle[0][0] = 1
        puzzle[0][1] = 1

        assert solve_sudoku(puzzle) is None

        _, steps = record(solve_sudoku_traced(puzzle))
        assert [step.kind for step in steps] == ["invalid"], (
            "it should notice before searching at all"
        )

    def test_sudoku_fills_the_most_constrained_cell_first(self):
        puzzle = [
            [5, 3, 4, 6, 7, 8, 9, 1, 0],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9],
        ]
        _, steps = record(solve_sudoku_traced(puzzle))
        chosen = [step for step in steps if step.kind == "choose-cell"]

        assert chosen[0].data["options"] == [2], "the only empty cell has exactly one option"

    def test_a_badly_sized_board_is_rejected(self):
        with pytest.raises(ValueError):
            solve_sudoku([[0, 0], [0, 0]])

    def test_permutations(self):
        assert permutations([]) == [[]]
        assert permutations([1]) == [[1]]
        assert sorted(permutations([1, 2, 3])) == sorted(
            list(perm) for perm in itertools.permutations([1, 2, 3])
        )

    def test_permutation_counts_are_factorial(self):
        import math

        for size in range(6):
            assert len(permutations(list(range(size)))) == math.factorial(size)

    def test_permutations_undo_every_choice(self):
        from dsalab.algorithms.backtracking import permutations_traced

        _, steps = record(permutations_traced([1, 2, 3]))
        counts = count_kinds(steps)

        assert counts["choose"] == counts["undo"], (
            "every choice must be undone, which is the whole technique"
        )

    def test_subsets(self):
        assert len(subsets([1, 2, 3])) == 8
        assert sorted(map(sorted, subsets([1, 2]))) == [[], [1], [1, 2], [2]]

    def test_graph_colouring(self):
        graph = Graph()
        graph.add_edges([(0, 1), (1, 2), (2, 0)])  # a triangle needs three colours

        assert graph_colouring(graph, 2) is None
        assignment = graph_colouring(graph, 3)
        assert assignment is not None
        assert all(assignment[edge.source] != assignment[edge.target]
                   for edge in graph.edges())

    def test_two_colouring_a_bipartite_graph_succeeds(self):
        graph = Graph()
        graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 0)])

        assert graph_colouring(graph, 2) is not None

    def test_hamiltonian_cycle(self):
        cycle = Graph()
        cycle.add_edges([(0, 1), (1, 2), (2, 3), (3, 0)])
        found = hamiltonian_cycle(cycle)

        assert found is not None
        assert found[0] == found[-1]
        assert sorted(found[:-1]) == [0, 1, 2, 3]

    def test_a_graph_with_no_hamiltonian_cycle(self):
        # A path, not a cycle: nothing links the two ends.
        path = Graph()
        path.add_edges([(0, 1), (1, 2), (2, 3)])

        assert hamiltonian_cycle(path) is None

    def test_rat_in_a_maze(self):
        maze = [
            [1, 0, 0, 0],
            [1, 1, 0, 1],
            [0, 1, 0, 0],
            [1, 1, 1, 1],
        ]
        path = rat_in_a_maze(maze)

        assert path is not None
        assert path[0] == (0, 0) and path[-1] == (3, 3)
        for row, column in path:
            assert maze[row][column] == 1

    def test_a_maze_with_no_way_through(self):
        blocked = [[1, 0], [0, 1]]

        assert rat_in_a_maze(blocked) is None


class TestBranchAndBound:
    def build_distances(self, count: int, rng: random.Random) -> list[list[float]]:
        distances = [[0.0] * count for _ in range(count)]
        for a in range(count):
            for b in range(a + 1, count):
                weight = float(rng.randint(1, 40))
                distances[a][b] = distances[b][a] = weight
        return distances

    def test_the_travelling_salesman_finds_the_true_optimum(self):
        # The essential test. A bound that is not optimistic prunes away the best
        # answer and returns something worse while claiming it is optimal, so it
        # has to be checked against exhaustive search rather than trusted.
        rng = random.Random(20260903)

        for _ in range(40):
            count = rng.randint(2, 7)
            distances = self.build_distances(count, rng)

            _, bound_cost = travelling_salesman(distances)
            _, brute_cost = travelling_salesman_brute_force(distances)

            assert bound_cost == pytest.approx(brute_cost)

    def test_the_tour_visits_every_city_once_and_returns(self):
        rng = random.Random(20260903)
        distances = self.build_distances(6, rng)
        tour, _ = travelling_salesman(distances)

        assert tour[0] == tour[-1]
        assert sorted(tour[:-1]) == list(range(6))

    def test_pruning_actually_happens(self):
        rng = random.Random(20260903)
        distances = self.build_distances(8, rng)
        _, steps = record(travelling_salesman_traced(distances))

        assert count_kinds(steps)["prune"] > 0
        summary = [step for step in steps if step.kind == "done"][0]
        assert summary.data["pruned"] > 0

    def test_trivial_tours(self):
        assert travelling_salesman([]) == ([], 0.0)
        assert travelling_salesman([[0.0]]) == ([0], 0.0)

    def test_the_spanning_tree_is_a_lower_bound_on_the_tour(self):
        # Removing one edge from a tour leaves a spanning tree, so no tour can be
        # cheaper than the minimum one.
        rng = random.Random(20260903)

        for _ in range(30):
            count = rng.randint(3, 7)
            distances = self.build_distances(count, rng)

            _, tour_cost = travelling_salesman(distances)
            assert spanning_tree_tour_bound(distances) <= tour_cost + 1e-9

    def test_knapsack_branch_and_bound_matches_the_table(self):
        rng = random.Random(20260903)

        for _ in range(100):
            count = rng.randint(1, 8)
            capacity = rng.randint(1, 20)
            items = [
                Item(f"i{index}", float(rng.randint(1, 10)), float(rng.randint(1, 30)))
                for index in range(count)
            ]

            by_bound, _ = knapsack_branch_and_bound(items, capacity)
            by_table, _ = knapsack_01(items, capacity)

            assert by_bound == pytest.approx(by_table)

    def test_knapsack_pruning_happens(self):
        rng = random.Random(20260903)
        items = [
            Item(f"i{index}", float(rng.randint(1, 10)), float(rng.randint(1, 30)))
            for index in range(14)
        ]
        _, steps = record(knapsack_branch_and_bound_traced(items, 30))

        assert count_kinds(steps)["prune"] > 0


class TestCounterexampleFinder:
    """The fourth headline feature: proving greedy wrong with real inputs."""

    def test_it_finds_the_classic_coin_change_failure(self):
        report = find_coin_change_counterexample()

        assert report.found_any
        smallest = report.smallest
        assert len(smallest.greedy_answer) > len(smallest.correct_answer)
        assert smallest.gap > 0

    def test_the_counterexample_it_finds_is_genuine(self):
        # Checking the finder itself: re-run both algorithms on the reported input
        # and confirm they really disagree the way it says.
        report = find_coin_change_counterexample()

        for example in report.counterexamples:
            coins_text = example.inputs.split("coins ")[1].split(", making")[0]
            coins = eval(coins_text)  # noqa: S307 - our own formatted output
            amount = int(example.inputs.split("making ")[1])

            assert len(greedy_coin_change(coins, amount)) == example.greedy_score
            assert len(coin_change(coins, amount)) == example.correct_score
            assert example.greedy_score > example.correct_score

    def test_it_finds_the_smallest_example_first(self):
        # The search is ordered smallest first on purpose: a three coin example
        # teaches something, a fifty coin one does not.
        report = find_coin_change_counterexample()

        assert "1, 3, 4" in report.smallest.inputs or "[1, 3, 4]" in report.smallest.inputs

    def test_it_finds_cases_where_greedy_gives_up_entirely(self):
        report = find_coin_change_failure()

        assert report.found_any
        assert report.smallest.greedy_answer is None
        assert report.smallest.correct_answer is not None

    def test_it_finds_a_knapsack_where_greedy_loses(self):
        report = find_knapsack_counterexample(attempts=500)

        assert report.found_any
        example = report.smallest
        assert example.correct_score > example.greedy_score

    def test_it_finds_a_schedule_where_the_wrong_greedy_rule_loses(self):
        report = find_activity_selection_counterexample(attempts=500)

        assert report.found_any
        assert report.smallest.correct_score > report.smallest.greedy_score

    def test_a_report_with_no_findings_says_so_honestly(self):
        # Searching a tiny space finds nothing, and the summary must not present
        # that as proof that greedy is correct.
        report = find_coin_change_counterexample(max_coin=2, max_coins_in_system=2,
                                                 max_amount=3)

        assert not report.found_any
        assert "not a proof" in report.summary()
        assert str(report.tried) in report.summary()

    def test_every_counterexample_explains_itself_in_plain_english(self):
        for report in (find_coin_change_counterexample(),
                       find_knapsack_counterexample(attempts=500)):
            for example in report.counterexamples:
                assert len(example.explanation) > 40
                assert example.explanation.endswith(".")
                assert str(example)
