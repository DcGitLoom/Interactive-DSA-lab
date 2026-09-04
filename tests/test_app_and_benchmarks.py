"""Tests for the benchmark harness and the app's logic.

The app's *logic* is tested here; its Streamlit layer is not, because it needs a
browser and a server and the tests would end up asserting things about widgets
rather than about behaviour. That is exactly why every decision lives in
`app/presentation.py` and `app/main.py` only draws: it keeps the untestable part
small enough to check by reading it.
"""

import json

import pytest

from app.presentation import (
    AVL_RULES,
    INPUT_EXPLANATIONS,
    INPUT_KINDS,
    RED_BLACK_RULES,
    available_algorithms,
    build_frames,
    colour_for,
    frame_at,
    inspect,
    measure_algorithm,
    race,
    suggest_input,
    tree_layout,
)
from benchmarks.harness import (
    BenchmarkResult,
    Comparison,
    measure,
    measure_operations,
    plot,
    relative_speed,
    save,
    stable_enough,
    time_once,
)
from dsalab.complexity import detect
from dsalab.structures.avl import AVLTree
from dsalab.structures.bst import BinarySearchTree
from dsalab.structures.red_black import RedBlackTree


class TestHarness:
    def test_it_times_something(self):
        elapsed = time_once(lambda: sum(range(10000)))

        assert elapsed > 0

    def test_garbage_collection_is_left_as_it_was_found(self):
        import gc

        assert gc.isenabled()
        time_once(lambda: None)
        assert gc.isenabled(), "timing must not leave collection switched off"

    def test_it_is_restored_even_when_the_timed_code_raises(self):
        import gc

        with pytest.raises(RuntimeError):
            time_once(lambda: (_ for _ in ()).throw(RuntimeError("boom")))

        assert gc.isenabled()

    def test_measuring_operations_detects_the_right_curve(self):
        result = measure_operations(
            "nested loop", "O(n^2)",
            lambda n: float(sum(1 for _ in range(n) for _ in range(n))),
            [16, 32, 64, 128, 256],
        )

        assert result.verdict.best.curve == "O(n^2)"
        assert result.matches_claim

    def test_a_wrong_claim_is_reported_as_a_disagreement(self):
        result = measure_operations(
            "nested loop claiming to be linear", "O(n)",
            lambda n: float(n * n), [16, 32, 64, 128, 256],
        )

        assert not result.matches_claim
        assert "DISAGREES" in result.summary()

    def test_neighbouring_curves_count_as_agreement(self):
        # n and n log n cannot be separated reliably by timing over a limited range
        # of sizes, so treating a near miss as a disagreement would make the report
        # dishonest in the other direction.
        sizes = [64, 128, 256, 512]
        import math

        result = BenchmarkResult(
            "borderline", "O(n)", sizes,
            [float(n) * math.log2(n) for n in sizes],
            detect(sizes, [float(n) * math.log2(n) for n in sizes]),
        )

        assert result.verdict.best.curve == "O(n log n)"
        assert result.matches_claim

    def test_setup_is_not_counted_against_the_operation(self):
        # Building the input is often far more expensive than the operation being
        # measured, and including it would drown the signal entirely.
        #
        # This is checked by comparing two measurements of the same cheap read:
        # one with the expensive list construction in build_input, where it should
        # not be timed, and one with it inside the operation, where it must be. The
        # second should be dramatically slower.
        #
        # An earlier version of this test fitted a curve to the first measurement
        # and expected O(1). That was a bad test: reading one element takes about a
        # hundred nanoseconds, which is close enough to the timer's resolution that
        # the fit describes the noise rather than the code, and it named a
        # different curve on different runs. Comparing two measurements against
        # each other avoids depending on absolute times at all.
        sizes = [200, 400, 800]

        outside = measure(
            "setup outside the timed region", "O(1)",
            lambda size: list(range(size * 200)),
            lambda prepared: prepared[0],
            sizes,
            repeats=3,
        )
        inside = measure(
            "setup inside the timed region", "O(n)",
            lambda size: size,
            lambda size: list(range(size * 200))[0],
            sizes,
            repeats=3,
        )

        assert inside.timings[-1] > outside.timings[-1] * 50, (
            f"building the input took {inside.timings[-1]:.6f}s and the read took "
            f"{outside.timings[-1]:.9f}s; if these were close, setup would be leaking "
            "into every measurement in the suite"
        )

    def test_relative_speed_is_reported_as_a_ratio(self):
        sizes = [10, 20, 40]
        slow = BenchmarkResult("slow", "O(n)", sizes, [1.0, 2.0, 4.0], detect(sizes,
                                                                              [1.0, 2.0, 4.0]))
        fast = BenchmarkResult("fast", "O(n)", sizes, [0.5, 1.0, 2.0], detect(sizes,
                                                                             [0.5, 1.0, 2.0]))

        speeds = relative_speed([slow, fast])
        assert speeds["fast"] == 1.0
        assert speeds["slow"] == 2.0

    def test_stability_checking(self):
        assert stable_enough([1.0, 1.02, 0.98])
        assert not stable_enough([1.0, 5.0, 0.2])
        assert stable_enough([])

    def test_results_are_saved_as_json(self, tmp_path):
        sizes = [10, 20, 40]
        comparison = Comparison("test", [
            BenchmarkResult("thing", "O(n)", sizes, [1.0, 2.0, 4.0],
                            detect(sizes, [1.0, 2.0, 4.0]), "a note")
        ])

        written = save([comparison], tmp_path)
        payload = json.loads(written.read_text())

        assert payload["comparisons"][0]["title"] == "test"
        assert payload["comparisons"][0]["results"][0]["measured"] == "O(n)"
        assert payload["comparisons"][0]["results"][0]["notes"] == "a note"

    def test_plotting_without_matplotlib_returns_nothing_rather_than_failing(self, tmp_path):
        # The numbers are the result and the picture is a convenience, so a machine
        # without matplotlib should still get its benchmarks.
        sizes = [10, 20, 40]
        comparison = Comparison("test", [
            BenchmarkResult("thing", "O(n)", sizes, [1.0, 2.0, 4.0],
                            detect(sizes, [1.0, 2.0, 4.0]))
        ])

        written = plot([comparison], tmp_path)
        assert isinstance(written, list)  # empty or populated, never an exception


class TestBenchmarkDefinitions:
    """The benchmarks themselves must at least run and produce sane output."""

    def test_every_comparison_runs_and_reports(self):
        from benchmarks.run import run_all

        comparisons = run_all(quick=True)

        assert len(comparisons) >= 6
        for comparison in comparisons:
            assert comparison.results, f"{comparison.title} measured nothing"
            for result in comparison.results:
                assert len(result.timings) == len(result.sizes)
                assert all(value >= 0 for value in result.timings)
                assert result.summary().endswith(".")

    def test_the_sorting_benchmarks_measure_what_they_claim(self):
        # This is the check that caught the day 20 bug: every sorting step used to
        # carry a copy of the whole array, so merge sort timed as quadratic while
        # its comparison count was still n log n.
        from benchmarks.run import sorting_comparison

        comparison = sorting_comparison([200, 400, 800, 1600])
        by_name = {result.name: result for result in comparison.results}

        for name in ("merge sort", "quick sort (median of three)", "heap sort"):
            assert by_name[name].matches_claim, (
                f"{name} measured as {by_name[name].verdict.best.curve} rather than "
                "n log n, which usually means something is copying the data per step"
            )

    def test_the_tree_benchmark_shows_why_balancing_matters(self):
        from benchmarks.run import tree_shape_comparison

        comparison = tree_shape_comparison([64, 128, 256, 512])
        by_name = {result.name: result for result in comparison.results}

        assert by_name["plain search tree"].verdict.best.curve == "O(n)"
        assert by_name["AVL tree"].verdict.best.curve == "O(log n)"


class TestAppLogic:
    def test_every_offered_algorithm_can_be_animated(self):
        for algorithm in available_algorithms():
            frames = build_frames([5, 2, 8, 1, 9], algorithm)

            assert frames, f"{algorithm} produced nothing to draw"
            assert frames[-1].values == sorted([5, 2, 8, 1, 9]), (
                f"{algorithm} did not end up sorted on screen"
            )

    def test_every_frame_has_something_to_show(self):
        for frame in build_frames([4, 2, 7, 1], "insertion"):
            assert frame.note
            assert frame.colour.startswith("#")
            assert len(frame.values) == 4

    def test_frames_only_highlight_positions_that_exist(self):
        for algorithm in available_algorithms():
            for frame in build_frames([3, 1, 4, 1, 5], algorithm):
                for index in frame.highlighted:
                    assert 0 <= index < len(frame.values), (
                        f"{algorithm} highlighted position {index} of {len(frame.values)}"
                    )

    def test_an_unknown_algorithm_is_rejected(self):
        with pytest.raises(ValueError):
            build_frames([1, 2], "telepathy")

    def test_sorting_an_already_sorted_list_still_animates(self):
        frames = build_frames([1, 2, 3], "bubble")

        assert frames
        assert frames[-1].values == [1, 2, 3]

    def test_colours_are_consistent_and_have_a_fallback(self):
        assert colour_for("swap") == colour_for("shift"), "moves should look the same"
        assert colour_for("compare") != colour_for("swap")
        assert colour_for("something nobody defined") == colour_for("default")

    def test_a_race_runs_both_on_identical_input(self):
        contest = race([5, 3, 8, 1], "insertion", "merge")

        assert contest.left.name == "insertion"
        assert contest.right.name == "merge"
        assert contest.left.frames[-1].values == contest.right.frames[-1].values

    def test_the_race_verdict_names_a_winner_and_explains_it(self):
        contest = race(suggest_input("reversed", 25), "insertion", "merge")

        assert contest.winner() == "merge", "reversed input is insertion sort's worst case"
        assert "times cheaper" in contest.verdict()

    def test_insertion_sort_wins_on_nearly_sorted_input(self):
        # The whole reason the race exists: the ranking depends on the input, and
        # seeing that is more useful than any table of complexities.
        contest = race(suggest_input("nearly sorted", 30), "insertion", "merge")

        assert contest.winner() == "insertion"

    def test_a_finished_competitor_holds_its_last_frame(self):
        contest = race([3, 1, 2], "insertion", "bubble")
        far_future = frame_at(contest.left, 10_000)

        assert far_future is not None
        assert far_future.values == sorted([3, 1, 2])

    def test_frame_at_on_an_empty_run(self):
        contest = race([], "insertion", "merge")

        assert frame_at(contest.left, 0) is None
        assert contest.length == 0

    def test_the_complexity_detective_agrees_with_the_docstrings(self):
        assert measure_algorithm("bubble").best.curve == "O(n^2)"
        assert measure_algorithm("merge").best.curve == "O(n log n)"

    def test_every_input_kind_produces_the_right_shape(self):
        for kind in INPUT_KINDS:
            values = suggest_input(kind, 20)
            assert len(values) == 20
            assert kind in INPUT_EXPLANATIONS

        assert suggest_input("sorted", 5) == [1, 2, 3, 4, 5]
        assert suggest_input("reversed", 5) == [5, 4, 3, 2, 1]
        assert len(set(suggest_input("few distinct", 30))) <= 3

    def test_an_unknown_input_kind_is_rejected(self):
        with pytest.raises(ValueError):
            suggest_input("interesting")

    def test_the_invariant_panel_lists_rules_that_hold(self):
        panel = inspect(RedBlackTree(range(20)), RED_BLACK_RULES)

        assert panel.healthy
        assert len(panel.rules) == len(RED_BLACK_RULES)
        assert all(panel.status_of(rule) == "holds" for rule in panel.rules)

    def test_the_panel_marks_the_rule_that_broke(self):
        tree = RedBlackTree(range(10))
        tree.root.colour = "red"

        panel = inspect(tree, RED_BLACK_RULES)
        assert not panel.healthy
        assert panel.status_of("rule 2: the root is black") == "broken"
        assert panel.status_of("rule 4: a red node never has a red child") == "holds"

    def test_the_panel_works_for_avl_trees_too(self):
        panel = inspect(AVLTree(range(30)), AVL_RULES)

        assert panel.healthy

    def test_tree_layout_gives_every_node_a_distinct_position(self):
        for tree, kind in (
            (BinarySearchTree([5, 3, 8, 1, 4]), "bst"),
            (AVLTree(range(15)), "avl"),
            (RedBlackTree(range(15)), "red_black"),
        ):
            nodes = tree_layout(tree, kind)

            assert len(nodes) == len(tree)
            positions = {(node["x"], node["y"]) for node in nodes}
            assert len(positions) == len(nodes), "two nodes would be drawn on top of each other"

    def test_the_layout_puts_the_tree_in_sorted_order_left_to_right(self):
        # An inorder walk gives the x coordinates, so reading the drawing left to
        # right should give the values in order.
        nodes = tree_layout(BinarySearchTree([5, 3, 8, 1, 4, 9]), "bst")
        by_position = [node["value"] for node in sorted(nodes, key=lambda node: node["x"])]

        assert by_position == sorted(by_position)

    def test_red_black_layout_carries_the_colours(self):
        nodes = tree_layout(RedBlackTree(range(10)), "red_black")

        assert all(node["colour"] in ("red", "black") for node in nodes)

    def test_an_empty_tree_lays_out_to_nothing(self):
        assert tree_layout(BinarySearchTree(), "bst") == []
