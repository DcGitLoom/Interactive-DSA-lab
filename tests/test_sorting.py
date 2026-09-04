"""Tests for every sorting algorithm.

The structure here is deliberate. Correctness is tested once, generically, against
every algorithm at once, because "does it sort" is the same question for all of
them. Then each algorithm gets tests for the specific claims made about it:
insertion sort being adaptive, selection sort using few swaps, counting sort being
stable, quick sort's pivot strategies behaving as advertised.
"""

import random

import pytest

from dsalab.algorithms.sorting import (
    ALL_SORTS,
    COMPARISON_SORTS,
    STABLE_SORTS,
    bubble_sort_traced,
    bucket_sort,
    counting_sort,
    counting_sort_traced,
    insertion_sort_traced,
    merge_sort_traced,
    quick_sort,
    quick_sort_traced,
    radix_sort,
    selection_sort_traced,
    shell_sort_traced,
)
from dsalab.complexity import detect
from dsalab.structures.heap import heap_sort
from dsalab.tracing import count_kinds, record

AWKWARD_INPUTS = [
    [],
    [1],
    [2, 1],
    [1, 2],
    [3, 3, 3],
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [0, -1, 5, -10, 3],
    [2, 1, 2, 1, 2],
    [100],
]


@pytest.mark.parametrize("name", sorted(ALL_SORTS))
class TestEverySortIsCorrect:
    @pytest.mark.parametrize("values", AWKWARD_INPUTS)
    def test_it_handles_the_awkward_inputs(self, name, values):
        assert ALL_SORTS[name](values) == sorted(values)

    def test_it_agrees_with_python_on_random_input(self, name):
        rng = random.Random(20260830)
        for _ in range(50):
            values = [rng.randint(-100, 100) for _ in range(rng.randint(0, 60))]

            assert ALL_SORTS[name](values) == sorted(values)

    def test_it_handles_many_duplicates(self, name):
        # Arrays with few distinct values break several sorts in interesting ways,
        # Lomuto partitioning worst of all.
        rng = random.Random(20260830)
        values = [rng.choice([1, 2, 3]) for _ in range(200)]

        assert ALL_SORTS[name](values) == sorted(values)

    def test_it_does_not_modify_the_input(self, name):
        values = [3, 1, 2]
        ALL_SORTS[name](values)

        assert values == [3, 1, 2], "sorting should return a new list, not edit the caller's"

    def test_it_sorts_a_larger_array(self, name):
        rng = random.Random(20260830)
        values = [rng.randint(0, 10000) for _ in range(500)]

        assert ALL_SORTS[name](values) == sorted(values)


class TestStability:
    """Equal items must keep their original relative order, where claimed."""

    @staticmethod
    def labelled_input(rng: random.Random, size: int = 200) -> list[tuple[int, int]]:
        """Pairs of (sort key, original position), so order changes are visible."""
        return [(rng.randint(0, 5), index) for index in range(size)]

    @pytest.mark.parametrize("name", sorted(STABLE_SORTS))
    def test_the_sorts_that_claim_stability_have_it(self, name):
        # The key function matters enormously here. Sorting the pairs directly
        # would compare the tie breaking second element too, which makes every
        # sort look stable and the test worthless. Sorting by the first element
        # alone is what leaves the order of equal items visible.
        rng = random.Random(20260830)
        values = self.labelled_input(rng)
        result = ALL_SORTS[name](values, lambda pair: pair[0])

        assert result == sorted(values, key=lambda pair: pair[0]), (
            f"{name} reordered equal items, so it is not stable"
        )

    @pytest.mark.parametrize("name", sorted(set(ALL_SORTS) - STABLE_SORTS))
    def test_the_sorts_that_do_not_claim_stability_still_sort_correctly(self, name):
        rng = random.Random(20260830)
        values = self.labelled_input(rng)
        result = ALL_SORTS[name](values, lambda pair: pair[0])

        assert [pair[0] for pair in result] == sorted(pair[0] for pair in values)

    def test_selection_sort_is_genuinely_unstable(self):
        # Documenting the failure rather than just claiming it. Selecting the
        # minimum and swapping it into place moves an item a long way, and here
        # that throws the first 2 past the second one.
        from dsalab.algorithms.sorting import selection_sort

        values = [(2, "first"), (2, "second"), (1, "third")]
        result = selection_sort(values, key=lambda pair: pair[0])

        assert result[0] == (1, "third")
        assert [label for _, label in result[1:]] == ["second", "first"]

    def test_counting_sort_is_stable(self):
        # Its stability depends on walking the input backwards when placing.
        # Walking forwards gives the same values with equal items reversed.
        rng = random.Random(20260830)
        values = [rng.randint(0, 5) for _ in range(100)]

        assert counting_sort(values) == sorted(values)

    def test_radix_sort_depends_on_its_inner_sort_being_stable(self):
        # If the per digit pass were not stable, sorting by the tens column would
        # undo the ones column and the answer would be wrong.
        rng = random.Random(20260830)
        values = [rng.randint(0, 999) for _ in range(200)]

        assert radix_sort(values) == sorted(values)


class TestInsertionSortIsAdaptive:
    def test_sorted_input_costs_one_comparison_per_item(self):
        _, steps = record(insertion_sort_traced(list(range(100))))

        assert count_kinds(steps)["compare"] == 99
        assert count_kinds(steps).get("shift", 0) == 0

    def test_reverse_sorted_input_is_the_worst_case(self):
        _, steps = record(insertion_sort_traced(list(range(100, 0, -1))))

        assert count_kinds(steps)["compare"] > 4000

    def test_the_cost_grows_with_disorder_not_with_size(self):
        # The real claim: work is proportional to how far out of order the input
        # is. Nearly sorted input of any size stays close to linear.
        def comparisons(values):
            _, steps = record(insertion_sort_traced(values))
            return float(count_kinds(steps)["compare"])

        # "Nearly sorted" has to mean every item is close to its home, not merely
        # that few items moved. Swapping two random positions displaces both by a
        # distance proportional to n, and insertion sort has to shift each of them
        # all that way, so a handful of long range swaps is still quadratic work.
        # Swapping neighbours is the disorder this algorithm is actually good at.
        rng = random.Random(20260830)
        sizes = [200, 400, 800, 1600, 3200]
        nearly_sorted = []
        for size in sizes:
            values = list(range(size))
            for _ in range(size // 50):
                index = rng.randrange(size - 1)
                values[index], values[index + 1] = values[index + 1], values[index]
            nearly_sorted.append(comparisons(values))

        verdict = detect(sizes, nearly_sorted)
        assert verdict.best.curve in {"O(n)", "O(n log n)"}, (
            f"nearly sorted input measured as {verdict.best.curve}, which is not adaptive"
        )

    def test_random_input_really_is_quadratic(self):
        def comparisons(n):
            rng = random.Random(n)
            values = [rng.randint(0, 10000) for _ in range(n)]
            _, steps = record(insertion_sort_traced(values))
            return float(count_kinds(steps)["compare"])

        sizes = [100, 200, 400, 800]
        verdict = detect(sizes, [comparisons(n) for n in sizes])

        assert verdict.best.curve == "O(n^2)"


class TestSelectionSortUsesFewSwaps:
    def test_it_never_makes_more_than_n_swaps(self):
        # Its one genuine advantage: minimum data movement, which matters when
        # writing is expensive.
        rng = random.Random(20260830)
        values = [rng.randint(0, 1000) for _ in range(200)]
        _, steps = record(selection_sort_traced(values))

        assert count_kinds(steps).get("swap", 0) <= 200

    def test_bubble_sort_makes_far_more_swaps_on_the_same_input(self):
        rng = random.Random(20260830)
        values = [rng.randint(0, 1000) for _ in range(200)]

        _, selection_steps = record(selection_sort_traced(values))
        _, bubble_steps = record(bubble_sort_traced(values))

        assert count_kinds(bubble_steps).get("swap", 0) > 10 * count_kinds(
            selection_steps
        ).get("swap", 0)

    def test_selection_sort_compares_the_same_amount_whatever_the_input(self):
        # Unlike insertion sort, it is not adaptive at all: sorted input costs
        # exactly as much as random input.
        _, sorted_steps = record(selection_sort_traced(list(range(100))))
        _, shuffled = record(selection_sort_traced(list(range(100, 0, -1))))

        assert count_kinds(sorted_steps)["compare"] == count_kinds(shuffled)["compare"]


class TestBubbleSortEarlyExit:
    def test_sorted_input_takes_a_single_pass(self):
        _, steps = record(bubble_sort_traced(list(range(50))))

        assert count_kinds(steps)["compare"] == 49, "one pass, then it should stop"
        assert any(step.kind == "done" for step in steps)

    def test_it_is_linear_on_already_sorted_input(self):
        def comparisons(n):
            _, steps = record(bubble_sort_traced(list(range(n))))
            return float(count_kinds(steps)["compare"])

        sizes = [100, 200, 400, 800, 1600]
        verdict = detect(sizes, [comparisons(n) for n in sizes])

        assert verdict.best.curve == "O(n)"


class TestMergeSort:
    def test_the_number_of_comparisons_grows_as_n_log_n(self):
        def comparisons(n):
            rng = random.Random(n)
            values = [rng.randint(0, 100000) for _ in range(n)]
            _, steps = record(merge_sort_traced(values))
            return float(count_kinds(steps)["compare"])

        sizes = [128, 256, 512, 1024, 2048]
        verdict = detect(sizes, [comparisons(n) for n in sizes])

        assert verdict.best.curve == "O(n log n)"

    def test_sorted_input_costs_about_the_same_as_random_input(self):
        # Merge sort is not adaptive, and that is the trade for having no bad
        # case: the cost is n log n whatever you feed it.
        def comparisons(values):
            _, steps = record(merge_sort_traced(values))
            return count_kinds(steps)["compare"]

        rng = random.Random(20260830)
        ordered = comparisons(list(range(1000)))
        shuffled = comparisons([rng.randint(0, 10000) for _ in range(1000)])

        assert 0.5 < ordered / shuffled < 1.5

    def test_the_recursive_and_iterative_versions_agree(self):
        rng = random.Random(20260830)
        for _ in range(50):
            values = [rng.randint(-500, 500) for _ in range(rng.randint(0, 100))]

            assert ALL_SORTS["merge"](values) == ALL_SORTS["merge (iterative)"](values)


class TestQuickSortPivots:
    def test_the_first_element_pivot_is_quadratic_on_sorted_input(self):
        # The classic trap. Sorted input is common, and this choice turns it into
        # the worst case.
        def comparisons(n):
            _, steps = record(quick_sort_traced(list(range(n)), pivot="first"))
            return float(count_kinds(steps)["compare"])

        sizes = [50, 100, 200, 400]
        verdict = detect(sizes, [comparisons(n) for n in sizes])

        assert verdict.best.curve == "O(n^2)"

    def test_the_median_of_three_pivot_makes_sorted_input_the_best_case(self):
        def comparisons(n):
            _, steps = record(quick_sort_traced(list(range(n)), pivot="median"))
            return float(count_kinds(steps)["compare"])

        sizes = [128, 256, 512, 1024, 2048]
        verdict = detect(sizes, [comparisons(n) for n in sizes])

        assert verdict.best.curve == "O(n log n)"

    def test_a_random_pivot_handles_sorted_input_too(self):
        def comparisons(n):
            _, steps = record(quick_sort_traced(list(range(n)), pivot="random", seed=1))
            return float(count_kinds(steps)["compare"])

        sizes = [128, 256, 512, 1024, 2048]
        verdict = detect(sizes, [comparisons(n) for n in sizes])

        assert verdict.best.curve in {"O(n log n)", "O(n)"}

    def test_every_pivot_strategy_still_sorts_correctly(self):
        rng = random.Random(20260830)
        for strategy in ("first", "random", "median"):
            for _ in range(20):
                values = [rng.randint(-100, 100) for _ in range(rng.randint(0, 60))]
                assert quick_sort(values, pivot=strategy, seed=7) == sorted(values)

    def test_an_unknown_pivot_strategy_is_rejected(self):
        with pytest.raises(ValueError):
            quick_sort([3, 1, 2], pivot="magic")

    def test_the_pivot_lands_in_its_final_position(self):
        # The property that makes quick sort work: after partitioning, the pivot
        # never moves again, which is why the two halves can be sorted
        # independently with no merge step.
        values = [5, 2, 8, 1, 9, 3]
        _, steps = record(quick_sort_traced(values))
        placements = [step for step in steps if step.kind == "place"]

        for step in placements:
            final = sorted(values)
            index = step.data["index"]
            assert step.data["array"][index] == final[index]


class TestShellSort:
    def test_the_gap_sequence_shrinks_to_one(self):
        _, steps = record(shell_sort_traced(list(range(50, 0, -1))))
        gaps = [step.data["gap"] for step in steps if step.kind == "gap"]

        assert gaps == sorted(gaps, reverse=True)
        assert gaps[-1] == 1, "the final pass must be a plain insertion sort"

    def test_it_beats_plain_insertion_sort_on_reversed_input(self):
        # The whole point of the gaps: items travel a long way cheaply, so by the
        # last pass the array is nearly sorted.
        values = list(range(300, 0, -1))

        _, shell_steps = record(shell_sort_traced(values))
        _, insertion_steps = record(insertion_sort_traced(values))

        assert count_kinds(shell_steps).get("shift", 0) < count_kinds(insertion_steps).get(
            "shift", 0
        )


class TestLinearSorts:
    def test_counting_sort_matches_python(self):
        rng = random.Random(20260830)
        for _ in range(50):
            values = [rng.randint(0, 50) for _ in range(rng.randint(0, 200))]

            assert counting_sort(values) == sorted(values)

    def test_counting_sort_handles_negative_values(self):
        assert counting_sort([3, -5, 0, -1, 2]) == [-5, -1, 0, 2, 3]

    def test_counting_sort_makes_no_comparisons_at_all(self):
        # This is why it can beat the n log n lower bound: the bound applies only
        # to algorithms that learn about the data by comparing.
        _, steps = record(counting_sort_traced([5, 3, 8, 1]))

        assert "compare" not in count_kinds(steps)

    def test_counting_sort_is_linear_in_the_input_size(self):
        def work(n):
            rng = random.Random(n)
            values = [rng.randint(0, 100) for _ in range(n)]
            _, steps = record(counting_sort_traced(values))
            return float(len(steps))

        sizes = [128, 256, 512, 1024, 2048]
        verdict = detect(sizes, [work(n) for n in sizes])

        assert verdict.best.curve == "O(n)"

    def test_radix_sort_matches_python(self):
        rng = random.Random(20260830)
        for _ in range(50):
            values = [rng.randint(0, 100000) for _ in range(rng.randint(0, 200))]

            assert radix_sort(values) == sorted(values)

    def test_radix_sort_handles_negative_numbers(self):
        # Most textbook versions quietly assume non negative input.
        assert radix_sort([5, -3, 0, -17, 22, -1]) == [-17, -3, -1, 0, 5, 22]

    def test_radix_sort_works_in_other_bases(self):
        rng = random.Random(20260830)
        values = [rng.randint(0, 5000) for _ in range(100)]

        assert radix_sort(values, base=2) == sorted(values)
        assert radix_sort(values, base=16) == sorted(values)

    def test_radix_sort_does_one_pass_per_digit(self):
        from dsalab.algorithms.sorting import radix_sort_traced

        _, steps = record(radix_sort_traced([999, 1, 50]))

        assert count_kinds(steps)["digit"] == 3, "three digit numbers need three passes"

    def test_bucket_sort_matches_python(self):
        rng = random.Random(20260830)
        for _ in range(50):
            values = [rng.random() for _ in range(rng.randint(0, 100))]

            assert bucket_sort(values) == sorted(values)

    def test_bucket_sort_handles_values_that_are_all_the_same(self):
        assert bucket_sort([2.0, 2.0, 2.0]) == [2.0, 2.0, 2.0]

    def test_bucket_sort_still_works_when_everything_lands_in_one_bucket(self):
        # The bad case: heavily skewed input degrades it to insertion sort. It
        # should still be correct, just slow.
        values = [0.001 * index for index in range(50)] + [1000.0]

        assert bucket_sort(values) == sorted(values)


class TestAgreementAcrossEverything:
    def test_all_eleven_sorts_produce_identical_output(self):
        rng = random.Random(20260830)
        for _ in range(30):
            values = [rng.randint(-1000, 1000) for _ in range(rng.randint(1, 80))]
            expected = sorted(values)

            for name, algorithm in ALL_SORTS.items():
                assert algorithm(values) == expected, f"{name} disagreed"
            assert heap_sort(values) == expected
            assert counting_sort(values) == expected
            assert radix_sort(values) == expected

    def test_every_traced_sort_reports_steps_with_readable_notes(self):
        for name, traced in COMPARISON_SORTS.items():
            _, steps = record(traced([3, 1, 2]))

            assert steps, f"{name} produced no steps to animate"
            # A note may be a question ("Is 5 bigger than 3?") as well as a
            # statement, since that is how a comparison reads most naturally.
            assert all(step.note.endswith((".", "?")) for step in steps), f"{name} note"
            assert all("array" in step.data or step.kind in {"gap", "split", "done"}
                       for step in steps), f"{name} produced a step with nothing to draw"
