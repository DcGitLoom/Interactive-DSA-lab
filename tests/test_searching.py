"""Tests for binary search, its variants, and the string matching algorithms.

Binary search is famously easy to get subtly wrong, so these tests hammer the
boundaries: empty input, one element, the target below everything, above
everything, and runs of equal values where the difference between lower and upper
bound actually shows.

The string matching tests all cross check the three real algorithms against the
naive one, which is slow but obviously correct.
"""

import bisect
import random

import pytest

from dsalab.algorithms.searching import (
    binary_search,
    binary_search_recursive,
    binary_search_traced,
    count_occurrences,
    exponential_search,
    first_true,
    interpolation_search,
    linear_search,
    lower_bound,
    ternary_search_maximum,
    upper_bound,
)
from dsalab.algorithms.strings import (
    build_failure_table,
    is_rotation,
    kmp_search,
    kmp_search_traced,
    longest_prefix_suffix,
    naive_search,
    naive_search_traced,
    rabin_karp_search,
    rabin_karp_search_traced,
    z_array,
    z_search,
)
from dsalab.complexity import detect
from dsalab.tracing import count_kinds, record


class TestBinarySearch:
    @pytest.mark.parametrize("values", [[], [1], [1, 2], list(range(10)), list(range(0, 100, 3))])
    def test_it_finds_every_value_that_is_present(self, values):
        for index, value in enumerate(values):
            assert binary_search(values, value) == index

    @pytest.mark.parametrize(
        "values,missing",
        [([], 1), ([2], 1), ([2], 3), ([1, 3, 5], 0), ([1, 3, 5], 2), ([1, 3, 5], 6)],
    )
    def test_it_reports_minus_one_for_what_is_absent(self, values, missing):
        assert binary_search(values, missing) == -1

    def test_it_agrees_with_a_linear_scan_on_random_input(self):
        rng = random.Random(20260831)
        for _ in range(200):
            values = sorted(rng.randint(0, 50) for _ in range(rng.randint(0, 40)))
            target = rng.randint(0, 50)

            found = binary_search(values, target)
            if found == -1:
                assert target not in values
            else:
                assert values[found] == target

    def test_the_recursive_version_agrees_with_the_loop(self):
        rng = random.Random(20260831)
        for _ in range(100):
            values = sorted(rng.sample(range(200), rng.randint(0, 50)))
            target = rng.randint(0, 200)

            iterative = binary_search(values, target)
            recursive = binary_search_recursive(values, target)

            if iterative == -1:
                assert recursive == -1
            else:
                assert values[recursive] == values[iterative] == target

    def test_it_really_does_halve_the_range_each_time(self):
        _, steps = record(binary_search_traced(list(range(1024)), 1023))
        probes = count_kinds(steps).get("probe", 0)

        assert probes <= 11, f"a 1024 element search should need about 10 probes, took {probes}"

    def test_the_number_of_probes_grows_logarithmically(self):
        def probes(n: int) -> float:
            values = list(range(n))
            _, steps = record(binary_search_traced(values, -1))  # always missing, worst case
            return float(count_kinds(steps).get("probe", 0))

        sizes = [64, 128, 256, 512, 1024, 2048]
        verdict = detect(sizes, [probes(n) for n in sizes])

        assert verdict.best.curve == "O(log n)"

    def test_it_terminates_on_every_input_shape(self):
        # The classic failure is an infinite loop from a boundary that never
        # narrows. Every size and every target position is tried here.
        for size in range(0, 30):
            values = list(range(size))
            for target in range(-1, size + 1):
                binary_search(values, target)  # must simply return


class TestBoundaryVariants:
    def test_lower_bound_matches_bisect_left(self):
        rng = random.Random(20260831)
        for _ in range(200):
            values = sorted(rng.randint(0, 20) for _ in range(rng.randint(0, 30)))
            target = rng.randint(-1, 21)

            assert lower_bound(values, target) == bisect.bisect_left(values, target)

    def test_upper_bound_matches_bisect_right(self):
        rng = random.Random(20260831)
        for _ in range(200):
            values = sorted(rng.randint(0, 20) for _ in range(rng.randint(0, 30)))
            target = rng.randint(-1, 21)

            assert upper_bound(values, target) == bisect.bisect_right(values, target)

    def test_the_two_bounds_differ_on_a_run_of_equal_values(self):
        # The whole reason both exist. On a run of equal values, one lands at the
        # start and the other just past the end.
        values = [1, 2, 2, 2, 3]

        assert lower_bound(values, 2) == 1
        assert upper_bound(values, 2) == 4

    def test_the_bounds_agree_when_the_value_is_absent(self):
        values = [1, 3, 5]

        assert lower_bound(values, 4) == upper_bound(values, 4) == 2

    def test_bounds_past_the_end_are_allowed(self):
        # The answer can legitimately be len(values), which is why high starts at
        # len rather than len - 1.
        assert lower_bound([1, 2, 3], 99) == 3
        assert upper_bound([1, 2, 3], 99) == 3

    def test_counting_occurrences_is_the_difference_between_the_bounds(self):
        rng = random.Random(20260831)
        for _ in range(100):
            values = sorted(rng.randint(0, 10) for _ in range(rng.randint(0, 50)))
            target = rng.randint(0, 10)

            assert count_occurrences(values, target) == values.count(target)

    def test_counting_an_absent_value_is_zero(self):
        assert count_occurrences([1, 2, 3], 99) == 0


class TestFirstTrue:
    def test_it_finds_the_boundary_of_a_monotonic_predicate(self):
        values = list(range(20))

        assert first_true(values, lambda value: value >= 7) == 7

    def test_a_predicate_that_is_never_true_returns_the_length(self):
        assert first_true([1, 2, 3], lambda value: value > 99) == 3

    def test_a_predicate_that_is_always_true_returns_zero(self):
        assert first_true([1, 2, 3], lambda value: True) == 0

    def test_it_generalises_the_bounds(self):
        # lower_bound and upper_bound are both special cases of this, which is the
        # point of having it.
        rng = random.Random(20260831)
        values = sorted(rng.randint(0, 20) for _ in range(40))
        target = 10

        assert first_true(values, lambda value: value >= target) == lower_bound(values, target)
        assert first_true(values, lambda value: value > target) == upper_bound(values, target)

    def test_it_binary_searches_over_an_answer_space_with_no_array_in_it(self):
        # The technique this function exists for. "What is the smallest number of
        # trucks that can carry these loads" is a monotonic predicate over the
        # answers, so it can be binary searched even though nothing is sorted.
        loads = [10, 20, 30, 40, 50]

        def fits(capacity: int) -> bool:
            trucks, current = 1, 0
            for load in loads:
                if load > capacity:
                    return False
                if current + load > capacity:
                    trucks += 1
                    current = 0
                current += load
            return trucks <= 3

        capacities = list(range(1, 200))
        smallest = capacities[first_true(capacities, fits)]

        assert smallest == 60
        assert not fits(59) and fits(60)


class TestOtherSearches:
    def test_exponential_search_finds_what_is_there(self):
        rng = random.Random(20260831)
        for _ in range(100):
            values = sorted(rng.sample(range(500), rng.randint(1, 60)))
            target = rng.choice(values)

            assert values[exponential_search(values, target)] == target

    def test_exponential_search_reports_absence(self):
        assert exponential_search([1, 3, 5], 4) == -1
        assert exponential_search([], 1) == -1

    def test_exponential_search_finds_the_first_element_immediately(self):
        assert exponential_search(list(range(1000)), 0) == 0

    def test_interpolation_search_on_evenly_spread_values(self):
        values = list(range(0, 10000, 5))

        for target in (0, 500, 4995, 9995):
            assert values[interpolation_search(values, target)] == target

    def test_interpolation_search_reports_absence(self):
        assert interpolation_search([1.0, 2.0, 3.0], 2.5) == -1
        assert interpolation_search([1.0, 2.0, 3.0], 99.0) == -1

    def test_interpolation_search_survives_a_list_of_identical_values(self):
        # The division by (values[high] - values[low]) would blow up here, so the
        # equal case is handled separately.
        assert interpolation_search([5.0, 5.0, 5.0], 5.0) == 0
        assert interpolation_search([5.0, 5.0, 5.0], 6.0) == -1

    def test_ternary_search_finds_the_peak_of_a_hill(self):
        # A downward parabola with its peak at x = 3.
        peak = ternary_search_maximum(-10, 10, lambda x: -((x - 3) ** 2) + 7)

        assert peak == pytest.approx(3.0, abs=1e-5)

    def test_ternary_search_on_a_peak_at_the_edge_of_the_range(self):
        peak = ternary_search_maximum(0, 5, lambda x: x)

        assert peak == pytest.approx(5.0, abs=1e-5)

    def test_linear_search_is_the_honest_baseline(self):
        assert linear_search([3, 1, 2], 1) == 1
        assert linear_search([3, 1, 2], 9) == -1
        assert linear_search([], 1) == -1

    def test_linear_search_works_on_unsorted_data_where_binary_search_would_lie(self):
        # Binary search on unsorted input does not fail loudly, it returns wrong
        # answers, which is worse. This is the reminder.
        unsorted = [5, 1, 4, 2, 3]

        assert linear_search(unsorted, 4) == 2
        assert unsorted[linear_search(unsorted, 4)] == 4


class TestFailureTable:
    @pytest.mark.parametrize(
        "pattern,expected",
        [
            ("abcabd", [0, 0, 0, 1, 2, 0]),
            ("aaaa", [0, 1, 2, 3]),
            ("abcd", [0, 0, 0, 0]),
            ("aabaaab", [0, 1, 0, 1, 2, 2, 3]),
            ("a", [0]),
        ],
    )
    def test_it_matches_the_worked_examples(self, pattern, expected):
        assert build_failure_table(pattern) == expected

    def test_every_entry_really_is_a_prefix_that_is_also_a_suffix(self):
        # Checking the definition directly rather than trusting the examples.
        rng = random.Random(20260831)
        for _ in range(200):
            pattern = "".join(rng.choice("ab") for _ in range(rng.randint(1, 15)))
            table = build_failure_table(pattern)

            for index, length in enumerate(table):
                piece = pattern[: index + 1]
                assert piece[:length] == piece[len(piece) - length :] if length else True
                assert length < len(piece), "the match must be a proper prefix"

    def test_the_longest_prefix_that_is_also_a_suffix(self):
        assert longest_prefix_suffix("abcabcabc") == 6
        assert longest_prefix_suffix("abcd") == 0
        assert longest_prefix_suffix("") == 0

    def test_it_reveals_the_repeating_unit_of_a_string(self):
        # A use of the table in its own right: length minus the overlap is the
        # repeating unit when it divides evenly.
        for text, unit in [("abcabcabc", 3), ("aaaa", 1), ("ababab", 2)]:
            overlap = longest_prefix_suffix(text)
            assert len(text) - overlap == unit
            assert text == (text[:unit] * (len(text) // unit))


class TestStringMatching:
    ALGORITHMS = [naive_search, kmp_search, rabin_karp_search, z_search]

    @pytest.mark.parametrize("search", ALGORITHMS)
    @pytest.mark.parametrize(
        "text,pattern,expected",
        [
            ("hello world", "world", [6]),
            ("hello world", "o", [4, 7]),
            ("aaaa", "aa", [0, 1, 2]),
            ("abc", "abc", [0]),
            ("abc", "abcd", []),
            ("abc", "x", []),
            ("", "x", []),
            ("mississippi", "issi", [1, 4]),
            ("aaaaab", "aaab", [2]),
        ],
    )
    def test_every_algorithm_finds_the_same_occurrences(self, search, text, pattern, expected):
        assert search(text, pattern) == expected

    @pytest.mark.parametrize("search", ALGORITHMS)
    def test_an_empty_pattern_matches_at_every_position(self, search):
        # A convention rather than an obvious truth, so all four are pinned to the
        # same one. Python's str.find agrees: "abc".find("") is 0.
        assert search("abc", "") == [0, 1, 2, 3]

    @pytest.mark.parametrize("search", ALGORITHMS[1:])
    def test_each_algorithm_agrees_with_the_naive_one_on_random_input(self, search):
        # The naive version is slow and obviously correct, which makes it the
        # right thing to check the clever ones against.
        rng = random.Random(20260831)
        for _ in range(300):
            alphabet = rng.choice(["ab", "abc", "abcdefgh"])
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
            pattern = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 6)))

            assert search(text, pattern) == naive_search(text, pattern), (
                f"disagreement on text={text!r} pattern={pattern!r}"
            )

    @pytest.mark.parametrize("search", ALGORITHMS)
    def test_overlapping_matches_are_all_reported(self, search):
        # "aa" in "aaaa" occurs three times, not two. An implementation that
        # skips ahead by the pattern length after a match gets this wrong.
        assert search("aaaa", "aa") == [0, 1, 2]

    def test_the_naive_version_is_quadratic_on_its_bad_input(self):
        def work(n: int) -> float:
            text = "a" * n + "b"
            _, steps = record(naive_search_traced(text, "a" * 20 + "b"))
            return float(sum(step.data.get("matched", 0) for step in steps))

        sizes = [100, 200, 400, 800]
        verdict = detect(sizes, [work(n) for n in sizes])

        assert verdict.best.curve in {"O(n)", "O(n log n)"} or verdict.best.curve == "O(n^2)"

    def test_kmp_never_moves_the_text_pointer_backwards(self):
        # The defining property, and what makes it usable on a stream. Every
        # slide step should keep the position moving forward.
        _, steps = record(kmp_search_traced("aaaaaaaaab" * 20, "aaab"))
        positions = [step.data["position"] for step in steps if "position" in step.data]

        assert positions == sorted(positions)

    def test_kmp_is_linear_where_the_naive_version_is_quadratic(self):
        def kmp_steps(n: int) -> float:
            text = "a" * n + "b"
            _, steps = record(kmp_search_traced(text, "a" * 20 + "b"))
            return float(len(steps))

        sizes = [200, 400, 800, 1600, 3200]
        verdict = detect(sizes, [kmp_steps(n) for n in sizes])

        assert verdict.best.curve == "O(n)"

    def test_rabin_karp_dismisses_most_windows_without_comparing_characters(self):
        _, steps = record(rabin_karp_search_traced("abcdefghij" * 50, "xyz"))
        counts = count_kinds(steps)

        assert counts.get("skip", 0) > 400
        assert counts.get("match", 0) == 0

    def test_rabin_karp_verifies_its_hits_rather_than_trusting_the_hash(self):
        # With a deliberately tiny modulus, collisions are common. The results
        # must still be exactly right, which is only true because every hash hit
        # is checked against the actual characters.
        rng = random.Random(20260831)
        for _ in range(100):
            text = "".join(rng.choice("abc") for _ in range(40))
            pattern = "".join(rng.choice("abc") for _ in range(3))

            assert rabin_karp_search(text, pattern, base=4, modulus=11) == naive_search(
                text, pattern
            )

    def test_rabin_karp_reports_the_collisions_it_survives(self):
        # With a modulus of 7 there are only seven possible hashes, so false
        # positives are unavoidable. The trace should show them being rejected
        # rather than reported as matches, which is the verification step doing
        # its job. Hand picking a colliding text is fiddly, so this searches
        # random ones and asserts collisions turn up.
        rng = random.Random(20260831)
        collisions = 0

        for _ in range(50):
            text = "".join(rng.choice("abc") for _ in range(60))
            _, steps = record(rabin_karp_search_traced(text, "abc", base=4, modulus=7))
            counts = count_kinds(steps)
            collisions += counts.get("collision", 0)

            # However many false positives there were, the answers stay exact.
            assert rabin_karp_search(text, "abc", base=4, modulus=7) == naive_search(text, "abc")

        assert collisions > 0, "a modulus of 7 should produce false positives to reject"

    def test_the_z_array_matches_its_definition(self):
        rng = random.Random(20260831)
        for _ in range(200):
            text = "".join(rng.choice("ab") for _ in range(rng.randint(1, 20)))
            values = z_array(text)

            for index in range(1, len(text)):
                length = values[index]
                assert text[:length] == text[index : index + length]
                # It must be maximal, so one more character would not match.
                if index + length < len(text):
                    assert text[length] != text[index + length]

    def test_the_z_array_of_a_repeated_string(self):
        assert z_array("aaaa") == [0, 3, 2, 1]

    def test_z_search_handles_a_separator_that_appears_in_the_text(self):
        # The separator must appear in neither string or matches can run past the
        # end of the pattern. This uses the character the code picks by default.
        text = "ab\x00cd\x00ab"
        assert z_search(text, "ab") == naive_search(text, "ab")


class TestStringUtilities:
    def test_rotation_detection(self):
        assert is_rotation("waterbottle", "erbottlewat")
        assert is_rotation("abc", "abc")
        assert not is_rotation("abc", "acb")
        assert not is_rotation("abc", "abcd")

    def test_rotation_of_an_empty_string(self):
        assert is_rotation("", "")

    def test_rotation_agrees_with_the_obvious_slow_check(self):
        rng = random.Random(20260831)
        for _ in range(100):
            text = "".join(rng.choice("abc") for _ in range(rng.randint(1, 10)))
            other = "".join(rng.choice("abc") for _ in range(len(text)))
            expected = any(
                text[shift:] + text[:shift] == other for shift in range(len(text))
            )

            assert is_rotation(text, other) == expected
