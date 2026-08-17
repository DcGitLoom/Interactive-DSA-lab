"""Tests for the recursion module.

Two things are being checked here. The obvious one is that the answers are
right. The less obvious one is that each function has the recursive *shape* it
claims to have, because that shape is the entire teaching point. A tail
recursive function that quietly records on the way back up would give the same
final list and be a completely different lesson.
"""

import math

import pytest

from dsalab.algorithms import recursion
from dsalab.tracing import count_kinds, record, run


def test_head_and_tail_recursion_produce_opposite_orders():
    down = run(recursion.print_down(4))
    up = run(recursion.print_up(4))

    assert down == [1, 2, 3, 4], "head recursion records on the way back up, so it ascends"
    assert up == [4, 3, 2, 1], "tail recursion records on the way down, so it descends"
    assert down == list(reversed(up))


def test_tail_recursion_does_all_its_work_before_bottoming_out():
    _, steps = record(recursion.print_up(3))
    kinds = [step.kind for step in steps]

    assert kinds.index("base") == len(kinds) - 1, "the base case should be the very last step"


def test_head_recursion_does_no_work_until_it_bottoms_out():
    _, steps = record(recursion.print_down(3))
    kinds = [step.kind for step in steps]

    assert kinds.index("base") < kinds.index("work"), "no work happens before the base case"


@pytest.mark.parametrize("n", [0, 1, 2, 5, 10])
def test_factorial_matches_the_standard_library(n):
    assert run(recursion.factorial(n)) == math.factorial(n)


def test_factorial_rejects_negative_input():
    with pytest.raises(ValueError):
        run(recursion.factorial(-1))


@pytest.mark.parametrize("n", [0, 1, 7, 100])
def test_total_matches_the_closed_form(n):
    assert run(recursion.total(n)) == n * (n + 1) // 2


def test_total_of_a_negative_number_is_zero():
    assert run(recursion.total(-5)) == 0


@pytest.mark.parametrize("base,exponent", [(2, 0), (2, 1), (2, 10), (3, 7), (5, 13), (1.5, 4)])
def test_power_matches_the_built_in_operator(base, exponent):
    assert run(recursion.power(base, exponent)) == pytest.approx(base**exponent)


def test_power_uses_a_logarithmic_number_of_squaring_steps():
    _, steps = record(recursion.power(2, 1024))
    squarings = count_kinds(steps).get("square", 0)

    assert squarings <= 12, "squaring should need about log2(1024) = 10 steps, not 1024"


def test_power_rejects_negative_exponents():
    with pytest.raises(ValueError):
        run(recursion.power(2, -1))


@pytest.mark.parametrize("n,expected", [(0, 0), (1, 1), (2, 1), (7, 13), (12, 144)])
def test_both_fibonacci_versions_agree(n, expected):
    assert run(recursion.fibonacci_naive(n)) == expected
    assert run(recursion.fibonacci_memoised(n)) == expected


def test_memoisation_removes_almost_all_of_the_work():
    _, naive_steps = record(recursion.fibonacci_naive(15))
    _, memo_steps = record(recursion.fibonacci_memoised(15))

    assert len(memo_steps) < len(naive_steps) / 20, (
        "memoising should turn exponential work into linear work, "
        f"but naive took {len(naive_steps)} steps and memoised took {len(memo_steps)}"
    )


def test_memoisation_actually_reuses_cached_answers():
    _, steps = record(recursion.fibonacci_memoised(10))

    assert count_kinds(steps).get("cache-hit", 0) > 0


@pytest.mark.parametrize("n,r,expected", [(5, 0, 1), (5, 5, 1), (5, 2, 10), (6, 3, 20)])
def test_combinations_matches_the_standard_library(n, r, expected):
    assert run(recursion.combinations(n, r)) == expected == math.comb(n, r)


def test_combinations_outside_the_valid_range_is_zero():
    assert run(recursion.combinations(4, 5)) == 0
    assert run(recursion.combinations(4, -1)) == 0


@pytest.mark.parametrize("disks", [0, 1, 2, 3, 6])
def test_hanoi_takes_exactly_two_to_the_n_minus_one_moves(disks):
    moves = run(recursion.towers_of_hanoi(disks))

    assert len(moves) == 2**disks - 1


def test_hanoi_never_places_a_larger_disk_on_a_smaller_one():
    pegs = {"A": list(range(4, 0, -1)), "B": [], "C": []}

    for disk, source, target in run(recursion.towers_of_hanoi(4)):
        assert pegs[source][-1] == disk, "a move must take the top disk of its peg"
        assert not pegs[target] or pegs[target][-1] > disk, "cannot stack big on small"
        pegs[target].append(pegs[source].pop())

    assert pegs["C"] == [4, 3, 2, 1], "every disk should end up on the target peg in order"


def test_taylor_series_approaches_the_real_value_of_e_to_the_x():
    assert run(recursion.taylor_e(1, 20)) == pytest.approx(math.e, rel=1e-9)
    assert run(recursion.taylor_e(2, 30)) == pytest.approx(math.exp(2), rel=1e-9)


def test_taylor_series_with_no_terms_is_zero():
    assert run(recursion.taylor_e(1, 0)) == 0.0


def test_more_taylor_terms_means_a_closer_answer():
    rough = abs(run(recursion.taylor_e(1, 4)) - math.e)
    better = abs(run(recursion.taylor_e(1, 10)) - math.e)

    assert better < rough


@pytest.mark.parametrize("m,n,expected", [(0, 0, 1), (1, 1, 3), (2, 2, 7), (3, 3, 61)])
def test_ackermann_known_values(m, n, expected):
    assert run(recursion.ackermann(m, n)) == expected


@pytest.mark.parametrize("n", [0, 1, 2, 9, 10])
def test_indirect_recursion_agrees_with_the_modulo_operator(n):
    assert run(recursion.is_even(n)) == (n % 2 == 0)
    assert run(recursion.is_odd(n)) == (n % 2 == 1)


def test_indirect_recursion_really_does_bounce_between_the_two_functions():
    _, steps = record(recursion.is_even(4))

    assert count_kinds(steps)["bounce"] == 4, "one handover per step down to zero"
