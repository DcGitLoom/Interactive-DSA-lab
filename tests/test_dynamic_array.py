"""Tests for the dynamic array.

Beyond the obvious behaviour, these check the two things that are easy to get
wrong and invisible from the outside: that growth really is doubling (so append
stays amortised constant), and that shrinking does not thrash.
"""

import pytest

from dsalab.complexity import detect
from dsalab.structures.dynamic_array import INITIAL_CAPACITY, DynamicArray
from dsalab.tracing import count_kinds, record


def test_a_new_array_is_empty():
    array = DynamicArray()

    assert len(array) == 0
    assert array.to_list() == []


def test_it_can_be_built_from_any_iterable():
    assert DynamicArray([1, 2, 3]).to_list() == [1, 2, 3]
    assert DynamicArray(range(4)).to_list() == [0, 1, 2, 3]
    assert DynamicArray("abc").to_list() == ["a", "b", "c"]


def test_append_then_read_back():
    array = DynamicArray()
    for value in range(10):
        array.append(value)

    assert len(array) == 10
    assert [array[i] for i in range(10)] == list(range(10))


def test_indexing_from_the_end_works_like_python_does_it():
    array = DynamicArray([10, 20, 30])

    assert array[-1] == 30
    assert array[-3] == 10


@pytest.mark.parametrize("bad_index", [3, 4, -4, 100])
def test_reading_outside_the_array_raises(bad_index):
    array = DynamicArray([1, 2, 3])

    with pytest.raises(IndexError):
        array[bad_index]


def test_reading_from_an_empty_array_raises():
    with pytest.raises(IndexError):
        DynamicArray()[0]


def test_assignment_replaces_in_place_without_changing_the_length():
    array = DynamicArray([1, 2, 3])
    array[1] = 99

    assert array.to_list() == [1, 99, 3]
    assert len(array) == 3


def test_capacity_doubles_rather_than_creeping_up_one_slot_at_a_time():
    array = DynamicArray()
    capacities = []
    for value in range(50):
        array.append(value)
        capacities.append(array.capacity)

    distinct = sorted(set(capacities))
    for smaller, bigger in zip(distinct, distinct[1:], strict=False):
        assert bigger == smaller * 2, f"capacity went {smaller} to {bigger}, which is not doubling"


def test_growth_happens_rarely_enough_for_append_to_be_amortised_constant():
    def copies_made_while_appending(n: int) -> float:
        array = DynamicArray()
        copied = 0
        for value in range(n):
            _, steps = record(array.append_traced(value))
            for step in steps:
                if step.kind == "grow":
                    copied += step.data["copied"]
        return float(copied)

    # If growth were linear rather than multiplicative, total copying would be
    # quadratic. Doubling makes the total copying grow linearly with n, which is
    # the whole justification for calling append O(1) amortised.
    verdict = detect([64, 128, 256, 512, 1024, 2048], [copies_made_while_appending(n) for n in
                                                       [64, 128, 256, 512, 1024, 2048]])

    assert verdict.best.curve == "O(n)"


def test_insert_shifts_everything_to_its_right():
    array = DynamicArray([1, 2, 4])
    array.insert(2, 3)

    assert array.to_list() == [1, 2, 3, 4]


def test_insert_at_the_front_and_at_the_end():
    array = DynamicArray([2, 3])
    array.insert(0, 1)
    array.insert(len(array), 4)

    assert array.to_list() == [1, 2, 3, 4]


def test_insert_into_an_empty_array():
    array = DynamicArray()
    array.insert(0, "only")

    assert array.to_list() == ["only"]


def test_insert_beyond_the_end_is_rejected():
    array = DynamicArray([1, 2])

    with pytest.raises(IndexError):
        array.insert(5, 99)


def test_inserting_at_the_front_shifts_every_existing_item():
    array = DynamicArray([1, 2, 3])
    _, steps = record(array.insert_traced(0, 0))

    assert count_kinds(steps)["shift"] == 3, "all three existing items must move right"


def test_appending_shifts_nothing():
    array = DynamicArray([1, 2, 3])
    _, steps = record(array.append_traced(4))

    assert count_kinds(steps).get("shift", 0) == 0


def test_pop_removes_from_the_end_by_default():
    array = DynamicArray([1, 2, 3])

    assert array.pop() == 3
    assert array.to_list() == [1, 2]


def test_pop_from_the_middle_closes_the_gap():
    array = DynamicArray([1, 2, 3, 4])

    assert array.pop(1) == 2
    assert array.to_list() == [1, 3, 4]


def test_pop_from_an_empty_array_raises():
    with pytest.raises(IndexError):
        DynamicArray().pop()


def test_the_array_shrinks_once_it_is_only_a_quarter_full():
    array = DynamicArray(range(32))
    full_capacity = array.capacity

    while len(array) > 4:
        array.pop()

    assert array.capacity < full_capacity, "memory should be released as the array empties"


def test_shrinking_never_falls_below_the_starting_capacity():
    array = DynamicArray(range(20))
    while len(array):
        array.pop()

    assert array.capacity == INITIAL_CAPACITY


def test_repeatedly_appending_and_popping_at_the_boundary_does_not_thrash():
    # This is the exact pattern that a shrink threshold of one half would turn
    # into an O(n) copy on every single operation. With the quarter threshold
    # there should be no resizing at all once the capacity has settled.
    array = DynamicArray(range(8))
    resizes = 0

    for _ in range(50):
        _, append_steps = record(array.append_traced(99))
        _, pop_steps = record(array.pop_traced())
        resizes += count_kinds(append_steps).get("grow", 0)
        resizes += count_kinds(pop_steps).get("shrink", 0)

    assert resizes <= 1, f"the boundary loop caused {resizes} resizes, which is thrashing"


def test_popping_clears_the_vacated_slot_so_the_object_can_be_collected():
    import gc
    import weakref

    class Tracked:
        pass

    array = DynamicArray()
    tracked = Tracked()
    array.append(tracked)
    reference = weakref.ref(tracked)

    array.pop()
    del tracked
    gc.collect()

    assert reference() is None, "the popped object is still being held alive by a stale slot"


def test_index_finds_the_first_match_and_reports_minus_one_when_absent():
    array = DynamicArray([5, 7, 5])

    assert array.index(5) == 0
    assert array.index(7) == 1
    assert array.index(99) == -1


def test_membership_testing():
    array = DynamicArray([1, 2, 3])

    assert 2 in array
    assert 99 not in array


def test_searching_stops_as_soon_as_it_finds_the_value():
    array = DynamicArray([1, 2, 3, 4, 5])
    _, steps = record(array.index_traced(2))

    assert count_kinds(steps)["compare"] == 2, "it should not keep looking after a match"


def test_remove_takes_out_the_first_occurrence_only():
    array = DynamicArray([1, 2, 1])
    array.remove(1)

    assert array.to_list() == [2, 1]


def test_removing_something_that_is_not_there_raises():
    with pytest.raises(ValueError):
        DynamicArray([1, 2]).remove(99)


def test_clear_empties_the_array_and_releases_the_memory():
    array = DynamicArray(range(100))
    array.clear()

    assert len(array) == 0
    assert array.capacity == INITIAL_CAPACITY
    assert array.to_list() == []


def test_iteration_visits_every_item_in_order():
    assert list(DynamicArray([3, 1, 2])) == [3, 1, 2]


def test_iterating_an_empty_array_yields_nothing():
    assert list(DynamicArray()) == []


def test_two_arrays_with_the_same_contents_are_equal():
    assert DynamicArray([1, 2]) == DynamicArray([1, 2])
    assert DynamicArray([1, 2]) != DynamicArray([1, 3])
    assert DynamicArray([1, 2]) != DynamicArray([1, 2, 3])


def test_comparing_against_a_plain_list_is_not_claimed_to_be_equal():
    assert DynamicArray([1, 2]) != [1, 2]


def test_it_holds_mixed_types_and_none_without_complaint():
    array = DynamicArray([1, "two", None, 3.5])

    assert array.to_list() == [1, "two", None, 3.5]
    assert None in array


def test_every_step_produced_carries_a_readable_note():
    array = DynamicArray()
    _, steps = record(array.append_traced(1))

    assert all(step.note.endswith(".") for step in steps)
