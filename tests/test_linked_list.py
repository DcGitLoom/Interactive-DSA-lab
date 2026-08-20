"""Tests for the three linked list variants.

Linked lists are where pointer bugs live, and most of them are invisible from the
front of the list. So these tests deliberately poke at the places where pointers
get forgotten: emptying a list completely, removing the last node, removing the
only node, and walking a doubly linked list backwards after every change.
"""

import pytest

from dsalab.structures.linked_list import (
    CircularLinkedList,
    DoublyLinkedList,
    SinglyLinkedList,
    merge_sorted,
    merge_sorted_traced,
)
from dsalab.tracing import count_kinds, record, run


class TestSinglyLinkedList:
    def test_a_new_list_is_empty(self):
        chain = SinglyLinkedList()

        assert len(chain) == 0
        assert chain.to_list() == []
        assert chain.head is None and chain.tail is None

    def test_building_from_an_iterable_keeps_the_order(self):
        assert SinglyLinkedList([1, 2, 3]).to_list() == [1, 2, 3]

    def test_prepend_puts_values_at_the_front(self):
        chain = SinglyLinkedList()
        for value in [1, 2, 3]:
            chain.prepend(value)

        assert chain.to_list() == [3, 2, 1]

    def test_append_puts_values_at_the_back(self):
        chain = SinglyLinkedList()
        for value in [1, 2, 3]:
            chain.append(value)

        assert chain.to_list() == [1, 2, 3]

    def test_the_tail_pointer_stays_correct_after_every_kind_of_change(self):
        # A stale tail pointer is the classic singly linked list bug. It does not
        # show up when reading the list, only on the next append, which is why
        # this test appends after each operation.
        chain = SinglyLinkedList([1, 2, 3])
        chain.pop_back()
        chain.append(9)
        assert chain.to_list() == [1, 2, 9]

        chain.remove(9)
        chain.append(8)
        assert chain.to_list() == [1, 2, 8]

        chain.reverse()
        chain.append(7)
        assert chain.to_list() == [8, 2, 1, 7]

    def test_emptying_the_list_clears_the_tail_as_well_as_the_head(self):
        chain = SinglyLinkedList([1])
        chain.pop_front()

        assert chain.head is None
        assert chain.tail is None, "a tail left pointing at a removed node corrupts the next append"

        chain.append(5)
        assert chain.to_list() == [5]

    def test_insert_at_every_valid_position(self):
        chain = SinglyLinkedList([1, 3])
        chain.insert(1, 2)
        chain.insert(0, 0)
        chain.insert(len(chain), 4)

        assert chain.to_list() == [0, 1, 2, 3, 4]

    def test_insert_out_of_range_is_rejected(self):
        with pytest.raises(IndexError):
            SinglyLinkedList([1]).insert(5, 9)

    def test_inserting_at_the_front_does_no_walking(self):
        chain = SinglyLinkedList([1, 2, 3, 4])
        _, steps = record(chain.insert_traced(0, 0))

        assert count_kinds(steps).get("walk", 0) == 0

    def test_inserting_in_the_middle_walks_to_the_position(self):
        chain = SinglyLinkedList([1, 2, 3, 4])
        _, steps = record(chain.insert_traced(3, 99))

        assert count_kinds(steps)["walk"] == 2

    def test_pop_front_and_pop_back(self):
        chain = SinglyLinkedList([1, 2, 3])

        assert chain.pop_front() == 1
        assert chain.pop_back() == 3
        assert chain.to_list() == [2]

    def test_popping_the_only_node_from_either_end(self):
        for pop in ("pop_front", "pop_back"):
            chain = SinglyLinkedList([42])

            assert getattr(chain, pop)() == 42
            assert len(chain) == 0
            assert chain.head is None and chain.tail is None

    def test_popping_an_empty_list_raises(self):
        with pytest.raises(IndexError):
            SinglyLinkedList().pop_front()
        with pytest.raises(IndexError):
            SinglyLinkedList().pop_back()

    def test_remove_takes_the_first_match_only(self):
        chain = SinglyLinkedList([1, 2, 1])

        assert chain.remove(1) is True
        assert chain.to_list() == [2, 1]

    def test_remove_reports_false_when_the_value_is_absent(self):
        chain = SinglyLinkedList([1, 2])

        assert chain.remove(99) is False
        assert chain.to_list() == [1, 2]

    def test_removing_the_head_and_the_tail(self):
        chain = SinglyLinkedList([1, 2, 3])
        chain.remove(1)
        chain.remove(3)

        assert chain.to_list() == [2]
        assert chain.head is chain.tail

    def test_get_by_index(self):
        chain = SinglyLinkedList([10, 20, 30])

        assert [chain.get(i) for i in range(3)] == [10, 20, 30]

    def test_get_out_of_range_raises(self):
        with pytest.raises(IndexError):
            SinglyLinkedList([1]).get(1)

    def test_find_reports_the_index_or_minus_one(self):
        chain = SinglyLinkedList([5, 6, 7])

        assert chain.find(6) == 1
        assert chain.find(99) == -1
        assert 7 in chain
        assert 99 not in chain

    def test_search_stops_at_the_first_match(self):
        chain = SinglyLinkedList([1, 2, 3, 4, 5])
        _, steps = record(chain.find_traced(2))

        assert count_kinds(steps)["compare"] == 2

    def test_reverse_turns_the_list_around(self):
        chain = SinglyLinkedList([1, 2, 3, 4])
        chain.reverse()

        assert chain.to_list() == [4, 3, 2, 1]

    def test_reversing_twice_gets_back_to_the_original(self):
        chain = SinglyLinkedList([1, 2, 3])
        chain.reverse()
        chain.reverse()

        assert chain.to_list() == [1, 2, 3]

    def test_reversing_an_empty_or_single_list_is_harmless(self):
        empty = SinglyLinkedList()
        empty.reverse()
        assert empty.to_list() == []

        single = SinglyLinkedList([1])
        single.reverse()
        assert single.to_list() == [1]

    def test_reverse_touches_every_node_exactly_once(self):
        chain = SinglyLinkedList([1, 2, 3, 4, 5])
        _, steps = record(chain.reverse_traced())

        assert count_kinds(steps)["reverse"] == 5

    @pytest.mark.parametrize(
        "values,expected",
        [([1], 1), ([1, 2], 1), ([1, 2, 3], 2), ([1, 2, 3, 4], 2), ([1, 2, 3, 4, 5], 3)],
    )
    def test_middle_is_found_in_one_pass(self, values, expected):
        assert SinglyLinkedList(values).middle() == expected

    def test_the_middle_of_an_empty_list_raises(self):
        with pytest.raises(IndexError):
            SinglyLinkedList().middle()

    def test_a_normal_list_has_no_cycle(self):
        assert SinglyLinkedList([1, 2, 3]).has_cycle() is False
        assert SinglyLinkedList().has_cycle() is False

    def test_a_deliberately_looped_list_is_detected(self):
        chain = SinglyLinkedList([1, 2, 3, 4])
        chain.tail.next = chain.head  # tie the end back to the start by hand

        assert chain.has_cycle() is True

    def test_a_node_pointing_at_itself_is_detected(self):
        chain = SinglyLinkedList([1])
        chain.head.next = chain.head

        assert chain.has_cycle() is True

    def test_two_lists_with_the_same_values_are_equal(self):
        assert SinglyLinkedList([1, 2]) == SinglyLinkedList([1, 2])
        assert SinglyLinkedList([1, 2]) != SinglyLinkedList([2, 1])
        assert SinglyLinkedList([1]) != [1]


class TestDoublyLinkedList:
    def test_pushing_at_both_ends(self):
        chain = DoublyLinkedList()
        chain.push_back(2)
        chain.push_back(3)
        chain.push_front(1)

        assert chain.to_list() == [1, 2, 3]

    def test_the_backward_walk_matches_the_forward_walk_reversed(self):
        chain = DoublyLinkedList([1, 2, 3, 4])

        assert chain.to_list_backwards() == [4, 3, 2, 1]

    def test_the_backward_links_stay_correct_after_every_removal(self):
        # A broken prev pointer is invisible when reading forwards, so every
        # removal is checked from both directions.
        chain = DoublyLinkedList([1, 2, 3, 4, 5])

        chain.pop_front()
        assert chain.to_list_backwards() == list(reversed(chain.to_list()))

        chain.pop_back()
        assert chain.to_list_backwards() == list(reversed(chain.to_list()))

        node = chain.head.next
        chain.unlink(node)
        assert chain.to_list_backwards() == list(reversed(chain.to_list()))
        assert chain.to_list() == [2, 4]

    def test_popping_from_the_back_is_available_unlike_the_singly_linked_case(self):
        chain = DoublyLinkedList([1, 2, 3])

        assert chain.pop_back() == 3
        assert chain.pop_front() == 1
        assert chain.to_list() == [2]

    def test_popping_an_empty_list_raises(self):
        with pytest.raises(IndexError):
            DoublyLinkedList().pop_front()
        with pytest.raises(IndexError):
            DoublyLinkedList().pop_back()

    def test_emptying_the_list_clears_both_ends(self):
        chain = DoublyLinkedList([1])
        chain.pop_back()

        assert chain.head is None and chain.tail is None
        chain.push_back(2)
        assert chain.to_list() == [2]

    def test_a_node_can_be_unlinked_without_knowing_its_position(self):
        chain = DoublyLinkedList()
        chain.push_back(1)
        middle = chain.push_back(2)
        chain.push_back(3)

        assert chain.unlink(middle) == 2
        assert chain.to_list() == [1, 3]

    def test_an_unlinked_node_stops_pointing_at_the_list_it_left(self):
        chain = DoublyLinkedList([1, 2, 3])
        node = chain.head.next
        chain.unlink(node)

        assert node.prev is None and node.next is None

    def test_move_to_front_promotes_a_node_in_place(self):
        chain = DoublyLinkedList()
        chain.push_back(1)
        chain.push_back(2)
        last = chain.push_back(3)

        chain.move_to_front(last)

        assert chain.to_list() == [3, 1, 2]
        assert chain.to_list_backwards() == [2, 1, 3]
        assert len(chain) == 3

    def test_moving_the_head_to_the_front_changes_nothing(self):
        chain = DoublyLinkedList([1, 2, 3])
        chain.move_to_front(chain.head)

        assert chain.to_list() == [1, 2, 3]
        assert len(chain) == 3

    def test_length_is_right_after_a_long_mix_of_operations(self):
        chain = DoublyLinkedList()
        for value in range(10):
            chain.push_back(value)
        for _ in range(3):
            chain.pop_front()
        for _ in range(2):
            chain.pop_back()

        assert len(chain) == 5 == len(chain.to_list())


class TestCircularLinkedList:
    def test_appending_builds_the_circle_in_order(self):
        circle = CircularLinkedList([1, 2, 3])

        assert circle.to_list() == [1, 2, 3]
        assert len(circle) == 3

    def test_the_last_node_points_back_at_the_first(self):
        circle = CircularLinkedList([1, 2, 3])

        assert circle.tail.next is circle.head
        assert circle.head.value == 1

    def test_a_circle_of_one_points_at_itself(self):
        circle = CircularLinkedList([7])

        assert circle.tail.next is circle.tail

    def test_prepend_puts_a_value_at_the_front_without_moving_the_tail(self):
        circle = CircularLinkedList([2, 3])
        circle.prepend(1)

        assert circle.to_list() == [1, 2, 3]
        assert circle.tail.value == 3

    def test_prepending_into_an_empty_circle_works(self):
        circle = CircularLinkedList()
        circle.prepend(1)

        assert circle.to_list() == [1]

    def test_iterating_stops_after_one_lap_rather_than_looping_forever(self):
        circle = CircularLinkedList([1, 2, 3])

        assert list(circle) == [1, 2, 3]

    def test_an_empty_circle_iterates_to_nothing(self):
        assert list(CircularLinkedList()) == []

    def test_removing_a_middle_value(self):
        circle = CircularLinkedList([1, 2, 3])

        assert circle.remove(2) is True
        assert circle.to_list() == [1, 3]
        assert circle.tail.next is circle.head, "the circle must stay closed"

    def test_removing_the_tail_moves_the_tail_pointer_back(self):
        circle = CircularLinkedList([1, 2, 3])
        circle.remove(3)

        assert circle.tail.value == 2
        assert circle.to_list() == [1, 2]

    def test_removing_the_only_value_empties_the_circle(self):
        circle = CircularLinkedList([1])

        assert circle.remove(1) is True
        assert circle.tail is None
        assert len(circle) == 0

    def test_removing_something_absent_reports_false(self):
        assert CircularLinkedList([1, 2]).remove(9) is False

    def test_josephus_with_a_step_of_two_matches_the_known_answer(self):
        # With 7 people counting every second one, the survivor is number 7.
        circle = CircularLinkedList(range(1, 8))

        assert run(circle.josephus(2)) == 7

    def test_josephus_with_a_step_of_one_leaves_the_last_person(self):
        circle = CircularLinkedList([1, 2, 3, 4])

        assert run(circle.josephus(1)) == 4

    def test_josephus_removes_everyone_but_one(self):
        circle = CircularLinkedList(range(6))
        _, steps = record(circle.josephus(3))

        assert count_kinds(steps)["eliminate"] == 5
        assert len(circle) == 1

    def test_josephus_on_an_empty_circle_raises(self):
        with pytest.raises(IndexError):
            run(CircularLinkedList().josephus(2))

    def test_josephus_rejects_a_step_below_one(self):
        with pytest.raises(ValueError):
            run(CircularLinkedList([1, 2]).josephus(0))


class TestMergeSorted:
    def test_merging_two_sorted_lists(self):
        left = SinglyLinkedList([1, 3, 5])
        right = SinglyLinkedList([2, 4, 6])

        assert merge_sorted(left, right).to_list() == [1, 2, 3, 4, 5, 6]

    def test_merging_with_an_empty_list_returns_the_other_one(self):
        assert merge_sorted(SinglyLinkedList([1, 2]), SinglyLinkedList()).to_list() == [1, 2]
        assert merge_sorted(SinglyLinkedList(), SinglyLinkedList([1, 2])).to_list() == [1, 2]

    def test_merging_two_empty_lists(self):
        assert merge_sorted(SinglyLinkedList(), SinglyLinkedList()).to_list() == []

    def test_merging_lists_of_very_different_lengths(self):
        left = SinglyLinkedList([1])
        right = SinglyLinkedList([2, 3, 4, 5])

        assert merge_sorted(left, right).to_list() == [1, 2, 3, 4, 5]

    def test_duplicates_survive_the_merge(self):
        left = SinglyLinkedList([1, 1, 2])
        right = SinglyLinkedList([1, 2])

        assert merge_sorted(left, right).to_list() == [1, 1, 1, 2, 2]

    def test_equal_values_take_from_the_left_list_first_so_the_merge_is_stable(self):
        left = SinglyLinkedList([1])
        right = SinglyLinkedList([1])
        _, steps = record(merge_sorted_traced(left, right))

        assert steps[0].data["from"] == "left"

    def test_the_whole_merge_is_reported_including_the_tail(self):
        left = SinglyLinkedList([1])
        right = SinglyLinkedList([2, 3, 4])
        _, steps = record(merge_sorted_traced(left, right))

        assert len(steps) == 4, "every value placed should produce a step"
