"""Tests for the heap, the priority queue and heap sort.

Two claims get measured rather than asserted: that building a heap from n items
is linear while pushing them one at a time is n log n, and that the priority
queue's position table stays correct through every swap.
"""

import heapq
import random

import pytest

from dsalab.complexity import detect
from dsalab.invariants import checked, verify
from dsalab.structures.heap import (
    BinaryHeap,
    PriorityQueue,
    heap_sort,
    heap_sort_traced,
    k_smallest,
)
from dsalab.tracing import count_kinds, record


class TestBinaryHeap:
    def test_an_empty_heap(self):
        heap = BinaryHeap()

        assert heap.is_empty()
        assert len(heap) == 0
        with pytest.raises(IndexError):
            heap.pop()
        with pytest.raises(IndexError):
            heap.peek()

    def test_the_smallest_item_comes_out_first(self):
        with checked():
            heap = BinaryHeap()
            for value in [5, 3, 8, 1, 9, 2]:
                heap.push(value)

        assert [heap.pop() for _ in range(6)] == [1, 2, 3, 5, 8, 9]

    def test_peek_does_not_remove(self):
        heap = BinaryHeap([5, 3, 8])

        assert heap.peek() == 3
        assert len(heap) == 3

    def test_it_agrees_with_the_standard_library_on_random_input(self):
        rng = random.Random(20260828)
        for _ in range(50):
            values = [rng.randint(-500, 500) for _ in range(rng.randint(1, 100))]
            mine = BinaryHeap(values)
            theirs = list(values)
            heapq.heapify(theirs)

            assert [mine.pop() for _ in range(len(values))] == sorted(values)
            assert [heapq.heappop(theirs) for _ in range(len(values))] == sorted(values)

    def test_duplicates_are_kept_rather_than_collapsed(self):
        heap = BinaryHeap([3, 1, 3, 1])

        assert [heap.pop() for _ in range(4)] == [1, 1, 3, 3]

    def test_a_key_function_turns_it_into_a_max_heap(self):
        heap = BinaryHeap([5, 3, 8, 1], key=lambda value: -value)

        assert [heap.pop() for _ in range(4)] == [8, 5, 3, 1]

    def test_a_key_function_can_order_by_a_field(self):
        tasks = [("write", 3), ("test", 1), ("ship", 2)]
        heap = BinaryHeap(tasks, key=lambda task: task[1])

        assert [heap.pop()[0] for _ in range(3)] == ["test", "ship", "write"]

    def test_the_heap_rule_holds_after_every_push_and_pop(self):
        rng = random.Random(20260828)
        with checked():
            heap = BinaryHeap()
            for _ in range(500):
                if heap.is_empty() or rng.random() < 0.6:
                    heap.push(rng.randint(0, 1000))
                else:
                    heap.pop()

    def test_a_hand_broken_heap_is_caught(self):
        heap = BinaryHeap([1, 2, 3, 4, 5])
        heap._items[0] = 99

        violations = heap.check_invariants()
        assert violations
        assert "parent" in violations[0].rule

    def test_pushing_keeps_the_tree_complete_so_the_array_has_no_gaps(self):
        heap = BinaryHeap()
        for value in range(20):
            heap.push(value)

        assert len(heap.to_list()) == 20, "a complete tree fills the array with no holes"

    def test_pop_fills_the_root_from_the_end_not_from_a_child(self):
        # Promoting a child would leave a hole one level down and break
        # completeness. The last item is used instead.
        heap = BinaryHeap([1, 2, 3, 4, 5, 6, 7])
        _, steps = record(heap.pop_traced())
        replace = [step for step in steps if step.kind == "replace"][0]

        assert replace.data["moved"] == 7, "the last item in the array fills the root"

    def test_push_pop_is_equivalent_to_a_push_then_a_pop(self):
        rng = random.Random(20260828)
        for _ in range(100):
            values = [rng.randint(0, 100) for _ in range(rng.randint(1, 20))]
            new = rng.randint(0, 100)

            combined = BinaryHeap(values)
            separate = BinaryHeap(values)

            assert combined.push_pop(new) == (separate.push(new) or separate.pop())
            assert sorted(combined.to_list()) == sorted(separate.to_list())

    def test_push_pop_on_an_empty_heap_returns_the_value_itself(self):
        assert BinaryHeap().push_pop(5) == 5


class TestHeapifyIsLinear:
    """The claim that surprises people, measured rather than asserted."""

    def test_building_all_at_once_is_linear(self):
        def swaps_to_build(n: int) -> float:
            rng = random.Random(n)
            values = [rng.randint(0, 10000) for _ in range(n)]
            heap = BinaryHeap.__new__(BinaryHeap)
            heap._key = lambda value: value
            heap._items = values
            _, steps = record(heap.heapify_traced())
            return float(count_kinds(steps).get("swap", 0))

        sizes = [128, 256, 512, 1024, 2048, 4096]
        verdict = detect(sizes, [swaps_to_build(n) for n in sizes])

        assert verdict.best.curve == "O(n)", f"heapify measured as {verdict.best.curve}"
        assert verdict.is_confident

    def test_pushing_one_at_a_time_costs_more_than_building_at_once(self):
        # Both end up with a valid heap. The difference is the number of swaps,
        # and it is the whole reason heapify exists as a separate operation.
        rng = random.Random(20260828)
        values = [rng.randint(0, 100000) for _ in range(4096)]

        one_at_a_time = BinaryHeap()
        pushed_swaps = 0
        for value in values:
            _, steps = record(one_at_a_time.push_traced(value))
            pushed_swaps += count_kinds(steps).get("swap", 0)

        at_once = BinaryHeap.__new__(BinaryHeap)
        at_once._key = lambda value: value
        at_once._items = list(values)
        _, steps = record(at_once.heapify_traced())
        built_swaps = count_kinds(steps).get("swap", 0)

        assert built_swaps < pushed_swaps
        assert built_swaps < 2 * len(values), "heapify should stay near n swaps"

    def test_heapify_produces_a_valid_heap_from_any_arrangement(self):
        rng = random.Random(20260828)
        for _ in range(100):
            values = [rng.randint(0, 500) for _ in range(rng.randint(0, 60))]
            heap = BinaryHeap(values)

            verify(heap)
            assert sorted(heap.to_list()) == sorted(values)

    def test_heapify_of_already_sorted_and_reverse_sorted_input(self):
        for values in (list(range(100)), list(range(100, 0, -1))):
            heap = BinaryHeap(values)
            verify(heap)
            assert [heap.pop() for _ in range(100)] == sorted(values)


class TestPriorityQueue:
    def test_items_come_out_in_priority_order(self):
        queue = PriorityQueue()
        queue.push("slow", 10)
        queue.push("urgent", 1)
        queue.push("normal", 5)

        assert [queue.pop()[1] for _ in range(3)] == ["urgent", "normal", "slow"]

    def test_an_empty_queue(self):
        queue = PriorityQueue()

        assert queue.is_empty()
        with pytest.raises(IndexError):
            queue.pop()
        with pytest.raises(IndexError):
            queue.peek()

    def test_decreasing_a_priority_moves_the_item_up(self):
        queue = PriorityQueue()
        queue.push("a", 10)
        queue.push("b", 5)
        queue.decrease_priority("a", 1)

        assert queue.pop()[1] == "a"

    def test_a_worse_priority_is_ignored_rather_than_applied(self):
        # Dijkstra offers routes to places it has already reached, and the
        # shorter one must win without the caller having to check first.
        queue = PriorityQueue()
        queue.push("a", 5)

        assert queue.decrease_priority("a", 10) is False
        assert queue.priority_of("a") == 5

    def test_pushing_an_item_already_present_improves_it(self):
        queue = PriorityQueue()
        queue.push("a", 10)
        queue.push("a", 3)

        assert len(queue) == 1
        assert queue.priority_of("a") == 3

    def test_asking_about_an_absent_item_raises(self):
        queue = PriorityQueue()

        with pytest.raises(KeyError):
            queue.priority_of("nobody")
        with pytest.raises(KeyError):
            queue.decrease_priority("nobody", 1)

    def test_membership_testing_is_available(self):
        queue = PriorityQueue()
        queue.push("a", 1)

        assert "a" in queue
        assert "b" not in queue

    def test_the_position_table_stays_correct_through_every_operation(self):
        # A stale position table gives a queue that works until it silently
        # starts looking in the wrong slot, so it is checked after every step.
        rng = random.Random(20260828)
        queue = PriorityQueue()
        live: dict[str, int] = {}

        for _ in range(600):
            action = rng.random()
            if action < 0.5 or not live:
                item = f"item{rng.randint(0, 80)}"
                priority = rng.randint(0, 1000)
                queue.push(item, priority)
                live[item] = min(live.get(item, priority), priority)
            elif action < 0.75:
                item = rng.choice(list(live))
                priority = rng.randint(0, 1000)
                queue.push(item, priority)
                live[item] = min(live[item], priority)
            else:
                priority, item = queue.pop()
                assert priority == live.pop(item)

            verify(queue)
            assert len(queue) == len(live)

        remaining = sorted((priority, item) for item, priority in live.items())
        assert [queue.pop() for _ in range(len(live))] == remaining

    def test_it_always_hands_back_the_current_minimum(self):
        rng = random.Random(20260828)
        queue = PriorityQueue()
        live: dict[str, int] = {}

        for index in range(200):
            item = f"n{index}"
            priority = rng.randint(0, 500)
            queue.push(item, priority)
            live[item] = priority

        while live:
            expected = min(live.values())
            priority, item = queue.pop()
            assert priority == expected
            del live[item]


class TestHeapSort:
    @pytest.mark.parametrize(
        "values",
        [
            [],
            [1],
            [2, 1],
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [3, 1, 3, 1, 3],
            [0, 0, 0],
            [-5, 10, -3, 0, 7],
        ],
    )
    def test_it_sorts_the_awkward_cases(self, values):
        assert heap_sort(values) == sorted(values)

    def test_it_agrees_with_python_on_random_input(self):
        rng = random.Random(20260828)
        for _ in range(200):
            values = [rng.randint(-1000, 1000) for _ in range(rng.randint(0, 80))]

            assert heap_sort(values) == sorted(values)

    def test_it_can_sort_descending(self):
        assert heap_sort([3, 1, 2], reverse=True) == [3, 2, 1]

    def test_it_does_not_modify_the_input(self):
        values = [3, 1, 2]
        heap_sort(values)

        assert values == [3, 1, 2]

    def test_it_extracts_exactly_one_item_per_position(self):
        _, steps = record(heap_sort_traced([5, 2, 8, 1]))

        assert count_kinds(steps)["extract"] == 3

    def test_the_number_of_swaps_grows_as_n_log_n(self):
        def swaps(n: int) -> float:
            rng = random.Random(n)
            values = [rng.randint(0, 100000) for _ in range(n)]
            _, steps = record(heap_sort_traced(values))
            return float(count_kinds(steps).get("swap", 0))

        sizes = [128, 256, 512, 1024, 2048]
        verdict = detect(sizes, [swaps(n) for n in sizes])

        assert verdict.best.curve == "O(n log n)"

    def test_sorted_input_costs_the_same_as_random_input(self):
        # Unlike quick sort, heap sort has no bad input. Its best case and worst
        # case are both n log n, which is exactly why it is used where a
        # guaranteed bound matters more than raw speed.
        def swaps(values):
            _, steps = record(heap_sort_traced(values))
            return count_kinds(steps).get("swap", 0)

        rng = random.Random(20260828)
        random_values = [rng.randint(0, 10000) for _ in range(1000)]

        ordered = swaps(list(range(1000)))
        shuffled = swaps(random_values)

        assert 0.5 < ordered / shuffled < 2.0, "no input should be much worse than another"


class TestKSmallest:
    def test_it_finds_the_k_smallest(self):
        assert k_smallest([5, 1, 9, 3, 7], 3) == [1, 3, 5]

    def test_k_of_zero_and_k_larger_than_the_input(self):
        assert k_smallest([3, 1, 2], 0) == []
        assert k_smallest([3, 1, 2], 10) == [1, 2, 3]

    def test_a_negative_k_is_rejected(self):
        with pytest.raises(ValueError):
            k_smallest([1, 2], -1)

    def test_it_agrees_with_sorting_the_whole_input(self):
        rng = random.Random(20260828)
        for _ in range(100):
            values = [rng.randint(-500, 500) for _ in range(rng.randint(1, 100))]
            k = rng.randint(0, len(values))

            assert k_smallest(values, k) == sorted(values)[:k]

    def test_it_works_on_a_stream_it_can_only_read_once(self):
        # The real justification: this never holds more than k items, so it works
        # on input far too large to sort or even to store.
        stream = (value * 7919 % 100003 for value in range(100000))

        assert len(k_smallest(stream, 5)) == 5
