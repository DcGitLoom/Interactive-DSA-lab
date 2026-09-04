"""Tests for the queues and the deque.

The circular queue tests concentrate on the wrap point, because that is where
ring buffer bugs live and where a test that only fills a fresh queue never
reaches.
"""

import random

import pytest

from dsalab.structures.deque import (
    Deque,
    DequeEmptyError,
    sliding_window_maximum,
    sliding_window_maximum_traced,
)
from dsalab.structures.queue import (
    CircularQueue,
    DynamicQueue,
    LinkedQueue,
    QueueEmptyError,
    QueueFromTwoStacks,
    QueueFullError,
)
from dsalab.tracing import count_kinds, record

GROWABLE_QUEUES = [DynamicQueue, LinkedQueue, QueueFromTwoStacks]


@pytest.mark.parametrize("kind", GROWABLE_QUEUES)
class TestEveryGrowableQueue:
    """Behaviour every queue must share, whatever it is built from."""

    def test_a_new_queue_is_empty(self, kind):
        queue = kind()

        assert queue.is_empty()
        assert len(queue) == 0

    def test_first_in_is_first_out(self, kind):
        queue = kind([1, 2, 3])

        assert queue.dequeue() == 1
        assert queue.dequeue() == 2
        assert queue.dequeue() == 3

    def test_peek_shows_the_oldest_without_removing_it(self, kind):
        queue = kind([1, 2])

        assert queue.peek() == 1
        assert len(queue) == 2

    def test_taking_from_an_empty_queue_raises(self, kind):
        with pytest.raises(QueueEmptyError):
            kind().dequeue()

    def test_peeking_at_an_empty_queue_raises(self, kind):
        with pytest.raises(QueueEmptyError):
            kind().peek()

    def test_it_can_be_reused_after_being_emptied(self, kind):
        queue = kind([1])
        queue.dequeue()
        queue.enqueue(9)

        assert queue.peek() == 9
        assert len(queue) == 1

    def test_to_list_reads_front_to_back(self, kind):
        assert kind([1, 2, 3]).to_list() == [1, 2, 3]

    def test_interleaved_adds_and_removes_keep_the_order(self, kind):
        queue = kind()
        mirror = []
        random.seed(20260822)

        for value in range(300):
            queue.enqueue(value)
            mirror.append(value)
            if random.random() < 0.4:
                assert queue.dequeue() == mirror.pop(0)

        assert queue.to_list() == mirror
        assert len(queue) == len(mirror)


class TestCircularQueue:
    def test_it_refuses_to_overflow(self):
        queue = CircularQueue(2, [1, 2])

        assert queue.is_full()
        with pytest.raises(QueueFullError):
            queue.enqueue(3)

    def test_a_capacity_below_one_is_rejected(self):
        with pytest.raises(ValueError):
            CircularQueue(0)

    def test_slots_are_reused_once_the_queue_wraps_around(self):
        # This is the whole point of the structure. Fill it, empty part of it,
        # then add more than the array has room for at the end.
        queue = CircularQueue(3, [1, 2, 3])
        queue.dequeue()
        queue.dequeue()
        queue.enqueue(4)
        queue.enqueue(5)

        assert queue.to_list() == [3, 4, 5]
        assert len(queue) == 3

    def test_the_order_survives_many_laps_around_the_ring(self):
        queue = CircularQueue(4)
        for value in range(100):
            queue.enqueue(value)
            assert queue.dequeue() == value

        assert queue.is_empty()

    def test_full_and_empty_are_told_apart_even_though_the_pointers_agree(self):
        # After wrapping, front and back can point at the same slot in both the
        # full and the empty state. The count is what distinguishes them.
        queue = CircularQueue(3, [1, 2, 3])
        assert queue.is_full() and not queue.is_empty()

        for _ in range(3):
            queue.dequeue()
        assert queue.is_empty() and not queue.is_full()

    def test_iteration_reads_front_to_back_across_the_wrap(self):
        queue = CircularQueue(4, [1, 2, 3])
        queue.dequeue()
        queue.enqueue(4)
        queue.enqueue(5)

        assert list(queue) == [2, 3, 4, 5]

    def test_a_dequeued_object_is_not_held_alive_by_a_stale_slot(self):
        import gc
        import weakref

        class Tracked:
            pass

        queue = CircularQueue(2)
        tracked = Tracked()
        queue.enqueue(tracked)
        reference = weakref.ref(tracked)

        queue.dequeue()
        del tracked
        gc.collect()

        assert reference() is None


class TestDynamicQueue:
    def test_it_grows_instead_of_refusing(self):
        queue = DynamicQueue(initial_capacity=2)
        for value in range(10):
            queue.enqueue(value)

        assert len(queue) == 10
        assert queue.to_list() == list(range(10))

    def test_capacity_doubles(self):
        queue = DynamicQueue(initial_capacity=2)
        for value in range(5):
            queue.enqueue(value)

        assert queue.capacity == 8

    def test_growing_after_the_ring_has_wrapped_keeps_the_order(self):
        # The bug this catches: copying the old buffer in memory order rather
        # than logical order. It only appears once the ring has wrapped, so a
        # test that fills a fresh queue would never find it.
        queue = DynamicQueue(initial_capacity=4)
        for value in range(4):
            queue.enqueue(value)
        queue.dequeue()
        queue.dequeue()
        queue.enqueue(4)
        queue.enqueue(5)  # the ring has now wrapped
        queue.enqueue(6)  # this one forces a grow

        assert queue.to_list() == [2, 3, 4, 5, 6]


class TestQueueFromTwoStacks:
    def test_it_behaves_exactly_like_a_queue(self):
        queue = QueueFromTwoStacks([1, 2, 3])

        assert queue.dequeue() == 1
        queue.enqueue(4)
        assert queue.dequeue() == 2
        assert queue.to_list() == [3, 4]

    def test_each_item_is_moved_between_the_stacks_at_most_once(self):
        # The amortised argument depends on this. If an item could be tipped
        # more than once the cost would not be constant.
        queue = QueueFromTwoStacks()
        for value in range(100):
            queue.enqueue(value)
        taken = [queue.dequeue() for _ in range(100)]

        assert taken == list(range(100))

    def test_adds_and_removes_can_be_interleaved_freely(self):
        queue = QueueFromTwoStacks()
        queue.enqueue(1)
        assert queue.dequeue() == 1

        queue.enqueue(2)
        queue.enqueue(3)
        assert queue.dequeue() == 2

        queue.enqueue(4)
        assert queue.dequeue() == 3
        assert queue.dequeue() == 4
        assert queue.is_empty()

    def test_peek_also_triggers_the_tip_when_needed(self):
        queue = QueueFromTwoStacks([1, 2])

        assert queue.peek() == 1
        assert queue.dequeue() == 1


class TestDeque:
    def test_it_adds_and_removes_at_both_ends(self):
        deque = Deque()
        deque.push_back(2)
        deque.push_back(3)
        deque.push_front(1)

        assert deque.to_list() == [1, 2, 3]
        assert deque.pop_front() == 1
        assert deque.pop_back() == 3
        assert deque.to_list() == [2]

    def test_used_at_one_end_only_it_is_a_stack(self):
        deque = Deque()
        for value in [1, 2, 3]:
            deque.push_front(value)

        assert [deque.pop_front() for _ in range(3)] == [3, 2, 1]

    def test_used_at_both_ends_it_is_a_queue(self):
        deque = Deque()
        for value in [1, 2, 3]:
            deque.push_back(value)

        assert [deque.pop_front() for _ in range(3)] == [1, 2, 3]

    def test_peeking_at_both_ends(self):
        deque = Deque([1, 2, 3])

        assert deque.peek_front() == 1
        assert deque.peek_back() == 3
        assert len(deque) == 3

    @pytest.mark.parametrize("method", ["pop_front", "pop_back", "peek_front", "peek_back"])
    def test_every_read_on_an_empty_deque_raises(self, method):
        with pytest.raises(DequeEmptyError):
            getattr(Deque(), method)()

    def test_emptying_and_refilling_from_the_other_end(self):
        deque = Deque([1])
        deque.pop_back()
        deque.push_front(2)

        assert deque.to_list() == [2]
        assert len(deque) == 1


class TestSlidingWindowMaximum:
    @pytest.mark.parametrize(
        "values,window,expected",
        [
            ([1, 3, -1, -3, 5, 3, 6, 7], 3, [3, 3, 5, 5, 6, 7]),
            ([1, 2, 3, 4], 1, [1, 2, 3, 4]),
            ([4, 3, 2, 1], 2, [4, 3, 2]),
            ([1, 1, 1], 2, [1, 1]),
            ([5], 1, [5]),
            ([1, 2, 3, 4], 4, [4]),
        ],
    )
    def test_it_matches_the_slow_but_obvious_method(self, values, window, expected):
        assert sliding_window_maximum(values, window) == expected

    def test_it_agrees_with_brute_force_on_random_input(self):
        random.seed(20260822)
        for _ in range(100):
            values = [random.randint(-50, 50) for _ in range(random.randint(1, 30))]
            window = random.randint(1, len(values))
            brute_force = [
                max(values[start : start + window]) for start in range(len(values) - window + 1)
            ]

            assert sliding_window_maximum(values, window) == brute_force

    def test_the_total_work_stays_linear_regardless_of_window_size(self):
        # The point of the algorithm. A brute force version would do about
        # n times window comparisons; this should stay near 2n whatever the
        # window is, because each position is added once and removed at most once.
        values = list(range(200, 0, -1))
        for window in (5, 50, 150):
            _, steps = record(sliding_window_maximum_traced(values, window))
            movements = count_kinds(steps).get("discard", 0) + count_kinds(steps).get("expire", 0)

            assert movements <= 2 * len(values)

    def test_a_window_larger_than_the_input_is_rejected(self):
        with pytest.raises(ValueError):
            sliding_window_maximum([1, 2], 5)

    def test_a_window_below_one_is_rejected(self):
        with pytest.raises(ValueError):
            sliding_window_maximum([1, 2], 0)
