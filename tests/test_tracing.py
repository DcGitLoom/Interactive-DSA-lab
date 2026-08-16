"""Tests for the step tracing layer.

The tracing layer is the foundation everything else stands on, so it gets tested
first and carefully. If `run` ever loses the return value, or `record` ever drops
a step, every animation and every benchmark in the project becomes a lie.
"""

from dsalab.tracing import Step, Traced, count_kinds, record, run


def counting_algorithm(n: int) -> Traced[int]:
    """A tiny traced algorithm used only for testing the tracing layer itself.

    It adds up the numbers from 0 to n - 1, yielding one step per addition, and
    returns the total.
    """
    total = 0
    for i in range(n):
        total += i
        yield Step("add", f"Add {i}, running total is now {total}.", {"i": i, "total": total})
    return total


def test_run_returns_the_answer_and_ignores_steps():
    assert run(counting_algorithm(5)) == 10


def test_run_on_an_algorithm_that_yields_nothing():
    assert run(counting_algorithm(0)) == 0


def test_record_returns_both_the_answer_and_every_step():
    answer, steps = record(counting_algorithm(4))

    assert answer == 6
    assert len(steps) == 4
    assert [step.data["i"] for step in steps] == [0, 1, 2, 3]


def test_record_keeps_the_steps_in_the_order_they_happened():
    _, steps = record(counting_algorithm(3))
    totals = [step.data["total"] for step in steps]

    assert totals == sorted(totals), "steps must arrive in the order the algorithm produced them"


def test_every_step_carries_a_plain_english_note():
    _, steps = record(counting_algorithm(3))

    for step in steps:
        assert step.note, "a step with no note would show up as a blank caption in the app"
        assert step.note.endswith("."), "notes are written as sentences"


def test_count_kinds_groups_steps_by_their_label():
    steps = [
        Step("compare", "a"),
        Step("swap", "b"),
        Step("compare", "c"),
    ]

    assert count_kinds(steps) == {"compare": 2, "swap": 1}


def test_count_kinds_of_an_empty_run_is_an_empty_dict():
    assert count_kinds([]) == {}


def test_step_data_defaults_to_an_empty_dict():
    step = Step("visit", "Nothing extra to show here.")

    assert step.data == {}


def test_steps_are_frozen_so_a_recorded_run_cannot_be_edited_afterwards():
    step = Step("compare", "Comparing two values.")

    try:
        step.kind = "swap"  # type: ignore[misc]
    except Exception as error:
        assert error.__class__.__name__ == "FrozenInstanceError"
    else:
        raise AssertionError("a recorded step should not be editable after the fact")
