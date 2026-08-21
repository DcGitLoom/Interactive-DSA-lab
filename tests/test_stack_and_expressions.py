"""Tests for stacks and for the expression algorithms built on them.

The expression tests lean hard on the awkward cases: precedence, right
associative exponentiation, unary looking input, mismatched brackets and
division by zero. Those are where conversion code is nearly always wrong, and
they are all invisible if you only test `2 + 3`.
"""

import pytest

from dsalab.algorithms.expressions import (
    brackets_balanced,
    brackets_balanced_traced,
    evaluate_infix,
    evaluate_postfix,
    evaluate_postfix_traced,
    infix_to_postfix,
    infix_to_prefix,
    tokenize,
)
from dsalab.structures.stack import ArrayStack, LinkedStack, StackEmptyError
from dsalab.tracing import count_kinds, record

STACKS = [ArrayStack, LinkedStack]


@pytest.mark.parametrize("kind", STACKS)
class TestBothStacks:
    """Every test here runs against both implementations.

    They are meant to be interchangeable, so anything true of one must be true of
    the other. Writing the tests once and parametrising them is what keeps that
    promise honest.
    """

    def test_a_new_stack_is_empty(self, kind):
        stack = kind()

        assert stack.is_empty()
        assert len(stack) == 0

    def test_last_in_is_first_out(self, kind):
        stack = kind()
        for value in [1, 2, 3]:
            stack.push(value)

        assert stack.pop() == 3
        assert stack.pop() == 2
        assert stack.pop() == 1

    def test_peek_shows_the_top_without_removing_it(self, kind):
        stack = kind([1, 2])

        assert stack.peek() == 2
        assert len(stack) == 2

    def test_popping_an_empty_stack_raises(self, kind):
        with pytest.raises(StackEmptyError):
            kind().pop()

    def test_peeking_at_an_empty_stack_raises(self, kind):
        with pytest.raises(StackEmptyError):
            kind().peek()

    def test_the_error_is_still_an_index_error_for_ordinary_code(self, kind):
        with pytest.raises(IndexError):
            kind().pop()

    def test_building_from_an_iterable_pushes_in_order(self, kind):
        stack = kind([1, 2, 3])

        assert stack.peek() == 3
        assert stack.to_list() == [1, 2, 3]

    def test_iteration_runs_from_the_top_down(self, kind):
        assert list(kind([1, 2, 3])) == [3, 2, 1]

    def test_it_becomes_empty_again_after_everything_is_popped(self, kind):
        stack = kind([1, 2])
        stack.pop()
        stack.pop()

        assert stack.is_empty()
        assert len(stack) == 0

    def test_it_can_be_reused_after_being_emptied(self, kind):
        stack = kind([1])
        stack.pop()
        stack.push(9)

        assert stack.peek() == 9

    def test_it_holds_any_type_including_none(self, kind):
        stack = kind()
        stack.push(None)
        stack.push("text")

        assert stack.pop() == "text"
        assert stack.pop() is None

    def test_a_long_run_of_pushes_and_pops_stays_consistent(self, kind):
        stack = kind()
        mirror = []
        for value in range(200):
            stack.push(value)
            mirror.append(value)
            if value % 3 == 0:
                assert stack.pop() == mirror.pop()

        assert stack.to_list() == mirror


def test_the_two_stacks_agree_on_everything():
    array_stack, linked_stack = ArrayStack(), LinkedStack()
    for value in range(50):
        array_stack.push(value)
        linked_stack.push(value)

    assert array_stack.to_list() == linked_stack.to_list()
    assert [array_stack.pop() for _ in range(50)] == [linked_stack.pop() for _ in range(50)]


class TestBracketMatching:
    @pytest.mark.parametrize(
        "text", ["", "()", "()[]{}", "([{}])", "a(b[c]d)e", "((()))", "no brackets here"]
    )
    def test_balanced_input_is_accepted(self, text):
        assert brackets_balanced(text) is True

    @pytest.mark.parametrize("text", ["(", ")", "(]", "([)]", "((", "))", "a(b]c"])
    def test_unbalanced_input_is_rejected(self, text):
        assert brackets_balanced(text) is False

    def test_the_case_a_counter_would_get_wrong(self):
        # Two openings and two closings, so a counting approach says yes. The
        # order is wrong, so a stack says no. This is the reason the stack is
        # needed at all.
        assert brackets_balanced("([)]") is False

    def test_it_stops_as_soon_as_it_finds_a_problem(self):
        _, steps = record(brackets_balanced_traced("(]))))))))"))

        assert steps[-1].kind == "mismatch"
        assert len(steps) == 2, "it should not keep reading after the answer is settled"

    def test_the_reported_depth_tracks_the_nesting(self):
        _, steps = record(brackets_balanced_traced("((()))"))
        depths = [step.data["depth"] for step in steps if "depth" in step.data]

        assert max(depths) == 3


class TestTokenising:
    def test_multi_digit_numbers_stay_together(self):
        assert tokenize("12 + 345") == ["12", "+", "345"]

    def test_decimals_stay_together(self):
        assert tokenize("1.5*2") == ["1.5", "*", "2"]

    def test_names_can_be_more_than_one_letter(self):
        assert tokenize("total + count_2") == ["total", "+", "count_2"]

    def test_whitespace_is_ignored(self):
        assert tokenize("  1  +  2 ") == ["1", "+", "2"]

    def test_brackets_become_their_own_tokens(self):
        assert tokenize("(1+2)") == ["(", "1", "+", "2", ")"]

    def test_an_unknown_character_is_rejected_with_its_position(self):
        with pytest.raises(ValueError, match="position"):
            tokenize("1 $ 2")


class TestInfixToPostfix:
    @pytest.mark.parametrize(
        "infix,expected",
        [
            ("2+3", "2 3 +"),
            ("2+3*4", "2 3 4 * +"),
            ("2*3+4", "2 3 * 4 +"),
            ("(2+3)*4", "2 3 + 4 *"),
            ("a+b*c-d", "a b c * + d -"),
            ("((a+b))", "a b +"),
            ("a*(b+c)/d", "a b c + * d /"),
            ("10", "10"),
        ],
    )
    def test_conversion_matches_the_expected_output(self, infix, expected):
        assert " ".join(infix_to_postfix(infix)) == expected

    def test_precedence_is_respected_without_brackets(self):
        assert " ".join(infix_to_postfix("2+3*4")) == "2 3 4 * +"

    def test_brackets_override_precedence_and_then_disappear(self):
        result = infix_to_postfix("(2+3)*4")

        assert "(" not in result and ")" not in result
        assert " ".join(result) == "2 3 + 4 *"

    def test_subtraction_groups_left_to_right(self):
        # 8 - 3 - 2 must mean (8 - 3) - 2 = 3, not 8 - (3 - 2) = 7.
        assert evaluate_postfix(infix_to_postfix("8-3-2")) == 3

    def test_exponentiation_groups_right_to_left(self):
        # This is the case that a naive equal precedence rule gets wrong.
        # 2 ^ 3 ^ 2 is 2 ^ 9 = 512, not (2 ^ 3) ^ 2 = 64.
        assert " ".join(infix_to_postfix("2^3^2")) == "2 3 2 ^ ^"
        assert evaluate_postfix(infix_to_postfix("2^3^2")) == 512

    def test_mixed_precedence_with_exponent(self):
        assert " ".join(infix_to_postfix("2+3^2*4")) == "2 3 2 ^ 4 * +"

    def test_an_unclosed_bracket_is_rejected(self):
        with pytest.raises(ValueError, match="never closed"):
            infix_to_postfix("(2+3")

    def test_a_stray_closing_bracket_is_rejected(self):
        with pytest.raises(ValueError, match="nothing open"):
            infix_to_postfix("2+3)")

    def test_square_and_curly_brackets_work_too(self):
        assert " ".join(infix_to_postfix("[2+3]*{4-1}")) == "2 3 + 4 1 - *"


class TestInfixToPrefix:
    @pytest.mark.parametrize(
        "infix,expected",
        [
            ("2+3", "+ 2 3"),
            ("2+3*4", "+ 2 * 3 4"),
            ("(2+3)*4", "* + 2 3 4"),
            ("a+b*c-d", "- + a * b c d"),
        ],
    )
    def test_conversion_matches_the_expected_output(self, infix, expected):
        assert " ".join(infix_to_prefix(infix)) == expected

    def test_left_grouping_survives_the_reversal(self):
        # The bug the reverse trick invites: a + b * c - d means (a + (b*c)) - d,
        # so the minus is the outermost operator and must come first in prefix.
        # Running the reversed tokens through the ordinary rules gets this wrong.
        assert " ".join(infix_to_prefix("a+b*c-d")) == "- + a * b c d"
        assert " ".join(infix_to_prefix("8-3-2")) == "- - 8 3 2"

    def test_right_grouping_also_survives_the_reversal(self):
        # Exponentiation grouping the other way is the mirror case, and a fix
        # that only flips one direction would break it.
        assert " ".join(infix_to_prefix("2^3^2")) == "^ 2 ^ 3 2"

    def test_prefix_keeps_the_same_meaning_as_the_infix_it_came_from(self):
        # Evaluating prefix is the postfix algorithm on the reversed tokens with
        # the operands swapped, so rather than write it, check the shape instead:
        # the operator that binds loosest must come first in prefix.
        assert infix_to_prefix("2+3*4")[0] == "+"
        assert infix_to_prefix("(2+3)*4")[0] == "*"


class TestPostfixEvaluation:
    @pytest.mark.parametrize(
        "postfix,expected",
        [
            ("2 3 +", 5),
            ("2 3 4 * +", 14),
            ("2 3 + 4 *", 20),
            ("8 3 - 2 -", 3),
            ("2 3 2 ^ ^", 512),
            ("7 2 /", 3.5),
            ("7 2 %", 1),
            ("42", 42),
        ],
    )
    def test_evaluation_matches_the_arithmetic(self, postfix, expected):
        assert evaluate_postfix(postfix) == pytest.approx(expected)

    def test_the_operands_are_taken_off_in_the_right_order(self):
        # The stack hands back the second operand first. Popping them the wrong
        # way round still gives the right answer for + and *, so subtraction is
        # the test that actually catches it.
        assert evaluate_postfix("8 3 -") == 5
        assert evaluate_postfix("8 2 /") == 4

    def test_names_can_be_supplied_as_values(self):
        assert evaluate_postfix("a b +", {"a": 2, "b": 40}) == 42

    def test_a_name_with_no_value_is_rejected(self):
        with pytest.raises(ValueError, match="no value"):
            evaluate_postfix("a 1 +")

    def test_too_few_operands_is_rejected(self):
        with pytest.raises(ValueError, match="needs two values"):
            evaluate_postfix("2 +")

    def test_leftover_operands_are_reported_rather_than_ignored(self):
        with pytest.raises(ValueError, match="single value"):
            evaluate_postfix("1 2 3 +")

    def test_brackets_in_postfix_are_rejected(self):
        with pytest.raises(ValueError, match="should not contain brackets"):
            evaluate_postfix("( 2 3 + )")

    def test_division_by_zero_is_reported_clearly(self):
        with pytest.raises(ZeroDivisionError):
            evaluate_postfix("1 0 /")

    def test_every_operator_application_produces_one_step(self):
        _, steps = record(evaluate_postfix_traced("2 3 4 * +"))

        assert count_kinds(steps)["apply"] == 2
        assert count_kinds(steps)["push"] == 3


class TestEvaluateInfix:
    @pytest.mark.parametrize(
        "expression,expected",
        [
            ("2+3*4", 14),
            ("(2+3)*4", 20),
            ("2*3+4*5", 26),
            ("100/4/5", 5),
            ("2^3^2", 512),
            ("((1+2)*(3+4))", 21),
            ("1.5*4", 6),
        ],
    )
    def test_it_agrees_with_python_itself(self, expression, expected):
        assert evaluate_infix(expression) == pytest.approx(expected)

    def test_it_agrees_with_python_on_randomly_built_expressions(self):
        import random

        random.seed(20260821)
        for _ in range(200):
            left, middle, right = (random.randint(1, 9) for _ in range(3))
            operator_one, operator_two = (random.choice("+-*/") for _ in range(2))
            expression = f"({left}{operator_one}{middle}){operator_two}{right}"

            assert evaluate_infix(expression) == pytest.approx(eval(expression))  # noqa: S307

    def test_names_flow_through_to_the_evaluation(self):
        assert evaluate_infix("x*y+1", {"x": 3, "y": 4}) == 13
