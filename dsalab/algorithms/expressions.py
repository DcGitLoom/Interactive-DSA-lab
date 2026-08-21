"""Expression handling: the classic thing stacks are for.

Four related problems, all solved with a stack:

1. Checking that brackets are balanced.
2. Converting infix notation (what people write) to postfix (what machines want).
3. Converting infix to prefix.
4. Evaluating a postfix expression.

The reason a stack fits so well is that all four are about *deferred work*. When
reading `2 + 3 * 4` left to right, you reach the `+` before you know whether it
can be applied, because a higher priority operator might be coming. So the `+`
waits somewhere until its turn, and the thing that always comes off next is the
most recently deferred one. That is a stack, exactly.

Why postfix exists at all: in `2 + 3 * 4` you cannot evaluate left to right,
because precedence rules say the multiplication happens first, and brackets can
override that anyway. Rewritten as postfix, `2 3 4 * +`, there are no brackets and
no precedence rules left to apply. You read left to right, and every operator acts
on the two values immediately before it. Precedence has been baked into the order
of the symbols. That is why compilers and calculators convert to postfix first.
"""

from __future__ import annotations

from dsalab.structures.stack import ArrayStack
from dsalab.tracing import Step, Traced, run

# Higher number means the operator binds more tightly and is applied first.
PRECEDENCE: dict[str, int] = {
    "+": 1,
    "-": 1,
    "*": 2,
    "/": 2,
    "%": 2,
    "^": 3,
}

# Almost every operator groups left to right: 8 - 3 - 2 means (8 - 3) - 2.
# Exponentiation is the odd one out and groups right to left, so 2 ^ 3 ^ 2 means
# 2 ^ (3 ^ 2) = 512, not (2 ^ 3) ^ 2 = 64. That single exception is why the
# comparison in the conversion below has to treat ^ differently, and getting it
# wrong is a bug that only shows up on chained exponents.
RIGHT_ASSOCIATIVE = {"^"}

BRACKET_PAIRS = {")": "(", "]": "[", "}": "{"}
OPENING_BRACKETS = set(BRACKET_PAIRS.values())


def brackets_balanced(text: str) -> bool:
    return run(brackets_balanced_traced(text))


def brackets_balanced_traced(text: str) -> Traced[bool]:
    """Check that every bracket is closed, in the right order, by the right kind.

    O(n) time and O(n) memory in the worst case, which is a string of nothing but
    opening brackets.

    Counting brackets instead of stacking them is the tempting shortcut and it is
    wrong. A counter says `([)]` is fine, because there are two openings and two
    closings. A stack catches it, because when the `)` arrives the most recent
    unclosed bracket is `[`, which does not match. Order is the whole point, and
    only a stack remembers order.
    """
    waiting = ArrayStack()

    for position, character in enumerate(text):
        if character in OPENING_BRACKETS:
            waiting.push(character)
            yield Step(
                "open",
                f"Found {character} at position {position}, so it goes on the stack "
                "to be closed later.",
                {"position": position, "character": character, "depth": len(waiting)},
            )
        elif character in BRACKET_PAIRS:
            if waiting.is_empty():
                yield Step(
                    "unmatched",
                    f"Found {character} at position {position} with nothing open. Unbalanced.",
                    {"position": position, "character": character},
                )
                return False

            opened = waiting.pop()
            if opened != BRACKET_PAIRS[character]:
                yield Step(
                    "mismatch",
                    f"{character} at position {position} tries to close {opened}, "
                    "which is the wrong kind of bracket.",
                    {"position": position, "expected": BRACKET_PAIRS[character], "found": opened},
                )
                return False

            yield Step(
                "close",
                f"{character} at position {position} correctly closes the {opened} "
                "that was still open.",
                {"position": position, "character": character, "depth": len(waiting)},
            )

    if not waiting.is_empty():
        yield Step(
            "unclosed",
            f"Reached the end with {len(waiting)} bracket(s) still open. Unbalanced.",
            {"still_open": len(waiting)},
        )
        return False

    yield Step("balanced", "Every bracket was opened and closed in the right order.", {})
    return True


def tokenize(expression: str) -> list[str]:
    """Split an expression into numbers, names, operators and brackets.

    Written by hand rather than with a regular expression, because a tokeniser is
    a small state machine and seeing it written out makes clear what a parser
    actually does. Multi digit numbers, decimals and multi letter names are all
    handled by gathering characters while they still belong to the current token.
    """
    tokens: list[str] = []
    index = 0

    while index < len(expression):
        character = expression[index]

        if character.isspace():
            index += 1
        elif character.isdigit() or character == ".":
            start = index
            while index < len(expression) and (
                expression[index].isdigit() or expression[index] == "."
            ):
                index += 1
            tokens.append(expression[start:index])
        elif character.isalpha() or character == "_":
            start = index
            while index < len(expression) and (
                expression[index].isalnum() or expression[index] == "_"
            ):
                index += 1
            tokens.append(expression[start:index])
        elif character in PRECEDENCE or character in OPENING_BRACKETS or character in BRACKET_PAIRS:
            tokens.append(character)
            index += 1
        else:
            raise ValueError(f"cannot make sense of {character!r} at position {index}")

    return tokens


def is_operand(token: str) -> bool:
    """True for a number or a name, false for an operator or a bracket."""
    return token not in PRECEDENCE and token not in OPENING_BRACKETS and token not in BRACKET_PAIRS


def infix_to_postfix(expression: str, right_associative: set[str] | None = None) -> list[str]:
    return run(infix_to_postfix_traced(expression, right_associative))


def infix_to_postfix_traced(
    expression: str, right_associative: set[str] | None = None
) -> Traced[list[str]]:
    """Convert infix to postfix using the shunting yard method. O(n).

    The rules, which are short enough to memorise:

    * An operand goes straight to the output. Its position never changes.
    * An opening bracket goes on the stack and waits.
    * A closing bracket pops operators to the output until the matching opening
      bracket appears, which is then thrown away. Brackets never reach the
      output, because postfix does not need them.
    * An operator pops any stacked operator that should be applied before it,
      then goes on the stack itself.
    * At the end, everything left on the stack goes to the output.

    The only subtle rule is the fourth. "Should be applied before it" means
    higher precedence, or equal precedence when the operator groups left to
    right. Exponentiation groups right to left, so an equal precedence `^` on the
    stack must *not* be popped, or `2 ^ 3 ^ 2` comes out as 64 instead of 512.

    `right_associative` names which operators group right to left. It defaults to
    the real answer for ordinary expressions, and exists as a parameter only
    because `infix_to_prefix` runs this same algorithm over a reversed expression
    and needs every operator's grouping flipped. See that function for why.
    """
    if right_associative is None:
        right_associative = RIGHT_ASSOCIATIVE
    tokens = tokenize(expression)
    output: list[str] = []
    operators = ArrayStack()

    for token in tokens:
        if is_operand(token):
            output.append(token)
            yield Step(
                "operand",
                f"{token} is a value, so it goes straight to the output.",
                {"token": token, "output": list(output)},
            )

        elif token in OPENING_BRACKETS:
            operators.push(token)
            yield Step(
                "open",
                f"{token} opens a group, so it waits on the stack.",
                {"token": token, "stack": operators.to_list()},
            )

        elif token in BRACKET_PAIRS:
            while not operators.is_empty() and operators.peek() not in OPENING_BRACKETS:
                popped = operators.pop()
                output.append(popped)
                yield Step(
                    "unwind",
                    f"Closing the group, so {popped} comes off the stack to the output.",
                    {"token": popped, "output": list(output)},
                )
            if operators.is_empty():
                raise ValueError("unbalanced brackets: found a closing bracket with nothing open")
            operators.pop()  # discard the matching opening bracket
            yield Step(
                "close",
                "The group is finished, and its brackets are dropped because "
                "postfix does not need them.",
                {"stack": operators.to_list()},
            )

        else:
            while (
                not operators.is_empty()
                and operators.peek() not in OPENING_BRACKETS
                and (
                    PRECEDENCE[operators.peek()] > PRECEDENCE[token]
                    or (
                        PRECEDENCE[operators.peek()] == PRECEDENCE[token]
                        and token not in right_associative
                    )
                )
            ):
                popped = operators.pop()
                output.append(popped)
                yield Step(
                    "pop",
                    f"{popped} binds at least as tightly as {token}, so it is applied first.",
                    {"token": popped, "output": list(output)},
                )
            operators.push(token)
            yield Step(
                "push",
                f"{token} waits on the stack until its operands are ready.",
                {"token": token, "stack": operators.to_list()},
            )

    while not operators.is_empty():
        popped = operators.pop()
        if popped in OPENING_BRACKETS:
            raise ValueError("unbalanced brackets: a group was opened and never closed")
        output.append(popped)
        yield Step(
            "flush",
            f"End of the expression, so the waiting {popped} goes to the output.",
            {"token": popped, "output": list(output)},
        )

    return output


def infix_to_prefix(expression: str) -> list[str]:
    """Convert infix to prefix, where the operator comes before its operands.

    Rather than writing a second algorithm, this reuses the postfix one with a
    trick worth knowing: reverse the input, swap every bracket for its partner,
    convert to postfix, then reverse the result.

    Reversing turns "operator after its operands" into "operator before them",
    and swapping the brackets keeps the groups intact through the reversal.

    The part that is easy to get wrong, and that I did get wrong first time:
    **reversing also reverses associativity.** In `a + b * c - d` the plus and
    minus group left to right, so the expression means `(a + (b * c)) - d`. Run
    the reversed tokens through the ordinary postfix rules and the equal
    precedence plus and minus get grouped the other way, producing prefix for
    `a + ((b * c) - d)`, which is a different expression with a different value.

    So the reversed pass has to treat every normally left grouping operator as
    right grouping, and exponentiation, which normally groups right to left, as
    left to right. Flipping the set is the whole fix.
    """
    swapped = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{"}
    tokens = tokenize(expression)
    flipped = " ".join(swapped.get(token, token) for token in reversed(tokens))
    reversed_associativity = set(PRECEDENCE) - RIGHT_ASSOCIATIVE
    return list(reversed(infix_to_postfix(flipped, reversed_associativity)))


def evaluate_postfix(tokens: list[str] | str, names: dict[str, float] | None = None) -> float:
    return run(evaluate_postfix_traced(tokens, names))


def evaluate_postfix_traced(
    tokens: list[str] | str, names: dict[str, float] | None = None
) -> Traced[float]:
    """Evaluate a postfix expression with a stack. O(n) time, O(n) memory.

    One pass, no precedence rules and no brackets, because the conversion already
    dealt with all of that. Values go on the stack; an operator takes the top two
    off, applies itself and pushes the answer back.

    The order of the two pops is the thing to get right. The stack gives the
    *second* operand first, so `a b -` needs `b` popped before `a`, and popping
    them the wrong way round gives the right answer for + and * and the wrong one
    for - and /, which is a bug that hides well.
    """
    if isinstance(tokens, str):
        tokens = tokenize(tokens)
    names = names or {}
    values = ArrayStack()

    for token in tokens:
        if is_operand(token):
            if token in names:
                value = float(names[token])
            else:
                try:
                    value = float(token)
                except ValueError:
                    raise ValueError(f"{token!r} has no value, so the expression cannot be worked out")
            values.push(value)
            yield Step(
                "push",
                f"{token} is a value, so {value:g} goes on the stack.",
                {"token": token, "value": value, "stack": values.to_list()},
            )
            continue

        if token in OPENING_BRACKETS or token in BRACKET_PAIRS:
            raise ValueError("a postfix expression should not contain brackets")

        if len(values) < 2:
            raise ValueError(f"the operator {token!r} needs two values but the stack has fewer")

        right = values.pop()
        left = values.pop()
        result = _apply(token, left, right)
        values.push(result)
        yield Step(
            "apply",
            f"{token} takes the top two values: {left:g} {token} {right:g} = {result:g}.",
            {"operator": token, "left": left, "right": right, "result": result},
        )

    if len(values) != 1:
        raise ValueError(
            f"the expression did not reduce to a single value, {len(values)} were left over"
        )

    return values.pop()


def _apply(operator: str, left: float, right: float) -> float:
    """Apply one binary operator to two numbers."""
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator == "/":
        if right == 0:
            raise ZeroDivisionError("division by zero in the expression")
        return left / right
    if operator == "%":
        if right == 0:
            raise ZeroDivisionError("remainder by zero in the expression")
        return left % right
    if operator == "^":
        return left**right
    raise ValueError(f"unknown operator {operator!r}")


def evaluate_infix(expression: str, names: dict[str, float] | None = None) -> float:
    """Work out the value of an ordinary infix expression.

    Convert then evaluate. Keeping these as two separate steps is deliberate: it
    is how real compilers are structured, and it means each half can be tested
    and animated on its own.
    """
    return evaluate_postfix(infix_to_postfix(expression), names)
