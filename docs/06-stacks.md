# Day 6: Stacks, and what they are actually for

Code: `dsalab/structures/stack.py`, `dsalab/algorithms/expressions.py`.
Tests: `tests/test_stack_and_expressions.py`.

## The restriction is the feature

A stack allows three things: push on top, pop off the top, look at the top. It
refuses to let you touch anything in the middle. That sounds like a weakness and
it is the entire point, because the refusal buys a guarantee: **whatever comes
out is always the most recent thing that went in.**

A surprising number of problems have exactly that shape:

* Undo history. The last action taken is the first one undone.
* Matching brackets. The next bracket you must close is always the most recently
  opened one.
* Function calls. The call stack is a real stack, and it is why a function always
  returns to whoever called it most recently. Recursion works because of this.
* Depth first search, and every recursive algorithm rewritten as a loop.

## Two implementations, and when the choice matters

| | ArrayStack | LinkedStack |
| - | - | - |
| push | O(1) amortised | O(1) worst case |
| pop | O(1) amortised | O(1) worst case |
| memory per element | just the value, plus spare capacity | value plus a node and a pointer |
| memory behaviour | one block, cache friendly | scattered, cache unfriendly |
| worst single operation | O(n) when it grows | always O(1) |

The array version is the right default. It is faster in practice for the reasons
covered on day 5: contiguous memory that the processor can prefetch.

The linked version earns its place in exactly one situation: when a single slow
operation is unacceptable. The array's amortised O(1) means the *average* is
constant, but one unlucky push copies the whole stack. If you are writing
something with a hard latency budget, an occasional long pause is a failure even
though the average is fine. Amortised and worst case are different promises, and
that difference is the whole reason both classes exist here.

Which end is the top is the actual design decision for the array version. Using
the end of the array makes push and pop the array's cheap operations. Using the
front would make both O(n), and the structure would be useless.

Both classes are tested by the same parametrised test class, because they are
meant to be interchangeable. Writing the tests once and running them against both
is what keeps that promise true rather than merely intended.

## Bracket matching, and why counting does not work

The tempting shortcut is to count openings and closings. It fails on `([)]`:
two openings, two closings, balanced by count, and complete nonsense. The stack
catches it, because when the `)` arrives the most recent unclosed bracket is `[`,
which does not match.

Order is the whole problem, and a counter has no memory of order. This is the
smallest good example of choosing a data structure by asking what information the
problem actually needs.

Cost is O(n) time and, in the worst case, O(n) memory for a string of nothing but
opening brackets.

## Infix, postfix and prefix

`2 + 3 * 4` cannot be evaluated left to right, because precedence says the
multiplication happens first, and brackets can override precedence anyway. So
evaluating infix directly means constantly looking ahead.

Postfix removes the problem. `2 3 4 * +` has no brackets and no precedence rules
left to apply. Read left to right, and every operator acts on the two values
immediately before it. All the precedence has been baked into the *order of the
symbols*. That is why compilers and calculators convert first and evaluate
second, and why this project keeps the two steps separate: it mirrors how a real
compiler is built and lets each half be tested and animated on its own.

### The shunting yard rules

* An operand goes straight to the output. Its position never changes.
* An opening bracket waits on the stack.
* A closing bracket pops operators to the output until its partner appears, which
  is then discarded. Brackets never reach the output.
* An operator pops any stacked operator that should be applied before it, then
  waits on the stack itself.
* At the end, everything left on the stack goes to the output.

Only the fourth rule is subtle. "Applied before it" means higher precedence, or
**equal precedence when the operator groups left to right**.

### The associativity trap, met twice

Nearly every operator groups left to right, so `8 - 3 - 2` means `(8 - 3) - 2`,
which is 3, not `8 - (3 - 2)`, which is 7. Exponentiation is the exception and
groups right to left, so `2 ^ 3 ^ 2` is `2 ^ 9 = 512`, not `(2 ^ 3) ^ 2 = 64`.
Handling that means not popping an equal precedence `^` off the stack.

Then the same idea bit again from the other side, and this one I got wrong first
time.

`infix_to_prefix` reuses the postfix algorithm with a trick: reverse the input,
swap each bracket for its partner, convert to postfix, reverse the result. It is
elegant and it produced the wrong answer for `a + b * c - d`, giving prefix for
`a + ((b * c) - d)` instead of `(a + (b * c)) - d`.

The reason is that **reversing an expression also reverses associativity.** In
the reversed token stream, operators that normally group left to right now have
to be treated as grouping right to left, and exponentiation the other way round.
The fix is to pass a flipped set of right grouping operators into the reversed
pass, which is why `infix_to_postfix` takes that as a parameter at all.

The lesson worth keeping: **a clever reuse of an existing algorithm inherits all
of its assumptions, including the ones you forgot were there.** The reverse trick
quietly assumed associativity was symmetric. Two tests now pin both directions
down, so a future tidy up cannot silently undo it.

## Evaluating postfix

One pass with a stack. Values go on; an operator takes the top two, applies
itself, pushes the answer back. O(n) time, O(n) memory.

The one thing to get right is the order of the two pops. **The stack gives back
the second operand first**, so `a b -` needs `b` popped before `a`. Getting this
backwards still produces correct answers for `+` and `*`, because they do not
care about order, and wrong answers for `-` and `/`. That is a bug that hides
beautifully behind a test suite that only checks addition, so there is a test
using subtraction and division specifically.

The validation is also deliberately strict. Too few operands, leftover operands
and brackets in postfix input are all reported as errors rather than ignored,
because an expression evaluator that silently returns a plausible wrong number is
far worse than one that refuses.

## A test worth stealing

`test_it_agrees_with_python_on_randomly_built_expressions` generates two hundred
random bracketed expressions, evaluates each with this code and with Python
itself, and asserts they match. It is a property based test in miniature: instead
of thinking up cases by hand, state a property that must always hold and throw
random input at it.

The seed is fixed, so a failure can be reproduced exactly rather than being a
test that fails once and then passes. That combination, random coverage with
reproducible failure, is worth having in any test suite.
