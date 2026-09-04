"""Recursion, in every shape the course covers.

Recursion is the first real topic because almost everything later leans on it:
tree traversals, quick sort, merge sort, backtracking and dynamic programming
are all recursion wearing different hats.

The mental model that makes recursion click:

A recursive function has two halves. The part written *before* the recursive
call runs on the way down, as the calls pile up on the stack. The part written
*after* the recursive call runs on the way back up, as the calls unwind. Two
functions with the same recursive call in them can behave completely differently
depending on which half does the work. `print_down` and `print_up` below are the
same function apart from one line moved, and they print opposite orders.

Every traced function here yields a step on the way down and on the way up, so
the visualiser can draw the call stack growing and shrinking.
"""

from __future__ import annotations

from dsalab.tracing import Step, Traced

# Every recursive function in Python is limited by the interpreter's recursion
# limit, which defaults to 1000 frames. That limit is not a language flaw, it is
# a guard: without it a runaway recursion would crash the process instead of
# raising a catchable error. Where a function here could realistically be given
# a large n, the docstring says what the practical limit is.


def print_down(n: int) -> Traced[list[int]]:
    """Head recursion: the work happens on the way back up.

    The recursive call comes first, so nothing is recorded until the calls have
    bottomed out at 0 and started to unwind. The result comes back ascending.
    """
    if n == 0:
        yield Step("base", "n is 0, so the calls stop piling up and start unwinding.", {"n": 0})
        return []

    yield Step("descend", f"Called with n={n}. Going deeper before doing any work.", {"n": n})
    below = yield from print_down(n - 1)
    yield Step("work", f"Back up at n={n}, now recording it.", {"n": n})
    return [*below, n]


def print_up(n: int) -> Traced[list[int]]:
    """Tail recursion: the work happens on the way down.

    Identical to `print_down` apart from where the recording line sits. Here it
    sits before the recursive call, so values are recorded as the stack grows
    and the result comes back descending.

    This shape is called tail recursion because the recursive call is the very
    last thing the function does. Languages that optimise tail calls can run it
    in constant stack space by reusing the same frame. CPython deliberately does
    not do this, so in Python a tail recursive function still costs one stack
    frame per call, which is why a loop is usually the better choice here.
    """
    if n == 0:
        yield Step("base", "n is 0, nothing left to record.", {"n": 0})
        return []

    yield Step("work", f"Recording n={n} before going deeper.", {"n": n})
    below = yield from print_up(n - 1)
    return [n, *below]


def factorial(n: int) -> Traced[int]:
    """n! computed by recursion.

    Time is O(n) and stack space is O(n), one frame per call.
    """
    if n < 0:
        raise ValueError("factorial is not defined for negative numbers")
    if n <= 1:
        yield Step("base", f"factorial({n}) is 1 by definition.", {"n": n, "result": 1})
        return 1

    yield Step("descend", f"factorial({n}) needs factorial({n - 1}) first.", {"n": n})
    smaller = yield from factorial(n - 1)
    result = n * smaller
    yield Step(
        "combine",
        f"factorial({n}) = {n} times {smaller} = {result}.",
        {"n": n, "result": result},
    )
    return result


def total(n: int) -> Traced[int]:
    """Sum of the numbers 1 to n, by recursion.

    Worth comparing against the closed form n * (n + 1) / 2, which gets the same
    answer in one multiplication. Recursion is a way of thinking, not always the
    right way of computing: when a formula exists, the formula wins.
    """
    if n <= 0:
        yield Step("base", "Nothing left to add, the sum of nothing is 0.", {"n": n})
        return 0

    yield Step("descend", f"sum({n}) needs sum({n - 1}) first.", {"n": n})
    smaller = yield from total(n - 1)
    result = n + smaller
    yield Step("combine", f"sum({n}) = {n} + {smaller} = {result}.", {"n": n, "result": result})
    return result


def power(base: float, exponent: int) -> Traced[float]:
    """base raised to exponent, by halving the exponent each time.

    The naive recursion multiplies by base exponent times, which is O(n). This
    version squares its way up instead, which is O(log n): to compute x^20 it
    computes x^10 once and squares it, rather than doing twenty multiplications.

    This trick is called exponentiation by squaring and it shows up again in
    modular arithmetic and in the Rabin Karp string matcher later in the project.
    """
    if exponent < 0:
        raise ValueError("this version handles non negative exponents only")
    if exponent == 0:
        yield Step("base", "Anything raised to 0 is 1.", {"exponent": 0, "result": 1.0})
        return 1.0

    half = yield from power(base, exponent // 2)
    if exponent % 2 == 0:
        result = half * half
        yield Step(
            "square",
            f"exponent {exponent} is even, so the answer is ({half}) squared = {result}.",
            {"exponent": exponent, "result": result},
        )
    else:
        result = half * half * base
        yield Step(
            "square",
            f"exponent {exponent} is odd, so square {half} and multiply by {base} once more.",
            {"exponent": exponent, "result": result},
        )
    return result


def fibonacci_naive(n: int) -> Traced[int]:
    """Fibonacci by tree recursion, the slow way, on purpose.

    Each call spawns two more calls, so the number of calls roughly doubles with
    every step up in n. That is O(2^n) time, and it is catastrophic: fib(40)
    already needs over three hundred million calls. Watch it in the app with
    n around 6 and you can see the same subproblem being recomputed over and
    over on different branches of the tree.

    This is the exact wastefulness that dynamic programming exists to remove,
    which is why the memoised version sits right below it.
    """
    if n < 2:
        yield Step("base", f"fib({n}) is {n} by definition.", {"n": n, "result": n})
        return n

    yield Step("descend", f"fib({n}) needs fib({n - 1}) and fib({n - 2}).", {"n": n})
    left = yield from fibonacci_naive(n - 1)
    right = yield from fibonacci_naive(n - 2)
    result = left + right
    yield Step(
        "combine",
        f"fib({n}) = {left} + {right} = {result}.",
        {"n": n, "result": result},
    )
    return result


def fibonacci_memoised(n: int, memo: dict[int, int] | None = None) -> Traced[int]:
    """Fibonacci with the answers remembered.

    Same recursion, one dictionary added. Every value of n is now computed at
    most once, which drops the cost from O(2^n) to O(n) time and O(n) space.

    This is top down dynamic programming in its smallest form. The full
    treatment, including the bottom up version, comes later in `dp.py`.
    """
    if memo is None:
        memo = {}

    if n in memo:
        yield Step(
            "cache-hit",
            f"fib({n}) was already worked out, reusing {memo[n]} instead of recomputing it.",
            {"n": n, "result": memo[n]},
        )
        return memo[n]

    if n < 2:
        memo[n] = n
        yield Step("base", f"fib({n}) is {n} by definition.", {"n": n, "result": n})
        return n

    left = yield from fibonacci_memoised(n - 1, memo)
    right = yield from fibonacci_memoised(n - 2, memo)
    memo[n] = left + right
    yield Step(
        "combine",
        f"fib({n}) = {left} + {right} = {memo[n]}, remembered for next time.",
        {"n": n, "result": memo[n]},
    )
    return memo[n]


def combinations(n: int, r: int) -> Traced[int]:
    """nCr using Pascal's rule: C(n, r) = C(n-1, r-1) + C(n-1, r).

    This is tree recursion again, and it recomputes just as wastefully as naive
    Fibonacci does. The point of including it is to show that the shape of the
    waste is the same, so the same fix works.
    """
    if r < 0 or r > n:
        yield Step("base", f"C({n}, {r}) is 0 because r is outside 0 to n.", {"n": n, "r": r})
        return 0
    if r == 0 or r == n:
        yield Step("base", f"C({n}, {r}) is 1, there is exactly one way.", {"n": n, "r": r})
        return 1

    left = yield from combinations(n - 1, r - 1)
    right = yield from combinations(n - 1, r)
    result = left + right
    yield Step(
        "combine",
        f"C({n}, {r}) = {left} + {right} = {result}.",
        {"n": n, "r": r, "result": result},
    )
    return result


def towers_of_hanoi(
    disks: int, source: str = "A", helper: str = "B", target: str = "C"
) -> Traced[list[tuple[int, str, str]]]:
    """Move a stack of disks from source to target, never putting a big disk on a small one.

    The recursive insight is that to move n disks you move the top n-1 out of the
    way onto the spare peg, move the bottom disk across, then move the n-1 back
    on top of it. The whole puzzle collapses into three lines.

    It takes exactly 2^n - 1 moves, and that is provably the minimum. With 64
    disks, at one move per second, it would take longer than the current age of
    the universe, which is the point of the legend the puzzle comes from.
    """
    if disks == 0:
        return []

    moves: list[tuple[int, str, str]] = []

    above = yield from towers_of_hanoi(disks - 1, source, target, helper)
    moves.extend(above)

    yield Step(
        "move",
        f"Move disk {disks} from peg {source} to peg {target}.",
        {"disk": disks, "from": source, "to": target},
    )
    moves.append((disks, source, target))

    below = yield from towers_of_hanoi(disks - 1, helper, source, target)
    moves.extend(below)

    return moves


def taylor_e(x: float, terms: int) -> Traced[float]:
    """e^x from its Taylor series, using Horner's method so nothing is recomputed.

    The plain series adds x^k / k! term by term, and computing each power and
    each factorial from scratch makes it O(n^2) multiplications. Horner's method
    rewrites the series so each step reuses the one below it:

        1 + x/1 * (1 + x/2 * (1 + x/3 * (...)))

    That is O(n) multiplications for the same answer. It is a small example of a
    theme that runs through the whole course: the cost is not in the formula, it
    is in how you arrange the arithmetic.
    """
    if terms <= 0:
        yield Step("base", "No terms requested, the series contributes nothing.", {"terms": 0})
        return 0.0

    result = 1.0
    for k in range(terms - 1, 0, -1):
        result = 1 + (x / k) * result
        yield Step(
            "term",
            f"Folded in the term for k={k}, running value is {result:.6f}.",
            {"k": k, "value": result},
        )
    return result


def ackermann(m: int, n: int) -> Traced[int]:
    """The Ackermann function, the standard example of nested recursion.

    Nested recursion means a recursive call whose argument is itself a recursive
    call. This function grows faster than any function you can build out of
    ordinary loops, which is exactly why it is famous: it proves that recursion
    can express things that simple iteration cannot.

    Keep the inputs tiny. ackermann(3, 3) is fine and returns 61. ackermann(4, 2)
    is a number with nineteen thousand digits and will exhaust the stack.
    """
    if m == 0:
        yield Step("base", f"m is 0, so the answer is n + 1 = {n + 1}.", {"m": m, "n": n})
        return n + 1

    if n == 0:
        yield Step("descend", f"n is 0, so this becomes A({m - 1}, 1).", {"m": m, "n": n})
        return (yield from ackermann(m - 1, 1))

    yield Step(
        "nested",
        f"A({m}, {n}) needs A({m}, {n - 1}) worked out first, then feeds it back in.",
        {"m": m, "n": n},
    )
    inner = yield from ackermann(m, n - 1)
    return (yield from ackermann(m - 1, inner))


def is_even(n: int) -> Traced[bool]:
    """Indirect recursion: this calls `is_odd`, which calls this one back.

    Neither function calls itself. They call each other, and the pair together
    forms the loop. It is a deliberately silly way to check evenness, but it is
    the clearest possible example of a recursion that is not visible from inside
    a single function, which is the thing worth recognising in real code.
    """
    if n == 0:
        yield Step("base", "0 is even.", {"n": 0, "result": True})
        return True
    yield Step("bounce", f"{n} is even only if {n - 1} is odd. Handing over.", {"n": n})
    return (yield from is_odd(n - 1))


def is_odd(n: int) -> Traced[bool]:
    """The other half of the indirect recursion pair. See `is_even`."""
    if n == 0:
        yield Step("base", "0 is not odd.", {"n": 0, "result": False})
        return False
    yield Step("bounce", f"{n} is odd only if {n - 1} is even. Handing back.", {"n": n})
    return (yield from is_even(n - 1))
