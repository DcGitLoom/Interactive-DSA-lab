# Day 2: Recursion

Code: `dsalab/algorithms/recursion.py`. Tests: `tests/test_recursion.py`.

## What recursion actually is

A recursive function is one that calls itself on a smaller version of the same
problem, and knows when to stop. That is the whole definition. Two parts are
required and leaving either one out breaks it:

1. **A base case.** A size of problem small enough to answer directly, with no
   further calls. Without one, the function calls itself forever.
2. **Progress towards the base case.** Every call must hand on a strictly
   smaller problem. `f(n)` calling `f(n)` is not recursion, it is a hang.

## The one idea that makes it click

Every recursive function has two halves, and it matters enormously which half
you put the work in.

```python
def down(n):          def up(n):
    if n == 0:            if n == 0:
        return                return
    down(n - 1)           print(n)      <- work happens here now
    print(n)             up(n - 1)
```

These are the same function with one line moved, and they print in opposite
orders. The reason is what the call stack is doing:

* Work written **before** the recursive call runs on the way **down**, while
  the stack is growing. The first call to reach that line is the outermost one,
  so you see n, then n-1, then n-2.
* Work written **after** the recursive call runs on the way **back up**, while
  the stack is unwinding. Nothing runs until the recursion has bottomed out, so
  you see 1, then 2, then 3.

Almost every confusing recursion bug is really a confusion about which of those
two halves the work is in. `print_down` and `print_up` in the code exist purely
to make that visible, and the app draws the stack growing and shrinking so you
can watch it happen.

## The shapes of recursion

The course names five shapes. They are not different techniques, they are just
descriptions of what the calls look like.

| Shape | What it means | Example in the code |
| - | - | - |
| Head recursion | The recursive call comes first, work happens while unwinding | `print_down` |
| Tail recursion | The recursive call is the last thing the function does | `print_up` |
| Tree recursion | A function makes more than one recursive call | `fibonacci_naive` |
| Indirect recursion | A calls B, and B calls A back | `is_even` and `is_odd` |
| Nested recursion | A recursive call whose argument is itself a recursive call | `ackermann` |

Tail recursion is worth one extra note. In languages that optimise tail calls,
a tail recursive function runs in constant stack space, because the compiler can
throw away the current frame before making the final call. CPython deliberately
does not do this. So in Python a tail recursive function still costs one stack
frame per call, and if the recursion is deep you will hit the default limit of
about 1000 frames. When you find yourself writing tail recursion in Python, a
plain loop is almost always the better answer.

## Cost

| Function | Time | Stack space | Why |
| - | - | - | - |
| `total`, `factorial` | O(n) | O(n) | One call per value, each doing constant work |
| `power` | O(log n) | O(log n) | The exponent halves every call |
| `fibonacci_naive` | O(2^n) | O(n) | Two calls per level, so calls roughly double per level |
| `fibonacci_memoised` | O(n) | O(n) | Each value of n is computed at most once |
| `combinations` | O(2^n) | O(n) | Tree recursion, same waste as naive Fibonacci |
| `towers_of_hanoi` | O(2^n) | O(n) | Provably needs 2^n - 1 moves, so this is optimal |
| `taylor_e` | O(n) | O(1) | Horner's arrangement, one multiply per term |
| `ackermann` | Grows faster than any primitive recursive function | Enormous | This is the point of it |

The stack space column is the one beginners forget. A recursive function that
looks like it uses no memory is using one stack frame per call, and each frame
holds its arguments and local variables. That is real memory, and it is the
reason a recursive traversal of a linked list with a million nodes crashes while
the loop version does not.

## Two things worth taking away

**Naive Fibonacci is the best argument for dynamic programming ever written.**
Run `fibonacci_naive(6)` in the visualiser and count how many times `fib(2)` is
computed. It is computed five times, on five different branches of the call
tree, each time from scratch. At n=30 the same waste has multiplied into over a
million redundant calls. Adding one dictionary, as `fibonacci_memoised` does,
takes it from O(2^n) to O(n) without changing the shape of the code at all. That
is the whole idea of dynamic programming in one line of difference, and
everything in `dp.py` later is a more careful version of the same move.

**Recursion is a way of thinking, not always a way of computing.** `total(n)`
adds the numbers 1 to n in n recursive calls. The formula n(n+1)/2 does it in
one multiplication and one division, with no stack at all. Recursion earns its
place when the problem is genuinely self similar, like a tree or a puzzle that
splits into smaller copies of itself. It does not earn its place just because
the problem can be phrased recursively.

## A trade off I actually hit

`towers_of_hanoi` originally yielded a step and returned nothing, with the caller
collecting moves from the steps. That worked for the app but made the tests
awkward, because verifying that the puzzle rules are respected meant digging
through step objects to find the move data. Returning the list of moves *and*
yielding the steps means the tests read as plain assertions about moves, while
the app still gets its animation. The general lesson: a traced function should
still return the answer a normal caller would want, not force everyone to
reconstruct it from the trace.
