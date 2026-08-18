# Day 3: Asymptotic notation, and measuring it for real

Code: `dsalab/complexity.py`. Tests: `tests/test_complexity.py`.

## The question complexity answers

Complexity is not about how fast a program is. It is about how the running time
changes when the input gets bigger. Those are different questions, and mixing
them up is the most common mistake people make with big O.

A concrete case. Suppose algorithm A takes 100n microseconds and algorithm B
takes n^2 microseconds. At n = 10, B wins easily: 100 against 1000. At n = 100
they tie. At n = 1000, A takes 0.1 seconds and B takes 1 second. At n = 1000000,
A takes 100 seconds and B takes eleven and a half days. The constant factor of
100 never changed. It just stopped mattering, because growth beats constants
eventually, and "eventually" arrives sooner than people expect.

That is why complexity throws away constants and lower order terms. It is not
sloppiness. It is deliberately answering the only question that survives changes
in hardware, language and compiler.

## The three notations

| Notation | Reads as | Means |
| - | - | - |
| O(f) | Big O | At most this fast growing. An upper bound. |
| Omega(f) | Big omega | At least this fast growing. A lower bound. |
| Theta(f) | Big theta | Exactly this fast growing. Upper and lower bound together. |

Formally, a function g is O(f) when there is some constant c and some size n0
such that beyond n0, g(n) is never more than c times f(n). The "beyond n0" part
is what lets us ignore small input behaviour, and the "some constant c" part is
what lets us ignore hardware speed.

In practice, people say big O when they mean big theta. Saying linear search is
O(n^2) is technically true, since an upper bound is allowed to be loose, but it
is useless. The convention is to state the tightest bound you can, so when
somebody says O(n) they generally mean it grows exactly like n.

## Best, worst and average case

These are a separate axis from the notation, and confusing the two axes causes
endless trouble. Notation describes a bound on a function. Case describes which
input you are talking about.

Take linear search for a value in an unsorted array of n items:

* **Best case:** the value is first. One comparison. Omega(1).
* **Worst case:** the value is last or missing. n comparisons. O(n).
* **Average case:** assuming the value is equally likely to be anywhere,
  about n/2 comparisons, which is still Theta(n) because constants do not count.

Worst case is what is usually quoted, for a good reason: it is a promise. An
average case is only true if your inputs match the distribution you assumed, and
real inputs have a nasty habit of not doing that. Quick sort is the famous
example. Its average is O(n log n) and its worst is O(n^2), and the worst case
used to be reachable in practice by feeding it already sorted data, which is
extremely common input.

## Amortised cost, which is not average cost

An amortised bound says that any *sequence* of operations costs a certain amount
per operation on average, even though individual operations vary. The difference
from average case is that amortised makes no assumption about the input at all.
It is a guarantee, not a statistical expectation.

The dynamic array on day 4 is the clean example. Appending is usually O(1), but
when the array is full it has to allocate a bigger block and copy everything
across, which is O(n). Yet appending n items in a row costs O(n) in total, so
each append is O(1) amortised. The expensive copies are rare enough, and the
cheap appends numerous enough, that the cost per operation averages out to a
constant no matter what you do. Day 4 works through the arithmetic.

## The complexity detective

The module built today does something textbooks cannot: it checks the claim
against measurements.

Give it a set of input sizes and how much work each one took, and it fits every
standard growth curve to the data and reports which one matches:

```python
from dsalab.complexity import measure_steps, doubling_sizes

def work(n):
    return sum(1 for _ in range(n) for _ in range(n))

verdict = measure_steps(work, doubling_sizes(8, 6))
print(verdict.summary())
# Measured growth matches O(n^2) (R squared 1.0000).
```

The fitting is ordinary least squares. For each candidate curve f it finds the a
and b that best satisfy `work = a * f(n) + b`, then scores the result with R
squared, which is the fraction of the variation in the measurements the curve
explains. 1.0 is perfect. The `a` absorbs hardware speed and the `b` absorbs
fixed setup cost, which is exactly the pair of things big O tells you to ignore.

### Why it also reports when it is not sure

Naming a winner is easy. Knowing when the winner is meaningless is the harder
half, so `is_confident` refuses to answer in two situations:

1. The best fit scores below 0.95, meaning the measurements are too noisy for
   any curve to describe them.
2. The best fit is barely ahead of the second best, meaning two curves are
   indistinguishable over the range of sizes measured.

That second case is real and common. Over sizes 100 to 140, n and n log n differ
by less than a tenth of a percent in fit quality, because log n hardly moves
across such a narrow band. The honest answer there is "cannot tell, measure a
wider range", and that is what it says.

### A trade off I actually hit

The confidence threshold was originally a gap of 0.02 in R squared between first
and second place, which felt like a sensible round number. It was wrong, and the
tests caught it. Fitting a cubic curve to perfectly quadratic data still scores
about 0.985, because a cubic can be stretched to lie close to a quadratic over a
finite range, so a perfect quadratic answer was being reported as uncertain. The
genuinely ambiguous case, n against n log n over a narrow range, has a gap of
about 0.00007. Those two are three orders of magnitude apart, so the threshold
moved to 0.001, which sits between them with room on both sides.

The lesson worth keeping: a threshold picked because it looks like a nice number
is a guess. Measure both the case you want to accept and the case you want to
reject, then put the line between them.

### Why counting beats timing

There are two ways to measure work here, and `measure_steps` is the better one.

Timing with `measure_time` is what people reach for first, and it is polluted by
everything else happening on the machine: other processes, garbage collection,
CPU frequency changes, cache state. It keeps the fastest of several runs, since
interference can only slow a run down and never speed it up, but it is still
noisy.

Counting operations is deterministic. Run it twice and you get the same numbers.
Because every algorithm in this project already reports its steps through the
tracing layer, counting work is free: the number of `compare` steps in a sort is
a real measurement of how much comparing it did, with no timer involved and no
counters added to the algorithm. The benchmarks report both, and where the two
disagree, that disagreement is itself worth looking at, since it usually means
constant factors or memory effects are doing something interesting.
