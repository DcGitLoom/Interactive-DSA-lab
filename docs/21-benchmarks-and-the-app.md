# Day 20: The benchmark harness and the visualiser

Code: `benchmarks/harness.py`, `benchmarks/run.py`, `app/presentation.py`,
`app/main.py`. Tests: `tests/test_app_and_benchmarks.py`.

The last day joins the three layers together: the library gets measured, and the
whole thing gets a front end.

## The benchmark harness does not just time things

A benchmark that prints "0.42 seconds" tells you almost nothing. This one takes
each implementation, times it across growing input sizes, hands the numbers to the
complexity detective from day 3, and reports whether the **measured growth curve
matches the one the code claims**:

```
merge sort: claimed O(n log n), measured O(n log n) (R squared 0.998), which matches.
plain search tree: claimed O(n), measured O(n) (R squared 1.000), which matches.
```

Every complexity claim in this project is now checked against reality rather than
taken on trust, and a run ends with a list of any disagreements.

Three measurement decisions, each with a reason:

**Keep the fastest run, not the average.** Interference from other processes,
scheduling and garbage collection can only make a run slower, never faster, so the
minimum is the number least polluted by things unrelated to the code.

**Disable garbage collection while timing.** A collection pause landing inside one
measurement and not another is noise about the allocator, not the algorithm. It is
restored in a `finally`, so an exception cannot leave it off.

**Build the input outside the timed region.** Constructing the data is often more
expensive than the operation, and timing it would drown the signal. There is a test
that measures the same read with the setup inside and outside the timed region and
asserts the difference is enormous.

And one honesty note that belongs in every benchmark of this kind: comparing this
project's Python merge sort against `list.sort` is comparing Python against C. The
gap says nothing about the algorithms. **What is comparable is the shape of the
curve**, which is exactly what the harness reports.

## The bug the benchmarks found on their first run

This is the best thing that happened on the last day.

Merge sort **timed as quadratic**. Its comparison count was still n log n, so the
algorithm was right, but the wall clock said otherwise, and doubling the input
really did quadruple the time.

The cause was the tracing layer, the thing this whole project is built on. Every
sorting step carried a snapshot of the array so the visualiser could draw that
frame:

```python
yield Step("compare", "...", {"indices": [i, j], "array": list(items)})
```

Copying an n element list on each of n log n steps is O(n^2 log n) work. **The
instrumentation had become the dominant cost of the thing it was instrumenting.**

The fix is a `snapshot()` helper in `dsalab/tracing.py` that copies only when the
collection is small enough to be worth animating, and returns None above that
limit. Nobody watches a ten thousand element sort frame by frame, and the
benchmarks never look at the data at all. After the change, merge, quick and heap
sort all measure as O(n log n).

Two things worth keeping from this:

1. **Observation changes what it observes.** That is not just a physics joke. Any
   logging, tracing or metrics layer sitting inside a hot loop is part of the cost
   of that loop, and it is easy for it to become most of it.
2. **Only the benchmark could have found this.** Every test passed throughout. The
   answers were right, the step counts were right, the complexity of the algorithm
   was right. The only symptom was the wall clock, and nothing was watching the
   wall clock until day 20.

There is now a test, `test_the_sorting_benchmarks_measure_what_they_claim`, that
fails if the timed curve for any n log n sort stops matching, so this cannot come
back quietly.

## A second, smaller mistake worth recording

The first run reported the hash table as disagreeing with its claimed O(1). It was
not wrong; my label was. The benchmark performs a number of lookups **proportional
to n**, so a structure with O(1) lookups totals O(n) across the run.

The claim being checked has to describe the thing actually measured. A benchmark
that does n operations is measuring n times the cost of one, and writing the per
operation complexity next to it produces a confident, meaningless disagreement.

Python's own Timsort was flagged too, for a different reason: at a few thousand
elements a C sort finishes in microseconds, which is close enough to the timer's
resolution that the fitted curve describes the noise. The fix was to measure it at
fifty times the size. **A measurement too fast to time is not a measurement.**

## The visualiser, and how much of it is testable

A Streamlit app is awkward to test: it wants a browser, a server and a session, and
the tests end up asserting things about widgets rather than about behaviour.

So the app is split in two:

* **`app/presentation.py`** holds every decision: which frames to draw, which
  positions to highlight, what to compare in a race, how to lay out a tree, which
  rules to show beside a structure. Plain Python, no Streamlit import, fully
  tested.
* **`app/main.py`** holds the drawing: widgets, charts, captions. It makes no
  decisions of its own.

That split keeps the interesting half under test and leaves the untested half small
enough to check by reading it. It is the same instinct as separating tracing from
drawing back on day 1.

The five tabs are the four headline features plus the plain animation:

**Watch a sort.** Step through any algorithm on a chosen input, with the plain
English note for each step. The input choice matters more than the algorithm
choice, so the inputs are offered by name with an explanation of what each one
shows.

**Race.** Two algorithms on **identical** input, side by side, with live counters.
Identical input is the whole point: racing on two different random arrays measures
the arrays as much as the algorithms. The winner changes with the input, and
watching insertion sort beat merge sort on nearly sorted data teaches more than any
table of complexities. Both of those outcomes are pinned by tests.

**Complexity detective.** Counts comparisons across growing sizes and reports the
curve the measurements actually match, with the full ranking of fits.

**Trees.** Build a plain search tree, an AVL tree or a red black tree from the same
input and watch the shape. Beside it, the rules the structure states about itself,
each marked as holding or broken. Choosing sorted input for the plain tree produces
a chain and a warning pointing at the other two, which is day 9 and day 10's story
in one interaction.

**Where greedy fails.** Runs the counterexample finder and shows the failing inputs
it found, with both answers and the explanation.

## A note on what is verified and what is not

Every claim in this document about the library, the harness and the app's logic is
backed by a test in the suite.

The Streamlit layer itself was **not run**, because streamlit is not installed in
the environment this was built in. `app/presentation.py` is exercised thoroughly by
the tests; `app/main.py` is layout code that has been read but not executed. That
distinction is worth stating plainly rather than leaving a reader to assume the
whole thing was demonstrated.
