# Interactive DSA Lab

Data structures and algorithms written from scratch in Python, then animated,
tested and benchmarked against the standard library.

Built by following Abdul Bari's data structures and algorithms course, one topic
per day for twenty days. Every commit is one day of study, so the history is a
record of how the understanding was built rather than a single dump of finished
code.

The whole thing was written locally over those twenty days, committing at the end
of each day's session, and pushed to GitHub in one go at the end. That is why the
commit dates run from 16 August to 4 September while the repository itself only
appears later: the work was done offline first and the remote came afterwards.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,app]"

pytest                            # 1,275 tests
python -m benchmarks.run          # measure every complexity claim in the project
streamlit run app/main.py         # the visualiser
```

## What makes this more than a visualiser

Most visualiser projects stop at the animation. This one has three layers, and the
second and third are the point.

**1. The library.** Every container and algorithm is implemented by hand. Where a
structure is being demonstrated it gets no help from a built in Python container:
the dynamic array allocates a raw `ctypes` block so the growth and copying are
real, the linked lists use actual node objects, the hash tables do their own
probing.

**2. The proof.** A pytest suite aimed at the awkward cases rather than the easy
ones: deleting a node with two children, a ring buffer wrapping before it grows, a
tombstone that would otherwise strand a key, all five red black rules verified
after every single operation in a two thousand operation run.

**3. The measurement.** A benchmark harness that times each implementation across
growing input sizes and reports whether the **measured growth curve matches the one
the code claims**. Not "0.42 seconds" but "claimed O(log n), measured O(log n),
R squared 0.998".

The animation sits on top of all three and reuses exactly the same code.

## The four things that separate it from a teaching demo

**Complexity detective** (`dsalab/complexity.py`). Give it input sizes and how much
work each took, and it least squares fits every standard growth curve and reports
which one the data matches. It also refuses to answer when two curves are
indistinguishable over the range measured, because naming a winner there would be
worse than useless.

**Race mode** (`app/presentation.py`). Two algorithms on identical input, frame by
frame, with live counters. The ranking changes with the input, and watching
insertion sort beat merge sort on nearly sorted data is more convincing than any
complexity table.

**Invariant checking** (`dsalab/invariants.py`). Each structure states its rules in
code. Inside a `checked()` block they are verified after every mutation, naming the
broken rule in plain English and the node that broke it, so corruption is caught at
the operation that caused it rather than surfacing later somewhere unrelated.

**Counterexample finder** (`dsalab/algorithms/greedy_vs_dp.py`). Every book says
greedy is not always optimal and almost none hand you a failing input. This
searches for them, smallest first:

```
Greedy coin change is not optimal
  Input:   coins [1, 3, 4], making 6
  Greedy:  [4, 1, 1] (score 3)
  Correct: [3, 3] (score 2)
```

It rediscovered that classic case on its own. When a search finds nothing it
reports weak evidence with the number of inputs tried, not a proof.

## How the tracing works

Every algorithm is a generator that yields a `Step` when something interesting
happens and returns the answer at the end.

```python
from dsalab.tracing import record, run
from dsalab.algorithms.sorting import merge_sort_traced

run(merge_sort_traced([5, 1, 4]))                       # [1, 4, 5], steps discarded
answer, steps = record(merge_sort_traced([5, 1, 4]))    # same answer, plus the replay
```

One implementation serves the tests, the benchmarks and the animation, so the
animation cannot drift out of agreement with the code that actually runs. Keeping a
separate instrumented copy is how visualisers end up teaching something their own
code does not do.

## What is covered

The full checklist is `docs/01-syllabus-coverage.md`, which follows the running
order of the course so nothing is quietly skipped.

**Structures**: dynamic array on raw memory, special and sparse matrices,
polynomials, singly and doubly and circular linked lists, stacks, queues, circular
buffers, deques, binary trees with all four traversals written twice, threaded
trees, binary search trees, AVL trees, red black trees, B-trees and 2-3 trees,
heaps and priority queues, hash tables with chaining and three probing strategies,
LRU cache, tries and compressed tries, segment trees with lazy propagation, Fenwick
trees, union find, graphs as adjacency list and matrix.

**Algorithms**: recursion in all five shapes, eleven sorts, binary search and its
variants, KMP, Rabin Karp, the Z algorithm, breadth and depth first search,
topological sort, cycle detection, Tarjan and Kosaraju, Prim and Kruskal, Dijkstra,
Bellman Ford, Floyd Warshall, A star, greedy methods, dynamic programming,
backtracking, branch and bound.

## The twenty days

| Day | Topic |
| - | - |
| 1 | Project setup and the step tracing system |
| 2 | Recursion in all five of its shapes |
| 3 | Asymptotic notation and the complexity detective |
| 4 | Arrays, amortised growth, special and sparse matrices, polynomials |
| 5 | Linked lists: singly, doubly, circular, and the two pointer tricks |
| 6 | Stacks, bracket matching, infix to postfix and prefix, evaluation |
| 7 | Queues, circular buffers, deques and sliding window maximum |
| 8 | Binary trees, four traversals twice over, threaded trees |
| 9 | Binary search trees, ordered queries, and the invariant checker |
| 10 | AVL trees, the four rotations, and a measured logarithmic height |
| 11 | Red black trees, all five rules checked after every operation |
| 12 | B-trees and 2-3 trees, the shape built for disk |
| 13 | Heaps, priority queues, heap sort, and the linear build proof |
| 14 | Hash tables, both collision strategies, and the LRU cache |
| 15 | Eleven sorting algorithms and the n log n lower bound |
| 16 | Binary search variants and string matching (KMP, Rabin Karp, Z) |
| 17 | Tries, segment and Fenwick trees, union find |
| 18 | Graphs: traversal, topological sort, SCC, MST, shortest paths |
| 19 | Greedy, dynamic programming, backtracking, branch and bound |
| 20 | The benchmark harness and the visualiser |

## A few things learned the hard way

Each came from a test or a benchmark disagreeing with me, and each is written up in
that day's document.

* **A ctypes memory leak** (day 4). Setting a popped slot to None is not enough: a
  `py_object` array keeps a hidden reference table of its own, so removed objects
  stayed alive forever. No test of the visible behaviour could catch it, so there
  is one that pops an object and asserts a weak reference to it goes dead.
* **The preemptive B-tree split needs an even order** (day 12). The textbook one
  pass insertion silently requires it, and a 2-3 tree is order 3. Found only
  because every structural check runs at six different orders.
* **A self referential invariant check is not a check** (day 17). The Fenwick tree
  check derived the array by reading the tree, then checked the tree against it. A
  Fenwick tree carries no redundancy, so it genuinely cannot check itself, and the
  method now says so.
* **A test that passed for the wrong reason** (day 15). The stability tests sorted
  `(key, position)` pairs directly, so the tiebreak was doing the stabilising
  rather than the algorithm, and even selection sort passed.
* **The tracing was distorting what it measured** (day 20). Every sorting step
  carried a copy of the whole array, so merge sort *timed* as quadratic while its
  comparison count stayed n log n. Only the benchmark could have found it; every
  test passed throughout.

## Documentation

`docs/` has one write up per topic in plain English: what the structure is, why it
exists, the cost of each operation with the reasoning behind it, and the trade offs
that came up while building it. Start with `docs/00-how-to-read-this-repo.md`.
