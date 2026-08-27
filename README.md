# Interactive DSA Lab

Data structures and algorithms written from scratch in Python, then animated,
tested and benchmarked against the standard library.

This is a learning project built by following Abdul Bari's data structures and
algorithms course, one topic per day. Every commit in the history is one day of
study, so the repository doubles as a record of how the understanding was built
up rather than a single dump of finished code.

## What makes this different from the usual visualiser

Most visualiser projects on GitHub stop at the animation. This one has three
layers, and the second and third are the point:

1. **The library.** Every container and every algorithm is implemented by hand.
   Where a structure is being demonstrated, no built in Python container is used
   underneath it, because that would be hiding the very thing the project is
   supposed to show.
2. **The proof.** A pytest suite covers the library, including the awkward
   cases: empty containers, single elements, duplicate keys, deletion of a node
   with two children, resize during collision, and so on.
3. **The measurement.** A benchmark harness times each implementation against
   the equivalent from the Python standard library across growing input sizes
   and plots the curves, so a claim like "this is O(log n)" is something you can
   see rather than something you have to take on trust.

The animation layer sits on top of all three and reuses the exact same code.

## How the tracing works

Every algorithm is a Python generator. It yields a `Step` whenever something
interesting happens and returns the answer at the end.

```python
from dsalab.tracing import record, run
from dsalab.algorithms.sorting import bubble_sort

run(bubble_sort([5, 1, 4]))          # [1, 4, 5], steps thrown away
answer, steps = record(bubble_sort([5, 1, 4]))  # same answer, plus the replay
```

That single implementation is what the tests test, what the benchmarks time and
what the app animates. There is no second animated copy of the algorithm that
can quietly drift out of agreement with the real one.

## Repository layout

```
dsalab/
  tracing.py       the Step type and the run and record helpers
  structures/      containers built from scratch
  algorithms/      procedures that operate on data
tests/             pytest suite for everything in dsalab
benchmarks/        timing harness and the plots it produces
app/               Streamlit app, the visual front end
docs/              one plain English write up per topic
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,app]"

pytest                      # run the test suite
python -m benchmarks.run    # time the library and write the plots
streamlit run app/main.py   # open the visualiser
```

## Progress

The build is being done one topic per day. The table fills in as the days go by.

| Day | Topic | Status |
| - | - | - |
| 1 | Project setup and the step tracing system | done |
| 2 | Recursion in all five of its shapes | done |
| 3 | Asymptotic notation and the complexity detective | done |
| 4 | Arrays, amortised growth, special and sparse matrices, polynomials | done |
| 5 | Linked lists: singly, doubly, circular, and the two pointer tricks | done |
| 6 | Stacks, bracket matching, infix to postfix and prefix, evaluation | done |
| 7 | Queues, circular buffers, deques and sliding window maximum | done |
| 8 | Binary trees, four traversals twice over, threaded trees | done |
| 9 | Binary search trees, ordered queries, and the invariant checker | done |
| 10 | AVL trees, the four rotations, and a measured logarithmic height | done |
| 11 | Red black trees, all five rules checked after every operation | done |
| 12 | B-trees and 2-3 trees, the shape built for disk | done |

## Documentation

Every topic gets a write up in `docs/` in plain English: what the structure is,
why it exists, the cost of each operation with the reasoning behind the cost,
and any trade off that came up while building it.
