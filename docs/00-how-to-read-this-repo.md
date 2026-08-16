# How to read this repository

Start here if you are picking the project up cold, including future me.

## The one idea you need first

Everything in the library is written as a generator that reports what it is
doing. A normal sort function looks like this:

```python
def bubble_sort(values):
    ...
    return values
```

The version in this project looks like this instead:

```python
def bubble_sort(values):
    ...
    yield Step("compare", "Is 5 bigger than 1?", {"i": 0, "j": 1})
    ...
    return values
```

The only difference is the `yield`. The algorithm still does the same work in
the same order. But because it hands control back after each interesting event,
a caller can watch the run happen instead of only seeing the answer.

Two helpers in `dsalab/tracing.py` do the consuming:

* `run(...)` drives the generator to the end and gives you the answer. The
  steps are thrown away. Tests and benchmarks use this.
* `record(...)` drives it to the end and keeps every step in a list. The app
  uses this so you can step forward and backward through the run.

## Why not write the algorithm twice

The obvious alternative is to keep a clean fast version for real use and a
separate instrumented version for the animation. That is how most visualiser
projects are built, and it has a quiet failure mode: you fix a bug in one copy
and forget the other, and now the animation is teaching something the code does
not actually do. With one generator there is nothing to keep in sync.

The cost of this choice is a small amount of speed, because yielding is not
free. That matters for benchmarking, so the benchmark harness always uses
`run(...)`, which never builds a list of steps. Where the yielding overhead
would still distort a measurement, the benchmark says so in its own notes.

## The parts of a step

A `Step` has three fields:

* `kind` is a short label such as `compare`, `swap`, `visit` or `insert`. The
  app picks colours from it and the tests count it.
* `note` is one sentence of plain English. This is the caption a learner reads.
* `data` is whatever the drawing code needs for that frame, for example the
  indices involved and a snapshot of the array.

## Reading order

1. `dsalab/tracing.py`, which is short and explains itself.
2. Any file in `dsalab/structures/`, because containers are easier to follow
   than algorithms.
3. The matching test file in `tests/`, which shows the edge cases that the
   structure is expected to survive.
4. The matching document in `docs/`, which explains the cost of each operation
   and why it costs that much.

## A note on the vocabulary

Where a document says an operation is O(1) or O(log n), it is talking about how
the running time grows as the amount of data grows, not about how fast it is in
absolute terms. An O(1) operation on a slow machine can easily be slower than an
O(log n) operation on a fast one for small inputs. The benchmarks exist to keep
that distinction honest, because they show real timings next to the theory.
