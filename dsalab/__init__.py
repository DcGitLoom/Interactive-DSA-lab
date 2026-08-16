"""Interactive DSA Lab.

Data structures and algorithms written from scratch in plain Python, each one
traced so it can be animated, tested and benchmarked.

The package is split in two halves:

* `dsalab.structures` holds the containers (dynamic array, linked lists, hash
  tables, trees, heaps, graphs and so on).
* `dsalab.algorithms` holds the procedures that work on data (sorting,
  searching, graph traversal, dynamic programming and the rest).

`dsalab.tracing` is the small shared layer that lets both halves report what
they are doing, step by step.
"""

from dsalab.tracing import Step, Traced, count_kinds, record, run

__all__ = ["Step", "Traced", "count_kinds", "record", "run"]

__version__ = "0.1.0"
