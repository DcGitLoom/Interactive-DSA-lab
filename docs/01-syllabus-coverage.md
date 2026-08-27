# Syllabus coverage

This is the checklist the project is built against. The left hand side follows
the running order of Abdul Bari's data structures and algorithms course, so
nothing from the course is quietly skipped. The extra topics at the bottom are
ones the course does not cover but that come up constantly in interviews and in
real systems, so they are included too.

A row is only ticked when three things exist: the implementation, tests that
cover its edge cases, and a document explaining the cost of each operation.

## Foundations

| Topic | Where it lives | Status |
| - | - | - |
| Recursion and its shapes (tail, tree, indirect, nested) | `dsalab/algorithms/recursion.py` | done |
| Asymptotic notation, and measuring it for real | `dsalab/complexity.py` | done |
| Static and dynamic memory, how Python objects sit in memory | `docs/04-arrays.md` | done |

## Arrays and matrices

| Topic | Where it lives | Status |
| - | - | - |
| Dynamic array (growth, amortised cost) | `dsalab/structures/dynamic_array.py` | done |
| Two dimensional arrays and row major layout | `docs/04-arrays.md` | done |
| Diagonal, triangular and band matrix representation | `dsalab/structures/special_matrix.py` | done |
| Sparse matrix representation and addition | `dsalab/structures/sparse_matrix.py` | done |
| Polynomial representation and evaluation | `dsalab/structures/polynomial.py` | done |

## Linked lists

| Topic | Where it lives | Status |
| - | - | - |
| Singly linked list | `dsalab/structures/linked_list.py` | done |
| Doubly linked list | `dsalab/structures/linked_list.py` | done |
| Circular linked list | `dsalab/structures/linked_list.py` | done |
| Reversal, cycle detection, merge of sorted lists | `dsalab/structures/linked_list.py` | done |

## Stacks and queues

| Topic | Where it lives | Status |
| - | - | - |
| Stack | `dsalab/structures/stack.py` | done |
| Parenthesis matching | `dsalab/algorithms/expressions.py` | done |
| Infix to postfix, infix to prefix, postfix evaluation | `dsalab/algorithms/expressions.py` | done |
| Queue, circular queue, queue from two stacks | `dsalab/structures/queue.py` | done |
| Deque, and sliding window maximum | `dsalab/structures/deque.py` | done |
| Priority queue | `dsalab/structures/heap.py` | pending |

## Trees

| Topic | Where it lives | Status |
| - | - | - |
| Binary tree, all four traversals, iterative and recursive | `dsalab/structures/binary_tree.py` | done |
| Building a tree from traversals, counting, height | `dsalab/structures/binary_tree.py` | done |
| Threaded binary tree | `dsalab/structures/threaded_tree.py` | done |
| Binary search tree | `dsalab/structures/bst.py` | done |
| AVL tree with all four rotations | `dsalab/structures/avl.py` | done |
| Red black tree | `dsalab/structures/red_black.py` | done |
| 2-3 tree and B tree | `dsalab/structures/btree.py` | done |
| Trie and compressed trie | `dsalab/structures/trie.py` | pending |
| Segment tree and Fenwick tree | `dsalab/structures/segment_tree.py` | pending |

## Heaps and hashing

| Topic | Where it lives | Status |
| - | - | - |
| Binary heap, heapify, heap sort | `dsalab/structures/heap.py` | pending |
| Hash table with separate chaining | `dsalab/structures/hash_table.py` | pending |
| Hash table with open addressing (linear, quadratic, double) | `dsalab/structures/hash_table.py` | pending |
| LRU cache built from a hash map and a linked list | `dsalab/structures/lru_cache.py` | pending |

## Sorting and searching

| Topic | Where it lives | Status |
| - | - | - |
| Bubble, insertion, selection | `dsalab/algorithms/sorting.py` | pending |
| Merge sort, iterative and recursive | `dsalab/algorithms/sorting.py` | pending |
| Quick sort and its pivot choices | `dsalab/algorithms/sorting.py` | pending |
| Heap sort, shell sort | `dsalab/algorithms/sorting.py` | pending |
| Counting sort, radix sort, bucket sort | `dsalab/algorithms/sorting.py` | pending |
| Why comparison sorting cannot beat n log n | `docs/13-sorting.md` | pending |
| Binary search and its variants (first, last, insertion point) | `dsalab/algorithms/searching.py` | pending |

## Graphs

| Topic | Where it lives | Status |
| - | - | - |
| Adjacency list and adjacency matrix | `dsalab/structures/graph.py` | pending |
| Breadth first and depth first search | `dsalab/algorithms/graph_traversal.py` | pending |
| Spanning trees, Prim, Kruskal | `dsalab/algorithms/mst.py` | pending |
| Union find with union by rank and path compression | `dsalab/structures/union_find.py` | pending |
| Dijkstra, Bellman Ford, Floyd Warshall, A star | `dsalab/algorithms/shortest_path.py` | pending |
| Topological sort, cycle detection | `dsalab/algorithms/graph_traversal.py` | pending |
| Tarjan's strongly connected components | `dsalab/algorithms/scc.py` | pending |

## Algorithm design techniques

| Topic | Where it lives | Status |
| - | - | - |
| Greedy: activity selection, fractional knapsack, Huffman coding | `dsalab/algorithms/greedy.py` | pending |
| Dynamic programming: LIS, 0/1 knapsack, matrix chain, coin change, LCS, edit distance | `dsalab/algorithms/dp.py` | pending |
| Backtracking: N queens, sudoku, permutations, graph colouring, Hamiltonian cycle | `dsalab/algorithms/backtracking.py` | pending |
| Branch and bound: travelling salesman | `dsalab/algorithms/branch_and_bound.py` | pending |
| Where greedy fails and dynamic programming does not | `dsalab/algorithms/greedy_vs_dp.py` | pending |
| NP hard and NP complete, in plain English | `docs/20-np-completeness.md` | pending |

## Extra topics the course does not cover

| Topic | Where it lives | Status |
| - | - | - |
| String matching: KMP, Rabin Karp, Z algorithm | `dsalab/algorithms/strings.py` | pending |
| Disjoint set applications, Kruskal's dependency on it | `dsalab/structures/union_find.py` | pending |
| Bit manipulation tricks used by the structures | `docs/21-bit-tricks.md` | pending |
| Amortised analysis, explained through the dynamic array | `docs/04-arrays.md` | done |

## The four things that make this more than a visualiser

These are the features that carry the project beyond drawing bars on a screen.
Each is described in its own document when it is built.

1. **Complexity detective.** The harness times a function across growing input
   sizes, fits the measurements against the standard growth curves, and reports
   which one actually matches. You get told that your hash table lookup measured
   as constant time rather than being asked to believe it.
2. **Race mode.** Two algorithms run on the identical input side by side, frame
   by frame, with live counters for comparisons and swaps. This is the fastest
   way to feel why insertion sort beats quick sort on tiny inputs and loses
   badly on large ones.
3. **Invariant checking.** Built on day 9 in `dsalab/invariants.py`. Every
   structure can state its own rules in code, and inside a `checked()` block the
   rules are verified after every single mutation, naming the broken rule in
   plain English and the node that broke it.
4. **Counterexample finder.** For problems where a simple greedy method looks
   right but is not, the lab searches small inputs until it finds one where
   greedy and dynamic programming disagree, then shows both answers. Being
   handed a concrete failing case is far more convincing than being told that
   greedy is not always optimal.
