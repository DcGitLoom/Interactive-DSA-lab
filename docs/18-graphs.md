# Day 18: Graphs

Code: `dsalab/structures/graph.py`, `dsalab/algorithms/graph_traversal.py`,
`scc.py`, `mst.py`, `shortest_path.py`. Tests: `tests/test_graphs.py`.

Every structure so far has been about holding values. A graph is about the
**connections** between them, and it is the model for roads, links, dependencies,
friendships, pipes, circuits and state machines.

## Choosing a representation

| | Adjacency list | Adjacency matrix |
| - | - | - |
| Memory | O(V + E) | O(V^2) regardless of edges |
| List a vertex's neighbours | O(degree) | O(V) |
| Check one specific edge | O(degree) | O(1) |
| Suits | sparse graphs | dense graphs, linear algebra |

The deciding number is **density**, and real graphs are far sparser than people
expect. A road network has about four roads per junction, not four thousand. For a
million junctions the list needs a few million entries and the matrix needs a
trillion cells: not a trade off, the difference between working and not working.

One small decision with large consequences: **a missing edge weighs infinity, not
zero.** Zero is a perfectly good weight, so using it to mean "no edge" makes a free
road indistinguishable from a missing one. Infinity also makes the shortest path
arithmetic work with no special cases, which is why Floyd Warshall comes out at
four lines.

## Breadth first and depth first are one algorithm

They differ in exactly one thing: what holds the vertices waiting to be visited.

* A **queue** takes the oldest waiting vertex, so everything one step away is
  visited before anything two steps away. The search spreads in rings.
* A **stack** takes the newest, so the search follows one path to its end.

From that single swap follows everything:

* **Breadth first finds the shortest path in an unweighted graph**, because it
  reaches every vertex by the fewest edges. Depth first cannot, at any price.
* **Depth first reveals structure**: cycles, topological order, strongly connected
  components. Breadth first cannot, because it never has one path on the stack to
  reason about.

### The bug that is invisible in the output

A vertex must be marked as seen **when it is queued**, not when it is visited. Mark
on visit and a vertex can be queued by several neighbours before it is taken out,
turning a linear traversal quadratic on a dense graph.

The answer is still correct, which is what makes it nasty. There is a test that
builds a star shaped graph and counts queue steps, because the step count is the
only place the difference shows.

## Cycle detection: two genuinely different problems

**Directed.** A cycle means an edge back to a vertex **still on the current path**,
not merely one seen before. Reaching a finished vertex is normal, it means two
paths converged. So three states are needed: unvisited, in progress, finished.
Using only two reports a cycle for any diamond shape, which is the classic wrong
version, and there is a test built on exactly a diamond.

**Undirected.** Every edge can be walked both ways, so the search always sees the
edge it just arrived by. That is not a cycle. Ignore the immediate parent, and then
any other edge to a visited vertex is real.

## Topological sort

Order the vertices so every edge points forwards: build systems, package managers,
spreadsheet recalculation, course prerequisites.

**Kahn's method** repeatedly takes a vertex with nothing pointing at it. Two
properties worth knowing:

* It detects cycles for free. If the loop finishes with vertices left over, every
  one of them is still waiting on something, which can only happen in a cycle.
* **The answer is usually not unique.** Independent tasks can go either way round,
  so the tests verify the defining property (every edge points forwards) rather
  than comparing against one expected list. That is the right shape of test
  whenever an answer is valid rather than singular.

**The depth first version** records each vertex when its recursion *returns* and
reverses at the end, because a vertex is finished only after everything it points
at is. Shorter, but it needs a separate cycle check and uses stack depth
proportional to the longest path.

## Strongly connected components

In a directed graph, reachability is one way, so "connected" means **mutually**
reachable. Collapsing each component to a point turns any directed graph into an
acyclic one that can be topologically sorted, which is the standard way to handle
cyclic dependencies. It is also how deadlock detection works.

**Tarjan** does it in one pass. Number vertices as they are first reached, then
track each one's **low link**: the smallest number reachable, including by one edge
back to a vertex still on the current path. A vertex whose low link equals its own
number is the root of a component.

The subtle part is the condition that a vertex may only take a low link from a
neighbour **still on the stack**. Without it, a vertex would inherit from a
component that is already finished and two separate components would merge. That
condition is the whole correctness argument.

**Kosaraju** does it in two passes and is much easier to believe: depth first
search recording finish order, reverse every edge, then traverse in reverse finish
order. In the reversed graph a traversal reaches everything that could reach the
start, and combining that with what the start can reach is exactly mutual
reachability.

Tarjan is faster, Kosaraju is more convincing. The tests check they agree on two
hundred random graphs, and separately verify that every member of a component can
actually reach every other member, which is the definition rather than the
algorithm.

## Minimum spanning trees, and why greedy works here

Both algorithms are greedy and both are **provably optimal**, which is unusual and
worth dwelling on, since day 19 is largely about greedy methods that are not. What
makes these work is the **cut property**:

> For any split of the vertices into two groups, the cheapest edge crossing the
> split belongs to some minimum spanning tree.

The proof is one paragraph. Suppose a minimum tree omits that cheapest crossing
edge. It must cross somewhere, so add the cheap edge, creating exactly one cycle,
and remove the more expensive crossing edge from that cycle. Still connected, total
no larger. So a tree containing the cheap edge is also minimal.

Both algorithms are just different ways of choosing which split to look at:

| | Prim | Kruskal |
| - | - | - |
| Method | grow one tree, take the cheapest edge leaving it | sort all edges, take each that closes no cycle |
| Cost | O(E log V) | O(E log E), dominated by the sort |
| Suits | dense graphs | sparse graphs |
| Disconnected input | needs restarting per component | produces a forest with no special handling |

**Prim is Dijkstra with one line changed.** Dijkstra's queue holds the cost of the
whole path from the start; Prim's holds the cost of the single edge that would
attach a vertex. Same queue, same loop, different question. Seeing that is worth
more than either algorithm alone.

**Kruskal is where day 17's union find pays off.** Two vertices are already
connected exactly when they are in the same group, so `union` returning False *is*
the cycle test. Without it you would run a traversal per edge and the whole thing
becomes quadratic.

The two can pick different edges when weights tie, so the tests compare the
**total** across two hundred random graphs rather than the edge sets.

## Shortest paths: four algorithms

| Algorithm | Finds | Cost | Negative weights |
| - | - | - | - |
| Dijkstra | one to all | O((V + E) log V) | **no** |
| Bellman Ford | one to all | O(V times E) | yes, detects negative cycles |
| Floyd Warshall | all pairs | O(V^3) | yes |
| A star | one to one | usually far less than Dijkstra | no |

### Dijkstra, and the exact sentence that explains its limit

Take the nearest unfinished vertex. **Its distance is now final**, because any
other route would have to pass through something already further away, and adding a
non negative edge cannot make it shorter.

That sentence contains the failure. The moment an edge can be negative, a vertex
settled early can be improved later, and the algorithm has moved on. **It does not
detect this, it just returns a wrong answer.**

So there is a test that builds a three vertex graph where Dijkstra confidently
returns 2 for a vertex whose true distance is 1, with Bellman Ford next to it
getting it right. Demonstrating the failure beats warning about it.

### Bellman Ford, and why V-1 passes

Brute force with a good stopping point: relax every edge, V - 1 times.

Why V - 1 is exactly right: a shortest path can never contain a cycle, since
removing the cycle would only make it cheaper, so it has at most V - 1 edges. Each
pass fixes at least one more edge of every shortest path.

Then the part Dijkstra cannot do at all. Run **one more pass**. If anything still
improves, there is a **negative cycle** and no shortest path exists. That is a real
question: in currency exchange a negative cycle is an arbitrage opportunity, and in
a system of costs and rebates it is a money printing bug.

### Floyd Warshall, and the loop order everyone gets wrong once

Three nested loops, and **the intermediate vertex loop must be outermost**. The
invariant being built is "the best path using only the vertices considered so far
as intermediate points", and each turn of the outer loop admits one more.

Swap the loops and the invariant collapses: you update paths through an
intermediate whose own best route is not yet known. The answers come out subtly too
large, and many of them are still correct, which makes it hard to spot.

### A star, and admissibility

Dijkstra expands the nearest vertex, so it searches in every direction including
directly away from the goal. A star orders the queue by **distance so far plus an
estimate of the distance remaining**, which sends it towards the goal.

The estimate must be **admissible**: never an overestimate. There is a test with a
deliberately inflated estimate where A star returns a route costing 5 when the best
costs 2, and reports nothing wrong. That is the point: this rule is easy to break
by accident when inventing a heuristic for a new problem, and breaking it fails
silently.

With an estimate of zero everywhere, A star **is** Dijkstra, and the tests check
they agree exactly. A tighter admissible estimate expands fewer vertices for the
same guaranteed answer, which is why Manhattan distance beats straight line
distance on a four direction grid.

## What the test file is really doing

Three patterns worth reusing anywhere:

1. **Two algorithms for one problem check each other.** Tarjan against Kosaraju,
   Prim against Kruskal, Dijkstra against Bellman Ford against Floyd Warshall, on
   hundreds of random graphs. Far better coverage than hand written expectations.
2. **Every documented failure has a test that demonstrates it.** Dijkstra on
   negative weights, A star with a bad heuristic, the two state cycle check on a
   diamond. A comment saying "this does not work with X" rots; a test does not.
3. **Where the answer is not unique, test the property.** Any valid topological
   order passes, because any valid topological order is a correct answer.
