# NP completeness, in plain English

This is the last topic of the course and the one most often taught as vocabulary
rather than as an idea. It is worth the effort, because it is the only part of the
subject that tells you when to **stop looking** for a fast algorithm.

## The question

Some problems have fast algorithms. Sorting is n log n, shortest paths are
essentially linear in the graph, and a minimum spanning tree is E log V.

Some problems do not, as far as anyone knows. Nobody can find the shortest tour
through 100 cities quickly, or fill a large knapsack optimally, or three colour a
big graph. **And nobody can prove those problems are actually hard.** That gap
between "we cannot do it" and "it cannot be done" is what this topic is about.

## The classes, without the formalism

**P** is the set of problems solvable in polynomial time: n, n^2, n^3, anything
where the exponent is a constant. Informally, the problems we call tractable.

**NP** is the set of problems where a proposed answer can be **checked** in
polynomial time. Not solved, checked.

The distinction is the whole idea, and sudoku makes it concrete. Solving a large
sudoku is hard. Checking a completed grid is trivial: look at every row, column and
box. Same for the travelling salesman: finding the best tour is hard, but given a
tour and a budget, adding up the distances takes no time at all.

Every problem in P is also in NP, since if you can solve it quickly you can check
an answer by solving it again. **Whether NP contains anything more is the open
question**, written P versus NP, and it has stood since 1971 with a million dollar
prize attached.

The honest summary of what most people believe: P is probably not NP, because
thousands of researchers have tried to find fast algorithms for these problems for
fifty years and nobody has, but that is evidence rather than proof.

## NP complete and NP hard

**NP complete** problems are the hardest ones in NP, and they have a remarkable
property: **they are all the same problem in disguise.** Any of them can be
translated into any other in polynomial time. So a fast algorithm for any single NP
complete problem gives a fast algorithm for all of them, and P equals NP.

That translation is called a reduction, and it is what the whole edifice rests on.
Cook proved in 1971 that boolean satisfiability is NP complete, and every other NP
complete problem has been proved so by reducing satisfiability, or something
already reduced from it, into it.

**NP hard** means "at least as hard as anything in NP", without needing to be in NP
itself. The travelling salesman *optimisation* problem is NP hard: given a tour you
cannot quickly check it is the best one, so it is not in NP, but it is at least as
hard as the decision version that is.

## The line runs through this project

The most striking thing about NP completeness is how **little** separates the easy
problems from the intractable ones. Every pair below appears in this repository:

| Easy | Intractable | The difference |
| - | - | - |
| Shortest path (day 18) | Longest simple path | asking for the maximum instead of the minimum |
| Euler path, visit every edge once | Hamiltonian cycle, visit every vertex once (day 19) | edges instead of vertices |
| Two colouring, bipartite check (day 18) | Three colouring (day 19) | one more colour |
| Fractional knapsack (day 19) | 0/1 knapsack (day 19) | items cannot be split |
| Minimum spanning tree (day 18) | Travelling salesman (day 19) | the tour must return home |

Look at the Euler and Hamiltonian pair in particular. Euler paths have a complete
answer you can check in linear time: one exists exactly when the graph is connected
and has zero or two vertices of odd degree. Hamiltonian cycles have nothing of the
kind and are NP complete. The two questions are near mirror images and their
difficulty is nothing alike.

**Difficulty is not visible in a problem statement.** That is the practical lesson,
and it is why recognising these problems matters more than the theory: if what you
are working on reduces to one of them, no amount of cleverness will produce a fast
exact algorithm, and you should spend your effort elsewhere.

## Where to spend that effort instead

Being NP hard does not mean giving up. Real systems solve these problems every day,
by relaxing one of the three things people usually assume:

**Give up on optimal, keep it fast.** An approximation algorithm with a proven
bound is often all you need. Day 19's `spanning_tree_tour_bound` is exactly this:
the minimum spanning tree gives a tour within twice optimal, quickly, because
removing one edge from any tour leaves a spanning tree. The Christofides algorithm
tightens that to 1.5.

**Give up on fast in the worst case, keep it optimal.** Branch and bound on day 19
is still exponential in theory and finishes twelve cities immediately in practice.
Modern satisfiability solvers routinely handle instances with millions of variables
despite the problem being the original NP complete one. The worst case is a
statement about the worst input, not about yours.

**Give up on generality.** Many NP hard problems are easy on the inputs you
actually have. Graph colouring is hard in general and easy on planar graphs. The
knapsack is hard in general and solved by a table when the capacity is small, which
is the pseudo polynomial point from day 19.

## Why this belongs at the end of the course

Everything before this day was about finding the best algorithm for a problem. This
day is about recognising when the best algorithm is one that gives up something,
and about knowing which thing to give up.

It also reframes the earlier material. Dijkstra is not merely a clever algorithm,
it is a clever algorithm **for a problem that permits one**. Its cleverness works
because shortest paths have optimal substructure and non negative weights make the
greedy choice safe. Take away either property, as negative weights do, and the
cleverness stops working, which is exactly what the day 18 test demonstrates.

The techniques in this project are not tools to apply by habit. They are tools that
fit particular shapes of problem, and the last thing worth learning is how to see
the shape first.
