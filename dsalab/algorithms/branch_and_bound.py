"""Branch and bound: backtracking that also refuses to explore losing branches.

Backtracking prunes branches that are **illegal**. Branch and bound prunes branches
that are legal but cannot possibly beat the best answer found so far.

The extra machinery is one function: a **bound**, which estimates the best possible
result achievable from a partial answer. If that optimistic estimate is still worse
than something already found, the whole branch is discarded unexamined.

The bound has one strict requirement: it must be **optimistic**. For a minimisation
problem it must never overestimate the remaining cost, or a branch containing the
true best answer gets discarded and the algorithm returns something worse while
insisting it is optimal.

That is the same rule as A star's admissible heuristic on day 18, and it fails the
same silent way. The two are close relatives: A star is branch and bound over paths.

Neither changes the worst case, which stays exponential. Both change what happens in
practice, often by orders of magnitude, and that gap between worst case and typical
is exactly why these problems are tractable in real use despite being NP hard.
"""

from __future__ import annotations

import itertools

from dsalab.algorithms.greedy import Item
from dsalab.tracing import Step, Traced, run

INFINITY = float("inf")


def travelling_salesman(distances: list[list[float]]) -> tuple[list[int], float]:
    return run(travelling_salesman_traced(distances))


def travelling_salesman_traced(distances: list[list[float]]) -> Traced[tuple[list[int], float]]:
    """The shortest tour visiting every city once and returning home.

    The classic NP hard problem. Trying every tour is (n-1)! / 2 possibilities: 12
    cities is about 20 million tours, 20 cities is 60 million billion.

    The bound used here is simple and effective: the cost so far, plus for every
    unvisited city the cheapest edge leaving it. No tour can be cheaper than that,
    since every remaining city must be left exactly once. That makes it optimistic,
    which is the requirement.

    Branch and bound does not make this problem polynomial. Nothing does, as far as
    anyone knows. It makes twelve cities finish quickly instead of slowly, and that
    is the honest claim.

    What people actually do for large instances: accept a good answer instead of the
    best one. The minimum spanning tree from day 18 gives a tour within twice
    optimal, and the Christofides algorithm gets within 1.5 times. `spanning_tree
    _tour_bound` below shows that connection.
    """
    count = len(distances)
    if count == 0:
        return [], 0.0
    if count == 1:
        return [0], 0.0

    cheapest_out = [
        min(distances[city][other] for other in range(count) if other != city)
        for city in range(count)
    ]

    best_tour: list[int] = []
    best_cost = INFINITY
    explored = 0
    pruned = 0

    def optimistic_total(cost_so_far: float, unvisited: set[int]) -> float:
        """Cost so far plus the cheapest way out of every city still to visit."""
        return cost_so_far + sum(cheapest_out[city] for city in unvisited)

    def explore(path: list[int], cost: float, unvisited: set[int]) -> Traced[None]:
        nonlocal best_tour, best_cost, explored, pruned
        explored += 1

        if not unvisited:
            total = cost + distances[path[-1]][path[0]]
            if total < best_cost:
                best_cost = total
                best_tour = path + [path[0]]
                yield Step(
                    "improve",
                    f"A complete tour costing {total} is the best so far, so every branch "
                    "whose optimistic estimate exceeds this can now be discarded.",
                    {"tour": list(best_tour), "cost": total},
                )
            return

        for city in sorted(unvisited, key=lambda other: distances[path[-1]][other]):
            step_cost = cost + distances[path[-1]][city]
            remaining = unvisited - {city}
            bound = optimistic_total(step_cost, remaining)

            if bound >= best_cost:
                pruned += 1
                yield Step(
                    "prune",
                    f"Going to city {city} costs at least {bound:.1f} even in the best "
                    f"case, which is no better than the {best_cost:.1f} already found, so "
                    "this whole branch is discarded without exploring it.",
                    {"city": city, "bound": bound, "best": best_cost},
                )
                continue

            yield from explore(path + [city], step_cost, remaining)

    yield from explore([0], 0.0, set(range(1, count)))

    yield Step(
        "done",
        f"Explored {explored} partial tours and pruned {pruned} branches. The best tour "
        f"costs {best_cost}.",
        {"explored": explored, "pruned": pruned, "cost": best_cost},
    )
    return best_tour, best_cost


def travelling_salesman_brute_force(distances: list[list[float]]) -> tuple[list[int], float]:
    """Every tour, no pruning. Kept so the pruning can be measured against it.

    Correct and hopeless, which is exactly what a reference implementation should
    be. The tests use it to confirm branch and bound finds the true optimum rather
    than merely a good tour, which is the thing a bad bound would break.
    """
    count = len(distances)
    if count <= 1:
        return list(range(count)), 0.0

    best_tour: list[int] = []
    best_cost = INFINITY

    for order in itertools.permutations(range(1, count)):
        tour = [0, *order, 0]
        cost = sum(distances[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    return best_tour, best_cost


def knapsack_branch_and_bound(items: list[Item], capacity: float) -> tuple[float, list[str]]:
    return run(knapsack_branch_and_bound_traced(items, capacity))


def knapsack_branch_and_bound_traced(
    items: list[Item], capacity: float
) -> Traced[tuple[float, list[str]]]:
    """0/1 knapsack by branch and bound rather than a table.

    The bound is the neat part and it reuses day 19's greedy work: **the fractional
    knapsack answer for the remaining items is an upper bound on the 0/1 answer.**
    Allowing fractions can only help, so the fractional value is never lower, which
    makes it optimistic in the required direction.

    That is a nice illustration of how these techniques stack. The greedy algorithm
    that was *wrong* for this problem turns out to be exactly the right tool for
    pruning the search that solves it correctly.

    Why use this rather than the dynamic programming version in `dp.py`? The table
    is O(n times capacity), which is fine for a capacity of 100 and useless for a
    capacity of a billion, or for weights that are not whole numbers. Branch and
    bound does not care about the size of the numbers, only about the number of
    items.
    """
    if capacity < 0:
        raise ValueError("capacity cannot be negative")

    ordered = sorted(items, key=lambda item: item.density, reverse=True)
    best_value = 0.0
    best_set: list[str] = []
    pruned = 0

    def fractional_bound(index: int, weight: float, value: float) -> float:
        """The best possible value from here, allowing fractions."""
        bound = value
        space = capacity - weight

        for item in ordered[index:]:
            if item.weight <= space:
                bound += item.value
                space -= item.weight
            else:
                bound += item.density * space
                break

        return bound

    def explore(index: int, weight: float, value: float, taken: list[str]) -> Traced[None]:
        nonlocal best_value, best_set, pruned

        if value > best_value:
            best_value = value
            best_set = list(taken)
            yield Step(
                "improve",
                f"A selection worth {value} is the best so far.",
                {"value": value, "items": list(taken)},
            )

        if index == len(ordered):
            return

        bound = fractional_bound(index, weight, value)
        if bound <= best_value:
            pruned += 1
            yield Step(
                "prune",
                f"Even allowing fractions, nothing from here can beat {best_value} (the "
                f"best case is {bound:.1f}), so this branch is abandoned. The greedy "
                "algorithm that gets this problem wrong is exactly what makes the bound.",
                {"bound": bound, "best": best_value},
            )
            return

        item = ordered[index]
        if weight + item.weight <= capacity:
            yield from explore(index + 1, weight + item.weight, value + item.value,
                               taken + [item.name])
        yield from explore(index + 1, weight, value, taken)

    yield from explore(0, 0.0, 0.0, [])
    yield Step("done", f"Best value {best_value}, after pruning {pruned} branch(es).",
               {"value": best_value, "pruned": pruned})
    return best_value, best_set


def spanning_tree_tour_bound(distances: list[list[float]]) -> float:
    """A lower bound on the best tour, from the minimum spanning tree.

    Removing any one edge from a tour leaves a path, which is a spanning tree. So
    **the best tour can never be cheaper than the minimum spanning tree**, and that
    tree is computable in O(E log V) by day 18's Kruskal.

    Two uses. As a bound it prunes branch and bound. And doubled, it gives a tour
    guaranteed to be within twice the optimum: walk the tree, skipping repeats. For
    a problem with no efficient exact solution, "provably within a factor of two,
    quickly" is often the right answer, and that shift from exact to approximate is
    what day 20's write up on NP completeness is really about.
    """
    from dsalab.algorithms.mst import kruskal, total_weight
    from dsalab.structures.graph import Graph

    count = len(distances)
    if count < 2:
        return 0.0

    graph = Graph()
    for a in range(count):
        for b in range(a + 1, count):
            graph.add_edge(a, b, distances[a][b])

    return total_weight(kruskal(graph))
