"""Shortest paths: four algorithms, and knowing which one to reach for.

Breadth first search on day 18 already finds the shortest path when every edge
costs the same. The moment edges have different weights that stops working, because
a route with more edges can be cheaper than one with fewer.

Four algorithms, and the differences are what matter:

| Algorithm | Finds | Cost | Negative weights |
| - | - | - | - |
| Dijkstra | one source to all | O((V + E) log V) | **no** |
| Bellman Ford | one source to all | O(V times E) | yes, and detects negative cycles |
| Floyd Warshall | every pair | O(V^3) | yes |
| A star | one source to one target | O((V + E) log V), usually far less | no |

The rules of thumb: Dijkstra unless there are negative weights. Bellman Ford when
there are, or when you need to know whether a negative cycle exists. Floyd Warshall
when you want every pair and the graph is small enough for a V^2 table. A star when
there is one specific target and you have a sensible estimate of the remaining
distance.
"""

from __future__ import annotations

from collections.abc import Callable

from dsalab.structures.graph import Graph, Vertex
from dsalab.structures.heap import PriorityQueue
from dsalab.tracing import Step, Traced, run

INFINITY = float("inf")


def rebuild_path(came_from: dict[Vertex, Vertex | None], goal: Vertex) -> list[Vertex]:
    """Walk the discovery pointers backwards to produce a path.

    Every algorithm here records one pointer per vertex rather than a whole path,
    which is O(V) memory instead of O(V^2), and the path is reconstructed only when
    somebody asks for it.
    """
    path: list[Vertex] = []
    current: Vertex | None = goal

    while current is not None:
        path.append(current)
        current = came_from.get(current)

    path.reverse()
    return path


def dijkstra(
    graph: Graph, start: Vertex
) -> tuple[dict[Vertex, float], dict[Vertex, Vertex | None]]:
    return run(dijkstra_traced(graph, start))


def dijkstra_traced(
    graph: Graph, start: Vertex
) -> Traced[tuple[dict[Vertex, float], dict[Vertex, Vertex | None]]]:
    """Shortest paths from one source, by always expanding the nearest unfinished vertex.

    O((V + E) log V) with a priority queue.

    The idea: keep a best known distance for every vertex, and repeatedly take the
    unfinished vertex with the smallest one. **When a vertex is taken, its distance
    is final**, because every other route to it would have to go through a vertex
    that is already further away, and adding a non negative edge cannot make it
    shorter.

    That sentence contains the whole reason **Dijkstra fails with negative
    weights.** The moment an edge can reduce a distance, a vertex taken early can
    be improved later, and the algorithm has already moved on. It does not detect
    the problem, it just returns a wrong answer, which is worse. There is a test
    that builds a small graph where it does exactly that, because the failure is
    much more convincing than the warning.

    Note the shape: this is Prim's algorithm from `mst.py` with one difference.
    Prim compares the weight of a single edge, Dijkstra compares the distance of a
    whole path. Same queue, same loop, different question.
    """
    if start not in graph:
        raise KeyError(f"{start!r} is not in the graph")

    distance: dict[Vertex, float] = {vertex: INFINITY for vertex in graph.vertices}
    came_from: dict[Vertex, Vertex | None] = {start: None}
    settled: set[Vertex] = set()

    distance[start] = 0.0
    frontier = PriorityQueue()
    frontier.push(start, 0.0)

    while not frontier.is_empty():
        current_distance, vertex = frontier.pop()

        if vertex in settled:
            continue
        settled.add(vertex)

        yield Step(
            "settle",
            f"{vertex!r} is the nearest unfinished vertex at distance {current_distance}, "
            "so that distance is final. Nothing further away can improve it, as long as "
            "no edge is negative.",
            {"vertex": vertex, "distance": current_distance, "settled": len(settled)},
        )

        for neighbour in graph.neighbours(vertex):
            if neighbour in settled:
                continue

            weight = graph.weight(vertex, neighbour)
            if weight < 0:
                yield Step(
                    "warning",
                    f"The edge {vertex!r} to {neighbour!r} has negative weight {weight}. "
                    "Dijkstra's guarantee does not hold here and the answer may be wrong. "
                    "Bellman Ford is the algorithm for this graph.",
                    {"from": vertex, "to": neighbour, "weight": weight},
                )

            offered = current_distance + weight
            if offered < distance[neighbour]:
                distance[neighbour] = offered
                came_from[neighbour] = vertex
                frontier.push(neighbour, offered)
                yield Step(
                    "relax",
                    f"Going through {vertex!r} reaches {neighbour!r} in {offered}, better "
                    "than anything found before, so its best known distance improves.",
                    {"vertex": neighbour, "distance": offered, "via": vertex},
                )

    return distance, came_from


def dijkstra_path(graph: Graph, start: Vertex, goal: Vertex) -> tuple[list[Vertex], float]:
    """The shortest path between two vertices and its total cost."""
    distance, came_from = dijkstra(graph, start)
    if distance.get(goal, INFINITY) == INFINITY:
        return [], INFINITY
    return rebuild_path(came_from, goal), distance[goal]


def bellman_ford(
    graph: Graph, start: Vertex
) -> tuple[dict[Vertex, float], dict[Vertex, Vertex | None]]:
    return run(bellman_ford_traced(graph, start))


def bellman_ford_traced(
    graph: Graph, start: Vertex
) -> Traced[tuple[dict[Vertex, float], dict[Vertex, Vertex | None]]]:
    """Shortest paths that cope with negative weights. O(V times E).

    Where Dijkstra is clever about which vertex to look at next, this is brute
    force with a good stopping point: **relax every edge, V - 1 times.**

    Why V - 1 passes is exactly right: a shortest path can never contain a cycle
    (removing the cycle would only make it cheaper or equal), so it has at most
    V - 1 edges. Each pass is guaranteed to get at least one more edge of every
    shortest path correct, so V - 1 passes finish the job whatever order the edges
    are considered in.

    Then comes the part Dijkstra cannot do at all. Run **one more pass**. If any
    distance still improves, there is a **negative cycle**: a loop whose total
    weight is below zero, which means going round it again is always cheaper and no
    shortest path exists.

    That is a real question, not a curiosity. In currency exchange, a negative
    cycle is an arbitrage opportunity. In a network of costs and rebates, it is a
    money printing bug.

    The early exit matters in practice: if a pass changes nothing, everything has
    settled and the remaining passes would do nothing, so it stops. On typical
    graphs that turns the worst case V times E into something far smaller.
    """
    if start not in graph:
        raise KeyError(f"{start!r} is not in the graph")

    distance: dict[Vertex, float] = {vertex: INFINITY for vertex in graph.vertices}
    came_from: dict[Vertex, Vertex | None] = {start: None}
    distance[start] = 0.0
    edges = graph.edges()

    if not graph.directed:
        # An undirected edge is two directed ones. Worth noting that an undirected
        # graph with any negative edge automatically has a negative cycle, since
        # you can walk back and forth along it forever.
        edges = edges + [type(edge)(edge.target, edge.source, edge.weight) for edge in edges]

    for pass_number in range(graph.vertex_count - 1):
        changed = False

        for edge in edges:
            if distance[edge.source] == INFINITY:
                continue
            offered = distance[edge.source] + edge.weight
            if offered < distance[edge.target]:
                distance[edge.target] = offered
                came_from[edge.target] = edge.source
                changed = True

        yield Step(
            "pass",
            f"Pass {pass_number + 1} of at most {graph.vertex_count - 1}: "
            + ("some distances improved." if changed else "nothing changed."),
            {"pass": pass_number + 1, "changed": changed,
             "distances": dict(distance)},
        )

        if not changed:
            yield Step(
                "settled",
                "A pass that changes nothing means everything has settled, so the "
                "remaining passes would do no work.",
                {"passes_used": pass_number + 1},
            )
            break

    # One extra pass. Any further improvement means a negative cycle.
    for edge in edges:
        if distance[edge.source] == INFINITY:
            continue
        if distance[edge.source] + edge.weight < distance[edge.target]:
            yield Step(
                "negative-cycle",
                f"After V-1 passes the edge {edge.source!r} to {edge.target!r} still "
                "improves things, which is only possible if a cycle has negative total "
                "weight. No shortest path exists.",
                {"edge": (edge.source, edge.target), "weight": edge.weight},
            )
            raise ValueError(
                f"the graph has a negative cycle reachable from {start!r}, so shortest "
                "paths are not defined"
            )

    return distance, came_from


def has_negative_cycle(graph: Graph, start: Vertex) -> bool:
    """Whether a negative cycle can be reached from `start`."""
    try:
        bellman_ford(graph, start)
        return False
    except ValueError:
        return True


def floyd_warshall(graph: Graph) -> dict[Vertex, dict[Vertex, float]]:
    return run(floyd_warshall_traced(graph))


def floyd_warshall_traced(graph: Graph) -> Traced[dict[Vertex, dict[Vertex, float]]]:
    """Shortest paths between every pair of vertices. O(V^3) time, O(V^2) memory.

    The whole algorithm is three nested loops, and the order of those loops is the
    entire idea:

        for middle in vertices:
            for start in vertices:
                for end in vertices:
                    if going through middle is shorter, take it

    The **middle loop must be outermost**, and that is the part everyone gets wrong
    at least once. The invariant being built is "the best path from start to end
    using only the vertices considered so far as intermediate points". Each turn of
    the outer loop allows one more vertex to be used in the middle of a path.

    Swap the loops and the invariant collapses: you would be updating paths using
    an intermediate whose own best route has not been worked out yet, and the
    answers come out too large in ways that are hard to spot, because many of them
    are still correct.

    Running Dijkstra from every vertex costs O(V times (V + E) log V), which is
    better on sparse graphs. Floyd Warshall wins on dense graphs, handles negative
    weights, and is four lines, which counts for something.
    """
    vertices = graph.vertices
    distance: dict[Vertex, dict[Vertex, float]] = {
        source: {target: INFINITY for target in vertices} for source in vertices
    }

    for vertex in vertices:
        distance[vertex][vertex] = 0.0
    for edge in graph.edges():
        distance[edge.source][edge.target] = min(
            distance[edge.source][edge.target], edge.weight
        )
        if not graph.directed:
            distance[edge.target][edge.source] = min(
                distance[edge.target][edge.source], edge.weight
            )

    for middle in vertices:
        for source in vertices:
            if distance[source][middle] == INFINITY:
                continue  # nothing can be improved by going through an unreachable point
            for target in vertices:
                through = distance[source][middle] + distance[middle][target]
                if through < distance[source][target]:
                    distance[source][target] = through

        yield Step(
            "allow",
            f"Allowed {middle!r} as an intermediate point. Every pair now holds the best "
            "route using only the vertices considered so far in the middle.",
            {"middle": middle},
        )

    for vertex in vertices:
        if distance[vertex][vertex] < 0:
            yield Step(
                "negative-cycle",
                f"{vertex!r} can reach itself with total weight {distance[vertex][vertex]}, "
                "which is a negative cycle.",
                {"vertex": vertex},
            )
            raise ValueError(f"the graph has a negative cycle through {vertex!r}")

    return distance


def a_star(
    graph: Graph,
    start: Vertex,
    goal: Vertex,
    estimate: Callable[[Vertex, Vertex], float],
) -> tuple[list[Vertex], float]:
    return run(a_star_traced(graph, start, goal, estimate))


def a_star_traced(
    graph: Graph,
    start: Vertex,
    goal: Vertex,
    estimate: Callable[[Vertex, Vertex], float],
) -> Traced[tuple[list[Vertex], float]]:
    """Dijkstra with a hint about which direction the goal is in.

    Dijkstra expands the nearest vertex, which means it searches equally in every
    direction, including directly away from the goal. If you have any idea where
    the goal is, that is wasted work.

    A star orders the queue by **distance so far plus an estimate of the distance
    remaining** rather than distance so far alone. Vertices that look like progress
    towards the goal get expanded first, and on a map style graph this can cut the
    work by an order of magnitude.

    The estimate is called a heuristic, and it must be **admissible**: it must never
    overestimate the true remaining distance. Straight line distance is the standard
    choice on a map, since no route can be shorter than flying directly.

    Why admissibility matters: if the estimate is too high for some vertex, A star
    can be talked out of expanding the vertex that is actually on the best path, and
    it returns a route that is not optimal. It does this quietly. There is a test
    that uses a deliberately inflated heuristic and shows a wrong answer coming out,
    because "must not overestimate" is a rule that is easy to break by accident when
    inventing a heuristic for a new problem.

    With an estimate of zero everywhere, this **is** Dijkstra: no information, so no
    guidance, and the tests check that the two then agree exactly.
    """
    if start not in graph or goal not in graph:
        raise KeyError("both the start and the goal must be in the graph")

    best_known: dict[Vertex, float] = {start: 0.0}
    came_from: dict[Vertex, Vertex | None] = {start: None}
    settled: set[Vertex] = set()

    frontier = PriorityQueue()
    frontier.push(start, estimate(start, goal))

    while not frontier.is_empty():
        _, vertex = frontier.pop()

        if vertex == goal:
            path = rebuild_path(came_from, goal)
            yield Step(
                "found",
                f"Reached {goal!r} with a total cost of {best_known[goal]}, having "
                f"expanded {len(settled)} of {graph.vertex_count} vertices.",
                {"path": path, "cost": best_known[goal], "expanded": len(settled)},
            )
            return path, best_known[goal]

        if vertex in settled:
            continue
        settled.add(vertex)

        for neighbour in graph.neighbours(vertex):
            offered = best_known[vertex] + graph.weight(vertex, neighbour)

            if offered < best_known.get(neighbour, INFINITY):
                best_known[neighbour] = offered
                came_from[neighbour] = vertex
                guess = estimate(neighbour, goal)
                frontier.push(neighbour, offered + guess)
                yield Step(
                    "consider",
                    f"{neighbour!r} costs {offered} to reach and is estimated {guess} from "
                    f"the goal, so it is queued at {offered + guess}. The estimate is what "
                    "makes the search head towards the goal rather than spreading evenly.",
                    {"vertex": neighbour, "cost": offered, "estimate": guess},
                )

    yield Step("unreachable", f"{goal!r} cannot be reached from {start!r}.", {})
    return [], INFINITY


def no_estimate(a: Vertex, b: Vertex) -> float:
    """The heuristic that knows nothing, turning A star back into Dijkstra."""
    return 0.0


def grid_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Straight line distance between two grid points, for A star on a map.

    Admissible whenever movement costs at least the straight line distance, which
    holds for the usual eight direction grid and certainly for four direction
    movement, where it is a considerable underestimate.
    """
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Grid distance when only horizontal and vertical moves are allowed.

    Tighter than straight line distance on a four direction grid, and still
    admissible, which makes it the better choice there: a tighter admissible
    estimate means fewer vertices expanded for the same guaranteed answer.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
