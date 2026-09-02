"""Minimum spanning trees: connect everything for the least total cost.

Given a connected undirected graph with weights, find the set of edges that keeps
everything connected for the smallest total weight. The answer always has exactly
V - 1 edges and never contains a cycle, which is why it is a tree.

The real uses are exactly what they sound like: laying cable or pipe to reach every
building for the least material, clustering (cut the k-1 heaviest edges of the tree
and you have k clusters), and approximating the travelling salesman problem.

Two algorithms, both greedy, and both provably optimal. That is unusual and worth
dwelling on, because day 19 is largely about greedy methods that are **not**
optimal. What makes these two work is the **cut property**:

> For any way of splitting the vertices into two groups, the cheapest edge crossing
> the split is in some minimum spanning tree.

The argument is short. Suppose a minimum spanning tree does not use that cheapest
crossing edge. It must cross the split somewhere, so add the cheap edge, which
creates exactly one cycle, and remove the more expensive crossing edge from that
cycle. Everything is still connected and the total is no larger. So a tree using
the cheap edge is also minimal.

Both algorithms are just different ways of choosing which split to look at:

* **Prim** grows one tree, always taking the cheapest edge leaving it.
* **Kruskal** sorts every edge and takes each one that does not close a cycle.
"""

from __future__ import annotations

from dsalab.structures.graph import Edge, Graph, Vertex
from dsalab.structures.heap import PriorityQueue
from dsalab.structures.union_find import UnionFind
from dsalab.tracing import Step, Traced, run


def total_weight(edges: list[Edge]) -> float:
    return sum(edge.weight for edge in edges)


def prim(graph: Graph, start: Vertex | None = None) -> list[Edge]:
    return run(prim_traced(graph, start))


def prim_traced(graph: Graph, start: Vertex | None = None) -> Traced[list[Edge]]:
    """Grow one tree, always adding the cheapest edge that reaches somewhere new.

    O(E log V) with the priority queue from day 13.

    The structure is exactly Dijkstra's algorithm with one line changed. Dijkstra
    keeps the cost of the whole path from the start; Prim keeps the cost of the
    single edge that would attach a vertex to the tree. Same queue, same loop, and
    they answer completely different questions. Seeing that is worth more than
    either algorithm on its own.

    The priority queue's `decrease_priority` is what makes this efficient: when a
    cheaper way to attach a vertex is found, its entry is updated in O(log n)
    rather than a duplicate being pushed.

    Prim suits **dense** graphs, because it looks at each vertex once and touches
    each edge a constant number of times, with no global sort.
    """
    if graph.directed:
        raise ValueError("a spanning tree is an undirected graph idea")
    if graph.vertex_count == 0:
        return []

    start = start if start is not None else graph.vertices[0]
    if start not in graph:
        raise KeyError(f"{start!r} is not in the graph")

    in_tree: set[Vertex] = set()
    cheapest_edge: dict[Vertex, Edge] = {}
    tree: list[Edge] = []

    frontier = PriorityQueue()
    frontier.push(start, 0.0)

    while not frontier.is_empty():
        cost, vertex = frontier.pop()
        if vertex in in_tree:
            continue

        in_tree.add(vertex)
        if vertex in cheapest_edge:
            tree.append(cheapest_edge[vertex])
            yield Step(
                "add",
                f"Added {vertex!r} to the tree using the edge of weight {cost}, the "
                "cheapest way to reach anything outside the tree.",
                {"vertex": vertex, "weight": cost, "size": len(tree)},
            )
        else:
            yield Step("start", f"Starting the tree at {vertex!r}.", {"vertex": vertex})

        for neighbour in graph.neighbours(vertex):
            if neighbour in in_tree:
                continue
            weight = graph.weight(vertex, neighbour)

            if neighbour not in frontier or weight < frontier.priority_of(neighbour):
                frontier.push(neighbour, weight)
                cheapest_edge[neighbour] = Edge(vertex, neighbour, weight)
                yield Step(
                    "offer",
                    f"{neighbour!r} can now be reached for {weight}, which is the best "
                    "offer it has had, so its place in the queue is updated rather than "
                    "duplicated.",
                    {"vertex": neighbour, "weight": weight, "from": vertex},
                )

    if len(in_tree) != graph.vertex_count:
        yield Step(
            "disconnected",
            f"Only {len(in_tree)} of {graph.vertex_count} vertices were reachable, so "
            "this is a spanning tree of one component rather than of the whole graph.",
            {"reached": len(in_tree)},
        )

    return tree


def kruskal(graph: Graph) -> list[Edge]:
    return run(kruskal_traced(graph))


def kruskal_traced(graph: Graph) -> Traced[list[Edge]]:
    """Sort every edge, then take each one that does not close a cycle.

    O(E log E) for the sort, which dominates, since the union find operations that
    follow are effectively constant time.

    The cycle test is the whole algorithm and it is where day 17's union find
    earns its place: two vertices are already connected exactly when they are in
    the same group, so `union` returning False **is** the cycle test. Without union
    find you would have to run a traversal per edge to check, which is O(V + E) per
    edge and turns the whole thing quadratic.

    Kruskal suits **sparse** graphs, where there are few edges to sort. It also has
    a property Prim lacks: it never needs the graph to be connected. On a
    disconnected graph it produces a minimum spanning **forest**, one tree per
    component, with no special handling at all.
    """
    if graph.directed:
        raise ValueError("a spanning tree is an undirected graph idea")

    vertices = graph.vertices
    position = {vertex: index for index, vertex in enumerate(vertices)}
    groups = UnionFind(len(vertices))

    edges = sorted(graph.edges(), key=lambda edge: edge.weight)
    yield Step(
        "sorted",
        f"Sorted all {len(edges)} edges by weight. The greedy choice is always the "
        "next cheapest one that does not close a cycle.",
        {"count": len(edges)},
    )

    tree: list[Edge] = []
    for edge in edges:
        a, b = position[edge.source], position[edge.target]

        if groups.union(a, b):
            tree.append(edge)
            yield Step(
                "add",
                f"{edge.source!r} and {edge.target!r} were in different groups, so the "
                f"edge of weight {edge.weight} joins them. {groups.groups} group(s) left.",
                {"edge": (edge.source, edge.target), "weight": edge.weight,
                 "groups": groups.groups},
            )
            if len(tree) == len(vertices) - 1:
                yield Step(
                    "done",
                    f"{len(vertices) - 1} edges connect {len(vertices)} vertices, so the "
                    "tree is complete and the remaining edges cannot help.",
                    {"total": total_weight(tree)},
                )
                break
        else:
            yield Step(
                "skip",
                f"{edge.source!r} and {edge.target!r} are already connected, so this edge "
                "would close a cycle and is skipped.",
                {"edge": (edge.source, edge.target), "weight": edge.weight},
            )

    return tree


def minimum_spanning_forest(graph: Graph) -> list[list[Edge]]:
    """One minimum spanning tree per connected component.

    Kruskal handles a disconnected graph without noticing, so this simply groups
    its output by component. Prim would need to be restarted once per component.
    """
    trees: dict[Vertex, list[Edge]] = {}
    vertices = graph.vertices
    position = {vertex: index for index, vertex in enumerate(vertices)}
    groups = UnionFind(len(vertices))

    for edge in sorted(graph.edges(), key=lambda edge: edge.weight):
        a, b = position[edge.source], position[edge.target]
        if groups.union(a, b):
            trees.setdefault(vertices[groups.find(a)], []).append(edge)

    # Regroup, since roots move around as groups merge.
    grouped: dict[int, list[Edge]] = {}
    for edges in trees.values():
        for edge in edges:
            grouped.setdefault(groups.find(position[edge.source]), []).append(edge)

    return list(grouped.values())
