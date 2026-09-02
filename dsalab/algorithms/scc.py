"""Strongly connected components: the hard version of "what is connected".

In an undirected graph, connectivity is easy: run a traversal, and everything you
reach is connected to everything else you reach.

In a **directed** graph it is much harder, because reachability is one way. A can
reach B without B reaching A. A strongly connected component is a group where
**every vertex can reach every other one**, following the edge directions.

Why anybody cares:

* Collapsing each component to a single point turns any directed graph into a
  directed acyclic graph, which can then be topologically sorted. That is the
  standard way to handle cyclic dependencies: find the cycles, treat each as one
  unit, order the units.
* Deadlock detection: a cycle of processes each waiting on the next is a strongly
  connected component in the wait for graph.
* In a compiler, mutually recursive functions form one component and must be
  compiled together.

Two algorithms are here because they take completely different routes to the same
answer, and the comparison is the interesting part.
"""

from __future__ import annotations

from dsalab.structures.graph import Graph, Vertex
from dsalab.structures.stack import ArrayStack
from dsalab.tracing import Step, Traced, run


def tarjan_scc(graph: Graph) -> list[list[Vertex]]:
    return run(tarjan_scc_traced(graph))


def tarjan_scc_traced(graph: Graph) -> Traced[list[list[Vertex]]]:
    """Tarjan's algorithm: every component in a single depth first search. O(V + E).

    The idea, which takes a moment to see but is worth the effort.

    Number the vertices in the order the search first reaches them. Then for each
    vertex work out its **low link**: the smallest number reachable from it,
    including by following one edge back to a vertex still on the current path.

    A vertex whose low link equals its own number is the **root of a component**,
    because nothing below it can escape to anything discovered earlier. Everything
    still on the stack above that root belongs to its component.

    The stack is the subtle part. It is not the recursion stack. It holds vertices
    that have been visited but not yet assigned to a component, and a vertex is
    only allowed to update its low link from a neighbour that is **still on that
    stack**. Without that condition, a vertex would take a low link from a
    neighbour in a component that is already finished, and two separate components
    would be merged into one. That condition is the whole correctness argument.

    One pass, which is why this is usually preferred over Kosaraju's two passes.
    """
    if not graph.directed:
        raise ValueError("strongly connected components are a directed graph question")

    order: dict[Vertex, int] = {}
    low: dict[Vertex, int] = {}
    on_stack: set[Vertex] = set()
    stack = ArrayStack()
    components: list[list[Vertex]] = []
    counter = 0

    def walk(vertex: Vertex) -> Traced[None]:
        nonlocal counter

        order[vertex] = low[vertex] = counter
        counter += 1
        stack.push(vertex)
        on_stack.add(vertex)

        yield Step(
            "discover",
            f"Reached {vertex!r} as number {order[vertex]}. Its low link starts as its "
            "own number and can only go down.",
            {"vertex": vertex, "order": order[vertex]},
        )

        for neighbour in graph.neighbours(vertex):
            if neighbour not in order:
                yield from walk(neighbour)
                low[vertex] = min(low[vertex], low[neighbour])
                yield Step(
                    "inherit",
                    f"{vertex!r} takes the low link {low[neighbour]} from {neighbour!r}, "
                    f"so its own is now {low[vertex]}.",
                    {"vertex": vertex, "from": neighbour, "low": low[vertex]},
                )
            elif neighbour in on_stack:
                low[vertex] = min(low[vertex], order[neighbour])
                yield Step(
                    "back-edge",
                    f"{vertex!r} points back at {neighbour!r}, which is still unassigned, "
                    f"so it can escape as far back as {low[vertex]}.",
                    {"vertex": vertex, "to": neighbour, "low": low[vertex]},
                )
            else:
                yield Step(
                    "ignore",
                    f"{neighbour!r} is already in a finished component, so the edge to it "
                    "is ignored. Taking a low link from it would merge two separate "
                    "components.",
                    {"vertex": vertex, "to": neighbour},
                )

        if low[vertex] == order[vertex]:
            component: list[Vertex] = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == vertex:
                    break

            components.append(component)
            yield Step(
                "component",
                f"{vertex!r} cannot escape past itself, so it is the root of a component: "
                f"{component!r}.",
                {"root": vertex, "component": component},
            )

    for vertex in graph.vertices:
        if vertex not in order:
            yield from walk(vertex)

    return components


def kosaraju_scc(graph: Graph) -> list[list[Vertex]]:
    return run(kosaraju_scc_traced(graph))


def kosaraju_scc_traced(graph: Graph) -> Traced[list[list[Vertex]]]:
    """Kosaraju's algorithm: two passes, and much easier to believe. O(V + E).

    1. Depth first search the graph, recording each vertex when it **finishes**.
    2. Reverse every edge.
    3. Depth first search the reversed graph, taking vertices in reverse finish
       order. Each traversal is exactly one component.

    Why it works, which is the appeal of this one: in the reversed graph, a
    traversal from a vertex reaches everything that could reach it in the original.
    Combining "what I can reach" with "what can reach me" gives exactly the mutual
    reachability that defines a component. The finish order is what guarantees you
    start each traversal in a component that has no unexplored components pointing
    into it, so the traversal cannot leak out.

    Tarjan is one pass and faster in practice; Kosaraju is two passes and much
    easier to convince yourself of. That is a fair trade to be able to make, and
    the tests check the two agree on hundreds of random graphs.
    """
    if not graph.directed:
        raise ValueError("strongly connected components are a directed graph question")

    finished: list[Vertex] = []
    seen: set[Vertex] = set()

    def first_pass(vertex: Vertex) -> None:
        seen.add(vertex)
        for neighbour in graph.neighbours(vertex):
            if neighbour not in seen:
                first_pass(neighbour)
        finished.append(vertex)

    for vertex in graph.vertices:
        if vertex not in seen:
            first_pass(vertex)

    yield Step(
        "finish-order",
        f"First pass done. Finish order: {finished!r}. The last to finish is in a "
        "component nothing else points into.",
        {"order": list(finished)},
    )

    flipped = graph.reversed()
    yield Step(
        "reverse",
        "Reversed every edge. A traversal here reaches everything that could reach the "
        "start in the original graph.",
        {},
    )

    assigned: set[Vertex] = set()
    components: list[list[Vertex]] = []

    for vertex in reversed(finished):
        if vertex in assigned:
            continue

        component: list[Vertex] = []
        waiting = ArrayStack([vertex])

        while not waiting.is_empty():
            current = waiting.pop()
            if current in assigned:
                continue
            assigned.add(current)
            component.append(current)
            for neighbour in flipped.neighbours(current):
                if neighbour not in assigned:
                    waiting.push(neighbour)

        components.append(component)
        yield Step(
            "component",
            f"Traversing the reversed graph from {vertex!r} found exactly one component: "
            f"{component!r}.",
            {"component": component},
        )

    return components


def condensation(graph: Graph) -> Graph:
    """Collapse each component to a single vertex, giving an acyclic graph.

    This is the practical payoff. Whatever cycles the original had, the result is
    always a directed acyclic graph, which means it can be topologically sorted.
    Any problem that needs an order but has cyclic dependencies is solved this way:
    condense, sort, then handle each component as a unit.

    Each new vertex is the frozenset of the original vertices in that component,
    which is hashable and prints readably.
    """
    components = tarjan_scc(graph)
    home: dict[Vertex, frozenset] = {}

    for component in components:
        label = frozenset(component)
        for member in component:
            home[member] = label

    condensed = Graph(directed=True)
    for component in components:
        condensed.add_vertex(frozenset(component))

    for edge in graph.edges():
        source, target = home[edge.source], home[edge.target]
        if source != target:
            condensed.add_edge(source, target)

    return condensed
