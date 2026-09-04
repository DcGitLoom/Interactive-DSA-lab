"""Graph traversal, and the four things that fall out of it.

Breadth first and depth first search are the same algorithm with one thing
changed: what holds the vertices waiting to be visited.

* A **queue** gives breadth first. The oldest waiting vertex is taken, so
  everything one step away is visited before anything two steps away. The search
  spreads outwards in rings.
* A **stack** gives depth first. The newest waiting vertex is taken, so the search
  follows one path as far as it goes before backing up.

That single swap is the whole difference, and it is worth seeing the two functions
side by side because they are otherwise identical. From it follows:

* Breadth first finds the **shortest path in an unweighted graph**, because it
  reaches every vertex by the fewest possible edges. Depth first does not, and no
  amount of care will make it.
* Depth first naturally reveals **structure**: cycles, topological order,
  articulation points, strongly connected components. Breadth first does not,
  because it never has a single path on the stack to reason about.

Everything else in this module is depth first search with bookkeeping added.
"""

from __future__ import annotations

from dsalab.structures.deque import Deque
from dsalab.structures.graph import Graph, Vertex
from dsalab.structures.stack import ArrayStack
from dsalab.tracing import Step, Traced, run


def breadth_first(graph: Graph, start: Vertex) -> list[Vertex]:
    return run(breadth_first_traced(graph, start))


def breadth_first_traced(graph: Graph, start: Vertex) -> Traced[list[Vertex]]:
    """Visit every reachable vertex, nearest first. O(V + E).

    The one detail that matters: a vertex is marked as seen **when it is queued**,
    not when it is visited. Marking on visit lets a vertex be queued several times
    by different neighbours before it is ever taken out, which in a dense graph
    turns a linear traversal into a quadratic one. It still produces the right
    answer, which is what makes it a nasty performance bug rather than an obvious
    error.
    """
    if start not in graph:
        raise KeyError(f"{start!r} is not in the graph")

    seen = {start}
    order: list[Vertex] = []
    waiting = Deque([start])

    while not waiting.is_empty():
        vertex = waiting.pop_front()
        order.append(vertex)
        yield Step(
            "visit",
            f"Visited {vertex!r}, taken from the front of the queue. "
            f"{len(waiting)} still waiting.",
            {"vertex": vertex, "waiting": len(waiting), "order": list(order)},
        )

        for neighbour in graph.neighbours(vertex):
            if neighbour not in seen:
                seen.add(neighbour)
                waiting.push_back(neighbour)
                yield Step(
                    "queue",
                    f"{neighbour!r} has not been seen, so it joins the back of the queue. "
                    "It is marked now rather than when it is visited, or it could be "
                    "queued several times over.",
                    {"vertex": neighbour, "from": vertex},
                )

    return order


def depth_first(graph: Graph, start: Vertex) -> list[Vertex]:
    return run(depth_first_traced(graph, start))


def depth_first_traced(graph: Graph, start: Vertex) -> Traced[list[Vertex]]:
    """Visit every reachable vertex, following each path to its end. O(V + E).

    Written with an explicit stack rather than recursion, so it can handle graphs
    deeper than Python's recursion limit, and so the comparison with the breadth
    first version above is exact: same code, `ArrayStack` instead of `Deque`.

    Neighbours are pushed in reverse so that the first neighbour is explored
    first, matching what the recursive version would do. Without that the two
    disagree on order while both being valid depth first searches, which makes
    them impossible to test against each other.
    """
    if start not in graph:
        raise KeyError(f"{start!r} is not in the graph")

    seen: set[Vertex] = set()
    order: list[Vertex] = []
    waiting = ArrayStack([start])

    while not waiting.is_empty():
        vertex = waiting.pop()
        if vertex in seen:
            # Unlike the queue version, a vertex can be stacked more than once
            # before being visited, so the check has to happen here too.
            continue

        seen.add(vertex)
        order.append(vertex)
        yield Step(
            "visit",
            f"Visited {vertex!r}, taken from the top of the stack, which is why the "
            "search goes deep rather than wide.",
            {"vertex": vertex, "waiting": len(waiting), "order": list(order)},
        )

        for neighbour in reversed(graph.neighbours(vertex)):
            if neighbour not in seen:
                waiting.push(neighbour)

    return order


def depth_first_recursive(graph: Graph, start: Vertex) -> list[Vertex]:
    """The recursive form, which is shorter and is what most people write.

    Kept alongside the iterative version because the tests check that the two
    agree, and because the recursive form is what the structural algorithms below
    are built on: they need work done on the way back up, which the call stack
    provides for free.
    """
    seen: set[Vertex] = set()
    order: list[Vertex] = []

    def walk(vertex: Vertex) -> None:
        seen.add(vertex)
        order.append(vertex)
        for neighbour in graph.neighbours(vertex):
            if neighbour not in seen:
                walk(neighbour)

    if start in graph:
        walk(start)
    return order


def connected_components(graph: Graph) -> list[list[Vertex]]:
    """Groups of vertices that can reach each other. O(V + E).

    Only meaningful for undirected graphs. In a directed graph the equivalent
    question is strongly connected components, which is a genuinely harder problem
    and is in `scc.py`.

    Traversal alone would only find the component containing the start vertex, so
    this restarts from every unvisited vertex. That restart loop is what turns a
    single traversal into a complete analysis, and it is the same shape as the
    outer loop in the topological sort and the SCC algorithms.
    """
    if graph.directed:
        raise ValueError(
            "connected components are for undirected graphs. For a directed graph "
            "you want strongly connected components, which is a different problem"
        )

    seen: set[Vertex] = set()
    groups: list[list[Vertex]] = []

    for vertex in graph.vertices:
        if vertex in seen:
            continue
        group = depth_first(graph, vertex)
        seen.update(group)
        groups.append(group)

    return groups


def shortest_unweighted_path(graph: Graph, start: Vertex, goal: Vertex) -> list[Vertex] | None:
    return run(shortest_unweighted_path_traced(graph, start, goal))


def shortest_unweighted_path_traced(
    graph: Graph, start: Vertex, goal: Vertex
) -> Traced[list[Vertex] | None]:
    """The path with the fewest edges, using breadth first search. O(V + E).

    Breadth first search visits vertices in order of distance from the start, so
    **the first time it reaches the goal, it has done so by the fewest edges**.
    That is the guarantee, and it is why no priority queue is needed here: when
    every edge costs the same, the order of discovery is already the order of
    distance. Dijkstra's algorithm is what you need the moment that stops being
    true.

    The path is rebuilt by recording, for each vertex, which vertex discovered it,
    then walking those links backwards from the goal. Storing one pointer per
    vertex is far cheaper than storing a whole path per vertex, and it is the same
    trick every path finding algorithm here uses.
    """
    if start not in graph or goal not in graph:
        raise KeyError("both the start and the goal must be in the graph")

    came_from: dict[Vertex, Vertex | None] = {start: None}
    waiting = Deque([start])

    while not waiting.is_empty():
        vertex = waiting.pop_front()

        if vertex == goal:
            path = []
            current: Vertex | None = goal
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            yield Step(
                "found",
                f"Reached {goal!r} after {len(path) - 1} edge(s). Because breadth first "
                "search arrives in order of distance, this is the shortest such path.",
                {"path": path, "length": len(path) - 1},
            )
            return path

        for neighbour in graph.neighbours(vertex):
            if neighbour not in came_from:
                came_from[neighbour] = vertex
                waiting.push_back(neighbour)
                yield Step(
                    "queue",
                    f"{neighbour!r} discovered from {vertex!r}, one step further out.",
                    {"vertex": neighbour, "from": vertex},
                )

    yield Step("unreachable", f"{goal!r} cannot be reached from {start!r}.", {})
    return None


def topological_sort(graph: Graph) -> list[Vertex]:
    return run(topological_sort_traced(graph))


def topological_sort_traced(graph: Graph) -> Traced[list[Vertex]]:
    """Order the vertices so every edge points forwards. O(V + E), by Kahn's method.

    This answers "in what order can these tasks be done, given that some depend on
    others". Build systems, package managers, spreadsheet recalculation and course
    prerequisites are all this problem.

    Kahn's method is a good example of an algorithm that is obvious once stated:
    repeatedly take a vertex with nothing left pointing at it, output it, and
    remove its outgoing edges, which may free up others.

    Two things worth noting:

    * It only works on a **directed acyclic graph**. A cycle means a set of tasks
      that all wait on each other, and no order exists. This is detected naturally:
      if the loop stops with vertices left over, every one of them is still waiting
      on something, which can only happen in a cycle. That makes Kahn's method a
      cycle detector as well as a sorter.
    * The answer is usually **not unique**. Independent tasks can go in either
      order, and any valid order is a correct answer, which is why the tests verify
      the ordering property rather than comparing against one expected list.
    """
    if not graph.directed:
        raise ValueError("only a directed graph can be topologically sorted")

    # Counting incoming edges in one pass, rather than calling in_degree per
    # vertex, which would be quadratic.
    waiting_on = {vertex: 0 for vertex in graph.vertices}
    for vertex in graph.vertices:
        for neighbour in graph.neighbours(vertex):
            waiting_on[neighbour] += 1

    ready = Deque([vertex for vertex in graph.vertices if waiting_on[vertex] == 0])
    order: list[Vertex] = []

    yield Step(
        "start",
        f"{len(ready)} vertex(es) have nothing pointing at them, so they can go first.",
        {"ready": list(ready)},
    )

    while not ready.is_empty():
        vertex = ready.pop_front()
        order.append(vertex)
        yield Step(
            "emit",
            f"{vertex!r} has no unmet dependencies left, so it goes next.",
            {"vertex": vertex, "order": list(order)},
        )

        for neighbour in graph.neighbours(vertex):
            waiting_on[neighbour] -= 1
            if waiting_on[neighbour] == 0:
                ready.push_back(neighbour)
                yield Step(
                    "freed",
                    f"{neighbour!r} was waiting only on {vertex!r}, so it is ready now.",
                    {"vertex": neighbour, "freed_by": vertex},
                )

    if len(order) != graph.vertex_count:
        stuck = [vertex for vertex in graph.vertices if vertex not in set(order)]
        yield Step(
            "cycle",
            f"{len(stuck)} vertex(es) are still waiting on something, which can only "
            "happen if they form a cycle. No valid order exists.",
            {"stuck": stuck},
        )
        raise ValueError(
            f"the graph has a cycle involving {stuck!r}, so no topological order exists"
        )

    return order


def topological_sort_depth_first(graph: Graph) -> list[Vertex]:
    """The same result from depth first search, for comparison.

    The insight: a vertex is finished only after everything it points at is
    finished. So recording each vertex when its recursion **returns**, and
    reversing the result at the end, gives a valid order.

    It is shorter than Kahn's method and uses no in degree counts, but it needs a
    separate cycle check, and it uses stack depth proportional to the longest path.
    Kahn's version detects cycles for free and uses no recursion, which is why it
    is the default here.

    This is day 2's point about recursion again: the work happens on the way back
    up, and that is the entire algorithm.
    """
    if not graph.directed:
        raise ValueError("only a directed graph can be topologically sorted")

    seen: set[Vertex] = set()
    order: list[Vertex] = []

    def walk(vertex: Vertex) -> None:
        seen.add(vertex)
        for neighbour in graph.neighbours(vertex):
            if neighbour not in seen:
                walk(neighbour)
        order.append(vertex)  # after the children, which is the whole point

    for vertex in graph.vertices:
        if vertex not in seen:
            walk(vertex)

    if has_cycle(graph):
        raise ValueError("the graph has a cycle, so no topological order exists")

    order.reverse()
    return order


def has_cycle(graph: Graph) -> bool:
    return run(has_cycle_traced(graph))


def has_cycle_traced(graph: Graph) -> Traced[bool]:
    """Whether the graph contains a cycle. O(V + E).

    The two cases are genuinely different problems, which is why the code branches
    on `directed` rather than pretending they are the same:

    **Directed.** A cycle exists when a depth first search finds an edge back to a
    vertex that is **still on the current path**, not merely one that has been
    visited before. Reaching an already finished vertex is completely normal, it
    just means two paths converge. So three states are needed, not two: unvisited,
    in progress, and finished. Using only visited and unvisited reports a cycle for
    any diamond shape, which is the classic wrong version.

    **Undirected.** Every edge can be walked in both directions, so the search
    always sees an edge back to where it just came from. That is not a cycle. The
    fix is to ignore the immediate parent, and then any other edge to a visited
    vertex is a real cycle.
    """
    in_progress: set[Vertex] = set()
    finished: set[Vertex] = set()

    def walk_directed(vertex: Vertex) -> Traced[bool]:
        in_progress.add(vertex)
        yield Step("enter", f"Started {vertex!r}. It is now on the current path.",
                   {"vertex": vertex, "path": list(in_progress)})

        for neighbour in graph.neighbours(vertex):
            if neighbour in in_progress:
                yield Step(
                    "cycle",
                    f"The edge from {vertex!r} back to {neighbour!r} points at something "
                    "still on the current path, which is a cycle.",
                    {"from": vertex, "to": neighbour},
                )
                return True
            if neighbour not in finished:
                if (yield from walk_directed(neighbour)):
                    return True

        in_progress.discard(vertex)
        finished.add(vertex)
        yield Step(
            "finish",
            f"Finished {vertex!r} and everything below it, so it leaves the current "
            "path. Later edges into it are convergence, not a cycle.",
            {"vertex": vertex},
        )
        return False

    def walk_undirected(vertex: Vertex, parent: Vertex | None) -> Traced[bool]:
        finished.add(vertex)

        for neighbour in graph.neighbours(vertex):
            if neighbour == parent:
                # The edge we arrived by, seen from the other side. Not a cycle.
                continue
            if neighbour in finished:
                yield Step(
                    "cycle",
                    f"{vertex!r} connects to {neighbour!r}, which was already visited by "
                    "another route, so there is a cycle.",
                    {"from": vertex, "to": neighbour},
                )
                return True
            if (yield from walk_undirected(neighbour, vertex)):
                return True

        return False

    for vertex in graph.vertices:
        if vertex in finished:
            continue
        found = (
            (yield from walk_directed(vertex))
            if graph.directed
            else (yield from walk_undirected(vertex, None))
        )
        if found:
            return True

    return False


def is_bipartite(graph: Graph) -> tuple[bool, dict[Vertex, int] | None]:
    """Whether the vertices can be split into two groups with no edge inside a group.

    Also called two colouring, and it is the one case of graph colouring that is
    easy: colouring with three colours is NP complete, but two is a simple
    traversal, colouring each vertex the opposite of its parent and failing if a
    neighbour already has the same colour.

    The characterisation is worth knowing: **a graph is bipartite exactly when it
    has no odd length cycle.** An even cycle alternates cleanly all the way round,
    an odd one always ends up with two neighbours the same colour.

    Real uses: matching two kinds of thing (students to projects, jobs to machines)
    and checking whether a set of conflicts can be split into two shifts.
    """
    colour: dict[Vertex, int] = {}

    for start in graph.vertices:
        if start in colour:
            continue

        colour[start] = 0
        waiting = Deque([start])

        while not waiting.is_empty():
            vertex = waiting.pop_front()
            for neighbour in graph.neighbours(vertex):
                if neighbour not in colour:
                    colour[neighbour] = 1 - colour[vertex]
                    waiting.push_back(neighbour)
                elif colour[neighbour] == colour[vertex]:
                    return False, None

    return True, colour


def reachable_from(graph: Graph, start: Vertex) -> set[Vertex]:
    """Everything reachable from a vertex, including itself."""
    return set(depth_first(graph, start))
