"""Graphs: the structure that models relationships rather than containment.

Every structure so far has been about holding values. A graph is about the
connections between them: roads between cities, links between pages, dependencies
between tasks, friendships, pipes, circuits, states in a machine.

Two representations, and choosing between them is the first real decision:

**Adjacency list.** Each vertex keeps a list of its neighbours. Memory is
O(V + E), and listing a vertex's neighbours is instant. Checking whether one
specific edge exists means scanning that vertex's list.

**Adjacency matrix.** A V by V grid where cell (a, b) holds the weight of the edge
from a to b. Checking a specific edge is O(1), but memory is O(V^2) whether or not
the edges exist, and listing a vertex's neighbours means scanning a whole row.

The deciding number is **density**. A graph with V vertices can have up to V^2
edges. Real graphs are usually sparse: a road network has maybe four roads per
junction, not four thousand. For a million junctions, the list needs a few million
entries and the matrix needs a trillion cells. That is not a trade off, it is the
difference between working and not working.

The matrix wins when the graph is genuinely dense, when you check specific edges
constantly, or when you want to do linear algebra on it, which is how algorithms
like Floyd Warshall and PageRank are expressed.

Both are implemented here because several algorithms later are naturally written
against one or the other, and because the memory difference is worth measuring
rather than describing.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from dsalab.invariants import Violation

Vertex = Hashable


@dataclass(frozen=True)
class Edge:
    """One connection, with a weight. Unweighted graphs simply use weight 1."""

    source: Vertex
    target: Vertex
    weight: float = 1.0

    def __repr__(self) -> str:
        arrow = f"{self.source!r} -> {self.target!r}"
        return arrow if self.weight == 1.0 else f"{arrow} ({self.weight})"


class Graph:
    """A graph as an adjacency list, directed or undirected, weighted or not.

    | Operation | Cost |
    | - | - |
    | add a vertex or an edge | O(1) |
    | list a vertex's neighbours | O(degree) |
    | check one specific edge | O(degree) |
    | remove an edge | O(degree) |
    | memory | O(V + E) |

    The `directed` flag is set once at construction and never changes, because
    almost every algorithm behaves differently on the two and a graph that could
    switch would be a permanent source of confusion. An undirected graph stores
    each edge in both directions, which is the honest representation: an undirected
    edge really is two directed ones that must stay in step.
    """

    def __init__(self, directed: bool = False) -> None:
        self.directed = directed
        self._neighbours: dict[Vertex, dict[Vertex, float]] = {}

    # Building

    def add_vertex(self, vertex: Vertex) -> None:
        """Add an isolated vertex. Does nothing if it is already there.

        Worth having separately from `add_edge`, because a vertex with no edges is
        a real thing that matters: it is its own connected component, and a
        traversal that only discovers vertices through edges would never see it.
        """
        self._neighbours.setdefault(vertex, {})

    def add_edge(self, source: Vertex, target: Vertex, weight: float = 1.0) -> None:
        """Add an edge, creating either endpoint if it is new.

        Adding an edge that already exists replaces its weight rather than
        duplicating it, which makes the structure a simple graph. Multigraphs, with
        several edges between the same pair, need a different representation and
        are not what any algorithm here expects.
        """
        self.add_vertex(source)
        self.add_vertex(target)
        self._neighbours[source][target] = weight
        if not self.directed:
            self._neighbours[target][source] = weight

    def add_edges(self, edges: Iterable[tuple]) -> None:
        """Add many edges, each a (source, target) or (source, target, weight)."""
        for edge in edges:
            if len(edge) == 2:
                self.add_edge(edge[0], edge[1])
            else:
                self.add_edge(edge[0], edge[1], edge[2])

    def remove_edge(self, source: Vertex, target: Vertex) -> bool:
        """Remove an edge. Returns whether it was there."""
        if source not in self._neighbours or target not in self._neighbours[source]:
            return False
        del self._neighbours[source][target]
        if not self.directed:
            del self._neighbours[target][source]
        return True

    def remove_vertex(self, vertex: Vertex) -> bool:
        """Remove a vertex and every edge touching it.

        In a directed graph this needs a scan of every other vertex, because an
        adjacency list records outgoing edges and there is no way to find the
        incoming ones without looking. That is O(V + E) and is the one operation
        the adjacency list is genuinely bad at.
        """
        if vertex not in self._neighbours:
            return False

        del self._neighbours[vertex]
        for others in self._neighbours.values():
            others.pop(vertex, None)
        return True

    # Reading

    @property
    def vertices(self) -> list[Vertex]:
        return list(self._neighbours)

    @property
    def vertex_count(self) -> int:
        return len(self._neighbours)

    @property
    def edge_count(self) -> int:
        """How many edges there are.

        In an undirected graph each edge is stored twice, so the total is halved.
        Forgetting that is a classic source of doubled edge counts and of
        algorithms that report the wrong density.
        """
        total = sum(len(targets) for targets in self._neighbours.values())
        return total if self.directed else total // 2

    def neighbours(self, vertex: Vertex) -> list[Vertex]:
        """Everything reachable in one step from `vertex`, in insertion order.

        Insertion order rather than sorted, because it is what a real adjacency
        list gives you and because the traversals should not depend on a
        convenient ordering. Tests that need determinism sort it themselves.
        """
        if vertex not in self._neighbours:
            raise KeyError(f"{vertex!r} is not in the graph")
        return list(self._neighbours[vertex])

    def weight(self, source: Vertex, target: Vertex) -> float:
        """The weight of one edge, or infinity when there is no such edge.

        Infinity rather than an error, because it is the value that makes the
        shortest path arithmetic work without special cases: a route through a
        missing edge automatically costs infinity and loses to every real route.
        """
        if source not in self._neighbours:
            return float("inf")
        return self._neighbours[source].get(target, float("inf"))

    def has_edge(self, source: Vertex, target: Vertex) -> bool:
        return source in self._neighbours and target in self._neighbours[source]

    def has_vertex(self, vertex: Vertex) -> bool:
        return vertex in self._neighbours

    def degree(self, vertex: Vertex) -> int:
        """How many edges leave this vertex. For undirected graphs, how many touch it."""
        return len(self._neighbours[vertex])

    def in_degree(self, vertex: Vertex) -> int:
        """How many edges arrive at this vertex.

        O(V + E), because an adjacency list only records outgoing edges. Kahn's
        topological sort needs these, so it computes them all in one pass rather
        than calling this per vertex, which would be quadratic.
        """
        return sum(1 for targets in self._neighbours.values() if vertex in targets)

    def edges(self) -> list[Edge]:
        """Every edge. In an undirected graph, each one once rather than twice."""
        collected: list[Edge] = []
        seen: set[tuple] = set()

        for source, targets in self._neighbours.items():
            for target, weight in targets.items():
                if not self.directed:
                    key = tuple(sorted((repr(source), repr(target))))
                    if key in seen:
                        continue
                    seen.add(key)
                collected.append(Edge(source, target, weight))

        return collected

    def reversed(self) -> Graph:
        """The same graph with every edge turned around.

        Needed by Kosaraju's algorithm for strongly connected components, and it is
        also how you find the vertices that can reach a given one rather than those
        it can reach.
        """
        flipped = Graph(directed=self.directed)
        for vertex in self._neighbours:
            flipped.add_vertex(vertex)
        for edge in self.edges():
            flipped.add_edge(edge.target, edge.source, edge.weight)
        return flipped

    @property
    def density(self) -> float:
        """Edges divided by the most edges this many vertices could have.

        Near 0 means sparse and an adjacency list is right. Near 1 means dense and
        a matrix starts to make sense. Most real graphs are far closer to 0 than
        people expect.
        """
        v = self.vertex_count
        if v < 2:
            return 0.0
        possible = v * (v - 1) if self.directed else v * (v - 1) / 2
        return self.edge_count / possible

    def to_matrix(self) -> AdjacencyMatrix:
        """Convert to the matrix form, which some algorithms are written against."""
        matrix = AdjacencyMatrix(self.vertices, directed=self.directed)
        for edge in self.edges():
            matrix.add_edge(edge.source, edge.target, edge.weight)
        return matrix

    def __contains__(self, vertex: Vertex) -> bool:
        return vertex in self._neighbours

    def __iter__(self) -> Iterator[Vertex]:
        return iter(self._neighbours)

    def __len__(self) -> int:
        return len(self._neighbours)

    def __repr__(self) -> str:
        kind = "directed" if self.directed else "undirected"
        return f"Graph({kind}, {self.vertex_count} vertices, {self.edge_count} edges)"

    def check_invariants(self) -> list[Violation]:
        violations: list[Violation] = []

        for source, targets in self._neighbours.items():
            for target, weight in targets.items():
                if target not in self._neighbours:
                    violations.append(Violation(
                        "every edge points at a vertex that exists",
                        f"{source!r} has an edge to {target!r}, which is not in the graph",
                    ))
                elif not self.directed:
                    back = self._neighbours[target].get(source)
                    if back is None:
                        violations.append(Violation(
                            "an undirected edge is recorded in both directions",
                            f"{source!r} to {target!r} exists but the reverse does not",
                        ))
                    elif back != weight:
                        violations.append(Violation(
                            "an undirected edge has the same weight in both directions",
                            f"{source!r} to {target!r} weighs {weight} but the reverse "
                            f"weighs {back}",
                        ))

        return violations


class AdjacencyMatrix:
    """A graph as a grid of weights, with no edge marked as infinity.

    Infinity for a missing edge rather than zero is deliberate and important. Zero
    is a perfectly reasonable **weight**, so using it to mean "no edge" makes a
    free road indistinguishable from a missing one. Infinity also makes the
    shortest path arithmetic work with no special cases, which is why Floyd
    Warshall on day 18 is four lines.

    | Operation | Cost |
    | - | - |
    | check or set one edge | O(1) |
    | list a vertex's neighbours | O(V) |
    | memory | O(V^2), whether the edges exist or not |
    """

    def __init__(self, vertices: Iterable[Vertex], directed: bool = False) -> None:
        self.directed = directed
        self._vertices = list(vertices)
        self._index = {vertex: position for position, vertex in enumerate(self._vertices)}

        size = len(self._vertices)
        self._weights = [[float("inf")] * size for _ in range(size)]
        for position in range(size):
            # The distance from a vertex to itself is zero, not infinity, and
            # Floyd Warshall depends on that being right from the start.
            self._weights[position][position] = 0.0

    @property
    def vertices(self) -> list[Vertex]:
        return list(self._vertices)

    @property
    def vertex_count(self) -> int:
        return len(self._vertices)

    def add_edge(self, source: Vertex, target: Vertex, weight: float = 1.0) -> None:
        a, b = self._index[source], self._index[target]
        self._weights[a][b] = weight
        if not self.directed:
            self._weights[b][a] = weight

    def weight(self, source: Vertex, target: Vertex) -> float:
        return self._weights[self._index[source]][self._index[target]]

    def has_edge(self, source: Vertex, target: Vertex) -> bool:
        a, b = self._index[source], self._index[target]
        return a != b and self._weights[a][b] != float("inf")

    def neighbours(self, vertex: Vertex) -> list[Vertex]:
        """O(V), because it has to scan the whole row even for a vertex with one edge."""
        position = self._index[vertex]
        return [
            self._vertices[other]
            for other in range(len(self._vertices))
            if other != position and self._weights[position][other] != float("inf")
        ]

    def rows(self) -> list[list[float]]:
        """The raw grid, for algorithms written in terms of the matrix itself."""
        return [list(row) for row in self._weights]

    def slots(self) -> int:
        """How many cells are stored, which is the memory cost. Always V^2."""
        return len(self._vertices) ** 2

    def to_graph(self) -> Graph:
        graph = Graph(directed=self.directed)
        for vertex in self._vertices:
            graph.add_vertex(vertex)
        for source in self._vertices:
            for target in self._vertices:
                if self.has_edge(source, target):
                    graph.add_edge(source, target, self.weight(source, target))
        return graph

    def __repr__(self) -> str:
        kind = "directed" if self.directed else "undirected"
        return f"AdjacencyMatrix({kind}, {self.vertex_count} vertices, {self.slots()} cells)"
