"""Tests for graphs and every graph algorithm.

Three recurring patterns here:

* Algorithms that solve the same problem two ways (Tarjan and Kosaraju, Prim and
  Kruskal, Dijkstra and Bellman Ford) are checked against each other on hundreds of
  random graphs.
* Where an algorithm has a documented failure mode, there is a test that
  **demonstrates the failure** rather than a comment claiming it: Dijkstra on
  negative weights, A star with an inadmissible heuristic.
* Results with more than one valid answer, such as a topological order, are checked
  by verifying the defining property rather than comparing against one list.
"""

import random

import pytest

from dsalab.algorithms.graph_traversal import (
    breadth_first,
    breadth_first_traced,
    connected_components,
    depth_first,
    depth_first_recursive,
    has_cycle,
    is_bipartite,
    shortest_unweighted_path,
    topological_sort,
    topological_sort_depth_first,
)
from dsalab.algorithms.mst import kruskal, kruskal_traced, prim, prim_traced, total_weight
from dsalab.algorithms.scc import condensation, kosaraju_scc, tarjan_scc
from dsalab.algorithms.shortest_path import (
    a_star,
    bellman_ford,
    dijkstra,
    dijkstra_path,
    dijkstra_traced,
    floyd_warshall,
    grid_distance,
    has_negative_cycle,
    manhattan_distance,
    no_estimate,
)
from dsalab.invariants import verify
from dsalab.structures.graph import AdjacencyMatrix, Graph
from dsalab.tracing import count_kinds, record


def random_graph(vertices: int, density: float, rng: random.Random,
                 directed: bool = False, weighted: bool = False) -> Graph:
    graph = Graph(directed=directed)
    for vertex in range(vertices):
        graph.add_vertex(vertex)

    for a in range(vertices):
        for b in range(vertices):
            if a == b:
                continue
            if not directed and b < a:
                continue
            if rng.random() < density:
                weight = rng.randint(1, 20) if weighted else 1.0
                graph.add_edge(a, b, weight)

    return graph


class TestGraphStructure:
    def test_building_an_undirected_graph(self):
        graph = Graph()
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")

        assert graph.vertex_count == 3
        assert graph.edge_count == 2
        assert sorted(graph.neighbours("b")) == ["a", "c"]
        verify(graph)

    def test_an_undirected_edge_is_stored_in_both_directions(self):
        graph = Graph()
        graph.add_edge("a", "b", 5)

        assert graph.has_edge("a", "b") and graph.has_edge("b", "a")
        assert graph.weight("b", "a") == 5

    def test_a_directed_edge_goes_one_way_only(self):
        graph = Graph(directed=True)
        graph.add_edge("a", "b")

        assert graph.has_edge("a", "b")
        assert not graph.has_edge("b", "a")
        assert graph.edge_count == 1

    def test_the_edge_count_of_an_undirected_graph_is_not_doubled(self):
        graph = Graph()
        graph.add_edges([("a", "b"), ("b", "c"), ("c", "a")])

        assert graph.edge_count == 3
        assert len(graph.edges()) == 3

    def test_an_isolated_vertex_is_kept(self):
        # It matters: an isolated vertex is its own component, and a traversal
        # that only finds vertices through edges would miss it entirely.
        graph = Graph()
        graph.add_vertex("lonely")
        graph.add_edge("a", "b")

        assert graph.vertex_count == 3
        assert connected_components(graph) == [["a", "b"], ["lonely"]] or len(
            connected_components(graph)
        ) == 2

    def test_adding_the_same_edge_twice_replaces_the_weight(self):
        graph = Graph()
        graph.add_edge("a", "b", 1)
        graph.add_edge("a", "b", 9)

        assert graph.edge_count == 1
        assert graph.weight("a", "b") == 9

    def test_removing_edges_and_vertices(self):
        graph = Graph()
        graph.add_edges([("a", "b"), ("b", "c"), ("c", "a")])

        assert graph.remove_edge("a", "b") is True
        assert graph.remove_edge("a", "b") is False
        assert graph.edge_count == 2

        assert graph.remove_vertex("c") is True
        assert graph.vertex_count == 2
        assert graph.edge_count == 0
        verify(graph)

    def test_a_missing_edge_weighs_infinity_rather_than_raising(self):
        # Infinity is what makes the shortest path arithmetic work without special
        # cases: a route through a missing edge automatically loses.
        graph = Graph()
        graph.add_edge("a", "b")

        assert graph.weight("a", "z") == float("inf")
        assert graph.weight("z", "a") == float("inf")

    def test_asking_for_the_neighbours_of_a_missing_vertex_raises(self):
        with pytest.raises(KeyError):
            Graph().neighbours("nobody")

    def test_reversing_a_directed_graph(self):
        graph = Graph(directed=True)
        graph.add_edges([("a", "b", 2), ("b", "c", 3)])
        flipped = graph.reversed()

        assert flipped.has_edge("b", "a") and flipped.has_edge("c", "b")
        assert not flipped.has_edge("a", "b")
        assert flipped.weight("b", "a") == 2

    def test_a_broken_undirected_graph_is_caught(self):
        graph = Graph()
        graph.add_edge("a", "b")
        del graph._neighbours["b"]["a"]  # break one direction by hand

        assert any("both directions" in violation.rule
                   for violation in graph.check_invariants())

    def test_density_reports_how_full_the_graph_is(self):
        complete = Graph()
        for a in range(5):
            for b in range(a + 1, 5):
                complete.add_edge(a, b)

        assert complete.density == 1.0
        assert Graph().density == 0.0


class TestAdjacencyMatrix:
    def test_it_holds_the_same_graph(self):
        graph = Graph()
        graph.add_edges([("a", "b", 3), ("b", "c", 4)])
        matrix = graph.to_matrix()

        assert matrix.weight("a", "b") == 3
        assert matrix.weight("b", "a") == 3
        assert matrix.weight("a", "c") == float("inf")
        assert sorted(matrix.neighbours("b")) == ["a", "c"]

    def test_a_missing_edge_is_infinity_and_not_zero(self):
        # Zero is a perfectly good weight, so using it to mean "no edge" would
        # make a free road indistinguishable from a missing one.
        matrix = AdjacencyMatrix(["a", "b"])
        matrix.add_edge("a", "b", 0)

        assert matrix.weight("a", "b") == 0
        assert matrix.has_edge("a", "b"), "a zero weight edge still exists"

    def test_the_distance_from_a_vertex_to_itself_starts_at_zero(self):
        matrix = AdjacencyMatrix(["a", "b"])

        assert matrix.weight("a", "a") == 0

    def test_it_round_trips_back_to_an_adjacency_list(self):
        graph = Graph(directed=True)
        graph.add_edges([("a", "b", 2), ("b", "c", 5), ("c", "a", 1)])
        back = graph.to_matrix().to_graph()

        assert sorted(map(str, back.edges())) == sorted(map(str, graph.edges()))

    def test_the_memory_difference_is_the_whole_argument(self):
        # A sparse graph: 500 vertices, 499 edges. The list holds about a thousand
        # entries, the matrix holds a quarter of a million cells.
        graph = Graph()
        for vertex in range(499):
            graph.add_edge(vertex, vertex + 1)

        assert graph.edge_count == 499
        assert graph.to_matrix().slots() == 500 * 500
        assert graph.density < 0.01


class TestTraversal:
    def build_sample(self) -> Graph:
        #     a
        #    / \
        #   b   c
        #  / \   \
        # d   e   f
        graph = Graph()
        graph.add_edges([("a", "b"), ("a", "c"), ("b", "d"), ("b", "e"), ("c", "f")])
        return graph

    def test_breadth_first_visits_in_rings(self):
        order = breadth_first(self.build_sample(), "a")

        assert order[0] == "a"
        assert set(order[1:3]) == {"b", "c"}, "everything one step away comes next"
        assert set(order[3:]) == {"d", "e", "f"}

    def test_depth_first_follows_a_path_to_its_end(self):
        order = depth_first(self.build_sample(), "a")

        assert order[0] == "a"
        assert order[:3] == ["a", "b", "d"], "it goes deep before it goes wide"

    def test_the_iterative_and_recursive_depth_first_searches_agree(self):
        rng = random.Random(20260902)
        for _ in range(100):
            graph = random_graph(rng.randint(1, 15), 0.3, rng)
            start = graph.vertices[0]

            assert depth_first(graph, start) == depth_first_recursive(graph, start)

    def test_both_traversals_visit_exactly_the_reachable_vertices(self):
        rng = random.Random(20260902)
        for _ in range(50):
            graph = random_graph(rng.randint(2, 20), 0.2, rng)
            start = graph.vertices[0]

            assert set(breadth_first(graph, start)) == set(depth_first(graph, start))

    def test_a_vertex_is_marked_when_queued_not_when_visited(self):
        # Marking on visit still gives the right answer but queues vertices many
        # times over, turning a linear traversal quadratic. The step count is the
        # only way to see it.
        graph = Graph()
        for vertex in range(1, 20):
            graph.add_edge(0, vertex)  # a star, so every vertex has a shared neighbour

        _, steps = record(breadth_first_traced(graph, 0))

        assert count_kinds(steps)["queue"] == 19, "each vertex should be queued once"

    def test_traversing_from_a_missing_vertex_raises(self):
        with pytest.raises(KeyError):
            breadth_first(Graph(), "nobody")

    def test_connected_components(self):
        graph = Graph()
        graph.add_edges([("a", "b"), ("c", "d"), ("d", "e")])
        graph.add_vertex("alone")

        groups = sorted(sorted(group) for group in connected_components(graph))
        assert groups == [["a", "b"], ["alone"], ["c", "d", "e"]]

    def test_components_of_a_directed_graph_are_refused(self):
        with pytest.raises(ValueError, match="undirected"):
            connected_components(Graph(directed=True))

    def test_breadth_first_finds_the_shortest_unweighted_path(self):
        graph = Graph()
        graph.add_edges([("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")])

        path = shortest_unweighted_path(graph, "a", "d")
        assert path == ["a", "d"], "the two edge route must lose to the one edge route"

    def test_it_reports_when_there_is_no_path(self):
        graph = Graph()
        graph.add_edge("a", "b")
        graph.add_edge("c", "d")

        assert shortest_unweighted_path(graph, "a", "d") is None

    def test_the_path_from_a_vertex_to_itself(self):
        graph = Graph()
        graph.add_edge("a", "b")

        assert shortest_unweighted_path(graph, "a", "a") == ["a"]

    def test_it_agrees_with_dijkstra_when_every_edge_costs_the_same(self):
        rng = random.Random(20260902)
        for _ in range(50):
            graph = random_graph(rng.randint(3, 15), 0.3, rng)
            start, goal = graph.vertices[0], graph.vertices[-1]

            path = shortest_unweighted_path(graph, start, goal)
            distance, _ = dijkstra(graph, start)

            if path is None:
                assert distance[goal] == float("inf")
            else:
                assert len(path) - 1 == distance[goal]


class TestCycleDetection:
    def test_a_directed_cycle_is_found(self):
        graph = Graph(directed=True)
        graph.add_edges([("a", "b"), ("b", "c"), ("c", "a")])

        assert has_cycle(graph) is True

    def test_a_directed_acyclic_graph_has_none(self):
        graph = Graph(directed=True)
        graph.add_edges([("a", "b"), ("b", "c"), ("a", "c")])

        assert has_cycle(graph) is False

    def test_a_diamond_is_not_a_cycle(self):
        # The classic wrong implementation uses only visited and unvisited, and
        # reports a cycle here because c is reached twice. Three states are needed.
        graph = Graph(directed=True)
        graph.add_edges([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])

        assert has_cycle(graph) is False

    def test_an_undirected_cycle_is_found(self):
        graph = Graph()
        graph.add_edges([("a", "b"), ("b", "c"), ("c", "a")])

        assert has_cycle(graph) is True

    def test_an_undirected_tree_has_no_cycle(self):
        # Every edge is walked in both directions, so the search always sees the
        # edge it arrived by. Ignoring the immediate parent is what stops that
        # being reported as a cycle.
        graph = Graph()
        graph.add_edges([("a", "b"), ("b", "c"), ("b", "d")])

        assert has_cycle(graph) is False

    def test_a_single_edge_is_not_an_undirected_cycle(self):
        graph = Graph()
        graph.add_edge("a", "b")

        assert has_cycle(graph) is False

    def test_a_self_loop_is_a_cycle(self):
        graph = Graph(directed=True)
        graph.add_edge("a", "a")

        assert has_cycle(graph) is True

    def test_a_tree_of_n_vertices_and_n_minus_one_edges_never_has_a_cycle(self):
        rng = random.Random(20260902)
        for _ in range(50):
            count = rng.randint(2, 30)
            graph = Graph()
            for vertex in range(1, count):
                graph.add_edge(vertex, rng.randrange(vertex))  # always attach to an earlier one

            assert graph.edge_count == count - 1
            assert has_cycle(graph) is False


class TestTopologicalSort:
    def is_valid_order(self, graph: Graph, order: list) -> bool:
        position = {vertex: index for index, vertex in enumerate(order)}
        return all(
            position[edge.source] < position[edge.target] for edge in graph.edges()
        )

    def test_it_orders_dependencies_before_dependents(self):
        graph = Graph(directed=True)
        graph.add_edges([("shirt", "tie"), ("tie", "jacket"), ("trousers", "shoes")])

        order = topological_sort(graph)
        assert self.is_valid_order(graph, order)
        assert len(order) == graph.vertex_count

    def test_any_valid_order_is_accepted_because_the_answer_is_not_unique(self):
        # Independent tasks can go either way round, so the test verifies the
        # property rather than comparing against one expected list.
        rng = random.Random(20260902)
        for _ in range(50):
            count = rng.randint(2, 20)
            graph = Graph(directed=True)
            for vertex in range(count):
                graph.add_vertex(vertex)
            for a in range(count):
                for b in range(a + 1, count):
                    if rng.random() < 0.2:
                        graph.add_edge(a, b)  # always forwards, so never cyclic

            assert self.is_valid_order(graph, topological_sort(graph))

    def test_the_two_methods_both_produce_valid_orders(self):
        rng = random.Random(20260902)
        for _ in range(50):
            count = rng.randint(2, 15)
            graph = Graph(directed=True)
            for vertex in range(count):
                graph.add_vertex(vertex)
            for a in range(count):
                for b in range(a + 1, count):
                    if rng.random() < 0.25:
                        graph.add_edge(a, b)

            assert self.is_valid_order(graph, topological_sort(graph))
            assert self.is_valid_order(graph, topological_sort_depth_first(graph))

    def test_a_cycle_is_reported_rather_than_producing_nonsense(self):
        graph = Graph(directed=True)
        graph.add_edges([("a", "b"), ("b", "c"), ("c", "a")])

        with pytest.raises(ValueError, match="cycle"):
            topological_sort(graph)
        with pytest.raises(ValueError, match="cycle"):
            topological_sort_depth_first(graph)

    def test_an_undirected_graph_is_refused(self):
        with pytest.raises(ValueError, match="directed"):
            topological_sort(Graph())

    def test_isolated_vertices_still_appear(self):
        graph = Graph(directed=True)
        graph.add_edge("a", "b")
        graph.add_vertex("alone")

        assert set(topological_sort(graph)) == {"a", "b", "alone"}


class TestBipartite:
    def test_an_even_cycle_is_bipartite(self):
        graph = Graph()
        graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 0)])

        ok, colouring = is_bipartite(graph)
        assert ok
        assert all(colouring[edge.source] != colouring[edge.target] for edge in graph.edges())

    def test_an_odd_cycle_is_not(self):
        # The characterisation: bipartite exactly when there is no odd cycle.
        graph = Graph()
        graph.add_edges([(0, 1), (1, 2), (2, 0)])

        assert is_bipartite(graph)[0] is False

    def test_a_tree_is_always_bipartite(self):
        graph = Graph()
        graph.add_edges([("a", "b"), ("b", "c"), ("b", "d"), ("d", "e")])

        assert is_bipartite(graph)[0] is True

    def test_a_disconnected_graph_is_handled_component_by_component(self):
        graph = Graph()
        graph.add_edges([(0, 1), (2, 3), (3, 4), (4, 2)])  # second part is an odd cycle

        assert is_bipartite(graph)[0] is False


class TestStronglyConnectedComponents:
    def normalise(self, components):
        return sorted(sorted(map(str, component)) for component in components)

    def test_a_simple_example(self):
        graph = Graph(directed=True)
        graph.add_edges([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("d", "e"), ("e", "d")])

        assert self.normalise(tarjan_scc(graph)) == [["a", "b", "c"], ["d", "e"]]

    def test_a_graph_with_no_cycles_has_one_component_per_vertex(self):
        graph = Graph(directed=True)
        graph.add_edges([("a", "b"), ("b", "c")])

        assert len(tarjan_scc(graph)) == 3

    def test_a_full_cycle_is_one_component(self):
        graph = Graph(directed=True)
        graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 0)])

        assert len(tarjan_scc(graph)) == 1

    def test_the_two_algorithms_agree_on_random_graphs(self):
        rng = random.Random(20260902)
        for _ in range(200):
            graph = random_graph(rng.randint(1, 12), rng.choice([0.1, 0.25, 0.5]), rng,
                                 directed=True)

            assert self.normalise(tarjan_scc(graph)) == self.normalise(kosaraju_scc(graph))

    def test_every_vertex_lands_in_exactly_one_component(self):
        rng = random.Random(20260902)
        for _ in range(50):
            graph = random_graph(rng.randint(1, 15), 0.2, rng, directed=True)
            components = tarjan_scc(graph)

            members = [vertex for component in components for vertex in component]
            assert sorted(members) == sorted(graph.vertices)
            assert len(members) == len(set(members))

    def test_members_of_a_component_really_can_all_reach_each_other(self):
        from dsalab.algorithms.graph_traversal import reachable_from

        rng = random.Random(20260902)
        for _ in range(30):
            graph = random_graph(rng.randint(2, 10), 0.35, rng, directed=True)

            for component in tarjan_scc(graph):
                for a in component:
                    reachable = reachable_from(graph, a)
                    for b in component:
                        assert b in reachable, f"{a!r} cannot reach {b!r} in its own component"

    def test_an_undirected_graph_is_refused(self):
        with pytest.raises(ValueError, match="directed"):
            tarjan_scc(Graph())

    def test_the_condensation_is_always_acyclic(self):
        # The practical payoff: collapsing components turns any directed graph
        # into one that can be topologically sorted.
        rng = random.Random(20260902)
        for _ in range(50):
            graph = random_graph(rng.randint(2, 12), 0.3, rng, directed=True)
            condensed = condensation(graph)

            assert has_cycle(condensed) is False
            topological_sort(condensed)  # must not raise


class TestMinimumSpanningTree:
    def test_a_worked_example(self):
        graph = Graph()
        graph.add_edges([
            ("a", "b", 1), ("b", "c", 2), ("a", "c", 4), ("c", "d", 3), ("b", "d", 5),
        ])

        assert total_weight(kruskal(graph)) == 6
        assert total_weight(prim(graph)) == 6

    def test_the_tree_has_exactly_one_fewer_edge_than_there_are_vertices(self):
        rng = random.Random(20260902)
        for _ in range(50):
            count = rng.randint(2, 20)
            graph = Graph()
            for vertex in range(1, count):
                graph.add_edge(vertex, rng.randrange(vertex), rng.randint(1, 50))

            assert len(kruskal(graph)) == count - 1
            assert len(prim(graph)) == count - 1

    def test_prim_and_kruskal_always_agree_on_the_total(self):
        # They can pick different edges when weights tie, but the total is unique.
        rng = random.Random(20260902)
        for _ in range(200):
            count = rng.randint(2, 15)
            graph = Graph()
            for vertex in range(1, count):
                graph.add_edge(vertex, rng.randrange(vertex), rng.randint(1, 30))
            for _ in range(count):
                a, b = rng.randrange(count), rng.randrange(count)
                if a != b:
                    graph.add_edge(a, b, rng.randint(1, 30))

            assert total_weight(prim(graph)) == total_weight(kruskal(graph))

    def test_the_tree_connects_everything(self):
        rng = random.Random(20260902)
        for _ in range(30):
            count = rng.randint(2, 15)
            graph = Graph()
            for vertex in range(1, count):
                graph.add_edge(vertex, rng.randrange(vertex), rng.randint(1, 30))

            tree = Graph()
            for edge in kruskal(graph):
                tree.add_edge(edge.source, edge.target, edge.weight)

            assert len(connected_components(tree)) == 1
            assert has_cycle(tree) is False

    def test_kruskal_skips_edges_that_would_close_a_cycle(self):
        # A fourth vertex is needed to see the skip. With only a triangle the
        # tree is complete after two edges and the loop stops before ever
        # considering the third, which is a real optimisation and not a bug.
        graph = Graph()
        graph.add_edges([("a", "b", 1), ("b", "c", 2), ("a", "c", 3), ("c", "d", 10)])
        _, steps = record(kruskal_traced(graph))

        assert count_kinds(steps)["skip"] == 1, "the edge closing the triangle is skipped"
        assert count_kinds(steps)["add"] == 3

    def test_kruskal_stops_as_soon_as_the_tree_is_complete(self):
        graph = Graph()
        graph.add_edges([("a", "b", 1), ("b", "c", 2), ("a", "c", 3)])
        _, steps = record(kruskal_traced(graph))

        assert any(step.kind == "done" for step in steps)
        assert count_kinds(steps).get("skip", 0) == 0, (
            "once two edges connect three vertices the remaining edges are never examined"
        )

    def test_prim_updates_an_offer_rather_than_duplicating_it(self):
        graph = Graph()
        graph.add_edges([("a", "b", 10), ("a", "c", 1), ("c", "b", 2)])
        _, steps = record(prim_traced(graph, "a"))
        offers = [step for step in steps if step.kind == "offer" and step.data["vertex"] == "b"]

        assert len(offers) == 2, "b is offered twice, at 10 and then at the better 2"
        assert offers[-1].data["weight"] == 2

    def test_a_disconnected_graph_gives_kruskal_a_forest(self):
        graph = Graph()
        graph.add_edges([("a", "b", 1), ("c", "d", 2)])
        forest = kruskal(graph)

        assert len(forest) == 2
        assert total_weight(forest) == 3

    def test_a_single_vertex_and_an_empty_graph(self):
        lonely = Graph()
        lonely.add_vertex("a")

        assert kruskal(lonely) == []
        assert prim(lonely) == []
        assert kruskal(Graph()) == []
        assert prim(Graph()) == []

    def test_a_directed_graph_is_refused(self):
        with pytest.raises(ValueError, match="undirected"):
            kruskal(Graph(directed=True))
        with pytest.raises(ValueError, match="undirected"):
            prim(Graph(directed=True))


class TestShortestPaths:
    def build_weighted(self) -> Graph:
        graph = Graph(directed=True)
        graph.add_edges([
            ("a", "b", 4), ("a", "c", 2), ("c", "b", 1), ("b", "d", 5),
            ("c", "d", 8), ("d", "e", 6), ("c", "e", 10),
        ])
        return graph

    def test_dijkstra_on_a_worked_example(self):
        distance, _ = dijkstra(self.build_weighted(), "a")

        assert distance["a"] == 0
        assert distance["c"] == 2
        assert distance["b"] == 3, "going a to c to b costs 3, better than the direct 4"
        assert distance["d"] == 8
        assert distance["e"] == 12

    def test_it_rebuilds_the_actual_path(self):
        path, cost = dijkstra_path(self.build_weighted(), "a", "b")

        assert path == ["a", "c", "b"]
        assert cost == 3

    def test_an_unreachable_vertex_stays_at_infinity(self):
        graph = Graph(directed=True)
        graph.add_edge("a", "b", 1)
        graph.add_vertex("island")

        distance, _ = dijkstra(graph, "a")
        assert distance["island"] == float("inf")
        assert dijkstra_path(graph, "a", "island") == ([], float("inf"))

    def test_dijkstra_agrees_with_bellman_ford_when_weights_are_positive(self):
        rng = random.Random(20260902)
        for _ in range(100):
            graph = random_graph(rng.randint(2, 12), 0.3, rng, directed=True, weighted=True)
            start = graph.vertices[0]

            assert dijkstra(graph, start)[0] == bellman_ford(graph, start)[0]

    def test_dijkstra_agrees_with_floyd_warshall(self):
        rng = random.Random(20260902)
        for _ in range(50):
            graph = random_graph(rng.randint(2, 10), 0.35, rng, directed=True, weighted=True)
            all_pairs = floyd_warshall(graph)

            for start in graph.vertices:
                assert dijkstra(graph, start)[0] == all_pairs[start]

    def test_dijkstra_gives_a_wrong_answer_with_a_negative_edge(self):
        # Demonstrating the failure rather than warning about it. Dijkstra settles
        # b at 2 and moves on, never learning that going through c costs 1.
        graph = Graph(directed=True)
        graph.add_edges([("a", "b", 2), ("a", "c", 5), ("c", "b", -4)])

        distance, _ = dijkstra(graph, "a")
        correct, _ = bellman_ford(graph, "a")

        assert distance["b"] == 2, "Dijkstra settles b early and never revisits it"
        assert correct["b"] == 1, "the real shortest path goes a, c, b for 1"
        assert distance["b"] != correct["b"]

    def test_dijkstra_warns_when_it_meets_a_negative_edge(self):
        graph = Graph(directed=True)
        graph.add_edges([("a", "b", 2), ("b", "c", -1)])
        _, steps = record(dijkstra_traced(graph, "a"))

        assert count_kinds(steps).get("warning", 0) >= 1

    def test_bellman_ford_handles_negative_edges_correctly(self):
        graph = Graph(directed=True)
        graph.add_edges([("a", "b", 4), ("a", "c", 5), ("c", "b", -3), ("b", "d", 2)])

        distance, _ = bellman_ford(graph, "a")
        assert distance["b"] == 2
        assert distance["d"] == 4

    def test_bellman_ford_detects_a_negative_cycle(self):
        graph = Graph(directed=True)
        graph.add_edges([("a", "b", 1), ("b", "c", -3), ("c", "a", 1)])

        assert has_negative_cycle(graph, "a") is True
        with pytest.raises(ValueError, match="negative cycle"):
            bellman_ford(graph, "a")

    def test_a_positive_cycle_is_not_a_problem(self):
        graph = Graph(directed=True)
        graph.add_edges([("a", "b", 1), ("b", "c", 1), ("c", "a", 1)])

        assert has_negative_cycle(graph, "a") is False

    def test_bellman_ford_stops_early_once_nothing_changes(self):
        from dsalab.algorithms.shortest_path import bellman_ford_traced

        graph = Graph(directed=True)
        for vertex in range(20):
            graph.add_edge(vertex, vertex + 1, 1)

        _, steps = record(bellman_ford_traced(graph, 0))
        assert any(step.kind == "settled" for step in steps)
        assert count_kinds(steps)["pass"] < 20

    def test_floyd_warshall_finds_every_pair(self):
        graph = Graph(directed=True)
        graph.add_edges([("a", "b", 3), ("b", "c", 4), ("a", "c", 10)])
        distance = floyd_warshall(graph)

        assert distance["a"]["c"] == 7, "going through b beats the direct edge"
        assert distance["a"]["a"] == 0
        assert distance["c"]["a"] == float("inf")

    def test_floyd_warshall_detects_a_negative_cycle(self):
        graph = Graph(directed=True)
        graph.add_edges([("a", "b", 1), ("b", "a", -3)])

        with pytest.raises(ValueError, match="negative cycle"):
            floyd_warshall(graph)


class TestAStar:
    def build_grid(self, width: int, height: int, blocked: set = frozenset()) -> Graph:
        graph = Graph()
        for x in range(width):
            for y in range(height):
                if (x, y) in blocked:
                    continue
                graph.add_vertex((x, y))
                for dx, dy in ((1, 0), (0, 1)):
                    other = (x + dx, y + dy)
                    if other[0] < width and other[1] < height and other not in blocked:
                        graph.add_edge((x, y), other, 1)
        return graph

    def test_it_finds_the_shortest_path_on_a_grid(self):
        grid = self.build_grid(6, 6)
        path, cost = a_star(grid, (0, 0), (5, 5), manhattan_distance)

        assert cost == 10, "ten steps on a six by six grid with no diagonals"
        assert path[0] == (0, 0) and path[-1] == (5, 5)

    def test_it_agrees_with_dijkstra_on_the_cost(self):
        rng = random.Random(20260902)
        for _ in range(20):
            blocked = {(rng.randrange(1, 7), rng.randrange(1, 7)) for _ in range(6)}
            blocked.discard((0, 0))
            blocked.discard((7, 7))
            grid = self.build_grid(8, 8, blocked)

            if (0, 0) not in grid or (7, 7) not in grid:
                continue

            _, cost = a_star(grid, (0, 0), (7, 7), manhattan_distance)
            distance, _ = dijkstra(grid, (0, 0))

            assert cost == distance[(7, 7)]

    def test_with_no_estimate_it_becomes_dijkstra(self):
        grid = self.build_grid(6, 6)

        _, guided = a_star(grid, (0, 0), (5, 5), manhattan_distance)
        _, blind = a_star(grid, (0, 0), (5, 5), no_estimate)

        assert guided == blind, "the answer must be the same either way"

    def test_the_estimate_makes_it_expand_fewer_vertices(self):
        from dsalab.algorithms.shortest_path import a_star_traced

        grid = self.build_grid(12, 12)

        _, guided_steps = record(a_star_traced(grid, (0, 0), (11, 11), manhattan_distance))
        _, blind_steps = record(a_star_traced(grid, (0, 0), (11, 11), no_estimate))

        guided = [s for s in guided_steps if s.kind == "found"][0].data["expanded"]
        blind = [s for s in blind_steps if s.kind == "found"][0].data["expanded"]

        assert guided < blind, (
            f"a star expanded {guided} vertices and Dijkstra expanded {blind}; the "
            "estimate should be doing some work"
        )

    def test_an_inadmissible_heuristic_gives_a_wrong_answer(self):
        # The rule that is easy to break by accident: the estimate must never
        # overestimate the true remaining distance.
        #
        # This graph makes the failure unmissable. The route start, middle, goal
        # costs 2, and the direct edge costs 5. A heuristic that claims middle is
        # 10 away from the goal, when it is really 1 away, pushes that route to
        # the back of the queue, so the direct edge is taken and a star returns 5
        # while insisting it is optimal.
        graph = Graph(directed=True)
        graph.add_edges([("start", "middle", 1), ("middle", "goal", 1), ("start", "goal", 5)])

        honest_guesses = {"start": 2, "middle": 1, "goal": 0}
        lying_guesses = {"start": 0, "middle": 10, "goal": 0}

        _, honest = a_star(graph, "start", "goal", lambda a, b: honest_guesses[a])
        _, cheating = a_star(graph, "start", "goal", lambda a, b: lying_guesses[a])

        assert honest == 2, "an admissible estimate still finds the best route"
        assert cheating == 5, (
            "the overestimate hid the better route, and nothing reported a problem, "
            "which is exactly why admissibility is a rule and not a suggestion"
        )

    def test_straight_line_distance_is_also_admissible(self):
        grid = self.build_grid(6, 6)

        _, with_line = a_star(grid, (0, 0), (5, 5), grid_distance)
        _, with_manhattan = a_star(grid, (0, 0), (5, 5), manhattan_distance)

        assert with_line == with_manhattan

    def test_an_unreachable_goal(self):
        graph = Graph()
        graph.add_edge((0, 0), (0, 1), 1)
        graph.add_vertex((9, 9))

        path, cost = a_star(graph, (0, 0), (9, 9), manhattan_distance)
        assert path == [] and cost == float("inf")
