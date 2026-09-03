"""Backtracking: try something, and undo it when it leads nowhere.

Backtracking is depth first search over the space of partial answers. Build an
answer one choice at a time; when a choice cannot possibly lead anywhere, undo it
and try the next one.

The undoing is the whole technique, and it is why every function here follows the
same shape:

    place a piece
    if it is still legal, recurse
    remove the piece      <- the backtrack

That last line is what people forget, and forgetting it does not raise an error. It
leaves rubbish behind that makes later branches look illegal, so the search quietly
returns too few answers, or none.

What separates backtracking from brute force is **pruning**: abandoning a branch
the moment it cannot work rather than completing it and checking at the end. For N
queens, checking legality as each queen is placed cuts the search from 4 billion
arrangements to a few thousand for an 8x8 board. The complexity is still
exponential in the worst case, but the constant is the difference between finishing
in a millisecond and not finishing.
"""

from __future__ import annotations

from collections.abc import Iterator

from dsalab.tracing import Step, Traced, run


def n_queens(size: int, first_only: bool = False) -> list[list[int]]:
    return run(n_queens_traced(size, first_only))


def n_queens_traced(size: int, first_only: bool = False) -> Traced[list[list[int]]]:
    """Place `size` queens on a board so that none attacks another.

    A solution is a list where position i holds the column of the queen in row i.
    That representation already prunes enormously: by giving each row exactly one
    queen, it makes row conflicts impossible without ever checking for them, so the
    search space drops from "choose n squares from n^2" to n^n, before any other
    pruning.

    **Choosing the representation is itself the first pruning step**, and it is a
    more powerful one than anything in the loop below.

    The remaining checks are columns and the two diagonals. Two queens share a
    diagonal exactly when the difference of their rows equals the difference of
    their columns, which is the one piece of arithmetic in the problem.

    Counts of solutions, as a sanity check: 1 for a 1x1 board, none for 2x2 or 3x3,
    2 for 4x4, 10 for 5x5, 4 for 6x6, 40 for 7x7 and 92 for 8x8. Those numbers are
    in the tests, because they are a much stronger check than eyeballing one board.
    """
    if size < 0:
        raise ValueError("a board cannot have a negative size")

    solutions: list[list[int]] = []
    columns: list[int] = []

    def safe(row: int, column: int) -> bool:
        for earlier_row, earlier_column in enumerate(columns):
            if earlier_column == column:
                return False
            if abs(earlier_row - row) == abs(earlier_column - column):
                return False
        return True

    def place(row: int) -> Traced[bool]:
        if row == size:
            solutions.append(list(columns))
            yield Step(
                "solution",
                f"All {size} queens are placed with no conflicts: {columns!r}.",
                {"solution": list(columns), "found": len(solutions)},
            )
            return first_only

        for column in range(size):
            if not safe(row, column):
                yield Step(
                    "reject",
                    f"A queen at row {row}, column {column} is attacked, so this whole "
                    "branch is abandoned without exploring it.",
                    {"row": row, "column": column},
                )
                continue

            columns.append(column)
            yield Step(
                "place",
                f"Placed a queen at row {row}, column {column}. {size - row - 1} left.",
                {"row": row, "column": column, "board": list(columns)},
            )

            if (yield from place(row + 1)):
                return True

            columns.pop()
            yield Step(
                "backtrack",
                f"Nothing works below row {row} with a queen at column {column}, so it "
                "is removed and the next column is tried. Undoing the move is the whole "
                "technique.",
                {"row": row, "column": column},
            )

        return False

    yield from place(0)
    return solutions


def solve_sudoku(board: list[list[int]]) -> list[list[int]] | None:
    return run(solve_sudoku_traced(board))


def solve_sudoku_traced(board: list[list[int]]) -> Traced[list[list[int]] | None]:
    """Fill a 9x9 grid so every row, column and box holds 1 to 9. Zero means empty.

    Backtracking again, with one addition worth knowing: instead of filling cells in
    order, it always fills the cell with the **fewest legal options**. That is
    called the most constrained variable heuristic, and it is enormously effective.

    The reasoning: a cell with one legal value is a forced move, so making it
    immediately costs nothing and shrinks everything else. A cell with nine options
    branches nine ways, and if the branch is doomed you find out nine times over.
    Doing the forced moves first also makes contradictions surface early, which is
    exactly what pruning wants.

    On hard puzzles this is the difference between milliseconds and minutes, and it
    generalises far beyond sudoku: the same idea drives real constraint solvers.
    """
    grid = [row[:] for row in board]

    if len(grid) != 9 or any(len(row) != 9 for row in grid):
        raise ValueError("a sudoku board must be 9 by 9")

    if not _givens_are_consistent(grid):
        # Checking the given numbers before starting is not a nicety, it is what
        # keeps this terminating in reasonable time. Backtracking only ever asks
        # whether a *new* placement is legal, so a board that already breaks the
        # rules is never noticed directly: the search fills in around the
        # contradiction and only fails once it reaches a cell the conflict
        # actually starves, which on a nearly empty board can be an enormous
        # amount of work first.
        #
        # I found this by writing a test with two ones in the same row and
        # watching it run for minutes. The right answer is that a puzzle
        # contradicting itself has no solution and can be rejected immediately.
        yield Step(
            "invalid",
            "The numbers already on the board break the rules, so no solution can "
            "exist and there is nothing to search.",
            {},
        )
        return None

    def legal(row: int, column: int, value: int) -> bool:
        for index in range(9):
            if grid[row][index] == value or grid[index][column] == value:
                return False

        box_row, box_column = 3 * (row // 3), 3 * (column // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_column, box_column + 3):
                if grid[r][c] == value:
                    return False
        return True

    def options(row: int, column: int) -> list[int]:
        return [value for value in range(1, 10) if legal(row, column, value)]

    def most_constrained() -> tuple[int, int, list[int]] | None:
        """The empty cell with the fewest choices, or None when the grid is full."""
        best = None
        for row in range(9):
            for column in range(9):
                if grid[row][column] != 0:
                    continue
                choices = options(row, column)
                if best is None or len(choices) < len(best[2]):
                    best = (row, column, choices)
                    if not choices:
                        return best  # a dead end, so stop looking
        return best

    def fill() -> Traced[bool]:
        target = most_constrained()
        if target is None:
            yield Step("solved", "Every cell is filled and every rule holds.", {})
            return True

        row, column, choices = target

        if not choices:
            yield Step(
                "dead-end",
                f"The cell at row {row}, column {column} has no legal value at all, so "
                "something earlier was wrong and this branch is abandoned.",
                {"row": row, "column": column},
            )
            return False

        yield Step(
            "choose-cell",
            f"The cell at row {row}, column {column} has only {len(choices)} legal "
            f"value(s), the fewest on the board, so it is filled first. Forced moves "
            "cost nothing and shrink everything else.",
            {"row": row, "column": column, "options": choices},
        )

        for value in choices:
            grid[row][column] = value
            yield Step("place", f"Tried {value} at row {row}, column {column}.",
                       {"row": row, "column": column, "value": value})

            if (yield from fill()):
                return True

            grid[row][column] = 0
            yield Step("backtrack", f"{value} at row {row}, column {column} led nowhere, "
                                    "so it is taken back out.",
                       {"row": row, "column": column, "value": value})

        return False

    solved = yield from fill()
    return grid if solved else None


def _givens_are_consistent(grid: list[list[int]]) -> bool:
    """Whether the numbers already on the board obey the rules.

    Only checks the filled cells against each other. An empty board is trivially
    consistent, and a consistent board may still be unsolvable, which is what the
    search is for.
    """
    def no_repeats(values: list[int]) -> bool:
        filled = [value for value in values if value != 0]
        return len(filled) == len(set(filled))

    for index in range(9):
        if not no_repeats(grid[index]):
            return False
        if not no_repeats([grid[row][index] for row in range(9)]):
            return False

    for box_row in (0, 3, 6):
        for box_column in (0, 3, 6):
            box = [grid[r][c] for r in range(box_row, box_row + 3)
                   for c in range(box_column, box_column + 3)]
            if not no_repeats(box):
                return False

    return True


def permutations(items: list) -> list[list]:
    return run(permutations_traced(items))


def permutations_traced(items: list) -> Traced[list[list]]:
    """Every ordering of the items. O(n!) results, which is unavoidable.

    Included as the smallest possible backtracking example, where the structure is
    visible with nothing else in the way: choose an unused item, recurse, put it
    back.

    The `used` list is what makes the undo explicit. Swapping elements in place is
    shorter and is the usual implementation, but it hides the backtrack inside the
    swap, and hiding the step this whole module is about seemed like the wrong
    choice here.
    """
    results: list[list] = []
    current: list = []
    used = [False] * len(items)

    def build() -> Traced[None]:
        if len(current) == len(items):
            results.append(list(current))
            yield Step("complete", f"Finished an ordering: {current!r}.",
                       {"permutation": list(current), "found": len(results)})
            return

        for index, item in enumerate(items):
            if used[index]:
                continue

            used[index] = True
            current.append(item)
            yield Step("choose", f"Took {item!r}, building {current!r}.",
                       {"item": item, "partial": list(current)})

            yield from build()

            current.pop()
            used[index] = False
            yield Step("undo", f"Put {item!r} back so the other orderings can use it.",
                       {"item": item})

    yield from build()
    return results


def subsets(items: list) -> list[list]:
    """Every subset. 2^n results, one binary choice per item: in or out.

    Where permutations ask "in what order", subsets ask "which ones", and the two
    shapes cover most of what backtracking is used for.
    """
    results: list[list] = []
    current: list = []

    def build(index: int) -> None:
        if index == len(items):
            results.append(list(current))
            return

        build(index + 1)          # leave this item out

        current.append(items[index])
        build(index + 1)          # take it
        current.pop()             # and undo

    build(0)
    return results


def graph_colouring(graph, colours: int) -> dict | None:
    return run(graph_colouring_traced(graph, colours))


def graph_colouring_traced(graph, colours: int) -> Traced[dict | None]:
    """Colour the vertices so no edge joins two of the same colour, with k colours.

    NP complete for three or more colours, which is worth putting next to day 18's
    bipartite check: **two colours is a simple traversal, three is intractable.**
    One extra colour moves the problem from easy to one with no known efficient
    solution, and there is no intuitive reason it should.

    The real uses are all scheduling in disguise: exam timetabling where clashing
    exams must differ, register allocation in a compiler, and radio frequency
    assignment.
    """
    vertices = graph.vertices
    assignment: dict = {}

    def colour(index: int) -> Traced[bool]:
        if index == len(vertices):
            yield Step("solved", f"Every vertex is coloured using {colours} colour(s).",
                       {"assignment": dict(assignment)})
            return True

        vertex = vertices[index]
        for choice in range(colours):
            clash = any(
                assignment.get(neighbour) == choice for neighbour in graph.neighbours(vertex)
            )
            if clash:
                yield Step(
                    "reject",
                    f"Colour {choice} clashes with a neighbour of {vertex!r}.",
                    {"vertex": vertex, "colour": choice},
                )
                continue

            assignment[vertex] = choice
            yield Step("assign", f"Gave {vertex!r} colour {choice}.",
                       {"vertex": vertex, "colour": choice})

            if (yield from colour(index + 1)):
                return True

            del assignment[vertex]
            yield Step("backtrack", f"Nothing works with {vertex!r} in colour {choice}.",
                       {"vertex": vertex, "colour": choice})

        return False

    found = yield from colour(0)
    return dict(assignment) if found else None


def hamiltonian_cycle(graph) -> list | None:
    """A cycle visiting every vertex exactly once, or None.

    Worth putting directly beside the Euler path, which visits every **edge** once.
    Finding an Euler path is easy: one exists exactly when the graph is connected
    and has zero or two vertices of odd degree, which is a check you can do in
    linear time.

    Finding a Hamiltonian cycle is **NP complete**. The two questions sound like
    mirror images of each other and their difficulty is nothing alike, which is one
    of the better illustrations that difficulty is not obvious from a problem
    statement.
    """
    vertices = graph.vertices
    if not vertices:
        return None

    start = vertices[0]
    path = [start]
    visited = {start}

    def extend() -> bool:
        if len(path) == len(vertices):
            return graph.has_edge(path[-1], start)

        for neighbour in graph.neighbours(path[-1]):
            if neighbour in visited:
                continue
            path.append(neighbour)
            visited.add(neighbour)

            if extend():
                return True

            path.pop()
            visited.discard(neighbour)

        return False

    return path + [start] if extend() else None


def rat_in_a_maze(maze: list[list[int]]) -> list[tuple[int, int]] | None:
    """A path from the top left to the bottom right of a grid, where 1 is passable.

    The course's introduction to backtracking, and a good one, because the undo step
    is physically obvious: step into a cell, and if the way ahead is blocked, step
    back out.
    """
    if not maze or not maze[0]:
        return None

    rows, columns = len(maze), len(maze[0])
    path: list[tuple[int, int]] = []
    visited: set[tuple[int, int]] = set()

    def walk(row: int, column: int) -> bool:
        if not (0 <= row < rows and 0 <= column < columns):
            return False
        if maze[row][column] == 0 or (row, column) in visited:
            return False

        path.append((row, column))
        visited.add((row, column))

        if (row, column) == (rows - 1, columns - 1):
            return True

        for step_row, step_column in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            if walk(row + step_row, column + step_column):
                return True

        path.pop()  # the backtrack
        return False

    return path if walk(0, 0) else None
