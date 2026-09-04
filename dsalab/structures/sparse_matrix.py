"""Sparse matrices: store only what is there.

The special matrices in `special_matrix.py` all assume the zeros sit in a known
pattern. A sparse matrix makes no such assumption. The non zero values can be
anywhere, they are just rare, so instead of a formula we keep an explicit list of
(row, column, value) triples.

This is worth doing when the matrix is genuinely mostly empty. The break even
point is easy to work out: the dense form needs n^2 numbers, the sparse form
needs roughly three numbers per non zero entry. So sparse wins once fewer than
about a third of the cells are filled. Below that it costs more memory than the
dense form and is slower to read, which is a real and often forgotten downside.

The layout used here is coordinate list form, kept sorted by row and then column.
Sorting is what makes addition efficient: two sorted lists can be merged in one
pass, exactly like the merge step of merge sort, rather than searching one list
for every element of the other.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from dsalab.tracing import Step, Traced, run


@dataclass(frozen=True)
class Entry:
    """One non zero cell of the matrix."""

    row: int
    column: int
    value: float


class SparseMatrix:
    """A matrix that stores only its non zero entries.

    | Operation | Cost | Why |
    | - | - | - |
    | read a cell | O(log k) | Binary search over k stored entries |
    | write a non zero cell | O(k) | Insertion has to keep the list sorted |
    | add two matrices | O(k1 + k2) | One merge pass, no searching |
    | multiply two matrices | O(k1 times row width) | Each entry meets a row |
    | memory | O(k) | Independent of how big the matrix claims to be |

    where k is the number of non zero entries.
    """

    def __init__(self, rows: int, columns: int, entries: Iterable[Entry] | None = None) -> None:
        if rows < 1 or columns < 1:
            raise ValueError("a matrix needs at least one row and one column")
        self.rows = rows
        self.columns = columns
        self._entries: list[Entry] = []
        for entry in entries or ():
            self[entry.row, entry.column] = entry.value

    # Reading

    def _find(self, row: int, column: int) -> int:
        """Binary search for the slot where (row, column) is or should be.

        Returns the index of the entry if present, otherwise the index where it
        would need to be inserted to keep the list sorted. Having one function
        answer both questions is what keeps the write path simple.
        """
        low, high = 0, len(self._entries)
        while low < high:
            middle = (low + high) // 2
            entry = self._entries[middle]
            if (entry.row, entry.column) < (row, column):
                low = middle + 1
            else:
                high = middle
        return low

    def __getitem__(self, position: tuple[int, int]) -> float:
        row, column = position
        self._check(row, column)
        index = self._find(row, column)
        if index < len(self._entries):
            found = self._entries[index]
            if found.row == row and found.column == column:
                return found.value
        return 0

    def __setitem__(self, position: tuple[int, int], value: float) -> None:
        row, column = position
        self._check(row, column)
        index = self._find(row, column)
        present = (
            index < len(self._entries)
            and self._entries[index].row == row
            and self._entries[index].column == column
        )

        if value == 0:
            # Storing an explicit zero would defeat the entire point, so a write
            # of zero removes the entry instead of recording it.
            if present:
                self._entries.pop(index)
            return

        if present:
            self._entries[index] = Entry(row, column, value)
        else:
            self._entries.insert(index, Entry(row, column, value))

    def _check(self, row: int, column: int) -> None:
        if not (0 <= row < self.rows and 0 <= column < self.columns):
            raise IndexError(
                f"({row}, {column}) is outside a {self.rows} by {self.columns} matrix"
            )

    def __iter__(self) -> Iterator[Entry]:
        return iter(self._entries)

    def __len__(self) -> int:
        """How many non zero entries are stored."""
        return len(self._entries)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SparseMatrix):
            return NotImplemented
        return (
            self.rows == other.rows
            and self.columns == other.columns
            and self._entries == other._entries
        )

    def __repr__(self) -> str:
        return (
            f"SparseMatrix({self.rows}x{self.columns}, "
            f"{len(self._entries)} non zero of {self.rows * self.columns})"
        )

    @property
    def density(self) -> float:
        """The fraction of cells that are non zero, from 0 to 1.

        Below about 0.33 the sparse form saves memory. Above it, the dense form
        is both smaller and faster, and this property is how you check rather
        than assume.
        """
        return len(self._entries) / (self.rows * self.columns)

    # Building and converting

    @classmethod
    def from_dense(cls, grid: list[list[float]]) -> SparseMatrix:
        """Build from an ordinary grid, keeping only the non zero cells."""
        if not grid or not grid[0]:
            raise ValueError("cannot build a matrix from an empty grid")
        width = len(grid[0])
        if any(len(row) != width for row in grid):
            raise ValueError("every row of the grid must be the same length")

        matrix = cls(len(grid), width)
        for row_index, row in enumerate(grid):
            for column_index, value in enumerate(row):
                if value != 0:
                    matrix[row_index, column_index] = value
        return matrix

    def to_dense(self) -> list[list[float]]:
        """Rebuild the full grid. Useful for tests and for small displays."""
        grid = [[0.0 for _ in range(self.columns)] for _ in range(self.rows)]
        for entry in self._entries:
            grid[entry.row][entry.column] = entry.value
        return grid

    def transpose(self) -> SparseMatrix:
        """Swap rows and columns. O(k log k), because the order has to be redone."""
        flipped = SparseMatrix(self.columns, self.rows)
        for entry in self._entries:
            flipped[entry.column, entry.row] = entry.value
        return flipped

    # Arithmetic

    def add(self, other: SparseMatrix) -> SparseMatrix:
        return run(self.add_traced(other))

    def add_traced(self, other: SparseMatrix) -> Traced[SparseMatrix]:
        """Add two sparse matrices in a single merge pass. O(k1 + k2).

        Because both lists are sorted by (row, column), the algorithm can walk
        them together with two pointers, exactly like merging two sorted halves
        in merge sort. The naive alternative, looping over one matrix and
        searching the other for each entry, would be O(k1 log k2) and would need
        a second pass to catch the entries only present in the second matrix.
        """
        if (self.rows, self.columns) != (other.rows, other.columns):
            raise ValueError(
                f"cannot add a {self.rows}x{self.columns} matrix to a "
                f"{other.rows}x{other.columns} one"
            )

        result = SparseMatrix(self.rows, self.columns)
        left = right = 0

        while left < len(self._entries) and right < len(other._entries):
            a, b = self._entries[left], other._entries[right]
            here, there = (a.row, a.column), (b.row, b.column)

            if here == there:
                total = a.value + b.value
                yield Step(
                    "add",
                    f"Both matrices have a value at ({a.row}, {a.column}): "
                    f"{a.value} + {b.value} = {total}.",
                    {"row": a.row, "column": a.column, "value": total},
                )
                # A sum of exactly zero must not be stored, or the matrix slowly
                # fills up with explicit zeros and stops being sparse.
                if total != 0:
                    result._entries.append(Entry(a.row, a.column, total))
                left += 1
                right += 1
            elif here < there:
                yield Step(
                    "copy",
                    f"Only the first matrix has a value at ({a.row}, {a.column}), "
                    "so it carries through unchanged.",
                    {"row": a.row, "column": a.column, "value": a.value},
                )
                result._entries.append(a)
                left += 1
            else:
                yield Step(
                    "copy",
                    f"Only the second matrix has a value at ({b.row}, {b.column}), "
                    "so it carries through unchanged.",
                    {"row": b.row, "column": b.column, "value": b.value},
                )
                result._entries.append(b)
                right += 1

        # Whatever is left in either list has no counterpart, so it copies
        # across. These still report a step: a tail that copied silently would
        # make the animation stop partway through the run, which is exactly the
        # kind of quiet gap the tests are here to catch.
        for entry in self._entries[left:] + other._entries[right:]:
            yield Step(
                "copy",
                f"Nothing left to merge against, so ({entry.row}, {entry.column}) "
                "carries through unchanged.",
                {"row": entry.row, "column": entry.column, "value": entry.value},
            )
            result._entries.append(entry)

        return result

    def multiply(self, other: SparseMatrix) -> SparseMatrix:
        """Matrix multiplication that only touches non zero entries.

        The dense algorithm is three nested loops and costs O(n^3) no matter how
        empty the matrices are. This version groups the second matrix by row,
        then for each non zero entry (i, k) of the first matrix walks only the
        non zero entries of row k of the second. Zeros contribute nothing to a
        sum of products, so skipping them changes nothing about the answer.

        The cost becomes proportional to the number of multiplications that
        actually matter, which for a sparse pair is dramatically less than n^3.
        """
        if self.columns != other.rows:
            raise ValueError(
                f"cannot multiply a {self.rows}x{self.columns} matrix by a "
                f"{other.rows}x{other.columns} one, the inner sizes must match"
            )

        rows_of_other: dict[int, list[Entry]] = {}
        for entry in other._entries:
            rows_of_other.setdefault(entry.row, []).append(entry)

        totals: dict[tuple[int, int], float] = {}
        for entry in self._entries:
            for partner in rows_of_other.get(entry.column, ()):
                key = (entry.row, partner.column)
                totals[key] = totals.get(key, 0) + entry.value * partner.value

        result = SparseMatrix(self.rows, other.columns)
        for (row, column), value in sorted(totals.items()):
            if value != 0:
                result._entries.append(Entry(row, column, value))
        return result
