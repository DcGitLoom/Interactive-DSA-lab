"""Matrices that waste memory if you store them the obvious way.

An n by n matrix stored in full needs n^2 slots. But many matrices that turn up
in real problems are mostly zeros, and the zeros sit in a predictable pattern.
When the pattern is known in advance, you can store only the values that can be
non zero and compute where each one lives with arithmetic.

The trade off is always the same: less memory, but every read and write now costs
an index calculation, and the code is harder to follow. For a diagonal matrix the
saving is from n^2 down to n, which is enormous and obviously worth it. For a
matrix that is only slightly sparse the saving is small and the complication is
not worth paying for.

Every class here keeps its values in a `DynamicArray`, which is the structure
built earlier the same day, rather than a Python list. Reads and writes are all
O(1), because working out the position is arithmetic, not searching.
"""

from __future__ import annotations

from dsalab.structures.dynamic_array import DynamicArray


class _SquareMatrix:
    """Shared plumbing: size checking, bounds checking and printing."""

    def __init__(self, size: int, slots: int) -> None:
        if size < 1:
            raise ValueError("a matrix needs at least one row")
        self.size = size
        self._values = DynamicArray(0 for _ in range(slots))

    def _check(self, row: int, column: int) -> None:
        if not (0 <= row < self.size and 0 <= column < self.size):
            raise IndexError(f"({row}, {column}) is outside a {self.size} by {self.size} matrix")

    def to_dense(self) -> list[list[float]]:
        """Rebuild the full n by n grid, for tests and for display."""
        return [[self[row, column] for column in range(self.size)] for row in range(self.size)]

    @property
    def stored_values(self) -> int:
        """How many numbers are actually being kept in memory."""
        return len(self._values)

    def __repr__(self) -> str:
        name = type(self).__name__
        return f"{name}(size={self.size}, stored={self.stored_values} of {self.size ** 2})"

    def __getitem__(self, position: tuple[int, int]) -> float:
        raise NotImplementedError

    def __setitem__(self, position: tuple[int, int], value: float) -> None:
        raise NotImplementedError


class DiagonalMatrix(_SquareMatrix):
    """Only the main diagonal can be non zero. Everything else is zero by rule.

    Storage is n instead of n^2. For a 1000 by 1000 matrix that is 1000 numbers
    instead of a million, which is the difference between a few kilobytes and
    several megabytes.

    The position of element (i, i) is simply i, so there is barely any arithmetic.
    """

    def __init__(self, size: int) -> None:
        super().__init__(size, size)

    def __getitem__(self, position: tuple[int, int]) -> float:
        row, column = position
        self._check(row, column)
        return self._values[row] if row == column else 0

    def __setitem__(self, position: tuple[int, int], value: float) -> None:
        row, column = position
        self._check(row, column)
        if row != column and value != 0:
            raise ValueError(
                "a diagonal matrix cannot hold a non zero value off the diagonal, "
                f"so ({row}, {column}) = {value} was refused"
            )
        if row == column:
            self._values[row] = value


class LowerTriangularMatrix(_SquareMatrix):
    """Only the main diagonal and everything below it can be non zero.

    Row i has i + 1 storable elements, so the total is 1 + 2 + ... + n, which is
    n(n + 1) / 2. That is a little under half of n^2, so the saving approaches
    fifty percent as n grows.

    The index arithmetic is the interesting part. To find where (i, j) lives,
    count all the elements in the rows above it, which is i(i + 1) / 2, then add
    j to step along the current row. Both of those come straight from the sum of
    the first i integers, which is why that formula is worth knowing by heart.
    """

    def __init__(self, size: int) -> None:
        super().__init__(size, size * (size + 1) // 2)

    def _offset(self, row: int, column: int) -> int:
        return row * (row + 1) // 2 + column

    def __getitem__(self, position: tuple[int, int]) -> float:
        row, column = position
        self._check(row, column)
        return self._values[self._offset(row, column)] if column <= row else 0

    def __setitem__(self, position: tuple[int, int], value: float) -> None:
        row, column = position
        self._check(row, column)
        if column > row:
            if value != 0:
                raise ValueError(
                    f"({row}, {column}) is above the diagonal, which a lower "
                    "triangular matrix keeps at zero"
                )
            return
        self._values[self._offset(row, column)] = value


class UpperTriangularMatrix(_SquareMatrix):
    """Only the main diagonal and everything above it can be non zero.

    The mirror image of the lower triangular case, and it stores the same
    n(n + 1) / 2 values. Row i now has n - i storable elements, so the offset
    counts a shrinking rather than a growing series.
    """

    def __init__(self, size: int) -> None:
        super().__init__(size, size * (size + 1) // 2)

    def _offset(self, row: int, column: int) -> int:
        # Elements in the rows above: n + (n-1) + ... + (n-row+1).
        before = row * self.size - (row * (row - 1)) // 2
        return before + (column - row)

    def __getitem__(self, position: tuple[int, int]) -> float:
        row, column = position
        self._check(row, column)
        return self._values[self._offset(row, column)] if column >= row else 0

    def __setitem__(self, position: tuple[int, int], value: float) -> None:
        row, column = position
        self._check(row, column)
        if column < row:
            if value != 0:
                raise ValueError(
                    f"({row}, {column}) is below the diagonal, which an upper "
                    "triangular matrix keeps at zero"
                )
            return
        self._values[self._offset(row, column)] = value


class SymmetricMatrix(LowerTriangularMatrix):
    """A matrix equal to its own mirror image, so (i, j) and (j, i) are the same.

    Only the lower triangle is stored, and a read of the upper triangle is
    answered by swapping the indices. The saving is the same n(n + 1) / 2, but
    unlike the triangular case nothing is being thrown away: every position still
    has a real value, it is just shared with its mirror.

    This is the representation used for undirected graph adjacency matrices
    later in the project, where an edge from a to b is the same edge as b to a.
    """

    def __getitem__(self, position: tuple[int, int]) -> float:
        row, column = position
        self._check(row, column)
        if column > row:
            row, column = column, row
        return self._values[self._offset(row, column)]

    def __setitem__(self, position: tuple[int, int], value: float) -> None:
        row, column = position
        self._check(row, column)
        if column > row:
            row, column = column, row
        self._values[self._offset(row, column)] = value


class TridiagonalMatrix(_SquareMatrix):
    """Non zero only on the main diagonal and the two next to it.

    This shape appears constantly in numerical work, for example when solving
    differential equations on a grid where each point only talks to its immediate
    neighbours.

    Storage is 3n - 2 values, which for large n is essentially linear. A 1000 by
    1000 tridiagonal matrix needs 2998 numbers instead of a million, a saving of
    over 99.7 percent.

    The layout used here is the lower diagonal first, then the main diagonal,
    then the upper diagonal, because keeping each diagonal contiguous is friendly
    to the CPU cache when sweeping along one of them.
    """

    def __init__(self, size: int) -> None:
        super().__init__(size, max(1, 3 * size - 2))

    def _offset(self, row: int, column: int) -> int:
        n = self.size
        if column == row - 1:
            return row - 1
        if column == row:
            return (n - 1) + row
        return (n - 1) + n + row

    def __getitem__(self, position: tuple[int, int]) -> float:
        row, column = position
        self._check(row, column)
        return self._values[self._offset(row, column)] if abs(row - column) <= 1 else 0

    def __setitem__(self, position: tuple[int, int], value: float) -> None:
        row, column = position
        self._check(row, column)
        if abs(row - column) > 1:
            if value != 0:
                raise ValueError(
                    f"({row}, {column}) is more than one step from the diagonal, "
                    "which a tridiagonal matrix keeps at zero"
                )
            return
        self._values[self._offset(row, column)] = value


class ToeplitzMatrix(_SquareMatrix):
    """Every diagonal holds a single repeated value, so (i, j) depends only on i - j.

    A matrix like this is completely described by its first row and first column,
    which is 2n - 1 numbers. Signal processing uses these heavily, because a
    convolution written as a matrix is Toeplitz.

    The stored order is the first column read downwards, then the rest of the
    first row, so the difference i - j maps straight onto an index.
    """

    def __init__(self, size: int) -> None:
        super().__init__(size, 2 * size - 1)

    def _offset(self, row: int, column: int) -> int:
        # Slots 0 to n-1 hold the diagonals on and below the main one, indexed by
        # how far below they sit. Slots n onwards hold the diagonals above it.
        difference = row - column
        return difference if difference >= 0 else self.size - 1 - difference

    def __getitem__(self, position: tuple[int, int]) -> float:
        row, column = position
        self._check(row, column)
        return self._values[self._offset(row, column)]

    def __setitem__(self, position: tuple[int, int], value: float) -> None:
        row, column = position
        self._check(row, column)
        # Writing one cell writes the whole diagonal, because in a Toeplitz
        # matrix they are the same number. That surprises people, so it is worth
        # stating plainly rather than hiding.
        self._values[self._offset(row, column)] = value
