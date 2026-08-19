"""Tests for the special matrix representations and the sparse matrix.

The theme running through these: a compressed representation has to behave
exactly like the full grid it replaces, and it has to actually save the memory it
claims to save. Both halves are checked.
"""

import pytest

from dsalab.structures.polynomial import Polynomial, Term
from dsalab.structures.sparse_matrix import Entry, SparseMatrix
from dsalab.structures.special_matrix import (
    DiagonalMatrix,
    LowerTriangularMatrix,
    SymmetricMatrix,
    ToeplitzMatrix,
    TridiagonalMatrix,
    UpperTriangularMatrix,
)


class TestDiagonalMatrix:
    def test_it_stores_only_the_diagonal(self):
        matrix = DiagonalMatrix(100)

        assert matrix.stored_values == 100, "storing 10000 cells would defeat the purpose"

    def test_reads_and_writes_on_the_diagonal(self):
        matrix = DiagonalMatrix(3)
        matrix[1, 1] = 5

        assert matrix[1, 1] == 5

    def test_everything_off_the_diagonal_reads_as_zero(self):
        matrix = DiagonalMatrix(3)
        matrix[0, 0] = 9

        assert matrix[0, 1] == 0
        assert matrix[2, 0] == 0

    def test_writing_a_non_zero_off_the_diagonal_is_refused(self):
        matrix = DiagonalMatrix(3)

        with pytest.raises(ValueError):
            matrix[0, 1] = 5

    def test_writing_a_zero_off_the_diagonal_is_harmless(self):
        matrix = DiagonalMatrix(3)
        matrix[0, 1] = 0  # already zero, so nothing to complain about

        assert matrix[0, 1] == 0

    def test_it_rebuilds_the_full_grid_correctly(self):
        matrix = DiagonalMatrix(3)
        matrix[0, 0], matrix[1, 1], matrix[2, 2] = 1, 2, 3

        assert matrix.to_dense() == [[1, 0, 0], [0, 2, 0], [0, 0, 3]]

    def test_a_size_of_zero_is_rejected(self):
        with pytest.raises(ValueError):
            DiagonalMatrix(0)

    def test_reading_outside_the_matrix_raises(self):
        with pytest.raises(IndexError):
            DiagonalMatrix(3)[3, 3]


class TestTriangularMatrices:
    def test_lower_triangular_stores_half_the_cells(self):
        matrix = LowerTriangularMatrix(10)

        assert matrix.stored_values == 55 == 10 * 11 // 2

    def test_lower_triangular_round_trips_every_valid_cell(self):
        matrix = LowerTriangularMatrix(4)
        expected = [[0] * 4 for _ in range(4)]

        value = 1
        for row in range(4):
            for column in range(row + 1):
                matrix[row, column] = value
                expected[row][column] = value
                value += 1

        assert matrix.to_dense() == expected

    def test_lower_triangular_refuses_a_value_above_the_diagonal(self):
        with pytest.raises(ValueError):
            LowerTriangularMatrix(3)[0, 2] = 7

    def test_upper_triangular_round_trips_every_valid_cell(self):
        matrix = UpperTriangularMatrix(4)
        expected = [[0] * 4 for _ in range(4)]

        value = 1
        for row in range(4):
            for column in range(row, 4):
                matrix[row, column] = value
                expected[row][column] = value
                value += 1

        assert matrix.to_dense() == expected

    def test_upper_triangular_refuses_a_value_below_the_diagonal(self):
        with pytest.raises(ValueError):
            UpperTriangularMatrix(3)[2, 0] = 7

    def test_no_two_cells_share_a_storage_slot(self):
        # If the index arithmetic collided, writing one cell would silently
        # change another. Filling every cell with a distinct value catches that.
        for kind, cells in (
            (LowerTriangularMatrix, [(r, c) for r in range(5) for c in range(r + 1)]),
            (UpperTriangularMatrix, [(r, c) for r in range(5) for c in range(r, 5)]),
        ):
            matrix = kind(5)
            for index, (row, column) in enumerate(cells, start=1):
                matrix[row, column] = index
            for index, (row, column) in enumerate(cells, start=1):
                assert matrix[row, column] == index, f"{kind.__name__} has colliding offsets"


class TestSymmetricMatrix:
    def test_writing_one_side_sets_the_mirror_too(self):
        matrix = SymmetricMatrix(3)
        matrix[2, 0] = 7

        assert matrix[0, 2] == 7

    def test_writing_the_upper_side_works_the_same_way(self):
        matrix = SymmetricMatrix(3)
        matrix[0, 2] = 4

        assert matrix[2, 0] == 4

    def test_the_rebuilt_grid_really_is_symmetric(self):
        matrix = SymmetricMatrix(4)
        matrix[1, 0], matrix[3, 2], matrix[2, 2] = 1, 2, 3
        grid = matrix.to_dense()

        for row in range(4):
            for column in range(4):
                assert grid[row][column] == grid[column][row]

    def test_it_stores_only_the_lower_triangle(self):
        assert SymmetricMatrix(10).stored_values == 55


class TestTridiagonalMatrix:
    def test_storage_is_linear_in_the_size(self):
        assert TridiagonalMatrix(1000).stored_values == 2998 == 3 * 1000 - 2

    def test_the_three_diagonals_round_trip(self):
        matrix = TridiagonalMatrix(4)
        for row in range(4):
            for column in range(4):
                if abs(row - column) <= 1:
                    matrix[row, column] = row * 10 + column

        for row in range(4):
            for column in range(4):
                expected = row * 10 + column if abs(row - column) <= 1 else 0
                assert matrix[row, column] == expected

    def test_cells_far_from_the_diagonal_are_zero_and_cannot_be_written(self):
        matrix = TridiagonalMatrix(4)

        assert matrix[0, 3] == 0
        with pytest.raises(ValueError):
            matrix[0, 3] = 1

    def test_a_one_by_one_tridiagonal_matrix_works(self):
        matrix = TridiagonalMatrix(1)
        matrix[0, 0] = 5

        assert matrix.to_dense() == [[5]]


class TestToeplitzMatrix:
    def test_writing_a_cell_fills_its_whole_diagonal(self):
        matrix = ToeplitzMatrix(4)
        matrix[1, 0] = 9

        assert matrix[2, 1] == 9
        assert matrix[3, 2] == 9

    def test_diagonals_are_independent_of_each_other(self):
        matrix = ToeplitzMatrix(4)
        for row in range(4):
            matrix[row, 0] = row + 1
        for column in range(4):
            matrix[0, column] = -(column + 1)

        assert matrix[0, 0] == -1
        assert matrix[1, 0] == 2
        assert matrix[0, 1] == -2
        assert matrix[3, 3] == -1, "the main diagonal is one value everywhere"

    def test_storage_is_two_n_minus_one(self):
        assert ToeplitzMatrix(50).stored_values == 99


class TestSparseMatrix:
    def test_an_empty_matrix_reads_as_all_zeros(self):
        matrix = SparseMatrix(3, 3)

        assert len(matrix) == 0
        assert matrix.to_dense() == [[0.0] * 3 for _ in range(3)]

    def test_writing_and_reading_back(self):
        matrix = SparseMatrix(4, 4)
        matrix[2, 3] = 7

        assert matrix[2, 3] == 7
        assert matrix[3, 2] == 0
        assert len(matrix) == 1

    def test_writing_zero_removes_the_entry_rather_than_storing_it(self):
        matrix = SparseMatrix(3, 3)
        matrix[1, 1] = 5
        matrix[1, 1] = 0

        assert len(matrix) == 0, "storing explicit zeros would make it stop being sparse"

    def test_overwriting_a_cell_does_not_add_a_second_entry(self):
        matrix = SparseMatrix(3, 3)
        matrix[1, 1] = 5
        matrix[1, 1] = 9

        assert len(matrix) == 1
        assert matrix[1, 1] == 9

    def test_entries_stay_sorted_by_row_then_column(self):
        matrix = SparseMatrix(4, 4)
        for row, column in [(3, 1), (0, 2), (2, 0), (0, 0)]:
            matrix[row, column] = 1

        positions = [(entry.row, entry.column) for entry in matrix]
        assert positions == sorted(positions)

    def test_round_trip_through_a_dense_grid(self):
        grid = [[0, 5, 0], [0, 0, 0], [3, 0, 7]]
        matrix = SparseMatrix.from_dense(grid)

        assert len(matrix) == 3
        assert matrix.to_dense() == [[0, 5, 0], [0, 0, 0], [3, 0, 7]]

    def test_a_ragged_grid_is_rejected(self):
        with pytest.raises(ValueError):
            SparseMatrix.from_dense([[1, 2], [3]])

    def test_density_reports_how_full_it_is(self):
        matrix = SparseMatrix.from_dense([[1, 0], [0, 0]])

        assert matrix.density == 0.25

    def test_reading_outside_the_matrix_raises(self):
        with pytest.raises(IndexError):
            SparseMatrix(2, 2)[5, 5]

    def test_addition_matches_the_dense_result(self):
        left = SparseMatrix.from_dense([[1, 0, 2], [0, 3, 0]])
        right = SparseMatrix.from_dense([[0, 4, 5], [6, 0, 0]])

        assert left.add(right).to_dense() == [[1, 4, 7], [6, 3, 0]]

    def test_addition_drops_terms_that_cancel_to_zero(self):
        left = SparseMatrix.from_dense([[5, 0], [0, 0]])
        right = SparseMatrix.from_dense([[-5, 0], [0, 0]])
        total = left.add(right)

        assert len(total) == 0, "a cancelled cell must not be kept as an explicit zero"
        assert total.to_dense() == [[0, 0], [0, 0]]

    def test_adding_matrices_of_different_shapes_is_rejected(self):
        with pytest.raises(ValueError):
            SparseMatrix(2, 2).add(SparseMatrix(3, 3))

    def test_addition_only_touches_the_non_zero_entries(self):
        from dsalab.tracing import record

        left = SparseMatrix.from_dense([[1, 0, 0, 0, 0]])
        right = SparseMatrix.from_dense([[0, 0, 0, 0, 2]])
        _, steps = record(left.add_traced(right))

        assert len(steps) == 2, "a five cell row with two values should take two steps, not five"

    def test_multiplication_matches_the_textbook_result(self):
        left = SparseMatrix.from_dense([[1, 2], [3, 4]])
        right = SparseMatrix.from_dense([[5, 6], [7, 8]])

        assert left.multiply(right).to_dense() == [[19, 22], [43, 50]]

    def test_multiplying_by_the_identity_changes_nothing(self):
        matrix = SparseMatrix.from_dense([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
        identity = SparseMatrix.from_dense([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        assert matrix.multiply(identity).to_dense() == matrix.to_dense()

    def test_multiplying_mismatched_shapes_is_rejected(self):
        with pytest.raises(ValueError):
            SparseMatrix(2, 3).multiply(SparseMatrix(2, 3))

    def test_multiplying_by_an_empty_matrix_gives_an_empty_matrix(self):
        matrix = SparseMatrix.from_dense([[1, 2], [3, 4]])

        assert len(matrix.multiply(SparseMatrix(2, 2))) == 0

    def test_transpose_swaps_the_indices(self):
        matrix = SparseMatrix.from_dense([[1, 2, 0], [0, 0, 3]])

        assert matrix.transpose().to_dense() == [[1, 0], [2, 0], [0, 3]]

    def test_transposing_twice_gets_back_to_the_start(self):
        matrix = SparseMatrix.from_dense([[1, 0, 2], [0, 3, 0]])

        assert matrix.transpose().transpose() == matrix

    def test_entries_can_be_supplied_at_construction(self):
        matrix = SparseMatrix(2, 2, [Entry(0, 1, 5.0)])

        assert matrix[0, 1] == 5.0


class TestPolynomial:
    def test_terms_are_kept_in_descending_order_of_exponent(self):
        polynomial = Polynomial([Term(1, 0), Term(3, 5), Term(2, 2)])

        assert [term.exponent for term in polynomial] == [5, 2, 0]

    def test_zero_coefficients_are_dropped(self):
        polynomial = Polynomial([Term(0, 3), Term(4, 1)])

        assert len(polynomial) == 1

    def test_repeated_exponents_are_combined(self):
        polynomial = Polynomial([Term(2, 3), Term(5, 3)])

        assert len(polynomial) == 1
        assert polynomial.evaluate(1) == 7

    def test_negative_exponents_are_rejected(self):
        with pytest.raises(ValueError):
            Polynomial([Term(1, -1)])

    def test_degree_of_the_zero_polynomial_is_minus_one(self):
        assert Polynomial().degree == -1
        assert Polynomial.from_coefficients([5]).degree == 0

    def test_building_from_dense_coefficients(self):
        polynomial = Polynomial.from_coefficients([1, 0, 3])  # 3x^2 + 1

        assert polynomial.degree == 2
        assert len(polynomial) == 2
        assert polynomial.evaluate(2) == 13

    def test_it_prints_the_way_a_person_would_write_it(self):
        assert str(Polynomial.from_coefficients([5, 0, 2])) == "2x^2 + 5"
        assert str(Polynomial.from_coefficients([-3, 1])) == "1x - 3"
        assert str(Polynomial()) == "0"

    def test_addition_merges_matching_exponents(self):
        left = Polynomial.from_coefficients([1, 2, 3])
        right = Polynomial.from_coefficients([4, 5])

        assert left.add(right) == Polynomial.from_coefficients([5, 7, 3])

    def test_addition_drops_terms_that_cancel(self):
        left = Polynomial([Term(3, 2)])
        right = Polynomial([Term(-3, 2)])

        assert len(left.add(right)) == 0

    def test_adding_the_zero_polynomial_changes_nothing(self):
        polynomial = Polynomial.from_coefficients([1, 2, 3])

        assert polynomial.add(Polynomial()) == polynomial

    def test_multiplication_matches_the_expansion_done_by_hand(self):
        # (x + 1)(x + 2) = x^2 + 3x + 2
        left = Polynomial.from_coefficients([1, 1])
        right = Polynomial.from_coefficients([2, 1])

        assert left.multiply(right) == Polynomial.from_coefficients([2, 3, 1])

    def test_multiplying_by_zero_gives_zero(self):
        assert len(Polynomial.from_coefficients([1, 2]).multiply(Polynomial())) == 0

    def test_evaluation_agrees_with_working_it_out_the_slow_way(self):
        polynomial = Polynomial.from_coefficients([5, 0, 3, 2])  # 2x^3 + 3x^2 + 5
        for x in (-2, 0, 1, 2.5, 10):
            slow = sum(
                coefficient * x**exponent
                for exponent, coefficient in enumerate([5, 0, 3, 2])
            )
            assert polynomial.evaluate(x) == pytest.approx(slow)

    def test_horner_uses_one_step_per_degree_not_per_power(self):
        from dsalab.tracing import record

        polynomial = Polynomial([Term(1, 10)])  # a single term of high degree
        _, steps = record(polynomial.evaluate_traced(2))

        assert len(steps) == 11, "one step per degree from 10 down to 0"
        assert polynomial.evaluate(2) == 1024

    def test_the_zero_polynomial_evaluates_to_zero(self):
        assert Polynomial().evaluate(99) == 0

    def test_the_derivative_follows_the_power_rule(self):
        # d/dx (2x^3 + 3x^2 + 5) = 6x^2 + 6x
        polynomial = Polynomial.from_coefficients([5, 0, 3, 2])

        assert polynomial.derivative() == Polynomial.from_coefficients([0, 6, 6])

    def test_the_derivative_of_a_constant_is_zero(self):
        assert len(Polynomial.from_coefficients([7]).derivative()) == 0
