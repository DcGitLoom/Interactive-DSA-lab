"""Polynomials, stored sparsely by term.

A polynomial like 5x^100 + 3 has two terms but a degree of 100. Storing it as a
list of 101 coefficients, almost all of them zero, is the same waste the sparse
matrix avoids, so the same fix applies: keep a list of (coefficient, exponent)
pairs and nothing else.

The list is kept sorted by exponent, highest first, which is how polynomials are
conventionally written and which makes addition a single merge pass.

Evaluation is done by Horner's method, which is the small idea from day 2 showing
up again where it matters more.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from dsalab.tracing import Step, Traced, run


@dataclass(frozen=True)
class Term:
    """One term of a polynomial, for example 5x^3."""

    coefficient: float
    exponent: int

    def __str__(self) -> str:
        if self.exponent == 0:
            return f"{self.coefficient:g}"
        if self.exponent == 1:
            return f"{self.coefficient:g}x"
        return f"{self.coefficient:g}x^{self.exponent}"


class Polynomial:
    """A polynomial in one variable, stored as its non zero terms only.

    | Operation | Cost | Why |
    | - | - | - |
    | add | O(t1 + t2) | One merge pass over two sorted term lists |
    | multiply | O(t1 times t2) | Every pair of terms meets once |
    | evaluate | O(t) | Horner's method, one multiply and one add per term |
    | degree | O(1) | The list is sorted, so the first term is the highest |

    where t is the number of non zero terms.
    """

    def __init__(self, terms: Iterable[Term] | None = None) -> None:
        collected: dict[int, float] = {}
        for term in terms or ():
            if term.exponent < 0:
                raise ValueError("negative exponents are not polynomials")
            collected[term.exponent] = collected.get(term.exponent, 0) + term.coefficient

        self._terms = [
            Term(coefficient, exponent)
            for exponent, coefficient in sorted(collected.items(), reverse=True)
            if coefficient != 0
        ]

    @classmethod
    def from_coefficients(cls, coefficients: Iterable[float]) -> Polynomial:
        """Build from a dense list where position 0 is the constant term.

        `Polynomial.from_coefficients([1, 0, 3])` is 3x^2 + 1. This is the
        convenient way to type a small polynomial, and the zeros are dropped on
        the way in.
        """
        return cls(
            Term(coefficient, exponent)
            for exponent, coefficient in enumerate(coefficients)
            if coefficient != 0
        )

    def __iter__(self) -> Iterator[Term]:
        return iter(self._terms)

    def __len__(self) -> int:
        """The number of non zero terms, which is not the degree."""
        return len(self._terms)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Polynomial):
            return NotImplemented
        return self._terms == other._terms

    @property
    def degree(self) -> int:
        """The highest exponent present, or -1 for the zero polynomial.

        Returning -1 rather than 0 for the zero polynomial matters: the constant
        5 has degree 0, and if zero also had degree 0 the two could not be told
        apart by degree alone.
        """
        return self._terms[0].exponent if self._terms else -1

    def __str__(self) -> str:
        if not self._terms:
            return "0"
        parts = [str(self._terms[0])]
        for term in self._terms[1:]:
            sign = "-" if term.coefficient < 0 else "+"
            magnitude = Term(abs(term.coefficient), term.exponent)
            parts.append(f" {sign} {magnitude}")
        return "".join(parts)

    def __repr__(self) -> str:
        return f"Polynomial({str(self)!r})"

    def add(self, other: Polynomial) -> Polynomial:
        return run(self.add_traced(other))

    def add_traced(self, other: Polynomial) -> Traced[Polynomial]:
        """Add two polynomials in one merge pass. O(t1 + t2).

        Both term lists are sorted by exponent, so this walks them together the
        way merge sort merges two sorted halves. Terms with matching exponents
        combine, and a pair that cancels to zero is dropped rather than stored.
        """
        merged: list[Term] = []
        left = right = 0

        while left < len(self._terms) and right < len(other._terms):
            a, b = self._terms[left], other._terms[right]

            if a.exponent == b.exponent:
                total = a.coefficient + b.coefficient
                yield Step(
                    "combine",
                    f"Both have an x^{a.exponent} term: "
                    f"{a.coefficient:g} + {b.coefficient:g} = {total:g}.",
                    {"exponent": a.exponent, "coefficient": total},
                )
                if total != 0:
                    merged.append(Term(total, a.exponent))
                left += 1
                right += 1
            elif a.exponent > b.exponent:
                yield Step(
                    "carry",
                    f"Only the first polynomial has an x^{a.exponent} term, so it carries over.",
                    {"exponent": a.exponent, "coefficient": a.coefficient},
                )
                merged.append(a)
                left += 1
            else:
                yield Step(
                    "carry",
                    f"Only the second polynomial has an x^{b.exponent} term, so it carries over.",
                    {"exponent": b.exponent, "coefficient": b.coefficient},
                )
                merged.append(b)
                right += 1

        # The tail of whichever list still has terms. These report a step too,
        # so the animation covers the whole run rather than stopping when the
        # shorter polynomial runs out.
        for term in self._terms[left:] + other._terms[right:]:
            yield Step(
                "carry",
                f"Nothing left to merge against, so the x^{term.exponent} term carries over.",
                {"exponent": term.exponent, "coefficient": term.coefficient},
            )
            merged.append(term)

        result = Polynomial()
        result._terms = merged
        return result

    def multiply(self, other: Polynomial) -> Polynomial:
        """Multiply two polynomials. O(t1 times t2).

        Every term of one meets every term of the other, coefficients multiply
        and exponents add. Terms that land on the same exponent are collected
        together, which the constructor does on the way in.

        Faster methods exist. Karatsuba multiplication and the fast Fourier
        transform bring dense polynomial multiplication down to O(n log n), and
        they matter enormously for large polynomials. The straightforward version
        is here because it is the one the course covers and the one worth being
        able to write from memory.
        """
        products: list[Term] = []
        for mine in self._terms:
            for theirs in other._terms:
                products.append(
                    Term(mine.coefficient * theirs.coefficient, mine.exponent + theirs.exponent)
                )
        return Polynomial(products)

    def evaluate(self, x: float) -> float:
        return run(self.evaluate_traced(x))

    def evaluate_traced(self, x: float) -> Traced[float]:
        """Work out the value of the polynomial at x, using Horner's method.

        The obvious way computes each power separately: for 3x^4 you multiply x
        by itself four times, then again for the next term, and so on. That is
        O(degree^2) multiplications for a dense polynomial.

        Horner's method rewrites the polynomial so that each step reuses the work
        of the last. For 2x^3 + 3x^2 + 0x + 5 it becomes:

            ((2x + 3)x + 0)x + 5

        which is one multiply and one add per degree, so O(degree) in total. The
        answer is identical, the arrangement is cheaper, and as a bonus it is
        also more accurate in floating point because there are fewer operations
        for rounding error to accumulate through.
        """
        if not self._terms:
            yield Step("done", "The zero polynomial evaluates to 0 everywhere.", {"result": 0.0})
            return 0.0

        result = 0.0
        exponent = self.degree
        terms_by_exponent = {term.exponent: term.coefficient for term in self._terms}

        while exponent >= 0:
            coefficient = terms_by_exponent.get(exponent, 0.0)
            result = result * x + coefficient
            yield Step(
                "horner",
                f"Multiply the running value by x and add the x^{exponent} "
                f"coefficient {coefficient:g}, giving {result:g}.",
                {"exponent": exponent, "coefficient": coefficient, "value": result},
            )
            exponent -= 1

        return result

    def derivative(self) -> Polynomial:
        """The derivative, by the power rule: the term c x^n becomes c n x^(n-1).

        Constant terms disappear, which falls out of the rule automatically since
        multiplying by an exponent of 0 gives 0.
        """
        return Polynomial(
            Term(term.coefficient * term.exponent, term.exponent - 1)
            for term in self._terms
            if term.exponent > 0
        )
