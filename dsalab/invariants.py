"""Invariant checking: let a structure state its own rules and verify them.

An invariant is a statement that must be true about a structure at all times,
between operations. A binary search tree's invariant is that every value in a
node's left subtree is smaller than the node and every value on the right is
larger. A red black tree has five. A heap has one.

These rules are usually written in comments and enforced by hope. Here they are
written in code, so they can actually be checked.

Two things this buys, and the second is the one that turned out to matter most:

1. **Tests that check the structure, not just the answers.** A tree can return
   every correct answer while being internally broken, because the damage only
   shows up on some later operation. Checking invariants after every mutation
   catches the corruption at the moment it happens, next to the line that caused
   it, instead of ten operations later in unrelated code.

2. **A teaching tool.** The visualiser can display the rules alongside the tree
   and mark which ones hold. When a red black tree rebalances, watching rule 4
   break and then be repaired by a rotation explains rotations better than any
   description of them does.

Checking is deliberately not free and not automatic. `checked` is a context
manager, so a test or the app can turn it on while an ordinary program pays
nothing. Some of these checks are O(n) and would turn an O(log n) insert into an
O(n) one if they ran always.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_CHECKING = False


@dataclass(frozen=True)
class Violation:
    """One broken rule, described well enough to act on.

    Attributes:
        rule: The name of the rule, as a learner would say it, for example
            "every value on the left is smaller".
        detail: Exactly what went wrong and where, for example which node.
    """

    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.detail}"


class InvariantError(AssertionError):
    """Raised when a structure is found to have broken its own rules.

    It subclasses AssertionError because that is what it is: a claim the code
    makes about itself, found to be false.
    """

    def __init__(self, structure: object, violations: list[Violation]) -> None:
        self.violations = violations
        listing = "\n".join(f"  - {violation}" for violation in violations)
        super().__init__(f"{type(structure).__name__} broke its own rules:\n{listing}")


@runtime_checkable
class HasInvariants(Protocol):
    """Anything that can state its own rules and check them."""

    def check_invariants(self) -> list[Violation]:
        """Return every broken rule, or an empty list when the structure is sound."""
        ...


def verify(structure: HasInvariants) -> None:
    """Raise if the structure has broken any of its rules. Always checks."""
    violations = structure.check_invariants()
    if violations:
        raise InvariantError(structure, violations)


def verify_if_checking(structure: HasInvariants) -> None:
    """Check the structure, but only inside a `checked()` block.

    Structures call this after each mutation. In ordinary use it is one boolean
    test and costs nothing; in a test or in the app it verifies everything.
    """
    if _CHECKING:
        verify(structure)


@contextlib.contextmanager
def checked() -> Iterator[None]:
    """Turn on invariant checking for the duration of a block.

    ```python
    with checked():
        tree.insert(5)   # the tree is verified after this line
    ```

    Nested blocks are safe, and the previous setting is restored on the way out
    even if an exception is raised, so a failing test cannot leave checking
    switched on for everything that follows it.
    """
    global _CHECKING
    previous = _CHECKING
    _CHECKING = True
    try:
        yield
    finally:
        _CHECKING = previous


def is_checking() -> bool:
    """Whether invariant checking is currently switched on."""
    return _CHECKING
