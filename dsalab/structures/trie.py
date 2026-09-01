"""Tries: a tree where the path spells the key.

Every structure so far treats a key as one indivisible thing to compare or hash. A
trie takes keys apart. Each edge is one character, and a word is the path from the
root down to a node marked as the end of a word.

Two consequences, and they are the whole reason the structure exists:

**Lookup cost depends on the key length, not on how many keys there are.** Finding
a ten letter word takes ten steps whether the trie holds a hundred words or ten
million. A balanced tree would need log n comparisons and each comparison itself
walks the string, so the trie is genuinely better as the collection grows.

**Prefixes come for free.** Every word starting with "car" lives under the node
you reach by walking c, a, r. Finding them is one walk plus a collection of the
subtree. A hash table cannot do this at all, and a balanced tree can only do it by
ordering keys and scanning a range, which is more work and more code.

That second point is why tries are behind autocomplete, spell checking, IP routing
tables and dictionary compression. Whenever the question is about prefixes rather
than exact keys, this is the structure.

The cost is memory. A plain trie node holds a map of children, and with one node
per character of every distinct prefix that adds up fast. The compressed trie
below is the standard fix: any chain of single child nodes collapses into one node
holding the whole run of characters.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from dsalab.invariants import Violation, verify_if_checking
from dsalab.tracing import Step, Traced, run


class TrieNode:
    """One node: the children by character, and whether a word ends here.

    `is_word` is separate from having no children on purpose. "car" and "carpet"
    can both be words, so a node in the middle of a path still needs to be able to
    say that a word ends there.
    """

    __slots__ = ("children", "is_word", "value")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_word = False
        self.value: Any = None

    def __repr__(self) -> str:
        return f"TrieNode({''.join(sorted(self.children))!r}{', word' if self.is_word else ''})"


class Trie:
    """A prefix tree over strings, optionally mapping each word to a value.

    | Operation | Cost | Note |
    | - | - | - |
    | insert a word of length m | O(m) | independent of how many words are stored |
    | lookup | O(m) | same |
    | delete | O(m) | plus tidying up nodes nobody needs any more |
    | words with a given prefix | O(m + k) | m to walk, k to collect the results |
    | longest prefix of a string that is a word | O(m) | the routing table operation |
    | memory | O(total characters) | the weakness, and why compression exists |
    """

    def __init__(self, words: Iterable[str] | None = None) -> None:
        self.root = TrieNode()
        self._size = 0
        for word in words or ():
            self.insert(word)

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"Trie({self._size} words)"

    def insert(self, word: str, value: Any = None) -> bool:
        return run(self.insert_traced(word, value))

    def insert_traced(self, word: str, value: Any = None) -> Traced[bool]:
        """Add a word. Returns False if it was already there. O(len(word))."""
        node = self.root

        for depth, character in enumerate(word):
            if character not in node.children:
                node.children[character] = TrieNode()
                yield Step(
                    "create",
                    f"No branch for {character!r} at depth {depth}, so a new node is added.",
                    {"character": character, "depth": depth, "prefix": word[: depth + 1]},
                )
            else:
                yield Step(
                    "reuse",
                    f"{word[: depth + 1]!r} is already a prefix of something, so this "
                    "branch is shared rather than duplicated.",
                    {"character": character, "depth": depth, "prefix": word[: depth + 1]},
                )
            node = node.children[character]

        if node.is_word:
            yield Step("duplicate", f"{word!r} was already stored.", {"word": word})
            return False

        node.is_word = True
        node.value = value
        self._size += 1
        yield Step("mark", f"Marked the end of {word!r}.", {"word": word})
        verify_if_checking(self)
        return True

    def _walk(self, prefix: str) -> TrieNode | None:
        """Follow a prefix and return the node it ends at, or None."""
        node = self.root
        for character in prefix:
            if character not in node.children:
                return None
            node = node.children[character]
        return node

    def contains(self, word: str) -> bool:
        return run(self.contains_traced(word))

    def contains_traced(self, word: str) -> Traced[bool]:
        """Whether an exact word is stored. O(len(word)).

        Note the difference from `starts_with`: reaching the end of the path is
        not enough, the node there must be marked as a word. A trie holding
        "carpet" contains the *path* for "car" but not the word.
        """
        node = self.root

        for depth, character in enumerate(word):
            if character not in node.children:
                yield Step(
                    "missing",
                    f"No branch for {character!r} at depth {depth}, so {word!r} is not here.",
                    {"depth": depth, "character": character},
                )
                return False
            node = node.children[character]
            yield Step(
                "descend",
                f"Followed {character!r}, now at the node for {word[: depth + 1]!r}.",
                {"depth": depth, "prefix": word[: depth + 1]},
            )

        if node.is_word:
            yield Step("found", f"{word!r} is stored here.", {"word": word})
            return True

        yield Step(
            "prefix-only",
            f"The path for {word!r} exists, but nothing ends here, so it is a prefix "
            "of some other word rather than a word itself.",
            {"word": word},
        )
        return False

    def __contains__(self, word: str) -> bool:
        return self.contains(word)

    def get(self, word: str, default: Any = None) -> Any:
        """The value stored with a word, or `default`."""
        node = self._walk(word)
        return node.value if node is not None and node.is_word else default

    def starts_with(self, prefix: str) -> bool:
        """Whether any stored word begins with this prefix. O(len(prefix))."""
        return self._walk(prefix) is not None

    def words_with_prefix(self, prefix: str) -> list[str]:
        """Every stored word beginning with `prefix`, in sorted order.

        This is the operation tries exist for, and the one no hash table can do.
        Cost is O(len(prefix)) to walk down plus O(total length of the results) to
        collect them, which is the best any structure could manage since the
        results have to be produced somehow.
        """
        node = self._walk(prefix)
        if node is None:
            return []

        found: list[str] = []

        def collect(current: TrieNode, built: str) -> None:
            if current.is_word:
                found.append(built)
            # Sorted so the output is deterministic, which matters for tests and
            # for an autocomplete list that should not jump around.
            for character in sorted(current.children):
                collect(current.children[character], built + character)

        collect(node, prefix)
        return found

    def longest_prefix_of(self, text: str) -> str | None:
        """The longest stored word that is a prefix of `text`, or None.

        This is the lookup an IP routing table does: given an address, find the
        most specific stored route that covers it. Also how a tokeniser finds the
        longest matching word at a position.

        O(len(text)) with a single walk, keeping track of the last word seen. Any
        other structure would need to try every prefix separately.
        """
        node = self.root
        best: str | None = None

        for index, character in enumerate(text):
            if character not in node.children:
                break
            node = node.children[character]
            if node.is_word:
                best = text[: index + 1]

        return best

    def delete(self, word: str) -> bool:
        """Remove a word and tidy up any nodes left with nothing to do.

        The tidying is the interesting part and is easy to get wrong in two
        opposite directions:

        * Delete too little, and the trie fills with dead paths, wasting memory
          and slowing every prefix walk.
        * Delete too much, and you break other words. Removing "carpet" must not
          remove the nodes for "car", and removing "car" must not remove the path
          "carpet" travels through.

        A node can only be removed when it has no children **and** is not itself
        the end of some other word. Working back up from the deleted word and
        stopping at the first node that fails either test is what gets this right.
        """
        path: list[tuple[TrieNode, str]] = []
        node = self.root

        for character in word:
            if character not in node.children:
                return False
            path.append((node, character))
            node = node.children[character]

        if not node.is_word:
            return False

        node.is_word = False
        node.value = None
        self._size -= 1

        # Walk back up, removing only nodes that nothing else needs.
        for parent, character in reversed(path):
            child = parent.children[character]
            if child.children or child.is_word:
                break
            del parent.children[character]

        verify_if_checking(self)
        return True

    def words(self) -> list[str]:
        """Every stored word, in sorted order.

        A trie sorts for free: walking the children in alphabetical order at each
        node produces the words in order, with no sorting step at all. That is a
        real advantage over a hash table, which has to collect and sort.
        """
        return self.words_with_prefix("")

    def __iter__(self) -> Iterator[str]:
        return iter(self.words())

    def node_count(self) -> int:
        """How many nodes exist, which is the memory cost of the structure."""

        def count(node: TrieNode) -> int:
            return 1 + sum(count(child) for child in node.children.values())

        return count(self.root)

    def check_invariants(self) -> list[Violation]:
        violations: list[Violation] = []

        def walk(node: TrieNode, built: str) -> int:
            words = 1 if node.is_word else 0

            if not node.is_word and not node.children and node is not self.root:
                violations.append(Violation(
                    "no node exists without a reason",
                    f"the node for {built!r} is neither a word nor on the way to one, "
                    "so a deletion failed to tidy up",
                ))

            for character, child in node.children.items():
                if len(character) != 1:
                    violations.append(Violation(
                        "each edge carries exactly one character",
                        f"an edge from {built!r} is labelled {character!r}",
                    ))
                words += walk(child, built + character)

            return words

        counted = walk(self.root, "")
        if counted != self._size:
            violations.append(Violation(
                "the recorded size matches the words stored",
                f"the trie says {self._size} words but {counted} were found",
            ))

        return violations


class CompressedTrieNode:
    """A node of a compressed trie, where each edge carries a whole run of characters."""

    __slots__ = ("children", "is_word", "value")

    def __init__(self) -> None:
        self.children: dict[str, tuple[str, CompressedTrieNode]] = {}
        self.is_word = False
        self.value: Any = None


class CompressedTrie:
    """A trie where chains of single child nodes collapse into one edge.

    Also called a radix tree or a Patricia trie. The observation behind it: in a
    plain trie storing "carpet", the nodes for c, a, r, p, e each have exactly one
    child, so five nodes carry no branching information at all. Collapsing them
    into a single edge labelled "carpet" stores the same information in one node.

    The saving is large on realistic data. A dictionary of English words has long
    unique tails on many entries, and this removes almost all of them. The tests
    measure the node count on the same word list both ways.

    What it costs is complexity in insertion. Adding a word that shares part of an
    edge means **splitting that edge**: the shared part stays, and the two
    different tails become separate children. That case does not exist in the
    plain trie at all, and it is where the bugs live, so the tests aim squarely at
    it.

    Lookup is still O(len(word)), just with fewer node hops and more character
    comparisons per hop, which is also better for the cache.
    """

    def __init__(self, words: Iterable[str] | None = None) -> None:
        self.root = CompressedTrieNode()
        self._size = 0
        for word in words or ():
            self.insert(word)

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"CompressedTrie({self._size} words, {self.node_count()} nodes)"

    @staticmethod
    def _shared_length(first: str, second: str) -> int:
        """How many characters the two strings agree on from the start."""
        limit = min(len(first), len(second))
        length = 0
        while length < limit and first[length] == second[length]:
            length += 1
        return length

    def insert(self, word: str, value: Any = None) -> bool:
        return run(self.insert_traced(word, value))

    def insert_traced(self, word: str, value: Any = None) -> Traced[bool]:
        """Add a word, splitting an edge where the paths diverge."""
        node = self.root
        remaining = word

        while True:
            if not remaining:
                if node.is_word:
                    yield Step("duplicate", f"{word!r} was already stored.", {"word": word})
                    return False
                node.is_word = True
                node.value = value
                self._size += 1
                yield Step("mark", f"Marked the end of {word!r}.", {"word": word})
                verify_if_checking(self)
                return True

            first = remaining[0]
            if first not in node.children:
                node.children[first] = (remaining, CompressedTrieNode())
                child = node.children[first][1]
                child.is_word = True
                child.value = value
                self._size += 1
                yield Step(
                    "create",
                    f"Nothing here starts with {first!r}, so the whole tail {remaining!r} "
                    "becomes a single edge rather than one node per character.",
                    {"edge": remaining},
                )
                verify_if_checking(self)
                return True

            label, child = node.children[first]
            shared = self._shared_length(label, remaining)

            if shared == len(label):
                # The whole edge is consumed, so carry on from the child.
                yield Step(
                    "follow",
                    f"The edge {label!r} matches, so it is followed in one hop rather "
                    f"than {len(label)}.",
                    {"edge": label},
                )
                node = child
                remaining = remaining[shared:]
                continue

            # The paths diverge partway along this edge, so it has to split.
            middle = CompressedTrieNode()
            node.children[first] = (label[:shared], middle)
            middle.children[label[shared]] = (label[shared:], child)

            yield Step(
                "split",
                f"{word!r} and the existing edge {label!r} agree for {shared} character(s) "
                f"and then diverge, so the edge splits into {label[:shared]!r} and "
                f"{label[shared:]!r}.",
                {"shared": label[:shared], "existing_tail": label[shared:]},
            )

            if shared == len(remaining):
                middle.is_word = True
                middle.value = value
            else:
                tail = remaining[shared:]
                leaf = CompressedTrieNode()
                leaf.is_word = True
                leaf.value = value
                middle.children[tail[0]] = (tail, leaf)

            self._size += 1
            verify_if_checking(self)
            return True

    def _walk(self, prefix: str) -> tuple[CompressedTrieNode, str] | None:
        """Follow a prefix, returning the node reached and any leftover edge text.

        The leftover matters: a prefix can end **partway along an edge**, which a
        plain trie never does. "car" in a trie holding only "carpet" ends in the
        middle of the edge labelled "carpet", and that is still a valid prefix.
        """
        node = self.root
        remaining = prefix

        while remaining:
            first = remaining[0]
            if first not in node.children:
                return None

            label, child = node.children[first]
            shared = self._shared_length(label, remaining)

            if shared == len(remaining):
                # Ended inside or exactly at the end of this edge.
                return child, label[shared:]
            if shared < len(label):
                return None

            node = child
            remaining = remaining[shared:]

        return node, ""

    def contains(self, word: str) -> bool:
        """Whether the exact word is stored. O(len(word))."""
        found = self._walk(word)
        if found is None:
            return False
        node, leftover = found
        # Leftover text means the word ended partway along an edge, so it is a
        # prefix of something rather than a stored word.
        return node.is_word and not leftover

    def __contains__(self, word: str) -> bool:
        return self.contains(word)

    def starts_with(self, prefix: str) -> bool:
        return self._walk(prefix) is not None

    def words_with_prefix(self, prefix: str) -> list[str]:
        """Every stored word beginning with `prefix`, in sorted order."""
        found = self._walk(prefix)
        if found is None:
            return []

        node, leftover = found
        collected: list[str] = []
        base = prefix + leftover

        def collect(current: CompressedTrieNode, built: str) -> None:
            if current.is_word:
                collected.append(built)
            for key in sorted(current.children):
                label, child = current.children[key]
                collect(child, built + label)

        collect(node, base)
        return sorted(collected)

    def words(self) -> list[str]:
        return self.words_with_prefix("")

    def __iter__(self) -> Iterator[str]:
        return iter(self.words())

    def node_count(self) -> int:
        def count(node: CompressedTrieNode) -> int:
            return 1 + sum(count(child) for _, child in node.children.values())

        return count(self.root)

    def check_invariants(self) -> list[Violation]:
        violations: list[Violation] = []

        def walk(node: CompressedTrieNode, built: str) -> int:
            words = 1 if node.is_word else 0

            for key, (label, child) in node.children.items():
                if not label:
                    violations.append(Violation(
                        "no edge is empty",
                        f"an edge from {built!r} carries no characters",
                    ))
                elif label[0] != key:
                    violations.append(Violation(
                        "each edge is filed under its first character",
                        f"an edge labelled {label!r} is filed under {key!r}",
                    ))

                if len(child.children) == 1 and not child.is_word:
                    violations.append(Violation(
                        "no node has a single child and no word, since that should "
                        "have been compressed",
                        f"the node after {built + label!r} could be merged with its child",
                    ))

                words += walk(child, built + label)

            return words

        counted = walk(self.root, "")
        if counted != self._size:
            violations.append(Violation(
                "the recorded size matches the words stored",
                f"the trie says {self._size} words but {counted} were found",
            ))

        return violations
