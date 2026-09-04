"""String matching: finding a pattern inside a text without rereading everything.

The naive method compares the pattern against every position in the text, and on
bad input it is O(n times m). The bad input is not exotic: searching for "aaaab"
inside a long run of "aaaa..." makes every position match four characters before
failing on the fifth.

The three real algorithms all avoid that, and each does it in a completely
different way. That is why they are worth studying together rather than picking
one:

* **KMP** precomputes what the pattern knows about itself, so that after a
  mismatch it can slide forward without ever moving the text pointer backwards.
* **Rabin Karp** compares numbers instead of strings, using a rolling hash that
  costs O(1) per position.
* **Z algorithm** computes, for every position, how much of the text matches the
  start of the pattern, reusing earlier answers to stay linear.

All three are O(n + m). The differences are in what else they can do: Rabin Karp
generalises to searching for many patterns at once and to two dimensions, KMP is
the basis for automaton style matching, and the Z array is a general tool that
turns up in many other string problems.
"""

from __future__ import annotations

from dsalab.tracing import Step, Traced, run


def naive_search(text: str, pattern: str) -> list[int]:
    return run(naive_search_traced(text, pattern))


def naive_search_traced(text: str, pattern: str) -> Traced[list[int]]:
    """Try the pattern at every position. O(n times m) in the worst case.

    Included as the baseline, because seeing where it wastes work is what makes
    the other three make sense. Watch it on text "aaaaaaaaab" with pattern "aaab"
    and the waste is obvious: after failing at position 0, it starts again at
    position 1 and re-examines characters it has already seen.

    That rereading is exactly what KMP eliminates.
    """
    found: list[int] = []
    if not pattern:
        return list(range(len(text) + 1))

    for start in range(len(text) - len(pattern) + 1):
        matched = 0
        while matched < len(pattern) and text[start + matched] == pattern[matched]:
            matched += 1

        yield Step(
            "attempt",
            f"At position {start}, {matched} of {len(pattern)} characters matched"
            + (". A hit." if matched == len(pattern) else ", then it failed."),
            {"start": start, "matched": matched, "hit": matched == len(pattern)},
        )

        if matched == len(pattern):
            found.append(start)

    return found


def build_failure_table(pattern: str) -> list[int]:
    """The table KMP uses, sometimes called the prefix function or failure function.

    `table[i]` is the length of the longest proper prefix of `pattern[:i+1]` that
    is also a suffix of it. "Proper" means it cannot be the whole thing.

    What that means in plain English: **if you have matched i+1 characters and the
    next one fails, how many characters do you already have correct?** The answer
    is table[i], because the last table[i] characters you matched are also the
    first table[i] characters of the pattern.

    For "abcabd" the table is [0, 0, 0, 1, 2, 0]. At index 4 it is 2, because
    "abcab" ends with "ab", which is also how the pattern starts. So a failure
    after matching "abcab" does not have to start over: two characters are already
    right.

    Building the table is itself a use of the same idea, comparing the pattern
    against itself, which is why the code below looks like a miniature version of
    the search.
    """
    table = [0] * len(pattern)
    length = 0
    index = 1

    while index < len(pattern):
        if pattern[index] == pattern[length]:
            length += 1
            table[index] = length
            index += 1
        elif length > 0:
            # Do not give up entirely: fall back to the next shorter candidate,
            # which the table already knows.
            length = table[length - 1]
        else:
            table[index] = 0
            index += 1

    return table


def kmp_search(text: str, pattern: str) -> list[int]:
    return run(kmp_search_traced(text, pattern))


def kmp_search_traced(text: str, pattern: str) -> Traced[list[int]]:
    """Knuth Morris Pratt. O(n + m) time, O(m) memory.

    The whole idea in one sentence: **the text pointer never moves backwards.**

    When a mismatch happens after matching k characters, the naive method throws
    away all k and restarts one position later. KMP asks the failure table how
    much of those k characters is still usable, slides the pattern forward by the
    difference, and carries on from the same place in the text.

    Because the text pointer only ever advances, the text is read exactly once,
    which is what makes it linear. It is also what makes KMP usable on a stream
    that cannot be rewound, which is a genuinely useful property the naive method
    does not have.
    """
    found: list[int] = []
    if not pattern:
        return list(range(len(text) + 1))

    table = build_failure_table(pattern)
    yield Step(
        "table",
        f"Built the failure table for {pattern!r}: {table}. Each entry says how "
        "many characters are still correct after a mismatch at that point.",
        {"table": table, "pattern": pattern},
    )

    matched = 0
    for position, character in enumerate(text):
        while matched > 0 and character != pattern[matched]:
            skipped = matched - table[matched - 1]
            yield Step(
                "slide",
                f"{character!r} does not continue the match, but the first "
                f"{table[matched - 1]} characters are still correct, so the pattern "
                f"slides forward {skipped} without the text pointer moving back.",
                {"position": position, "from": matched, "to": table[matched - 1]},
            )
            matched = table[matched - 1]

        if character == pattern[matched]:
            matched += 1

        if matched == len(pattern):
            start = position - len(pattern) + 1
            found.append(start)
            yield Step(
                "match",
                f"Full match found at position {start}.",
                {"start": start, "position": position},
            )
            matched = table[matched - 1]

    return found


def rabin_karp_search(
    text: str, pattern: str, base: int = 256, modulus: int = 1_000_000_007
) -> list[int]:
    return run(rabin_karp_search_traced(text, pattern, base, modulus))


def rabin_karp_search_traced(
    text: str, pattern: str, base: int = 256, modulus: int = 1_000_000_007
) -> Traced[list[int]]:
    """Compare hashes instead of strings, with a rolling hash. O(n + m) expected.

    Treat a window of text as a number in base 256, the way a string of digits is
    a number in base 10. Comparing two numbers is O(1), so if the numbers differ
    the strings certainly differ and the whole window is dismissed with one
    comparison.

    The trick that makes it work is the **rolling** part. Recomputing the hash of
    each window from scratch would be O(m) per position and no better than naive.
    Instead the next window's hash comes from the current one in constant time:
    subtract the character leaving on the left, multiply by the base to shift
    everything up, and add the character arriving on the right. That is the same
    move as "12345 to 23456" in decimal.

    Two honest caveats:

    * Hashes can collide, so **every hash match must be verified** by an actual
      character comparison. Skipping that gives an algorithm that is fast and
      occasionally wrong. With a large prime modulus collisions are rare, so the
      expected cost stays linear, but the worst case is O(n times m).
    * A poor modulus makes collisions common and the worst case likely. The value
      here is a large prime, which is the standard choice.

    Where it genuinely beats KMP is **many patterns at once**: hash all of them,
    keep the hashes in a set, and one pass over the text checks every pattern
    simultaneously. It also extends naturally to two dimensions for image
    matching, which KMP does not.
    """
    found: list[int] = []
    n, m = len(text), len(pattern)

    if m == 0:
        return list(range(n + 1))
    if m > n:
        return found

    # base^(m-1), used to remove the leaving character's contribution.
    high_order = pow(base, m - 1, modulus)

    pattern_hash = 0
    window_hash = 0
    for index in range(m):
        pattern_hash = (pattern_hash * base + ord(pattern[index])) % modulus
        window_hash = (window_hash * base + ord(text[index])) % modulus

    yield Step(
        "hash",
        f"The pattern hashes to {pattern_hash}. Any window with a different hash "
        "cannot possibly match, and is dismissed in one comparison.",
        {"pattern_hash": pattern_hash},
    )

    for start in range(n - m + 1):
        if window_hash == pattern_hash:
            # Verify. Hashes can collide, and an unverified match is a bug.
            if text[start : start + m] == pattern:
                found.append(start)
                yield Step("match", f"Hash matched at position {start} and the characters "
                                    "check out.", {"start": start})
            else:
                yield Step(
                    "collision",
                    f"The hash matched at position {start} but the characters do not. "
                    "This is a false positive, which is why every hit is verified.",
                    {"start": start},
                )
        else:
            yield Step(
                "skip",
                f"Window at {start} hashes to {window_hash}, which is not the pattern's, "
                "so it is dismissed without comparing any characters.",
                {"start": start, "hash": window_hash},
            )

        if start < n - m:
            # Roll: drop the leftmost character, shift up, add the new rightmost.
            window_hash = (window_hash - ord(text[start]) * high_order) % modulus
            window_hash = (window_hash * base + ord(text[start + m])) % modulus
            window_hash %= modulus

    return found


def z_array(text: str) -> list[int]:
    """For each position, how many characters from there match the start of the string.

    `z[i]` is the length of the longest substring starting at i that is also a
    prefix of the whole string. By convention z[0] is left as 0, since the whole
    string trivially matches itself.

    Computing this naively is O(n^2). The linear version keeps a window [left,
    right) which is the rightmost prefix match found so far, and reuses it: if
    position i falls inside that window, the answer at i is already known from the
    matching position nearer the front, and only needs extending if it reaches the
    window's edge.

    That reuse is the same instinct as KMP's failure table, which is why the two
    algorithms are close relatives. The Z array is the more general tool: it is
    used for pattern matching here, but also for finding periods, longest common
    prefixes and string compression.
    """
    n = len(text)
    z = [0] * n
    left = right = 0

    for index in range(1, n):
        if index < right:
            # Inside the known window, so start from what the mirrored position
            # already told us, without exceeding the window.
            z[index] = min(right - index, z[index - left])

        while index + z[index] < n and text[z[index]] == text[index + z[index]]:
            z[index] += 1

        if index + z[index] > right:
            left, right = index, index + z[index]

    return z


def z_search(text: str, pattern: str) -> list[int]:
    """Find every occurrence using the Z array. O(n + m).

    The trick is to search a single combined string: pattern, then a separator
    that appears in neither, then the text. Any position in the text part whose Z
    value equals the pattern length is a match, because that many characters there
    agree with the start of the combined string, which is the pattern.

    The separator is essential. Without it a Z value could run past the end of the
    pattern and into the text, reporting matches that are not there.
    """
    if not pattern:
        return list(range(len(text) + 1))

    separator = "\x00"
    if separator in text or separator in pattern:
        separator = max(chr(0x10FFFE), max(text + pattern)) if text + pattern else "\x01"

    combined = pattern + separator + text
    values = z_array(combined)
    offset = len(pattern) + 1

    return [
        index - offset
        for index in range(offset, len(combined))
        if values[index] == len(pattern)
    ]


def longest_prefix_suffix(text: str) -> int:
    """The longest proper prefix of `text` that is also a suffix of it.

    Falls straight out of the KMP failure table, and it answers a question that
    turns up on its own: the shortest repeating unit of a string is its length
    minus this value, when that divides evenly. For "abcabcabc" the answer is 6,
    so the repeating unit is 9 - 6 = 3 characters, which is "abc".
    """
    if not text:
        return 0
    return build_failure_table(text)[-1]


def is_rotation(first: str, second: str) -> bool:
    """Whether one string is a rotation of the other, in O(n).

    The neat observation: every rotation of a string appears inside that string
    doubled. "waterbottle" doubled contains "erbottlewat". So this is one string
    search rather than n comparisons, which is a good example of turning a
    problem into one you have already solved.
    """
    return len(first) == len(second) and bool(kmp_search(first + first, second))
