# Day 16: Binary search, its variants, and string matching

Code: `dsalab/algorithms/searching.py`, `dsalab/algorithms/strings.py`.
Tests: `tests/test_searching.py`.

## Binary search is harder than it looks

Jon Bentley found that around ninety percent of professional programmers could not
write a correct binary search when asked, and the version in Java's standard
library carried an overflow bug for nine years before anyone noticed. Both classic
mistakes are worth naming:

**The overflow.** `(low + high) // 2` overflows when both are large, in any
language with fixed width integers. `low + (high - low) // 2` cannot. Python's
integers are unbounded so it cannot bite here, and the correct form is used anyway
because the habit is what transfers to other languages.

**The boundary.** Whether the loop is `while low <= high` or `while low < high`,
and whether the update is `middle - 1` or `middle`, decides both correctness and
termination. Off by one and you either miss the answer or loop forever. There is a
test that runs every array size from 0 to 30 against every possible target
position, purely to prove the loop always terminates.

One requirement people forget: **the input must already be sorted.** Binary search
on unsorted data does not fail loudly, it quietly returns wrong answers, which is
much worse. And if you sort just to search once, you have spent O(n log n) to save
O(n) and a linear scan was better. Binary search pays off when you search the same
data many times.

## The variants matter more than the plain search

Real questions are rarely "is this value present". They are "where would it go",
"what is the first entry after this timestamp", "how many are below this
threshold".

| Function | Answers | Python equivalent |
| - | - | - |
| `binary_search` | where is this exact value | none directly |
| `lower_bound` | first position not below the target | `bisect_left` |
| `upper_bound` | first position above the target | `bisect_right` |
| `count_occurrences` | how many times it appears, in O(log n) | none |
| `first_true` | where a monotonic predicate turns true | the general case |

`lower_bound` and `upper_bound` differ by a single character, `<` against `<=`,
and that character decides which end of a run of equal values you land on.
Together they give the full range of every occurrence of a value, so counting
becomes a subtraction rather than a scan.

Both use `while low < high` with `high = middle`, not `middle - 1`, because **the
answer can legitimately be the position after the last element**. That is why
`high` starts at `len(values)` rather than `len(values) - 1`, and there is a test
for exactly that case.

### The most useful reframing in this file

`first_true` is the general form that all of the above are special cases of. It
needs only that the predicate is **monotonic**: once true, it stays true.

Thinking of binary search as "find the boundary of a monotonic predicate" rather
than "find a value in a sorted array" is what makes it usable on problems with no
array in them at all:

* The smallest truck capacity that fits a set of loads into three trips.
* The first software version where a test starts failing, which is what
  `git bisect` does.
* The minimum speed that finishes a journey in time.

There is a test that binary searches over truck capacities, where nothing is
sorted and there is no list of answers to look in. That technique, binary
searching the answer space, is worth more than the original algorithm.

## The other searches, and what they cost

**Exponential search** doubles to find a bound, then binary searches inside it.
O(log i) where i is the position of the answer rather than O(log n) of the whole
collection, and it is the only option when the collection has **no known length**,
such as a stream or a paginated API. The doubling is the same idea as the dynamic
array's growth on day 4.

**Interpolation search** guesses where the value should be rather than always
splitting in the middle. Looking for 950 in a list from 1 to 1000, the answer is
obviously near the end. O(log log n) on evenly spread data, about five probes for
a million items against twenty for binary search, and **O(n) on skewed data**
because a bad guess barely narrows the range. It is a gamble on the shape of the
data in the same way bucket sort is, and binary search's guarantee is usually
worth more than a better average.

**Ternary search** finds the peak of a function that rises then falls. Binary
search needs a monotonic predicate, which a peak is not, but the same halving
instinct works with a different comparison: take two points a third of the way in
from each end, and whichever side is lower cannot contain the peak. Each round
removes a third of the range.

**Linear search** is here as the honest baseline. It beats binary search more
often than people expect, because it needs no sorting and walks memory in order.

## String matching, three ways

The naive method tries the pattern at every position and is O(n times m) on bad
input. The bad input is not exotic: "aaaab" inside a long run of "aaaa..." matches
four characters at every position before failing on the fifth.

All three real algorithms are O(n + m) and each avoids the waste differently,
which is why they are worth learning together.

### KMP: never move the text pointer backwards

The failure table records, for each prefix of the pattern, the length of the
longest proper prefix that is also a suffix. In plain English: **if you have
matched k characters and the next one fails, how many are still correct?**

For "abcabd" the table is `[0, 0, 0, 1, 2, 0]`. At index 4 it is 2, because
"abcab" ends with "ab", which is also how the pattern starts. So a failure after
matching "abcab" does not start over, two characters are already right.

The consequence is that the text pointer only ever advances, so the text is read
exactly once. That also makes KMP usable on a **stream that cannot be rewound**,
which the naive method cannot do at all. There is a test asserting the text
positions in the trace are monotonically increasing, which is the property stated
directly.

### Rabin Karp: compare numbers, not strings

Treat a window of text as a number in base 256. Comparing two numbers is O(1), so
a window whose hash differs is dismissed in one comparison without looking at a
single character. There is a test showing four hundred windows dismissed this way.

The rolling part is what makes it work: the next window's hash comes from the
current one in constant time, by subtracting the departing character, shifting up,
and adding the arriving one. The same move as going from 12345 to 23456 in
decimal.

**Every hash match must be verified against the actual characters.** Hashes
collide, and skipping the check gives an algorithm that is fast and occasionally
wrong. There is a test that runs it with a deliberately terrible modulus of 7, so
false positives are frequent, and asserts both that collisions occur and that the
answers stay exactly right.

Where it genuinely beats KMP is **many patterns at once**: hash all of them into a
set and one pass over the text checks every pattern simultaneously. It also
extends to two dimensions for image matching, which KMP does not.

### The Z array: the general tool

`z[i]` is how many characters starting at position i match the start of the
string. Computing it naively is O(n^2); the linear version keeps a window of the
rightmost prefix match found so far and reuses it, which is the same reuse
instinct as KMP's table.

Searching with it is a trick worth knowing: build `pattern + separator + text` and
any position whose Z value equals the pattern length is a match. **The separator is
essential**, or a match could run past the end of the pattern into the text.

The Z array is more generally useful than either of the others: periods, longest
common prefixes and several compression algorithms all fall out of it.

## Two tests worth pointing at

**Overlapping matches.** "aa" occurs three times in "aaaa", not two. An
implementation that skips ahead by the pattern length after a match gets this
wrong, and all four algorithms are tested against it.

**The empty pattern.** Whether it matches everywhere or nowhere is a convention
rather than a truth, so all four are pinned to the same one, matching what
Python's own `str.find` does.

And the general shape of the string tests: the three clever algorithms are all
cross checked against the **naive one**, on three hundred random text and pattern
pairs. The naive version is slow and obviously correct, which makes it exactly the
right reference. Testing a clever algorithm against a simple one you trust is
usually better than testing it against hand written expected values, because it
covers cases you would never think to write down.
