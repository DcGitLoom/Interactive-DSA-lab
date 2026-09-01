# Day 17: Tries, range queries and union find

Code: `dsalab/structures/trie.py`, `segment_tree.py`, `union_find.py`.
Tests: `tests/test_trie_and_ranges.py`.

Three structures that have nothing in common except that each is the only good
answer to a question the earlier structures cannot handle at all.

## Tries: when the key has parts

Every structure so far treats a key as one indivisible thing to compare or hash. A
trie takes keys apart: each edge is one character, and a word is a path from the
root.

Two consequences:

**Lookup cost depends on the key length, not the collection size.** Finding a ten
letter word takes ten steps whether the trie holds a hundred words or ten million.
There is a test that measures exactly this, comparing the step count for the same
lookup in a one word trie and a five thousand word trie, and they are identical.

**Prefixes come for free.** Every word starting with "car" lives under the node you
reach by walking c, a, r. A hash table cannot do this at all, because hashing
destroys any relationship between similar keys. That is why tries are behind
autocomplete, spell checking and IP routing tables.

A third property falls out unasked: **a trie sorts for free.** Walking children in
alphabetical order produces the words in order with no sorting step, which a hash
table cannot do either.

### The two ways deletion goes wrong

Removing a word means removing the nodes nobody needs any more, and there are
opposite failures:

* **Delete too little** and the trie fills with dead paths, wasting memory and
  slowing every prefix walk.
* **Delete too much** and you break other words. Removing "carpet" must not remove
  the nodes for "car", and removing "car" must not break the path "carpet" travels
  through.

The rule that gets it right: a node can be removed only when it has no children
**and** is not itself the end of another word. Walk back up from the deleted word
and stop at the first node that fails either test. There is a test for each
failure direction.

### The compressed trie, and where its bugs live

A plain trie storing "carpet" uses five nodes that have exactly one child each and
carry no branching information. Collapsing them into a single edge labelled
"carpet" stores the same thing in one node. On a realistic word list the saving is
large: the tests measure 300 random words and the compressed version uses less than
half the nodes, and a single twelve letter word needs 2 nodes instead of 13.

The cost is a case that does not exist in a plain trie. Inserting a word that
shares part of an edge means **splitting that edge**: the shared part stays and the
two different tails become separate children. Inserting "cartoon" into a trie
holding "carpet" splits the edge into "car" plus "pet" and "toon".

That is where every compressed trie bug lives, so the tests aim at it directly:
splitting an edge, inserting a word that **ends inside** an edge ("car" into
"carpet"), repeated splits of the same edge, and inserting the same pair of words
in both orders to check the result is identical either way.

## Segment trees and Fenwick trees: questions about ranges

The problem: answer many queries like "sum of positions 3 to 17" while the array is
also changing. The two obvious approaches each fail at one half:

| Approach | Query | Update |
| - | - | - |
| Scan the range | O(n) | O(1) |
| Keep prefix sums | O(1) | O(n), every prefix after the change is stale |
| Segment or Fenwick tree | O(log n) | O(log n) |

Both structures work by storing partial answers for **nested blocks** rather than
for single positions or for the whole array.

### Segment tree

Each node holds the answer for one range: the root covers everything, its children
the two halves, down to leaves covering one position. Any query breaks into at most
2 log n prebuilt blocks. There is a test confirming a query over 1024 items uses at
most twenty blocks.

The query has three cases, and the middle one is the whole algorithm:

* No overlap: contribute nothing.
* **Fully inside: return the stored answer without going deeper.**
* Partial overlap: ask both children.

Why the array is 4n and not 2n: the tree is only perfectly balanced when n is a
power of two, and for other n the recursion can go one level deeper. 2n works for
many inputs and then quietly writes out of bounds on others, which is a horrible
bug to chase, so the generous bound is standard.

The requirement on the combining operation is **associativity, not commutativity**.
The tree always combines neighbouring ranges in order, so matrix multiplication is
fine while subtraction is not.

### Lazy propagation

Adding 5 to positions 3 through 100000 as individual updates is 99998 operations.
The fix is one idea: **when a range update covers a node completely, do not push it
down. Leave a note and stop.** The note says "everything below me has this much
added, I have not told them yet", and it is only pushed one level when something
needs to look inside.

Work is deferred until it is unavoidable, and if nobody queries inside that range
it is never done at all. This is the same instinct as the tombstones on day 14 and
the amortised copying on day 4: **do the expensive thing only when someone forces
you to.** A test adds to all 4096 positions and asserts at most two blocks were
touched.

### Fenwick tree, and the neatest bit trick in the project

A Fenwick tree does prefix sums in a third of the code and a quarter of the memory,
using one idea about binary numbers:

**Slot i stores the sum of a block ending at i, whose length is the lowest set bit
of i.** Slot 12 (binary 1100, lowest set bit 4) holds positions 9 to 12. Slot 8
(1000) holds positions 1 to 8.

So a prefix sum is assembled by repeatedly stripping the lowest set bit: 13 = 1101
reads slots 13, 12 and 8, and those three blocks tile 1 to 13 exactly. The number
of steps is the number of set bits, at most log n. There is a test asserting those
exact three slots are read.

`i & -i` extracts the lowest set bit. In two's complement `-i` is `~i + 1`, which
inverts everything above the lowest set bit and leaves everything below it zero, so
the AND leaves exactly that bit standing.

The limitation is real: **range sums are computed as a difference of two prefix
sums**, so the operation must have an inverse. A Fenwick tree cannot answer a range
minimum, because there is no way to subtract one minimum from another. That is the
practical dividing line between the two structures.

| | Segment tree | Fenwick tree |
| - | - | - |
| Operations | any associative one | ones with an inverse, in practice sums |
| Range updates | yes, with lazy propagation | not directly |
| Memory | 4n | n + 1 |
| Code | about 80 lines | about 20 |

### The check that was worthless, and what it taught me

I wrote a `check_invariants` for the Fenwick tree that verified every slot held the
sum of its block. It looked thorough. It was worthless: it derived the array using
`to_list()`, which reads the tree, then checked the tree against it. Corrupting a
slot changed both sides equally and no violation ever appeared.

**A self referential check is not a check**, and the deeper reason is worth
knowing: a Fenwick tree carries **no redundancy**. Every value is stored once,
spread across slots, so any array of slot values is a valid Fenwick tree of some
array. There is nothing to contradict. A segment tree can check itself because a
parent must equal its two children combined, and that redundancy is exactly what a
consistency check needs.

So the method now verifies only what is genuinely independent (the array length,
and that no slot describes a block running off the front), and there is a test
asserting the corruption is undetectable, with the reason written down. The
contents are checked instead by cross checking every range sum against a segment
tree built separately from the same values.

The general lesson: **before writing a consistency check, ask what redundancy it
can compare against.** If there is none, the check is theatre.

## Union find, and the algorithm with an absurd complexity

The problem: given statements like "a and b are connected", answer "are x and y
connected". This is Kruskal's cycle test, undirected cycle detection, percolation,
image segmentation and type unification in a compiler.

Keep each group as a tree where every node points at its parent and the root names
the group. Merging is then a single pointer assignment. Two one line optimisations
turn that from good to essentially free:

**Union by size**: hang the smaller tree under the larger. Without it, merging in a
bad order builds a chain of length n. There is a test that constructs exactly that
order and measures depth: the naive version exceeds 100, the optimised one stays at
2.

**Path compression**: after finding a root, point every node passed straight at it.
A test builds a 99 deep chain by hand, finds the deepest node once for 99 hops, and
finds it again for 1.

Together the amortised cost is O(alpha(n)), where alpha is the inverse of the
Ackermann function from day 2. **That value is at most 4 for any n that fits in the
universe**, so each operation is constant time in practice. The proof is famously
difficult and the code is fifteen lines, which is a combination you rarely see.

Union by **size** is used rather than by rank because they give the same guarantee
and size answers a question people actually ask, while rank is an internal number
that means nothing outside the algorithm.

`UnionFindWithoutOptimisations` is kept deliberately so the improvements can be
measured rather than claimed. Keeping a bad implementation around as a measuring
stick is worth doing: it is far more convincing than a comment, and it keeps the
benchmark honest.

The `union` return value matters beyond this file: it is False exactly when the two
items were already connected, which is precisely the cycle test Kruskal's algorithm
needs tomorrow.
