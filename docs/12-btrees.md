# Day 12: B-trees and 2-3 trees

Code: `dsalab/structures/btree.py`. Tests: `tests/test_btree.py`.

## Why binary is the wrong shape for disk

Every tree so far has been binary, which is right when the data is in memory. It
is wrong when the data is on a disk, and the reason explains this whole structure.

A disk does not read a byte at a time. The hardware reads a whole block, typically
four or eight kilobytes, and the cost is dominated by *finding* the block rather
than by its size. **Reading one byte and reading four kilobytes cost about the
same.**

Put a balanced binary tree of a million records on disk. Its height is around 20,
so a lookup is 20 block reads, and each read hands you a block containing one key
and two pointers, wasting the other four kilobytes entirely.

A B-tree fills the block. Give each node 200 keys and 201 children, and a million
records fit in a tree of height 3. **Three reads instead of twenty.** The
comparisons inside a node are effectively free, because the block is already in
memory.

That is the entire idea: when the expensive operation is fetching a node, make
nodes as big as a fetch.

| Order | Height for 10,000 keys | Height for 1,000,000 keys |
| - | - | - |
| 2 (binary) | about 14 | about 20 |
| 8 | 4 | 7 |
| 128 | 2 | 3 |

There is a test asserting the order 128 tree holding ten thousand keys is at most
three levels deep.

## The rules

For a B-tree of order m:

* At most m - 1 keys and m children per node.
* At least ceil(m/2) - 1 keys in every node except the root. **This lower bound is
  the half people forget**, and it is what keeps the tree dense enough for the
  height to stay at log_m(n). Without it, a million keys could legally sit in a
  chain of nodes holding one key each.
* A node with k keys has exactly k + 1 children, unless it is a leaf.
* Keys within a node are sorted, and the subtree between two keys holds values
  between them.
* All leaves are at the same depth.

A **2-3 tree is exactly a B-tree of order 3**: one or two keys per node, two or
three children. It is not implemented separately, because there is nothing to
implement. `two_three_tree()` sets the order to 3 and the tests treat it as the
special case it is.

## The tree grows upwards

This is the genuinely unusual part. Every other structure in this project grows
at the leaves. A B-tree grows at the **root**.

When a node overflows it splits into two halves and pushes its middle key up to
the parent. When the root itself splits, a brand new root is created holding one
key, and the whole tree gains a level.

That is why every leaf stays at the same depth without any rebalancing: no path
ever gets longer on its own, they all get longer together. Deletion mirrors it
exactly, and a merge that empties the root is the only way the tree ever gets
shorter.

## The bug that taught me the most this week

I wrote insertion the way CLRS does it: **preemptive splitting**. On the way down,
split every full node you pass, so the leaf you reach is guaranteed to have room.
One pass, each node touched exactly once, which is exactly what you want for a
tree on disk.

The tests failed at orders 3, 5 and 13, and passed at 4, 6 and 8.

That pattern is the clue. Preemptive splitting splits a node holding the
**maximum** of m-1 keys into two halves plus a promoted middle key. For both
halves to meet the minimum of ceil(m/2)-1, the maximum has to be **odd**, which
means the order has to be **even**.

Work it through at order 5: the maximum is 4 keys, the minimum is 2. A full node
of 4 keys splits as 2 + promoted + 1, and that trailing node has one key where two
are required. The scheme cannot work at odd orders, and no amount of careful
coding fixes it.

The same constraint applies to deletion. Repairing on the way down means merging
two minimum sized nodes plus a separator, which is 2 x (ceil(m/2)-1) + 1 keys, and
that only fits inside a node at even orders. Repairing on the way back up merges a
node that is one **below** minimum with a minimum sibling, giving 2 x (ceil(m/2)-1)
keys, which fits at every order.

Since a 2-3 tree is order 3, supporting odd orders was not optional, so both
insertion and deletion now work bottom up: insert, and split if the node
overflows; delete, and repair if the node falls short. The cost is that a split or
a merge can cascade, so a node may be touched twice.

**The lesson: the elegant one pass version has a precondition nobody states out
loud.** Half the textbook presentations of B-trees quietly assume an even order
throughout and never mention it. The tests found this because they run every
structural check at six different orders, odd and even, rather than picking one
convenient order and trusting it.

## Deletion: borrow before merge

An underfull node has two possible repairs:

**Borrow** from a sibling with a key to spare. The sibling's nearest key moves up
into the parent, and the parent's separating key comes down into the short node.
It is a rotation. It touches three nodes, leaves the height alone, and never
propagates, so it is always preferred.

**Merge** when neither neighbour can spare a key. The two nodes and the parent's
separating key combine into one. This removes a key from the parent, which may
leave it short too, so a merge can cascade all the way up, and if the root empties
the tree loses a level.

Deleting a key from an **internal** node uses the same trick as the binary search
tree on day 9: an internal key is a separator between two subtrees and cannot
simply be removed, so it is replaced by its predecessor, which lives in a leaf, and
that leaf key is deleted instead.

## Cost summary

| Operation | Comparisons | Node fetches |
| - | - | - |
| search | O(m log_m n) | O(log_m n) |
| insert | O(m log_m n) | O(log_m n) |
| delete | O(m log_m n) | O(log_m n) |
| traversal | O(n) | O(n / m) |

Both columns are logarithmic, and in memory the left one is what you feel. On disk
only the right one matters, and it is smaller by a factor of log2(m). That gap is
the entire reason B-trees run essentially every database index and filesystem in
existence.

## What B+ trees change

Worth a paragraph, since real databases use the variant rather than the plain form.

In a B+ tree, **all the data lives in the leaves** and the internal nodes hold only
separator keys, with the leaves linked together in a chain. Two consequences:
internal nodes hold more keys because they carry no data, making the tree even
shallower, and a range scan becomes "find the start, then walk the leaf chain",
with no need to climb back up through the tree. Range scans are most of what a
database index does, so that second point is decisive.
