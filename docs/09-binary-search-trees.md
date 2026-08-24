# Day 9: Binary search trees, and structures that check themselves

Code: `dsalab/structures/bst.py`, `dsalab/invariants.py`. Tests: `tests/test_bst.py`.

## One rule, and everything follows

For every node: everything in the left subtree is smaller, everything in the
right subtree is larger.

That rule alone gives search, insertion and deletion in time proportional to the
**height** of the tree, because each comparison throws away an entire subtree. It
is binary search made structural.

## The honest weakness

Notice that every cost above is "proportional to the height", and **nothing in
this structure controls the height**.

Insert values in sorted order and every new node goes to the right of the last
one. The result is a chain of height n: a linked list wearing a tree costume,
with worse memory overhead and no advantage at all.

Sorted input is not exotic. It is one of the most common inputs there is:
importing an ordered file, inserting timestamps, replaying a log, loading
primary keys from a database. A plain search tree is genuinely dangerous in
production for this reason.

Rather than write that as a warning, the tests measure it. `test_the_height_of_a
_sorted_insertion_grows_linearly` builds trees from sorted input at five sizes,
hands the heights to the complexity detective from day 3, and gets back O(n) with
confidence. The companion test does the same with shuffled input and gets back
something logarithmic. Random insertion order gives an expected height of about
4.3 times log2(n), which is a constant factor from ideal and therefore still
O(log n).

| Operation | Balanced tree | Degenerate tree |
| - | - | - |
| search, insert, delete | O(log n) | O(n) |
| minimum, maximum | O(log n) | O(n) |
| floor, ceiling | O(log n) | O(n) |
| range query returning k values | O(log n + k) | O(n) |
| inorder traversal | O(n) | O(n) |

Day 10 makes the left hand column a guarantee.

## Why a tree rather than a hash table

A hash table beats a balanced tree on plain lookup: O(1) against O(log n). If
exact key lookup is all you need, use the hash table.

A tree wins where **order** matters, and these operations are the reason it
exists:

* `minimum` and `maximum`.
* `floor(x)`, the largest value not above x, and `ceiling(x)`, the smallest not
  below. Useful whenever you have a measurement and want the nearest known point.
* `range_query(low, high)`, every value in a range, in sorted order.
* Ordered iteration, straight out of an inorder traversal.

A hash table cannot do any of these without looking at every key, because hashing
deliberately destroys the ordering. The range query here costs O(log n + k) thanks
to pruning: if a node's value is already below `low`, its entire left subtree is
too small and is never entered.

## Deletion, the only fiddly operation

Three cases:

1. **A leaf.** Detach it.
2. **One child.** The child takes the removed node's place. Everything in that
   subtree is already on the right side of the parent, so the ordering holds.
3. **Two children.** The node cannot just be removed, because its parent has one
   link and there are two subtrees to rehome. Instead the node's value is
   replaced by its **inorder successor**, the smallest value in the right
   subtree, and the successor is then deleted from below.

Why the successor is the only sensible choice: it is larger than everything in
the left subtree (it lives in the right one) and smaller than everything else in
the right subtree (it is that subtree's minimum). It is therefore, along with the
predecessor, the only value that can sit in that position without breaking the
ordering.

And it terminates: the successor is the leftmost node of the right subtree, so it
has **no left child**, so removing it is case 1 or case 2. The two child case can
never trigger another two child case.

The usual wrong version promotes the right child directly rather than that
subtree's leftmost descendant. It works whenever the right child has no left
child of its own, which is often enough to pass a careless test suite, so there
are two tests here covering both shapes.

## Invariant checking

This is the third of the four features that lift the project above a visualiser,
and building it here made everything afterwards easier.

An invariant is something that must be true about a structure at all times. They
are usually written in comments and enforced by hope. Here a structure states its
rules in code:

```python
with checked():
    tree.insert(5)      # the tree verifies itself after this line
    tree.delete(3)      # and this one
```

Inside a `checked()` block, every mutation is followed by a full verification, and
a broken rule raises immediately with the rule named in plain English and the
offending node identified.

Two things this buys:

**Tests that check the structure, not the answers.** A tree can return every
correct answer while being internally corrupt, with the damage only surfacing
several operations later somewhere unrelated. The strongest test in this file
deletes forty values one at a time in random order and verifies the whole tree
after each one, so a bad deletion is caught at that deletion.

**A teaching tool.** The app can list the rules beside the tree and mark which
hold. Watching a red black tree break rule 4 and then repair it with a rotation
explains rotations better than any prose description.

Checking is off by default and switched on by a context manager, because some of
these checks are O(n) and would turn an O(log n) insert into an O(n) one if they
ran always. The block restores the previous setting even when an exception is
raised, so a failing test cannot leave checking switched on for everything after
it, and there is a test for that specifically.

### The subtle part of checking a search tree

The obvious implementation compares each node against its immediate children.
**That check is wrong**, and it is wrong in a way that passes casual inspection.

A value can be on the correct side of its parent and the wrong side of its
grandparent. Put 60 as the right child of 40, where 40 is the left child of 50:
60 is correctly larger than 40, and it sits in the left subtree of 50 where
everything must be below 50. Checking parent against child alone sees nothing
wrong.

The correct check carries a **running range** down the tree. Every node inherits a
lower and upper bound from the path taken to reach it, and must fall inside them.
There is a test built on exactly the 40 and 60 arrangement above, because it is
the case that separates the right check from the plausible one.

## A design decision: duplicates are refused

`insert` returns False for a value already present rather than storing it twice.
That is a choice, not the only option: some trees keep a count per node, and some
allow equal values on one designated side.

Refusing keeps the invariant strict, left is *strictly* smaller and right is
*strictly* larger, which makes both the deletion logic and the invariant check
simpler and unambiguous. It also matches how a set behaves, which is what people
usually want from this structure. If duplicates mattered, a count per node would
be the right change, and it would not disturb anything else.

## Why the traversals here are iterative

`BinaryTree` on day 8 offers recursive and iterative traversals. `BinarySearchTree`
uses the iterative ones for `inorder` and `height`, and that is deliberate: the
failure mode of this structure is a degenerate tree, and a degenerate tree of ten
thousand nodes is ten thousand levels deep, well past Python's recursion limit.
Using the recursive version here would mean the structure crashes precisely in the
case it is most likely to end up in. There is a test that builds a 5000 node chain
and traverses it, which would fail immediately with recursion.
