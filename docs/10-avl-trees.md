# Day 10: AVL trees

Code: `dsalab/structures/avl.py`. Tests: `tests/test_avl.py`.

## The rule

One sentence on top of the search tree rule:

> For every node, the heights of its two subtrees differ by at most 1.

The difference is called the **balance factor**, defined here as
height(left) - height(right), so it must always be -1, 0 or +1. When an insertion
or deletion pushes it to -2 or +2, the tree repairs itself with a rotation before
returning.

## Why that rule guarantees a logarithmic height

This is worth deriving rather than accepting, because the derivation is short and
it explains where the famous 1.44 comes from.

Ask the opposite question. What is the **fewest** nodes an AVL tree of height h
can have? Call it N(h). The root has two subtrees, and to be as sparse as
possible one has height h-1 and the other h-2, because anything more lopsided
would break the rule. So:

```
N(h) = 1 + N(h-1) + N(h-2)
```

That is the Fibonacci recurrence. Fibonacci numbers grow roughly as 1.618^h, so
the minimum number of nodes grows **exponentially** in the height. Turn it around
and the height grows only **logarithmically** in the number of nodes, specifically
at most about 1.44 times log2(n).

So even the worst possible AVL tree is within 44 percent of a perfect one. There
is a test asserting the bound at eight different sizes, and another that hands the
measured heights to the complexity detective and gets back O(log n) with
confidence, next to the day 9 measurement that got back O(n) for the same input.

## The four rotation cases

| Case | Situation | Fix |
| - | - | - |
| Left left | Too tall on the left, left child leans left | one right rotation |
| Right right | Too tall on the right, right child leans right | one left rotation |
| Left right | Too tall on the left, left child leans right | rotate the child left, then rotate right |
| Right left | Too tall on the right, right child leans left | rotate the child right, then rotate left |

The two single cases are easy. The two double cases are where people go wrong, so
here is why the single rotation is not enough for them:

**A single rotation lifts the child's outer subtree.** In the left right case the
extra depth is in the child's *inner* subtree, so a single right rotation leaves
that subtree exactly as deep as it was and simply moves the imbalance to the
other side. Rotating the child left first turns the inner subtree into an outer
one, which converts the problem into a left left case that the single rotation
does fix.

### Why rotations preserve the ordering

```
      y                x
     / \              / \
    x   C    -->     A   y
   / \                  / \
  A   B                B   C
```

Only one subtree changes hands: B moves from x's right to y's left. That is legal
because everything in B is larger than x and smaller than y, which is exactly the
range that y's left subtree is allowed to hold.

The one line proof: read both diagrams left to right and the values appear as
`A x B y C` in both. Same inorder traversal, so the search tree property is
untouched. Any rotation you can draw this way is safe.

### One detail that causes silent corruption

Inside a rotation the two heights must be recomputed **bottom up**: the node that
moved down first, then the node that moved up. Do it the other way round and the
second calculation uses a stale value.

What makes this dangerous is that nothing visibly breaks. The tree still holds the
right values in the right order, so every lookup and every traversal is correct.
Only the balance decisions afterwards are wrong, and they slowly let the tree
degrade. There is a test that walks the whole tree after five hundred insertions
recomputing every height from scratch and comparing, because that is the only way
to see this.

## Insertion and deletion, and where recursion earns its place

Insertion is the clearest example in the project of day 2's point that a
recursive function has two halves:

* **On the way down**, it finds where the value belongs. This is the plain search
  tree behaviour.
* **On the way back up**, it recomputes each ancestor's height and rebalances if
  needed.

The upward half is only possible because the return trip visits precisely the
nodes whose subtrees changed, in order from the insertion point up to the root.
An iterative version would have to record that path explicitly. Here recursion
provides it for free.

**Insertion needs at most one rotation** (or one double rotation), because a
single rotation restores the subtree to the height it had before the insertion,
so nothing above it can still be out of balance.

**Deletion may need a rotation at every level.** Removing a node makes a subtree
shorter, and unlike insertion, rebalancing does not necessarily restore the
previous height, so the imbalance can propagate all the way to the root. It is
still O(log n), but it is why deletion is the more expensive operation.

## AVL against red black, which is tomorrow

| | AVL | Red black |
| - | - | - |
| Height bound | 1.44 log2(n) | 2 log2(n) |
| Lookups | slightly faster, tree is shorter | slightly slower |
| Insert rotations | at most 1 | at most 2 |
| Delete rotations | up to log n | at most 3 |
| Suits | read heavy work | write heavy work |

That is the entire practical difference, and it is why Java's `TreeMap`, C++'s
`std::map` and the Linux kernel's scheduler all use red black trees, while
databases and in memory indexes that are read far more often than written often
prefer AVL.

## What the tests are actually doing

Two different jobs, and both are needed:

**Hand built cases.** Four tests, one per rotation case, each with an insertion
order chosen to trigger exactly that case, asserting the resulting shape and the
number of rotations. These pin down the behaviour that is easy to get subtly
wrong.

**Random stress with checking on.** Two thousand mixed insertions and deletions
against a plain Python set, with invariant checking switched on so every single
operation verifies the search order, the stored heights, the balance factors and
the size. This is what finds the cases nobody thought of, and because the
invariant check runs inside the operation, a failure points at the operation that
caused it rather than at some later symptom.
