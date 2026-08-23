# Day 8: Binary trees, traversals, and threaded trees

Code: `dsalab/structures/binary_tree.py`, `dsalab/structures/threaded_tree.py`.
Tests: `tests/test_binary_tree.py`.

## Why trees at all

A binary tree is a node with a value and up to two children. The reason this
shape matters is arithmetic: a tree of height h holds at most 2^(h+1) - 1 nodes,
which turned around says n nodes can be arranged in a tree of height about
log2(n).

That is the entire promise. A search that discards half the remaining
possibilities at each step finishes in about log2(n) steps: twenty comparisons
for a million items instead of a million.

The promise has a catch that day 10 exists to fix. **Nothing here forces a tree to
be short.** Insert values in increasing order into a plain search tree and you get
a chain of height n, which is a linked list wearing a tree costume, and every
operation goes back to O(n). There are tests for exactly that shape, because a
degenerate tree is where traversal code with an explicit stack usually breaks.

## The four traversals

| Traversal | Order | Uses it for |
| - | - | - |
| Preorder | node, left, right | Copying a tree, writing it to a file, prefix expressions |
| Inorder | left, node, right | Sorted output from a search tree |
| Postorder | left, right, node | Freeing a tree, computing sizes bottom up, evaluating expression trees |
| Level order | row by row | Shortest path in an unweighted graph, printing a tree by level |

Two facts are worth memorising:

* **Preorder's first value is always the root, and postorder's last value always
  is.** That is what makes tree rebuilding possible.
* **Inorder on a binary search tree comes out sorted.** This single fact is why
  search trees are more than fast lookup: ranges, ordered iteration and "the next
  value after this one" all fall out of it.

Level order is the odd one out. The other three are depth first and use a stack,
whether an explicit one or the call stack that recursion provides. Level order is
breadth first and uses a queue, and it has no natural recursive form at all. This
is the clearest demonstration of the day 7 point: **the choice of stack or queue
is what decides whether a search goes deep or wide.**

`levels()` groups the output one list per row using a small trick worth knowing:
record how many nodes are in the queue before starting a row. That count is
exactly the width of the current level, because every node of the next level is
added behind them, so no depth information needs to be stored on the nodes.

## Recursive and iterative, side by side

Every traversal is written twice. The recursive version is what you should write:
it is three lines and obviously correct. The iterative version matters when the
tree could be deep enough to overflow the call stack, which in Python is around a
thousand frames.

The pairs also make a point that is easy to miss: **recursion is not magic.** The
iterative versions do exactly what the recursive ones do, with the invisible call
stack replaced by a stack you can see.

Two details that catch people out:

**Iterative preorder pushes the right child before the left.** That looks
backwards until you remember a stack reverses things. Push left first and you get
a mirror image of the correct answer.

**Iterative postorder is genuinely awkward**, because a node must be visited after
both subtrees, so you cannot tell on arrival whether it is ready. The version here
sidesteps it: run a preorder that takes the right child first, producing
node-right-left, then reverse the result. Reversed, that is left-right-node, which
is postorder exactly. The honest alternative needs two stacks or a "last visited"
pointer and is much more code.

The tests cross check every recursive version against its iterative twin on a
hundred randomly shaped trees. They are two implementations of one definition, so
any disagreement means one is wrong, and random trees find cases neither was
written with in mind.

## Rebuilding a tree from its traversals

Preorder gives you the root. Finding that value in the inorder list splits it in
two: everything before is the left subtree, everything after is the right. The
sizes of those halves say how much of the preorder list belongs to each side, and
the same reasoning recurses.

Three things worth knowing about this:

1. **One traversal is not enough.** Many different trees share a preorder.
2. **Preorder plus inorder works. Postorder plus inorder works. Preorder plus
   postorder does not.** Those two cannot tell whether a lone child is a left or a
   right child, so the tree is not uniquely determined.
3. **It needs distinct values.** With repeats, the split point is ambiguous and
   more than one tree fits. The implementation refuses rather than silently
   picking one, and there is a test asserting the refusal, because quietly
   choosing among several valid answers is far worse than saying no.

The implementation is O(n) using a lookup table of inorder positions. Searching
the inorder list on each call, which is the version most people write first, is
O(n^2).

## The array representation

A tree can also live in a flat array with no pointers at all:

```
parent of i = (i - 1) // 2
left child  = 2i + 1
right child = 2i + 2
```

Navigation becomes arithmetic, and nodes sit near their children in memory, which
is cache friendly in the way day 5 described.

The catch is that the array needs a slot for every position in the tree, filled or
not. A chain of n nodes would need 2^n slots. So this only makes sense for trees
that are **complete**: every level full except possibly the last, which fills from
the left.

That is exactly the shape of a binary heap, which is why heaps are stored this way
and why this representation is introduced here rather than skipped.

## Threaded binary trees

Start with a count. A binary tree of n nodes has 2n child pointers, and exactly
n - 1 of them point at something, since every node but the root has exactly one
parent. So **n + 1 pointers in every binary tree are null**, usually more than
half of them, doing nothing.

A threaded tree uses them. A null left pointer is made to point at the node before
this one in inorder, a null right pointer at the node after. Those are called
threads, and one flag per side records whether a pointer is a real child or a
thread. Without the flags the two are indistinguishable and following a thread
believing it is a child loops forever, which is the characteristic bug and has a
test aimed at it.

The payoff is **inorder traversal with no stack and no recursion, in O(1)
memory**. Finishing a node, if its right pointer is a thread you step straight to
the successor; if it is a real child, the successor is the leftmost node of that
subtree. Compare that with the iterative traversal, whose stack can grow to the
height of the tree.

The cost is honest: every insertion has to maintain the threads, so writes get
more complicated and a little slower. That was a serious win when memory was
scarce, and still matters in embedded work. On a modern desktop the recursive
traversal is usually fine, so this is more often studied than deployed.

It is included because the underlying idea generalises well beyond trees: **when a
structure has wasted space, ask whether that space can carry information.**

`wasted_pointer_count()` exists so the n + 1 claim can be tested rather than
asserted, and it holds for every tree size tried.

## Cost summary

| Operation | Cost | Memory while running |
| - | - | - |
| Any traversal, recursive | O(n) | O(h) call stack |
| Any traversal, iterative | O(n) | O(h) explicit stack |
| Threaded inorder | O(n) | O(1) |
| height, size, leaves | O(n) | O(h) |
| Rebuild from two traversals | O(n) | O(n) |

where h is the height, which is log2(n) if the tree is balanced and n if it is not.
Day 9 builds the search tree that makes the height matter, and day 10 makes the
good case a guarantee.
