# Day 5: Linked lists

Code: `dsalab/structures/linked_list.py`. Tests: `tests/test_linked_list.py`.

## The trade being made

An array is one continuous block, which is why indexing is instant and why
inserting in the middle is slow: everything after the insertion point has to
physically move.

A linked list gives up the continuous block. Each value sits in its own node
somewhere in memory, and each node holds a pointer to the next. Every cost flips:

| Operation | Array | Linked list |
| - | - | - |
| read position i | O(1) | O(n) |
| insert or delete at the front | O(n) | O(1) |
| insert or delete at a known node | O(n) | O(1) |
| find a value | O(n) | O(n) |
| memory per element | just the value | the value plus a pointer |

Neither wins outright. The question is always which operation your code does most.

## The cost that complexity notation hides

Big O says walking a linked list and walking an array are both O(n). Real
machines disagree, sometimes by a factor of ten.

An array sits in one block, so reading it in order marches straight through
memory. The CPU notices the pattern and fetches the next chunk before it is
asked for. A linked list's nodes are scattered wherever the allocator put them,
so every step is a jump to an unpredictable address, and the processor has to
stop and wait for memory each time.

On top of that, every node carries a pointer as well as its value, so a linked
list of small values can use twice the memory of the equivalent array. That is
why languages default to array backed lists, and why "linked list is O(1) to
insert" is a true statement that still loses to an array in a great many real
programs. Complexity tells you how something scales, not how fast it is.

`Node` uses `__slots__` for this reason. Without it every node carries an
attribute dictionary costing around fifty bytes, which for a structure that
creates one object per element is a large tax for nothing.

## Singly linked list

The one design decision worth arguing about is keeping a tail pointer. Without
one, appending means walking the entire list to find the end, which makes
building a list of n items cost O(n^2). With one, appending is O(1).

The cost is bookkeeping. Every operation that could change the last node has to
remember to update the tail, and forgetting is invisible when you read the list
back. The list looks perfect, and then the next append links onto a node that
was removed. Two tests exist purely for this:
`test_the_tail_pointer_stays_correct_after_every_kind_of_change` appends after
each kind of mutation, and
`test_emptying_the_list_clears_the_tail_as_well_as_the_head` covers the case
where the list becomes empty and the tail must be set back to None.

### Popping from the back is the weak spot

Even with a tail pointer, removing the last node is O(n). To unlink a node you
need the node *before* it, and a singly linked list cannot look backwards, so you
walk from the head to find it. This single limitation is the entire reason the
doubly linked list exists.

### Reversal, and the order of four lines

Reversing in place is O(n) time and O(1) memory, using three pointers. The order
inside the loop is what makes it work:

```
following = current.next     save it first, or the rest of the list is lost
current.next = previous      turn this node around
previous = current           step both pointers forward
current = following
```

Move the first line anywhere else and the moment you reassign `current.next` you
have dropped every node after it, with no way to reach them again. This is the
classic interview question and the classic way to fail it.

### The tortoise and hare, used twice

Two pointers moving at different speeds solve two different problems here.

**Finding the middle.** The obvious method walks the list to count, then walks
half of it again. Two passes, and impossible on data you can only read once. Move
one pointer one step and another two steps, and when the fast one falls off the
end the slow one is exactly halfway. One pass.

**Detecting a cycle.** If the list loops, both pointers end up inside the loop,
and since the fast one gains one position per step it must eventually land on the
slow one. The alternative is recording every visited node in a set, which also
works but costs O(n) memory. Floyd's method costs two pointers, and that
difference is why it is the one worth knowing.

## Doubly linked list

One extra pointer per node buys three things:

1. Removing from the back is O(1) instead of O(n).
2. The list can be walked in either direction.
3. A node can be unlinked when all you hold is that node itself.

The third is the one that matters most. It is what makes an LRU cache possible: a
hash table hands you the node directly, and you pull it out of the list in
constant time. A singly linked list cannot do that at any price, because it has
no way to find what came before.

The cost is roughly twice as many pointer updates per change, each a chance to
get it wrong. The characteristic bug is a broken `prev` pointer, which is
completely invisible when reading the list forwards. So every removal test here
checks `to_list_backwards()` against the reverse of `to_list()`. A list that
disagrees with itself in the two directions is broken even though every forward
read looks perfect.

## Circular linked list

The last node points back at the first, so there is no end to fall off. Anything
walking it must count rather than test for None, and forgetting that is an easy
infinite loop, which is why `__iter__` here loops exactly `self._length` times.

Only the tail is stored, not the head. That is not arbitrary: `tail.next` is the
head, so one pointer gives constant time access to both ends. Appending and
prepending then turn out to be the same two pointer assignments, differing only
in whether the tail pointer moves, which is a neat way of showing that a circle
has no real beginning.

The Josephus problem is the classic demonstration. Stand n people in a circle,
count round removing every step'th one, and ask who survives. After the last
position the count carries straight into the first with no special case at all,
which is exactly what the structure is for. There is a known O(n) formula for the
answer, but the simulation is what shows the structure earning its keep.

## Merging two sorted lists

This is where linked lists genuinely beat arrays. Merging two sorted arrays needs
somewhere to put the result, so it costs O(n + m) extra memory. Merging two
linked lists only relinks nodes that already exist, so it costs nothing extra.

That is why merge sort on a linked list is a truly in place O(n log n) sort,
while merge sort on an array is not. It comes back on day 14.

The merge is also **stable**: when two values are equal, the one from the left
list is taken first. That is the `<=` rather than `<` in the comparison, one
character that decides whether equal items keep their original relative order.
Stability matters when sorting records by one field after already sorting by
another, and there is a test asserting it, because it is exactly the kind of
detail that gets broken by a later tidy up.

## A trade off I actually hit

`merge_sorted` drains the leftover tail of whichever list still has values, and
the first version did that in a plain loop with no steps yielded. The answers
were all correct and every result test passed. The step counting test caught it:
merging a one element list with a four element list produced two steps instead of
five, meaning the animation would freeze while the data kept changing.

Same lesson as day 4, now met twice, so it is a rule rather than a coincidence:
**if the trace is an output, test the trace.** Correct answers say nothing about
whether the trace is complete.
