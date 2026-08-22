# Day 7: Queues and deques

Code: `dsalab/structures/queue.py`, `dsalab/structures/deque.py`.
Tests: `tests/test_queue_and_deque.py`.

## Stack or queue is a choice about shape

A stack hands back the newest arrival. A queue hands back the oldest. That single
difference decides the shape of the search built on top:

* A stack goes **deep**. Take the newest thing, follow it as far as it leads.
  That is depth first search.
* A queue goes **wide**. Take the oldest thing, so everything one step away is
  handled before anything two steps away. That is breadth first search, and it is
  why breadth first search finds the shortest path in an unweighted graph while
  depth first search does not.

Swapping a stack for a queue in the traversal code on day 16 changes nothing else
and completely changes the algorithm. That is worth seeing once.

Queues also turn up anywhere work must be served fairly: print jobs, incoming
requests, task scheduling, keystrokes waiting to be handled.

## Why a queue in an array has to be circular

Take a plain array queue of capacity 5. Add five items, remove three. The front
is now at index 3, the back is at index 5, and there are three free slots at the
start of the array and none at the end. Adding a sixth item leaves two bad
options: refuse, even though the array is mostly empty, or shift everything back
to the start, which makes every add O(n).

A circular queue lets the back wrap round to index 0 when it runs off the end, so
those three free slots at the front get used. No shifting, no waste, everything
stays O(1).

The whole thing comes from one operator: `(index + 1) % capacity` is what turns an
array into a ring.

### Telling full from empty

A ring has one genuinely awkward detail. When the front and back pointers land on
the same slot, the queue could be completely empty or completely full, and the
pointers look identical in both cases. Two standard fixes:

| Fix | Cost | Benefit |
| - | - | - |
| Waste one slot, so full means back is one behind front | one slot | no extra state to keep correct |
| Keep a count of items | one integer, updated on every operation | `len()` is O(1) and reports how full the buffer is |

This project keeps the count. The deciding argument was not memory, it was that
the count is useful on its own: `len()` should be O(1), and "how full is this
buffer" is exactly the number monitoring code wants. The wasted slot approach
would make `len()` a modulo calculation with a special case, which is more code
for less information.

### The bug that only appears after wrapping

`DynamicQueue` doubles when full, which means copying the old ring into a bigger
one. The copy must walk the items **in logical order, front first**, not in memory
order, because after a wrap the items are split across the end of the buffer.
Copying in memory order silently reorders the queue.

What makes this dangerous is that it only shows up once the ring has wrapped at
least once. Fill a fresh queue until it grows and everything looks perfect.
`test_growing_after_the_ring_has_wrapped_keeps_the_order` forces a wrap first and
then a growth, which is the only arrangement that catches it.

The general habit worth taking from this: **for any structure with a wrap or a
boundary, write the test that crosses it.** Testing the easy path proves nothing
about the hard one.

## The queue built from two stacks

This one is included for the analysis rather than for practical use, because the
reasoning is the clearest example of amortised cost there is.

Arrivals get pushed onto an inbox stack. When something is needed and the outbox
is empty, the entire inbox is tipped into the outbox. Tipping reverses the order,
so the oldest item ends up on top, which is exactly what a queue should give back.

A single dequeue can therefore cost O(n). Yet the amortised cost is O(1), because:

> **Every item is moved exactly twice in its whole life.** Once from inbox to
> outbox, once out of the outbox. It can never be tipped twice, because tipping
> only happens when the outbox is empty, and an item in the outbox stays there
> until it leaves.

So n operations do at most 2n moves, which is constant per operation. Notice this
is a guarantee over *any* sequence of calls, not an average over likely inputs. An
adversary who knows the implementation cannot construct a bad case. That is
precisely the difference between amortised and average, and it is why amortised
bounds are worth so much more.

## Deques, and the algorithm that justifies them

A deque adds and removes at both ends in O(1). A stack and a queue are both
special cases of it, which the tests demonstrate directly by using a deque as
each.

The implementation is the doubly linked list from day 5, so every operation is
worst case O(1) with no copying and no capacity limit. A circular buffer deque
would have better memory behaviour and amortised rather than worst case bounds,
the same trade as the two stack implementations on day 6.

### Sliding window maximum

This is the best argument for the structure existing. Given a list and a window
size k, report the maximum of every window of k consecutive values.

The obvious method checks all k values in each of the n - k + 1 windows, so it is
O(n times k). For a million values and a window of a thousand that is a billion
comparisons.

The deque version is O(n) total, whatever k is. It holds **positions**, kept so
that their values are always in decreasing order, maintained by two rules:

1. Before adding a new position, discard every position at the back whose value
   is smaller than or equal to the new one.
2. Before reading the answer, discard the front position if it has fallen out of
   the window.

Rule 1 is the insight, and it is worth stating carefully. A position discarded
this way is not merely unhelpful right now, it is **permanently useless**: the new
value is both larger *and* stays in the window longer, so the discarded position
can never be the maximum of any future window. Nothing is lost by throwing it
away.

With those rules the front of the deque is always the answer for the current
window. Each position enters once and leaves at most once, so the total work is at
most 2n regardless of the window size, and there is a test asserting exactly that
across three very different window sizes.

The deque is the right structure because rule 1 works at the back and rule 2 works
at the front. Nothing else gives O(1) at both ends.

## Cost summary

| Structure | enqueue | dequeue | peek | memory |
| - | - | - | - | - |
| CircularQueue | O(1) | O(1) | O(1) | fixed, never grows |
| DynamicQueue | O(1) amortised | O(1) | O(1) | doubles as needed |
| LinkedQueue | O(1) | O(1) | O(1) | one node per item |
| QueueFromTwoStacks | O(1) | O(1) amortised | O(1) amortised | two stacks |
| Deque | O(1) both ends | O(1) both ends | O(1) both ends | one node per item |
