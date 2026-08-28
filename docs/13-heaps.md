# Day 13: Heaps, priority queues and heap sort

Code: `dsalab/structures/heap.py`. Tests: `tests/test_heap.py`.

## A structure that gives up almost everything

A heap cannot find an arbitrary value in less than O(n). It has no ordering you
can walk. It answers no range queries. Compared with the search trees of the last
four days it looks like a step backwards.

What it does instead is one thing, extremely well: **hand back the smallest item,
repeatedly, in O(log n), while accepting new items just as fast.**

That is the shape of a priority queue, and priority queues are everywhere:
Dijkstra's shortest paths, Prim's minimum spanning tree, Huffman coding, task
schedulers, event simulation, the k largest of a stream.

## The weak rule is the point

```
every node is no larger than its children
```

Notice what is missing. Nothing about siblings. Nothing about cousins. A search
tree orders everything; a heap orders only along each root to leaf path.

That weakness is deliberate. **A weaker invariant is cheaper to restore**, and it
is still strong enough to pin the minimum at the root, where it can be read in
O(1). Giving up the ordering you do not need is what buys the speed on the one
operation you do.

## Why it lives in an array

A heap is always a **complete** tree: every level full except the last, which
fills from the left. Complete trees have no gaps, so the array layout from day 8
fits perfectly:

```
parent of i = (i-1)//2      left = 2i+1      right = 2i+2
```

No node objects, no pointers, navigation by arithmetic, and a parent sits near its
children in memory. Day 8 noted that the array representation is unusable for
general trees because a sparse tree needs exponentially many slots. A heap is
never sparse, so it gets all of the benefit and none of the cost.

Keeping the tree complete is why `pop` works the way it does. You cannot promote
the smaller child into the empty root, because that leaves a hole one level down.
So the **last** item in the array is moved to the root, which keeps the tree
complete by construction, and then sinks. **Shape first, ordering second**, and
that priority is what keeps the array valid at all times.

## Building a heap is O(n), not O(n log n)

This is the fact worth knowing, and it is genuinely surprising.

Sift down every node that has children, working backwards from the last such node
to the root. Working backwards matters: when a node is sifted down, both its
subtrees are already heaps, which is exactly what sift down needs.

The obvious cost estimate says n nodes times log n each, so O(n log n). That is
far too pessimistic, because **most nodes are near the bottom, and nodes near the
bottom barely move**:

| Level from the bottom | How many nodes | How far each can fall |
| - | - | - |
| 0 (leaves) | n/2 | 0 |
| 1 | n/4 | 1 |
| 2 | n/8 | 2 |
| h | n / 2^(h+1) | h |

Total work is n times the sum of h / 2^(h+1) over all h, and that sum converges to
1. So the total is O(n).

Pushing n items one at a time really is O(n log n), because there every item can
climb the full height. Building all at once is linear. The tests measure both:
`test_building_all_at_once_is_linear` counts actual swaps at six sizes and hands
them to the complexity detective, which reports O(n) with confidence, and a second
test confirms building at once uses fewer swaps than pushing one at a time.

## The mistake that hides

In `_sift_down`, the item must swap with the **better of its two children**, not
with any child that beats it.

Swap with the wrong child and a larger value ends up above a smaller one. Nothing
crashes. The heap still answers `pop` with plausible values. They are just not
always the right ones, and the error appears far from its cause. This is the same
class of bug as the stale heights on day 10 and the broken colours on day 11, and
it is why invariant checking exists: the rule is checked after every push and pop
in the stress tests.

## The priority queue and the position table

A plain heap can push and pop, which covers most uses. Dijkstra on day 18 wants
one more operation: **lower the priority of something already in the queue** when a
shorter route to a place is found.

On a plain heap that means finding the item, which is O(n) and would dominate the
whole algorithm. So `PriorityQueue` keeps a dictionary from item to array
position, updated on every swap. Finding an item becomes O(1) and changing its
priority O(log n).

The cost is real: extra memory, and **every single swap must keep the table
correct**. Miss one and the queue works perfectly until it quietly starts looking
in the wrong slot. The invariant check verifies every recorded position against
the array, and the stress test runs six hundred mixed operations verifying after
each one.

The common alternative is **lazy deletion**: never update anything, just push the
improved entry as a duplicate and skip stale entries as they come out. Simpler,
often faster in practice, at the cost of a queue holding more entries than there
are items. Both are legitimate; the explicit version is here because it makes the
decrease operation visible rather than hidden in a skip condition.

One deliberate bit of API design: pushing an item that is already present with a
*worse* priority is ignored rather than treated as an error. That is exactly what
Dijkstra wants, since it offers routes to places it has already reached and the
shorter one should simply win, without every caller having to check first.

## Heap sort

O(n log n) always, best and worst case alike, sorting **in place** with O(1) extra
memory. That combination is rare: merge sort matches the time but needs O(n) extra
memory, quick sort matches the memory but has an O(n^2) worst case.

The in place trick is elegant. Build a **max** heap, then swap the root with the
last item and shrink the heap by one. The largest item is now in its final place at
the end, and the heap occupies the shrinking prefix. The array sorts itself from
the back forwards, with the heap and the sorted region sharing one array and never
overlapping.

So why is quick sort still the usual choice? **Cache behaviour.** Sift down jumps
from index i to 2i+1, which for a large array is a different cache line almost
every step. Quick sort scans linearly, which the hardware prefetches perfectly.
Heap sort typically loses by a factor of two or three despite identical
complexity. Day 15 measures it.

There is one test worth pointing at: `test_sorted_input_costs_the_same_as_random
_input`. Heap sort has no bad input, which is precisely why it is chosen where a
guaranteed bound matters more than raw average speed.

## k smallest of a stream

To find the k **smallest**, keep a **max** heap of the best k so far. That sounds
backwards and is exactly right: the largest of the k sits at the root, so each new
value is compared against it in O(1) and only better values get in. The heap never
grows past k.

Cost is O(n log k) time and O(k) memory, against O(n log n) and O(n) for sorting
everything. When the input is a billion item stream and k is ten, that is the
difference between possible and impossible, and there is a test that runs it over
a stream it can only read once.
