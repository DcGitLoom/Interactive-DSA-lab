# Day 15: Sorting, eleven ways

Code: `dsalab/algorithms/sorting.py` (heap sort is in `structures/heap.py`).
Tests: `tests/test_sorting.py`.

## Why comparison sorting cannot beat n log n

This is the most important proof in the course, and it is short enough to hold in
your head.

A comparison sort learns about the data **only** by asking "is a bigger than b".
Picture the algorithm as a decision tree: each internal node is one comparison,
each branch is a yes or no answer, and each leaf is a final arrangement of the
input.

Now count. An input of n distinct items has **n! possible orderings**, and the
algorithm has to be able to produce every one of them, or there is some input it
sorts incorrectly. So the tree needs at least n! leaves.

A binary tree of height h has at most 2^h leaves. Therefore:

```
2^h >= n!
h   >= log2(n!)
```

And log2(n!) is approximately n log2(n) - 1.44n by Stirling's approximation, so:

```
h = Omega(n log n)
```

The height of the tree is the number of comparisons in the worst case. So **no
comparison sort can do better than n log n in the worst case**, ever, no matter how
clever it is. Merge sort and heap sort achieve that bound, which means they are
optimal and there is no point looking for something better in this model.

The phrase "in this model" is the escape hatch, and counting sort walks straight
through it: it never compares two values at all, so the argument does not apply.
More on that below.

## The eleven algorithms

| Algorithm | Best | Average | Worst | Memory | Stable | Adaptive |
| - | - | - | - | - | - | - |
| Bubble | O(n) | O(n^2) | O(n^2) | O(1) | yes | yes |
| Selection | O(n^2) | O(n^2) | O(n^2) | O(1) | no | no |
| Insertion | O(n) | O(n^2) | O(n^2) | O(1) | yes | yes |
| Binary insertion | O(n log n) compares | O(n^2) moves | O(n^2) | O(1) | yes | no |
| Shell | O(n log n) | about O(n^1.5) | depends on gaps | O(1) | no | partly |
| Merge | O(n log n) | O(n log n) | O(n log n) | O(n) | yes | no |
| Quick | O(n log n) | O(n log n) | O(n^2) | O(log n) | no | no |
| Heap | O(n log n) | O(n log n) | O(n log n) | O(1) | no | no |
| Counting | O(n + k) | O(n + k) | O(n + k) | O(n + k) | yes | no |
| Radix | O(d(n + b)) | same | same | O(n + b) | yes | no |
| Bucket | O(n) | O(n) | O(n^2) | O(n) | yes | no |

Every row of that table is checked by a test. The complexity claims are measured
with the day 3 detector rather than copied from a book.

## The quadratic three, and why one of them survives

**Bubble sort** is the one everybody learns and nobody should use. Its only merit
is the early exit: a pass with no swaps means the array is sorted, making it O(n)
on already sorted input. Insertion sort is also O(n) there and better everywhere
else.

**Selection sort** is O(n^2) comparisons no matter what, and not adaptive at all,
so sorted input costs exactly as much as random input. But it makes only **O(n)
swaps**, the fewest of any sort here, and that is a real advantage when moving data
is expensive relative to comparing it: large records, or flash memory where writes
wear out the hardware. A test confirms bubble sort makes over ten times as many
swaps on the same input.

**Insertion sort** is the one that earns its place, for two reasons:

1. It is **adaptive**. The inner loop stops as soon as the item is in place, so
   the work is proportional to how far out of order the input is rather than to
   its size.
2. It has tiny constants: no recursion, no extra memory, one comparison and one
   move per step, walking memory in order.

That is why real library sorts, Python's Timsort and C++'s introsort included,
switch to insertion sort once a partition drops below a few dozen items.

### A test that taught me what "nearly sorted" means

The adaptive test failed at first. I generated "nearly sorted" input by swapping a
few pairs of random positions, and the measurement came back O(n^2).

The measurement was right and my input was wrong. **Swapping two random positions
displaces both items by a distance proportional to n**, and insertion sort has to
shift each of them all that way, so even a handful of long range swaps is
quadratic work. Nearly sorted has to mean *every item is close to its home*, not
merely that few items moved. Swapping neighbours instead gives genuinely local
disorder, and the measurement comes back linear.

That distinction is the whole content of the word adaptive, and I only understood
it properly because a measurement disagreed with me.

## Binary insertion sort: an improvement that is not

It cuts comparisons from O(n^2) to O(n log n) by binary searching for the
insertion point. The **moves** are still O(n^2), because the item still has to be
physically shifted, and since moving usually costs about as much as comparing, the
total barely changes.

Worse, it gives up the adaptive behaviour: plain insertion sort does one comparison
per already sorted item, this one does log n regardless, making it **slower on
nearly sorted input**, which was the whole point of insertion sort.

It genuinely helps only when comparison is expensive relative to moving, such as
sorting long strings or comparing by an expensive computed key. A good lesson in
optimising the wrong half of an algorithm.

## Shell sort, and an open problem

Insertion sort is slow because a distant item moves one place at a time. So sort
items that are a large gap apart first, letting things travel a long way cheaply,
and shrink the gap to 1, by which point the array is nearly sorted and insertion
sort is fast.

The complexity depends entirely on the gap sequence, and this is the remarkable
part: **nobody knows the best sequence.** Shell's original halving gives O(n^2).
Knuth's 3k+1, used here, gives O(n^1.5). The best known reach about O(n^1.33). The
true optimum is still an open problem decades later, which is a rare thing to be
able to say about an algorithm this simple.

## The n log n three, and why quick sort wins in practice

All three hit the optimal bound, so the choice between them is about everything
else.

**Merge sort** is the only one that is both stable and guaranteed n log n. It
costs O(n) extra memory, and it is uniquely good at two things: sorting linked
lists, where merging just relinks nodes and needs no extra memory at all, and
sorting data too big for memory, by sorting chunks and merging the files in one
streaming pass.

**Heap sort** is guaranteed n log n in O(1) memory, with no bad input at all.

**Quick sort** has an O(n^2) worst case and is still the fastest in practice. The
reason is memory rather than mathematics: partitioning scans straight through the
array, which the cache and prefetcher handle perfectly, while merge sort touches a
second array and heap sort jumps between index i and 2i+1. Complexity says these
three are equal; hardware says they are not.

### The pivot is the whole algorithm

| Strategy | Sorted input | Notes |
| - | - | - |
| First element | **O(n^2)** | The classic teaching version, and a trap |
| Random | O(n log n) expected | No input is bad if the attacker cannot guess the seed |
| Median of three | O(n log n) | Sorted input becomes the *best* case |

The first element version is not a theoretical concern. Sorted input is extremely
common, and this choice turns the most ordinary input into the worst case. There
is a test that measures it and gets back O(n^2) with confidence, next to the
median of three version measuring O(n log n) on identical input.

Median of three is what most real implementations use: cheap to compute, and on
sorted input the middle element is the true median, so the partitions are perfect.

Two partition schemes are included. **Lomuto's** is easier to read and is what gets
taught. **Hoare's** original does about three times fewer swaps and copes far
better with arrays full of equal values, where Lomuto degrades to O(n^2) because
everything goes to one side. Hoare's catch is that the pivot does not end up in its
final position, so the recursion boundary is different, and getting that wrong is
the classic infinite loop.

## The linear sorts, which do not compare at all

**Counting sort** counts occurrences and writes the values back out. O(n + k) where
k is the range of the values. There is a test asserting it produces **no comparison
steps whatsoever**, which is precisely why the lower bound does not apply: it uses
values as array indices, an operation the decision tree model does not contain.

The cost is the assumption. Values must be integers in a known, reasonably small
range. Sorting a thousand values spread over a billion needs a billion counters.
The rule of thumb is that it wins when k is around n or smaller.

Its stability is not automatic. It comes from turning counts into running totals
and walking the **input backwards** when placing. Forwards gives the same sorted
values with equal items reversed, which no test of the sorted output would ever
catch.

**Radix sort** sorts by one digit at a time, least significant first, and the key
insight is that **its correctness depends on the per digit sort being stable**.
Sorting by the tens column only preserves the ones column ordering if ties keep
their order. Swap in an unstable inner sort and radix sort silently produces wrong
answers. Stability is usually presented as a nice property; here it is load
bearing.

Most textbook versions quietly assume non negative input. This one splits out the
negatives, sorts their absolute values and reverses that half, and there is a test
for it.

**Bucket sort** spreads values across buckets by range and sorts each one. O(n)
when the values are spread evenly, O(n^2) when they all land in one bucket, which
makes it the most input dependent sort here. Each bucket is finished with insertion
sort, which is exactly right because buckets are small and nearly sorted.

## Stability, and the one character that decides it

A sort is stable when equal elements keep their original relative order. It sounds
pedantic until you sort a table by one column having already sorted it by another:
a stable sort preserves the first ordering as a tiebreak, an unstable one loses it.

Stability is usually decided by a **single comparison operator**. Merge sort's
`<=` takes from the left half on ties, which is the half that came first. Change it
to `<` and stability is gone with no other visible difference. Insertion sort's `>`
does the same job.

### The test that was quietly worthless

My first stability test sorted a list of `(key, original position)` pairs and
compared against Python's sorted output. Every algorithm passed, including
selection sort, which is definitely not stable.

The bug was that sorting the pairs **directly** compares the second element too
when the first ones tie, so the tiebreak was doing the stabilising, not the
algorithm. Passing a key function that looks only at the first element is what
leaves the order of equal items visible, and then selection sort fails exactly as
it should.

A test that passes for the wrong reason is worse than no test, because it stops
you looking.

## What a real library does

None of these eleven is what Python actually uses. `sorted()` uses **Timsort**,
which combines the ideas here:

* It scans for **runs** of already ordered data, exploiting the fact that real
  data is often partly sorted.
* Short runs are extended and sorted with **binary insertion sort**.
* Runs are combined with **merge sort**, keeping stability.
* Merges are balanced by size, with a stack of pending runs, so the worst case
  stays n log n.

The result is O(n) on already sorted input, O(n log n) worst case, and stable. It
is essentially "use the adaptive one when the data is easy, the guaranteed one when
it is not", which is the conclusion this whole day points at.
