# Day 4: Arrays, memory, and the matrices that refuse to waste it

Code: `dsalab/structures/dynamic_array.py`, `special_matrix.py`, `sparse_matrix.py`,
`polynomial.py`. Tests: `tests/test_dynamic_array.py`, `tests/test_matrices.py`.

## What an array really is

An array is one continuous block of memory holding equally sized slots. That
single sentence explains every property arrays have.

Because the slots are the same size and sit next to each other, the address of
slot i is just `start + i * slot_size`. That is one multiplication and one
addition, no matter whether i is 0 or 4 million. This is why reading an array by
index is O(1), and it is the only reason. It is not because arrays are somehow
special, it is because a multiplication does not care how big the number is.

Because the block is continuous and allocated once, it cannot grow. The memory
after the end of the block belongs to something else. If you want a bigger array,
you allocate a new bigger block and copy everything across. There is no way
around this at the hardware level, and everything about dynamic arrays follows
from having to live with it.

### Static and dynamic memory

A local variable declared inside a function lives on the **stack**, which is
managed automatically: the space appears when the function is entered and
vanishes when it returns. It is fast, but the size must be known in advance and
the lifetime is tied to the function call.

Anything that needs to outlive the function that created it, or whose size is
only known at run time, lives on the **heap**. The heap is a large pool that you
request blocks from explicitly and hand back when done. It is more flexible and
slower, and forgetting to hand memory back is what a memory leak is.

In Python every object lives on the heap and the garbage collector hands memory
back for you, so the distinction is invisible most of the time. It became very
visible today, and that story is below.

## The dynamic array

`DynamicArray` stores its items in a `ctypes` array of object pointers, which is
a genuine fixed size block that cannot resize itself. Using a Python list under
the hood would have been three lines of code and would have taught nothing,
because Python's list is itself a dynamic array and all the interesting work
would have been happening inside it, out of sight.

### Growth: why doubling, and why it makes append O(1)

When the block fills, the array allocates a bigger one and copies everything
across, which is O(n). So how can append be called constant time?

Because the expensive copies get rarer at exactly the rate that makes the total
work linear. Append n items with doubling, starting from capacity 1, and the
copies happen at sizes 1, 2, 4, 8, and so on up to n. The total number of
elements copied is:

```
1 + 2 + 4 + 8 + ... + n
```

That sum is less than 2n. Not 2n per append, 2n in total, across all n appends.
Spread over n appends that is under 2 copies each, which is a constant. So the
amortised cost of an append is O(1), and this is a guarantee about any sequence
of appends, not a statement about the average input.

Now try growing by a fixed amount instead, say 10 slots at a time. The copies
happen at 10, 20, 30, and the total copying is 10 + 20 + ... + n, which is
n^2 / 20. That is quadratic total work, so each append costs O(n) on average and
appending a million items becomes unusable. The difference between multiplying
and adding is the entire ballgame here.

The test `test_growth_happens_rarely_enough_for_append_to_be_amortised_constant`
checks this by counting real copies at several sizes and handing the numbers to
the complexity detective from day 3. It confirms the total copying is O(n), so
the amortised claim is measured rather than asserted.

### Shrinking: why a quarter and not a half

When items are removed the block should eventually be released, or an array that
briefly held a million items keeps that memory forever. The obvious rule is to
halve the block when it is half empty. That rule is a trap.

Picture a block of capacity 8 holding 4 items. Delete one, it is now half empty,
so it shrinks to capacity 4 holding 3. Append one, now it is full at 4, so it
doubles back to 8. Append, delete, append, delete, and every single operation
triggers a full copy. The amortised guarantee is destroyed by a two line loop.
This is called thrashing.

Shrinking only at a quarter full leaves a gap between the two triggers. After a
shrink the array is half full, so it needs n/2 appends to trigger a grow or n/4
deletes to trigger another shrink. Either way, an O(n) copy is always paid for by
a linear number of cheap operations, so the amortised bound holds. The test
`test_repeatedly_appending_and_popping_at_the_boundary_does_not_thrash` runs
exactly the pathological loop and asserts that almost no resizing happens.

### The bug I did not expect: a leak inside ctypes

This was the most interesting hour of the day.

When an item is popped, the array must stop referring to it, or the object can
never be garbage collected even though it is logically gone. So `pop` sets the
vacated slot to None. With a Python list that would be the end of it.

It was not enough. A `ctypes` array of `py_object` keeps a **second, hidden
dictionary** called `_objects`, mapping slot number to the object stored there.
It has to: the raw memory holds only a pointer, and a raw pointer does not keep
an object alive, so ctypes maintains its own references on your behalf.
Overwriting the slot updates the raw memory and leaves the old entry sitting in
`_objects`, holding the removed object alive forever.

Nothing about the array's visible behaviour shows this. Length is right, contents
are right, every ordinary test passes. The test that caught it takes a weak
reference to an object, pops it, forces a collection, and asserts the weak
reference has gone dead. `_release_slot` now clears both places.

The general lesson is the one worth keeping: **a correct answer is not the same
as correct behaviour.** Resource handling is invisible to tests that only check
outputs, so it needs tests that check the resource directly.

## Cost summary for the dynamic array

| Operation | Cost | Why |
| - | - | - |
| read or write by index | O(1) | Address arithmetic |
| append | O(1) amortised | Occasional O(n) copy, paid for by n cheap appends |
| pop from the end | O(1) amortised | Same argument |
| insert at position i | O(n) | Every element to the right shifts one slot |
| delete at position i | O(n) | Every element to the right shifts back |
| search by value | O(n) | Unsorted, so there is no shortcut |
| memory | O(n) | Between n and 2n slots, depending where growth landed |

Notice that insert and delete are slow for the same reason reads are fast:
continuous memory. You cannot have gaps in a block and still compute an address
by multiplying, so making room means physically moving things. Linked lists on
day 5 make the opposite trade.

## Matrices that store only what they must

An n by n matrix takes n^2 slots stored plainly. Many useful matrices are mostly
zeros in a predictable pattern, and when the pattern is known you can store only
the values that can be non zero and compute where each one lives.

| Representation | Stored values | Saving at n = 1000 |
| - | - | - |
| Dense | n^2 | none, 1000000 numbers |
| Diagonal | n | 99.9 percent |
| Triangular, symmetric | n(n + 1) / 2 | about 50 percent |
| Tridiagonal | 3n - 2 | 99.7 percent |
| Toeplitz | 2n - 1 | 99.8 percent |

Every one of these keeps reads and writes at O(1), because finding the storage
slot is arithmetic rather than searching. The triangular case is the one worth
working through by hand: to find where (i, j) lives you count the elements in all
the rows above, which is 1 + 2 + ... + i, which is i(i + 1) / 2, then add j to
step along the current row.

The cost of all this is that the code is harder to read and every representation
has rules about what you are allowed to write where. The classes refuse a write
that would break their rule rather than silently ignoring it, because a matrix
that quietly drops your value is far worse than one that complains.

A test worth mentioning: `test_no_two_cells_share_a_storage_slot` fills every
valid cell with a distinct number and reads them all back. Off by one errors in
index arithmetic cause two cells to share a slot, which is invisible if you only
test a few values, and immediately obvious with this test.

## Sparse matrices, when the zeros have no pattern

If the non zero values can be anywhere, no formula can find them, so store the
coordinates explicitly as a list of (row, column, value) triples kept sorted.

The break even point is worth knowing rather than guessing. Dense costs n^2
numbers, sparse costs about three numbers per entry, so **sparse only wins below
roughly one third density**. Above that it uses more memory than the dense form
and is slower to read, since a read becomes a binary search instead of address
arithmetic. The `density` property exists so you can check rather than assume.

Keeping the entries sorted is what makes addition fast. Two sorted lists merge in
one pass with two pointers, exactly like the merge step of merge sort, so adding
two sparse matrices costs O(k1 + k2) and never looks at a zero. A five cell row
holding two values takes two steps, not five, and there is a test that asserts
precisely that.

One rule shows up in three separate places today and is worth stating on its own:
**never store a computed zero**. When two entries cancel during addition, the
result must be dropped rather than recorded as an explicit zero. Otherwise a
matrix slowly fills with zeros that take up space and slow everything down, and
it stops being sparse without anyone noticing.

## Polynomials are the same idea in one dimension

`5x^100 + 3` has two terms and degree 100. Stored as a dense coefficient list it
needs 101 numbers, 99 of them zero. So the same fix applies: keep only the non
zero terms, sorted by exponent, and addition becomes the same single merge pass.

Evaluation uses Horner's method, which was introduced on day 2 and matters more
here. The obvious approach computes each power separately, which is O(degree^2)
multiplications. Horner rewrites `2x^3 + 3x^2 + 0x + 5` as `((2x + 3)x + 0)x + 5`,
which is one multiply and one add per degree, so O(degree). Same answer, cheaper
arrangement, and slightly more accurate in floating point as a bonus because
there are fewer operations for rounding error to build up through.

## A second trade off worth recording

The trace for sparse addition originally stopped early. The merge loop yields a
step per comparison, but once one list runs out the remaining entries were copied
across in a plain loop with no steps at all. The answer was perfectly correct, so
every test of the result passed. Only the test that counts steps caught it, and
what it was really catching was that the animation would stop partway through the
run while the data kept changing underneath it.

That is worth generalising: when the trace is a first class output of the code,
it needs testing like any other output. "The answer is right" is not sufficient
when the trace is something users are going to watch.
