# Day 14: Hash tables and the LRU cache

Code: `dsalab/structures/hash_table.py`, `dsalab/structures/lru_cache.py`.
Tests: `tests/test_hashing.py`.

## No comparisons at all

Every search structure so far compared its way to an answer. A hash table
computes the address:

```
index = hash(key) % number_of_buckets
```

One arithmetic operation, one array access. Not O(log n) comparisons. **No
comparisons.** That is why a hash table beats every tree on plain lookup.

Two honest caveats:

**O(1) is average, not guaranteed.** Two keys can land in the same bucket, and
everything interesting here is about what happens then.

**Order is gone.** No minimum, no floor, no ceiling, no range query, no sorted
iteration. Hashing deliberately scatters keys, so day 9's `floor` and
`range_query` cannot be written here at any price short of scanning everything.
That is the trade: a hash table for exact lookup, a balanced tree when order
matters.

## Why 0.75

This is the single most argued about number in the project, so here is the actual
reasoning rather than "because Java does it".

The load factor is entries divided by buckets. For **chaining**, it is exactly the
average chain length, so the cost of a lookup is roughly 1 + load/2 comparisons.
The relationship is linear and gentle:

| Load | Average chain | Memory wasted on empty buckets |
| - | - | - |
| 0.5 | 0.5 | about half the table |
| 0.75 | 0.75 | about a quarter |
| 1.0 | 1.0 | very little |
| 2.0 | 2.0 | none, but every lookup walks two entries |

There is no cliff. Choosing 0.75 is a balance between two costs that are both
mild: resize too eagerly and you waste memory and rehash more often than needed;
resize too late and chains lengthen. Three quarters sits where the wasted space is
modest and the average chain is still well under one. It is a judgement call and
0.7 or 0.8 would be defensible. **Nothing breaks at 0.75, which is exactly why the
number is a matter of taste for chaining.**

**Open addressing is a completely different story**, and it gets 0.5 here. There
is no chain to grow into: a collision has to go and sit in somebody else's slot,
which makes the next collision more likely. The expected number of probes for an
unsuccessful linear probe lookup is roughly:

```
(1 + 1/(1 - load)^2) / 2
```

| Load | Expected probes |
| - | - |
| 0.5 | 2.5 |
| 0.75 | 8.5 |
| 0.9 | 50.5 |
| 0.95 | 200.5 |

That is a cliff, not a slope. The `test_probing_cost_climbs_steeply_as_the_table
_fills` test measures real probe counts at load 0.5 and 0.9 and asserts the
difference is dramatic, so the number is backed by a measurement rather than a
formula copied out of a book.

So: **the threshold is not one number, it is a property of the collision strategy.**
Using 0.75 for open addressing, which people do because they remember the number,
gives eight probes per failed lookup where they expect one.

## Chaining against open addressing

| | Chaining | Open addressing |
| - | - | - |
| Where collisions go | a list hanging off the bucket | another slot in the array |
| Behaviour past the threshold | degrades gently | falls off a cliff |
| Memory | one pointer per entry, scattered | one array, no extra allocation |
| Cache behaviour | poor, chains are scattered nodes | excellent, especially linear probing |
| Deletion | trivial, remove from the list | awkward, needs tombstones |
| Load factor above 1 | allowed | impossible |

Neither wins. Python's dict uses open addressing, Java's `HashMap` uses chaining
that upgrades long chains into trees. Both are correct choices for their
priorities.

## The three probe sequences

**Linear**: index+1, index+2, and so on. Best cache behaviour, since it walks
straight through memory. Suffers **primary clustering**: any run of occupied slots
grows at both ends, because every key landing anywhere in the run must walk past
all of it, and long runs then attract more keys, so clusters merge into bigger
clusters.

**Quadratic**: index+1, index+4, index+9. Jumping further breaks up primary
clusters. Costs cache behaviour, and still suffers **secondary clustering**, since
keys with the same starting slot follow the identical path.

**Double hashing**: the step size comes from a second hash of the key, so two keys
sharing a start almost never share a path. Best distribution, worst cache
behaviour, two hash computations per lookup.

Two details that are easy to get wrong and are enforced in the code:

* The table size is a **power of two** and quadratic probing uses the triangular
  numbers, step(step+1)/2, rather than step^2. That combination provably visits
  every slot; plain squares do not, and can loop forever on a table that still has
  free slots. There is a test asserting every probe sequence reaches every slot.
* The second hash in double hashing must **never be zero** and is forced odd, so
  the step is coprime with a power of two size. A step of zero probes the same
  slot forever.

## Deletion, and why tombstones exist

This is the best "why is this so awkward" moment of the day.

Suppose A and B both hash to slot 3. A takes slot 3, B probes on to slot 4. Now
delete A and simply blank slot 3. A later search for B starts at slot 3, finds it
empty, and concludes B is not in the table.

**B is still there and has become unreachable.**

The fix is a tombstone: a marker meaning "empty now, but keep searching". Lookups
walk past it, insertions can reuse it. There is a test that builds exactly this
situation with deliberately colliding keys, deletes the first one, and checks the
rest are still findable. The invariant check goes further and verifies that every
stored key is still reachable along its own probe sequence, which is the general
form of the same bug.

Chaining has none of this. Remove the entry from its list and you are done. That
simplicity is a genuine argument in chaining's favour, and it is the sort of thing
complexity tables never show.

### The tombstone bug I introduced myself

Tombstones count towards the load factor, and they should: they still have to be
probed past, so a table full of them is slow even though it holds nothing.

But my first `_grow` simply doubled every time the threshold was crossed. Write and
delete the same five keys repeatedly, and the table doubles from 16 slots to 32
while holding **five entries**. The load factor was reporting real pressure and the
response was wrong.

The fix is to ask why the table is full. If there are at least as many tombstones
as live entries, the table is **clogged rather than full**, so it is rebuilt at the
same size, sweeping the graves away for free. Otherwise the pressure is real and it
doubles.

My first version of that rule compared live entries against half the load
threshold, which is fussier and happens to land exactly on the boundary in the
common case of deleting n entries and inserting n more, so it doubled anyway.
Counting graves against entries says what is actually meant.

## The worst case is an attack, not an accident

The `Colliding` class in the tests hashes everything to the same value, which
turns any hash table into a linear scan. The table stays correct and becomes no
better than a list.

That is not a hypothetical. Sending keys chosen to collide is a real denial of
service technique: every request becomes O(n) and the server falls over under a
load it should handle easily. It is why Python randomises string hashing per
process, and it is worth knowing before building a service on top of a hash map.

## The LRU cache: two structures covering each other's gaps

A cache with a size limit must evict something when full. Least recently used is
the usual policy, and the requirement is that **both** `get` and `put` are O(1).

Neither structure can do it alone:

* A **hash map** finds a key in O(1) but knows nothing about recency.
* A **doubly linked list** maintains recency order in O(1) at both ends, but
  finding a particular key in it is O(n).

Together they cover each other exactly. The list holds entries in recency order,
most recent at the front. The hash map maps each key to **the list node itself**,
not to the value. So a lookup finds the node in O(1), and because the list is
doubly linked, that node can be unlinked and moved to the front in O(1) too.

This is what day 5's insistence about the doubly linked list mattered for: being
able to unlink a node you already hold, with no idea where it sits, is the property
that makes this cache possible. A singly linked list cannot do it at any price.

Two design details worth stating:

**Membership testing does not count as a use.** Asking whether something is cached
is not the same as using it, so `in` does not touch the recency order or the hit
counters. Getting that wrong makes the statistics meaningless, since every check
would look like a hit. There is a test for it.

**Statistics are kept.** Hits, misses and evictions, because the number that
actually matters for a cache is the hit rate, and a cache you cannot measure is a
cache you cannot tune.

The characteristic failure of a structure built from two others is that they
**drift apart**: the map holds a key whose node was unlinked, or the list holds a
node the map forgot. Lookups keep working for a while, then quietly return stale
values or leak memory. The invariant check verifies the two agree on every entry,
and the stress test runs two thousand operations against a hand written reference
implementation, checking the exact recency order after every single one.
