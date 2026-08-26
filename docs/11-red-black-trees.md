# Day 11: Red black trees

Code: `dsalab/structures/red_black.py`. Tests: `tests/test_red_black.py`.

## Balance without measuring

An AVL tree measures. Every node stores its height and the tree rebalances
whenever two subtrees differ by more than one.

A red black tree stores **one bit per node**, a colour, and enforces five rules
that never mention height at all:

1. Every node is red or black.
2. The root is black.
3. Every empty position counts as black.
4. A red node never has a red child.
5. Every path from a node down to any empty position passes through the same
   number of black nodes. That count is the black height.

## Why one bit is enough

Rules 4 and 5 together force the balance, and the argument is two sentences:

* Rule 5 says every root to leaf path has the same number of black nodes, so the
  **shortest** possible path is all black.
* Rule 4 says reds cannot be adjacent, so the **longest** possible path alternates
  red and black, and is therefore at most twice as long as the shortest.

No path is more than twice any other, which bounds the height at 2 log2(n + 1).

That is looser than AVL's 1.44 log2(n), so lookups are slightly slower. There is
a test that builds both trees from the same input and asserts the red black one is
at least as tall, because that is the price being paid.

## What the price buys

| | AVL | Red black |
| - | - | - |
| Height | 1.44 log2(n) | 2 log2(n) |
| Rotations per insert | at most 1 | at most 2 |
| Rotations per delete | up to log n | **at most 3** |

The deletion row is the whole reason this structure exists. AVL deletion can
rotate at every level on the way back up; red black deletion never rotates more
than three times regardless of tree size. For write heavy work that difference
dominates, which is why `std::map`, Java's `TreeMap` and the Linux kernel all use
red black trees.

Both of those bounds are tested directly by counting rotation steps across
hundreds of operations.

## Why new nodes are red

This single decision is what makes insertion cheap.

A red node adds nothing to the black height, so **rule 5 is never broken by an
insertion**. Rule 5 is the expensive rule, because it is global: it constrains
every path in the tree at once. The only rule a red insertion can break is rule
4, no two reds in a row, and that is a purely local problem involving the node,
its parent and its uncle.

Colouring a new node black would break rule 5 on every path through it, meaning a
repair across the whole tree.

## Insertion repair: it all depends on the uncle

**Red uncle.** Recolour: parent and uncle go black, grandparent goes red. No
pointer moves at all. But the grandparent is now red and may clash with its own
parent, so the problem moves two levels up and the loop repeats.

**Black uncle.** Recolouring cannot work, because making the parent black would
add a black node to this side only and break rule 5. A rotation is needed, and
after it the loop always ends. If the new node is on the inside, one extra
rotation straightens it first, exactly like the AVL double rotation cases.

That asymmetry gives the structure its characteristic cost profile: **a single
insertion can recolour O(log n) times but can never rotate more than twice**,
because a rotation always terminates the loop while a recolouring does not.

A test asserts exactly this, and writing it taught me something. My first version
compared total recolourings against total rotations after inserting a thousand
sorted values, and it failed: the counts came out roughly equal. On sorted input
nearly every insertion hits the black uncle case and rotates, so in aggregate the
two are similar. **The asymmetry is per operation, not in total.** The test now
records the worst single insertion, and the comment records the mistake, because
the wrong version looked perfectly reasonable.

## Deletion repair: the extra black

Deletion is harder than insertion, and the reason is worth stating plainly: a red
red clash is local, while a missing black is global, affecting every path through
that position.

The idea that makes it tractable is to imagine the node now sitting in the vacated
spot as carrying an **extra black**, a debt of one black node that its paths are
short by. The job is to discharge that debt. Four cases, depending on the sibling:

| Case | Sibling | What happens |
| - | - | - |
| 1 | Red | Not a fix by itself. Rotate to produce a black sibling and one of the cases below. |
| 2 | Black, both children black | Paint the sibling red. Both sides are now short by one, which is consistent, so the debt moves up to the parent and the loop repeats. |
| 3 | Black, far child black, near child red | Rotate the sibling to turn this into case 4. |
| 4 | Black, far child red | One rotation at the parent clears the debt. The loop ends. |

Cases 1 and 3 do not solve anything. They exist purely to funnel every situation
into cases 2 and 4, which are the ones that make progress. Case 2 is the only one
that iterates, which is why deletion is O(log n) overall while still using at most
three rotations.

## The sentinel

Every empty position points at one shared black sentinel node rather than None.
This is not a micro optimisation, it is what makes deletion writable at all: the
fixup has to ask about the colour and the parent of a position that no longer
holds anything, and a sentinel can answer while None cannot. It also makes rule 3
structural rather than a special case scattered through the code.

## Why invariant checking mattered most here

This is the hardest structure in the project, and it is also the one where a bug
is least visible. **A red black tree can hold every value in the correct order
while its colours are nonsense.** Every lookup and every traversal keeps returning
the right answer, and the only symptom is that the balance quietly degrades over
time.

So the invariant check names each rule separately, and almost every test runs
inside `checked()`, verifying all five rules plus the search order plus every
parent pointer after every single operation. The strongest test runs two thousand
mixed insertions and deletions against a plain Python set with checking on.

Separately named rules also make the visualiser useful: the app can list the five
rules beside the tree and mark which currently hold. Watching rule 4 break on an
insertion and be repaired by a rotation is a far better explanation of why
rotations exist than any description of them.
