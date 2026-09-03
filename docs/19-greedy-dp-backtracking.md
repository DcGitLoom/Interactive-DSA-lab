# Day 19: Greedy, dynamic programming, backtracking, branch and bound

Code: `dsalab/algorithms/greedy.py`, `dp.py`, `backtracking.py`,
`branch_and_bound.py`, `greedy_vs_dp.py`.
Tests: `tests/test_algorithm_design.py`.

Four techniques, and the whole day is really about one question: **how much of the
search space can you avoid looking at, and what does that cost you in correctness?**

| Technique | What it does | Optimal |
| - | - | - |
| Greedy | takes the best looking choice, never reconsiders | only when the problem has the greedy choice property |
| Dynamic programming | considers every choice, remembers the answers | yes |
| Backtracking | tries choices, undoes the illegal ones | yes, exhaustively |
| Branch and bound | backtracking that also discards losing branches | yes, if the bound is optimistic |

## Greedy, and the two properties it needs

Greedy is provably optimal only when:

1. **The greedy choice property.** Some optimal solution contains the choice that
   looks best right now. This is proved by an exchange argument: take any optimal
   solution, swap in the greedy choice, show the result is no worse.
2. **Optimal substructure.** After that choice, what remains is a smaller version of
   the same problem.

Day 18's minimum spanning trees have both, which is why Prim and Kruskal are
correct. Here the problems split into two groups, and **the split is the lesson**:

**Greedy is optimal**: activity selection (take the earliest finishing), fractional
knapsack (take the densest), Huffman coding (merge the two rarest).

**Greedy is not optimal**: 0/1 knapsack, coin change with arbitrary coins.

Those look almost identical to the ones above. Fractional and 0/1 knapsack differ
by one word, and that word is the difference between an O(n log n) greedy answer
and an NP hard problem.

### Activity selection: the rule matters more than the greed

Four rules that all sound reasonable:

| Rule | Optimal |
| - | - |
| Earliest finishing time | **yes** |
| Shortest activity | no: a short activity can straddle two others and block both |
| Earliest starting time | no: one activity starting at 9 and running all day blocks everything |
| Fewest conflicts | no, though it is close |

"Greedy works here" is not the lesson. **"Greedy works here with this particular
rule"** is, and the counterexample finder demonstrates the failure of the shortest
first rule with a real schedule rather than describing it.

The exchange argument for the correct rule is the cleanest one in the subject: take
any optimal schedule and replace its first activity with the one finishing
earliest. That one finishes no later, so it conflicts with nothing that followed,
and the count is unchanged.

### Huffman coding, and why you cannot tell by looking

Merge the two least frequent symbols repeatedly. They end up deepest, so they get
the longest codes, which is right because they are used least. The exchange
argument goes through, so it is optimal.

It sits in the same file as greedy coin change, which is **not** optimal. There is
no way to tell which is which by looking at them. **You have to do the proof.**
That is the real content of the day.

Two properties the tests check rather than assume: no code is a prefix of another
(which is what lets a decoder work with no separators), and encode then decode
returns the original, on a hundred random texts.

One detail with a story: the tie breaking counter. Two symbols of equal frequency
have no natural order, so without a tie breaker the codes change between runs while
staying equally optimal, and nothing can be tested. Determinism is a testability
requirement, not a correctness one, and it is worth adding deliberately rather than
discovering its absence in a flaky test.

## The counterexample finder

This is the fourth of the features meant to lift the project above a visualiser,
and it is the one I most wanted when learning this material.

Every textbook says greedy is not always optimal. Almost none hand you a failing
input. So `greedy_vs_dp.py` searches for them, smallest first, and reports both
answers with an explanation. Running it produces:

```
Greedy coin change is not optimal
  Input:   coins [1, 3, 4], making 6
  Greedy:  [4, 1, 1] (score 3)
  Correct: [3, 3] (score 2)
  Why:     Taking the largest coin that fits leaves a remainder needing two more
           coins. Refusing the biggest coin needs only two in total.
```

It found the classic 1, 3, 4 example on its own, which was a good sign the search
was working rather than merely running.

Four searches, and the second is the one that surprised me most:

1. **Greedy coin change is not optimal.** Three coins instead of two.
2. **Greedy coin change fails entirely.** With coins of 2 and 3 making 4, greedy
   takes a 3, cannot make 1, and reports the amount impossible. It is not: 2 and 2
   works. A method returning a worse answer is a nuisance; a method returning "no
   answer exists" when one does is a different category of wrong.
3. **Greedy 0/1 knapsack loses.** Taking the densest item leaves space nothing fits
   into.
4. **The wrong greedy rule for activity selection loses.** The sharpest one,
   because the failure is not greed but the **choice of rule**.

Three deliberate design decisions:

* **Smallest input first.** A counterexample with three coins teaches something; one
  with fifty coins and a target of 9,417 proves the same point and teaches nothing.
* **The report distinguishes "found nothing" from "proved nothing exists".** A
  search that finds nothing after 2000 inputs is weak evidence, and the summary
  says so in those words, including the count. There is a test asserting the phrase
  "not a proof" appears.
* **The finder is tested against itself.** One test re-runs both algorithms on each
  reported input and confirms they really disagree as claimed, because a
  counterexample generator that generates wrong counterexamples would be worse than
  none.

## Dynamic programming: brute force plus a notebook

It applies when there are **overlapping subproblems** (otherwise memoising buys
nothing and divide and conquer is the right shape) and **optimal substructure**.

The hard part is never the code. It is deciding **what a subproblem is**, and the
longest increasing subsequence is the clearest small example. The subproblem has to
be "the longest run **ending at position i**", not "the longest run within the
first i values". The second cannot be extended, because it does not say what the
last element is, so there is no way to check whether a new value may follow.

Three problems worth their place:

**Matrix chain multiplication.** (AB)C and A(BC) give the same answer at wildly
different costs: 7,500 multiplications against 75,000 for the same three matrices.
Database query planners solve exactly this when ordering joins.

**Edit distance.** Every cell asks the same question about the last character. The
tests check it is symmetric and obeys the triangle inequality, which are real
properties of a distance and a much stronger check than a few known values.

**0/1 knapsack.** O(n times capacity), which is **pseudo polynomial**: polynomial in
the *value* of the capacity but exponential in the number of digits used to write
it. Doubling the capacity doubles the work; adding a digit multiplies it by ten.
The problem is NP hard and this algorithm does not contradict that, which is worth
understanding rather than glossing over.

## Backtracking: the undo is the technique

```
place a piece
if still legal, recurse
remove the piece      <- the backtrack
```

Forgetting the last line does not raise an error. It leaves rubbish behind that
makes later branches look illegal, so the search quietly returns too few answers.
There is a test asserting the number of "choose" steps equals the number of "undo"
steps.

**The representation is the first and biggest pruning step.** Storing N queens as
"the column of the queen in row i" makes row conflicts impossible without ever
checking for them, cutting the space from "choose n squares from n^2" to n^n before
any other pruning happens. That single decision does more than every check in the
loop.

The solution counts (1, 0, 0, 2, 10, 4, 40, 92 for boards 1 to 8) are in the tests,
because a count is a far stronger check than eyeballing one board.

### The sudoku bug worth recording

The solver fills the cell with the **fewest legal options** first, which is the
most constrained variable heuristic and is enormously effective: a cell with one
option is a forced move that costs nothing and shrinks everything else.

Then I wrote a test with two ones in the same row, expecting an immediate "no
solution", and it ran for minutes.

The reason is subtle and general. **Backtracking only ever checks whether a new
placement is legal, so a board that already breaks the rules is never noticed
directly.** The search happily fills in around the contradiction and only fails
when it reaches a cell the conflict starves, which on a nearly empty board is an
enormous amount of work away.

The fix is to validate the given numbers before searching, which is also the
correct behaviour: a puzzle contradicting itself has no solution and needs no
search at all. The lesson generalises to any constraint solver: **check the input
is consistent before assuming the search will discover it.**

## Branch and bound: pruning what is legal but losing

Backtracking prunes **illegal** branches. Branch and bound also prunes branches that
are legal but cannot beat the best answer found so far, using a **bound**: an
optimistic estimate of the best result reachable from a partial answer.

The bound must never be pessimistic, or a branch containing the true best answer is
discarded and the algorithm returns something worse while insisting it is optimal.
That is exactly A star's admissibility rule from day 18, failing the same silent
way. A star is branch and bound over paths.

Because a bad bound fails silently, the travelling salesman tests compare against
**exhaustive search** on forty random instances rather than trusting the answer.
That is the only test that can catch it.

The nicest connection of the day: **the bound for 0/1 knapsack is the fractional
knapsack answer.** Allowing fractions can only help, so it is never an
underestimate. The greedy algorithm that is *wrong* for this problem turns out to
be exactly the right tool for pruning the search that solves it correctly.

Neither technique changes the worst case, which stays exponential. Both change what
happens in practice, often by orders of magnitude, and that gap between worst case
and typical is why NP hard problems are solved routinely despite being NP hard.
