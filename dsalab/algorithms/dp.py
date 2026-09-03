"""Dynamic programming: stop recomputing the same thing.

Day 2 showed naive Fibonacci computing fib(2) five times in one small call tree.
Dynamic programming is the general fix for that waste, and it applies whenever a
problem has:

1. **Overlapping subproblems.** The same smaller problem comes up many times. If
   every subproblem is distinct, remembering answers buys nothing, and divide and
   conquer (merge sort, quick sort) is the right approach instead.
2. **Optimal substructure.** The best answer to the whole is built from best
   answers to the parts. Greedy needs this too; dynamic programming differs by
   considering **every** choice at each step rather than committing to one.

Two ways to write it, and both are here for several problems because the difference
is real:

* **Top down (memoised).** Write the natural recursion, cache the answers. Easy to
  get right, only computes what it needs, costs stack depth.
* **Bottom up (tabulated).** Fill a table from the smallest case upwards. No
  recursion, easier to optimise the memory, but you have to work out the right
  order to fill it in.

The honest summary of the technique: **dynamic programming is brute force plus a
notebook.** Everything hard about it is in deciding what a subproblem is and what
order to solve them in.
"""

from __future__ import annotations

from dsalab.algorithms.greedy import Item
from dsalab.tracing import Step, Traced, run


def longest_increasing_subsequence(values: list[int]) -> list[int]:
    return run(longest_increasing_subsequence_traced(values))


def longest_increasing_subsequence_traced(values: list[int]) -> Traced[list[int]]:
    """The longest run of increasing values, not necessarily next to each other.

    O(n^2) here. The subproblem is "the longest increasing subsequence **ending at
    position i**", and the answer to that is one more than the best among all
    earlier positions holding a smaller value.

    Defining the subproblem as "ending at i" rather than "within the first i
    values" is the whole trick. The second version cannot be extended, because it
    does not say what the last element is, so there is no way to check whether a
    new value may follow it. **Choosing the subproblem is the hard part of dynamic
    programming**, and this is the clearest small example of it.

    There is an O(n log n) version using binary search over the smallest possible
    tail for each length, which is faster and much less obvious. The quadratic
    version is here because it shows the structure.
    """
    if not values:
        return []

    best = [1] * len(values)
    previous = [-1] * len(values)

    for index in range(1, len(values)):
        for earlier in range(index):
            if values[earlier] < values[index] and best[earlier] + 1 > best[index]:
                best[index] = best[earlier] + 1
                previous[index] = earlier
                yield Step(
                    "extend",
                    f"{values[index]} can follow {values[earlier]}, giving a run of "
                    f"{best[index]} ending at position {index}.",
                    {"index": index, "length": best[index], "after": earlier},
                )

    end = max(range(len(values)), key=lambda index: best[index])
    sequence: list[int] = []
    while end != -1:
        sequence.append(values[end])
        end = previous[end]

    sequence.reverse()
    return sequence


def knapsack_01(items: list[Item], capacity: int) -> tuple[float, list[str]]:
    return run(knapsack_01_traced(items, capacity))


def knapsack_01_traced(items: list[Item], capacity: int) -> Traced[tuple[float, list[str]]]:
    """Fill a bag of limited capacity with whole items, maximising value.

    O(n times capacity) time and memory.

    Each item gets one question: take it or leave it. The best value using the
    first i items with capacity c is the better of:

    * leaving item i, which is the best with i-1 items and the same capacity, or
    * taking it, which is its value plus the best with i-1 items and the capacity
      reduced by its weight.

    Compare that with the greedy approach in `greedy.py`, which sorts by value per
    unit weight and takes what fits. Greedy is optimal for the **fractional**
    version and can be arbitrarily bad here, because a partly filled bag wastes the
    remaining space. `greedy_vs_dp.py` finds concrete cases automatically.

    A note on the complexity, because it matters more than it looks. O(n times
    capacity) is called **pseudo polynomial**: it is polynomial in the *value* of
    the capacity but exponential in the number of bits used to write it down.
    Doubling the capacity doubles the work, but doubling the number of digits
    squares it. This problem is NP hard, and this algorithm does not contradict
    that, which is worth understanding rather than glossing over.
    """
    if capacity < 0:
        raise ValueError("capacity cannot be negative")

    count = len(items)
    table = [[0.0] * (capacity + 1) for _ in range(count + 1)]

    for index in range(1, count + 1):
        item = items[index - 1]
        weight = int(item.weight)

        for space in range(capacity + 1):
            without = table[index - 1][space]

            if weight > space:
                table[index][space] = without
            else:
                with_it = table[index - 1][space - weight] + item.value
                table[index][space] = max(without, with_it)

        yield Step(
            "row",
            f"Considered {item.name!r} (weight {weight}, value {item.value}). The best "
            f"value using the first {index} item(s) is now {table[index][capacity]}.",
            {"item": item.name, "best": table[index][capacity]},
        )

    # Walk the table backwards to find out which items were actually taken. The
    # table holds the values, not the choices, and reconstructing from it costs
    # nothing extra in memory.
    chosen: list[str] = []
    space = capacity
    for index in range(count, 0, -1):
        if table[index][space] != table[index - 1][space]:
            item = items[index - 1]
            chosen.append(item.name)
            space -= int(item.weight)

    chosen.reverse()
    return table[count][capacity], chosen


def coin_change(coins: list[int], amount: int) -> list[int] | None:
    return run(coin_change_traced(coins, amount))


def coin_change_traced(coins: list[int], amount: int) -> Traced[list[int] | None]:
    """The fewest coins making an amount, correctly this time. O(amount times coins).

    The subproblem is the fewest coins for every amount from 0 upwards, and each is
    one more than the best of the amounts reachable by removing a single coin.

    Unlike greedy, this **always** finds an answer when one exists, and always the
    best one. Greedy on coins of 1, 3, 4 makes 6 with three coins where two are
    enough, and greedy on coins of 3, 4 fails to make 6 at all.
    """
    if amount < 0:
        raise ValueError("cannot make a negative amount")
    if not coins:
        return [] if amount == 0 else None

    best = [float("inf")] * (amount + 1)
    last_coin = [0] * (amount + 1)
    best[0] = 0

    for value in range(1, amount + 1):
        for coin in coins:
            if coin <= value and best[value - coin] + 1 < best[value]:
                best[value] = best[value - coin] + 1
                last_coin[value] = coin

        if best[value] != float("inf"):
            yield Step(
                "solve",
                f"{value} can be made with {int(best[value])} coin(s), ending with a "
                f"{last_coin[value]}.",
                {"amount": value, "coins": int(best[value])},
            )

    if best[amount] == float("inf"):
        yield Step("impossible", f"{amount} cannot be made from these coins at all.",
                   {"amount": amount})
        return None

    used: list[int] = []
    remaining = amount
    while remaining > 0:
        used.append(last_coin[remaining])
        remaining -= last_coin[remaining]

    return sorted(used, reverse=True)


def matrix_chain_order(dimensions: list[int]) -> tuple[int, str]:
    return run(matrix_chain_order_traced(dimensions))


def matrix_chain_order_traced(dimensions: list[int]) -> Traced[tuple[int, str]]:
    """The cheapest way to bracket a chain of matrix multiplications. O(n^3).

    Multiplying matrices is associative: (AB)C and A(BC) give the same answer. They
    do **not** cost the same. For matrices of 10x100, 100x5 and 5x50:

    * (AB)C costs 10x100x5 + 10x5x50 = 7500 multiplications.
    * A(BC) costs 100x5x50 + 10x100x50 = 75000.

    Ten times the work for the same result, purely from where the brackets go. On
    longer chains the gap is far larger.

    The number of possible bracketings is the Catalan number, which grows
    exponentially, so trying them all is hopeless. The subproblem is the cheapest
    way to multiply the chain from i to j, and the answer tries every split point
    between them, which is why there are three loops: chain length, start position,
    split point.

    The chain length loop comes first because a longer chain's answer depends on
    shorter ones, so they must all be ready. That is the same "get the order right"
    requirement as Floyd Warshall on day 18.

    Real uses: query planners deciding the order of database joins solve exactly
    this problem, since joining in the wrong order is the same kind of catastrophe.
    """
    count = len(dimensions) - 1
    if count < 1:
        return 0, ""

    cost = [[0] * count for _ in range(count)]
    split_at = [[0] * count for _ in range(count)]

    for length in range(2, count + 1):
        for start in range(count - length + 1):
            end = start + length - 1
            cost[start][end] = float("inf")

            for split in range(start, end):
                total = (
                    cost[start][split]
                    + cost[split + 1][end]
                    + dimensions[start] * dimensions[split + 1] * dimensions[end + 1]
                )
                if total < cost[start][end]:
                    cost[start][end] = total
                    split_at[start][end] = split

            yield Step(
                "solve",
                f"The cheapest way to multiply matrices {start} to {end} costs "
                f"{cost[start][end]}, splitting after matrix {split_at[start][end]}.",
                {"start": start, "end": end, "cost": cost[start][end]},
            )

    def bracket(start: int, end: int) -> str:
        if start == end:
            return f"M{start}"
        split = split_at[start][end]
        return f"({bracket(start, split)}{bracket(split + 1, end)})"

    return cost[0][count - 1], bracket(0, count - 1)


def longest_common_subsequence(first: str, second: str) -> str:
    return run(longest_common_subsequence_traced(first, second))


def longest_common_subsequence_traced(first: str, second: str) -> Traced[str]:
    """The longest sequence of characters appearing in both, in order. O(n times m).

    This is what `diff` is built on, and what version control uses to work out
    which lines changed. It is also used in bioinformatics to compare sequences.

    The recursion is short: if the last characters match, they are part of the
    answer and the problem shrinks on both sides. If not, the answer is the better
    of dropping the last character of one or the other.
    """
    rows, columns = len(first), len(second)
    table = [[0] * (columns + 1) for _ in range(rows + 1)]

    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            if first[row - 1] == second[column - 1]:
                table[row][column] = table[row - 1][column - 1] + 1
            else:
                table[row][column] = max(table[row - 1][column], table[row][column - 1])

        yield Step(
            "row",
            f"After matching {first[:row]!r} against all of the second string, the "
            f"longest common run is {table[row][columns]} character(s).",
            {"row": row, "length": table[row][columns]},
        )

    result: list[str] = []
    row, column = rows, columns
    while row > 0 and column > 0:
        if first[row - 1] == second[column - 1]:
            result.append(first[row - 1])
            row -= 1
            column -= 1
        elif table[row - 1][column] >= table[row][column - 1]:
            row -= 1
        else:
            column -= 1

    result.reverse()
    return "".join(result)


def edit_distance(first: str, second: str) -> int:
    return run(edit_distance_traced(first, second))


def edit_distance_traced(first: str, second: str) -> Traced[int]:
    """The fewest single character edits turning one string into another. O(n times m).

    Also called Levenshtein distance. Three edits are allowed: insert, delete and
    replace, each costing 1.

    Every cell of the table asks the same question about the last character: if the
    two match, nothing needs doing and the cost is whatever the shorter pair cost.
    If not, take the cheapest of the three edits.

    Used by spell checkers, fuzzy search, DNA comparison and "did you mean" in
    command line tools.

    The memory can be reduced from O(n times m) to O(min(n, m)), because each row
    only depends on the one above it. The full table is kept here so the animation
    can show it, which is a genuine case of the visualiser affecting the design,
    and the docstring saying so is better than a silent inefficiency.
    """
    rows, columns = len(first), len(second)
    table = [[0] * (columns + 1) for _ in range(rows + 1)]

    # An empty string needs one insertion per character of the other.
    for row in range(rows + 1):
        table[row][0] = row
    for column in range(columns + 1):
        table[0][column] = column

    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            if first[row - 1] == second[column - 1]:
                table[row][column] = table[row - 1][column - 1]
            else:
                table[row][column] = 1 + min(
                    table[row - 1][column],      # delete from the first
                    table[row][column - 1],      # insert into the first
                    table[row - 1][column - 1],  # replace
                )

        yield Step(
            "row",
            f"{first[:row]!r} is {table[row][columns]} edit(s) away from {second!r}.",
            {"row": row, "distance": table[row][columns]},
        )

    return table[rows][columns]


def subset_sum(values: list[int], target: int) -> list[int] | None:
    """Whether some subset adds up exactly to `target`, and which one.

    O(n times target). The decision version of this is a classic NP complete
    problem, and this is another pseudo polynomial algorithm: fine when the target
    is small, useless when it is a 200 digit number.
    """
    reachable: dict[int, list[int]] = {0: []}

    for value in values:
        for total in sorted(reachable, reverse=True):
            if total + value <= target and total + value not in reachable:
                reachable[total + value] = reachable[total] + [value]

    return reachable.get(target)


def fibonacci_bottom_up(n: int) -> int:
    """Fibonacci filling a table upwards, using O(1) memory.

    The bookend to day 2. Naive recursion is O(2^n), memoised recursion is O(n)
    time and O(n) memory, and this is O(n) time and O(1) memory, because only the
    last two values are ever needed.

    That last step is the classic dynamic programming memory optimisation: once the
    table is filled in a known order, most of it can be thrown away as you go.
    """
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative numbers")
    if n < 2:
        return n

    previous, current = 0, 1
    for _ in range(n - 1):
        previous, current = current, previous + current
    return current
