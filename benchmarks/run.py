"""Run every benchmark and write the results.

    python -m benchmarks.run           # the standard set
    python -m benchmarks.run --quick   # smaller sizes, for a fast check

Each comparison below answers a specific question rather than just producing
numbers, and the question is in the title.
"""

from __future__ import annotations

import random
import sys

from benchmarks.harness import (
    Comparison,
    measure,
    measure_operations,
    plot,
    relative_speed,
    save,
)
from dsalab.algorithms.searching import binary_search, linear_search
from dsalab.algorithms.sorting import (
    insertion_sort,
    merge_sort,
    quick_sort,
)
from dsalab.structures.avl import AVLTree
from dsalab.structures.bst import BinarySearchTree
from dsalab.structures.dynamic_array import DynamicArray
from dsalab.structures.hash_table import ChainedHashTable
from dsalab.structures.heap import BinaryHeap, heap_sort
from dsalab.structures.linked_list import SinglyLinkedList
from dsalab.structures.red_black import RedBlackTree
from dsalab.structures.trie import Trie

SIZES = [500, 1000, 2000, 4000, 8000]
QUICK_SIZES = [200, 400, 800, 1600]


def shuffled(size: int, seed: int = 20260904) -> list[int]:
    values = list(range(size))
    random.Random(seed + size).shuffle(values)
    return values


def ascending(size: int) -> list[int]:
    """Already sorted input, which is the best case for some sorts and the worst
    case for others, and is far more common in real data than random input."""
    return list(range(size))


def sorting_comparison(sizes: list[int]) -> Comparison:
    """Which sorts survive at scale, and does each one grow the way it claims."""
    comparison = Comparison("Sorting, random input")

    comparison.results.append(measure(
        "merge sort", "O(n log n)", shuffled, merge_sort, sizes,
        notes="Guaranteed n log n and stable, at the cost of O(n) extra memory.",
    ))
    comparison.results.append(measure(
        "quick sort (median of three)", "O(n log n)", shuffled, quick_sort, sizes,
        notes="Usually the fastest of these, because partitioning scans memory in order.",
    ))
    comparison.results.append(measure(
        "heap sort", "O(n log n)", shuffled, heap_sort, sizes,
        notes="Same complexity, in place, and slower in practice because sifting jumps "
              "between index i and 2i+1 and misses the cache almost every step.",
    ))
    comparison.results.append(measure(
        "insertion sort", "O(n^2)", shuffled, insertion_sort, sizes[:4],
        notes="Only measured at the smaller sizes, since quadratic growth makes the rest "
              "pointless. It beats everything here on tiny or nearly sorted input.",
    ))
    # The C sort needs much larger inputs before it takes long enough to measure.
    # At the sizes used above it runs in microseconds, which is small enough that
    # timer resolution and scheduling noise dominate, and the fitted curve then
    # describes the noise rather than the algorithm. Scaling the sizes up is the
    # honest fix; the first version of this benchmark reported Timsort as
    # quadratic purely because it was too fast to time.
    comparison.results.append(measure(
        "Python list.sort (Timsort, in C)", "O(n log n)", shuffled, sorted,
        [size * 50 for size in sizes],
        notes="The reference, measured at fifty times the size because at the sizes above "
              "it is too fast to time reliably. It is far faster in absolute terms "
              "because it is C rather than Python, so only the shape of the curve is a "
              "fair comparison.",
    ))

    return comparison


def sorted_input_comparison(sizes: list[int]) -> Comparison:
    """What already sorted input does to each algorithm.

    This is where the pivot choice on day 15 and the adaptivity on day 15 both show
    up as real timings rather than as claims.
    """
    comparison = Comparison("Sorting, already sorted input")

    comparison.results.append(measure(
        "insertion sort", "O(n)", ascending, insertion_sort, sizes,
        notes="Adaptive: the inner loop stops immediately, so sorted input is linear.",
    ))
    comparison.results.append(measure(
        "merge sort", "O(n log n)", ascending, merge_sort, sizes,
        notes="Not adaptive. It does the same work whatever the input, which is the "
              "price of having no bad case.",
    ))
    comparison.results.append(measure(
        "quick sort (median of three)", "O(n log n)", ascending, quick_sort, sizes,
        notes="Median of three turns sorted input into the best case. Taking the first "
              "element as the pivot would make it the worst case instead.",
    ))

    return comparison


def lookup_comparison(sizes: list[int]) -> Comparison:
    """Hash table against balanced tree against linear scan, for plain lookup."""
    comparison = Comparison("Looking up a key")

    def build_table(size: int) -> ChainedHashTable:
        table = ChainedHashTable()
        for value in range(size):
            table[value] = value
        return table

    def build_avl(size: int) -> AVLTree:
        return AVLTree(shuffled(size))

    def build_red_black(size: int) -> RedBlackTree:
        return RedBlackTree(shuffled(size))

    # The claimed curve is for the *whole* benchmark, not one operation. Each of
    # these does a number of lookups proportional to n, so a structure with O(1)
    # lookups totals O(n) and one with O(log n) lookups totals O(n log n). Getting
    # this wrong was my first mistake here: the report flagged the hash table as
    # disagreeing with O(1) when it was doing n/7 lookups per run.
    comparison.results.append(measure(
        "hash table", "O(n)", build_table,
        lambda table: [table.get(value, None) for value in range(0, len(table), 7)],
        sizes,
        notes="Constant per lookup, so the total grows with the number of lookups only.",
    ))
    comparison.results.append(measure(
        "AVL tree", "O(n log n)", build_avl,
        lambda tree: [value in tree for value in range(0, len(tree), 7)], sizes,
        notes="Slower than the hash table, and it can answer range and order questions "
              "that the hash table cannot answer at all.",
    ))
    comparison.results.append(measure(
        "red black tree", "O(n log n)", build_red_black,
        lambda tree: [value in tree for value in range(0, len(tree), 7)], sizes,
        notes="A looser height bound than AVL, bought with far cheaper rebalancing.",
    ))

    return comparison


def search_comparison(sizes: list[int]) -> Comparison:
    """Binary search against a linear scan, on sorted data."""
    comparison = Comparison("Searching a sorted list")

    # Fifty searches per run whatever the size, so the total is fifty times the
    # cost of one search: O(log n) for binary search and O(n) for the scan.
    comparison.results.append(measure(
        "binary search", "O(log n)", ascending,
        lambda values: [binary_search(values, target)
                        for target in range(0, len(values), max(1, len(values) // 50))],
        sizes,
    ))
    comparison.results.append(measure(
        "linear scan", "O(n)", ascending,
        lambda values: [linear_search(values, target)
                        for target in range(0, len(values), max(1, len(values) // 50))],
        sizes,
        notes="Beats binary search on small inputs, since it walks memory in order and "
              "needs no sorted input to begin with.",
    ))

    return comparison


def structure_comparison(sizes: list[int]) -> Comparison:
    """Where a linked list beats an array and where it loses badly."""
    comparison = Comparison("Building and indexing a sequence")

    comparison.results.append(measure(
        "dynamic array, append", "O(n)", lambda size: size,
        lambda size: [DynamicArray(range(size))], sizes,
        notes="n appends, each O(1) amortised, so the total is linear despite the "
              "occasional full copy.",
    ))
    comparison.results.append(measure(
        "linked list, append", "O(n)", lambda size: size,
        lambda size: [SinglyLinkedList(range(size))], sizes,
        notes="Also linear thanks to the tail pointer, but with an object per element "
              "and no cache friendliness.",
    ))
    comparison.results.append(measure(
        "dynamic array, index every tenth", "O(n)",
        lambda size: DynamicArray(range(size)),
        lambda array: [array[index] for index in range(0, len(array), 10)], sizes,
        notes="Address arithmetic, so each read is constant time.",
    ))
    comparison.results.append(measure(
        "linked list, index every tenth", "O(n^2)",
        lambda size: SinglyLinkedList(range(size)),
        lambda chain: [chain.get(index) for index in range(0, len(chain), 10)], sizes[:4],
        notes="Each read walks from the head, so n/10 reads cost O(n^2) in total. This "
              "is the clearest measurement of what giving up contiguous memory costs.",
    ))

    return comparison


def heap_build_comparison(sizes: list[int]) -> Comparison:
    """The day 13 claim that building a heap all at once is linear."""
    comparison = Comparison("Building a heap")

    def swaps_building_at_once(size: int) -> float:
        from dsalab.tracing import count_kinds, record

        heap = BinaryHeap.__new__(BinaryHeap)
        heap._key = lambda value: value
        heap._items = shuffled(size)
        _, steps = record(heap.heapify_traced())
        return count_kinds(steps).get("swap", 0)

    def swaps_pushing_one_at_a_time(size: int) -> float:
        from dsalab.tracing import count_kinds, record

        heap = BinaryHeap()
        total = 0
        for value in shuffled(size):
            _, steps = record(heap.push_traced(value))
            total += count_kinds(steps).get("swap", 0)
        return total

    comparison.results.append(measure_operations(
        "heapify, all at once", "O(n)", swaps_building_at_once, sizes,
        notes="Counted in swaps rather than seconds, because counting is deterministic. "
              "Most nodes are near the bottom and barely move, which is why this is "
              "linear rather than n log n.",
    ))
    comparison.results.append(measure_operations(
        "pushing one at a time", "O(n log n)", swaps_pushing_one_at_a_time, sizes,
        notes="Every item can climb the full height, so this really is n log n.",
    ))

    return comparison


def tree_shape_comparison(sizes: list[int]) -> Comparison:
    """The day 9 and day 10 story: what balancing is worth, in one measurement."""
    comparison = Comparison("Tree height with sorted input")

    comparison.results.append(measure_operations(
        "plain search tree", "O(n)",
        lambda size: float(BinarySearchTree(range(size)).height()), sizes,
        notes="Sorted input turns it into a linked list, which is the failure the AVL "
              "tree exists to prevent.",
    ))
    comparison.results.append(measure_operations(
        "AVL tree", "O(log n)",
        lambda size: float(AVLTree(range(size)).height()), sizes,
        notes="The same input, kept within about 1.44 log2(n) by rotations.",
    ))
    comparison.results.append(measure_operations(
        "red black tree", "O(log n)",
        lambda size: float(RedBlackTree(range(size)).height()), sizes,
        notes="Taller than the AVL tree and cheaper to maintain.",
    ))

    return comparison


def trie_comparison(sizes: list[int]) -> Comparison:
    """The day 17 claim that a trie lookup does not care how many words are stored."""
    comparison = Comparison("Prefix lookup")

    def build_trie(size: int) -> Trie:
        rng = random.Random(size)
        words = ["".join(rng.choice("abcdefgh") for _ in range(8)) for _ in range(size)]
        return Trie(words)

    def build_list(size: int) -> list[str]:
        rng = random.Random(size)
        return ["".join(rng.choice("abcdefgh") for _ in range(8)) for _ in range(size)]

    comparison.results.append(measure(
        "trie, words with a prefix", "O(n)", build_trie,
        lambda trie: trie.words_with_prefix("ab"), sizes,
        notes="Walking to the prefix costs the same whatever the trie holds. The growth "
              "measured here is the cost of producing the matches, which grows with how "
              "many words share the prefix, so linear is the honest claim for the whole "
              "operation rather than for the lookup part.",
    ))
    comparison.results.append(measure(
        "list scan, words with a prefix", "O(n)", build_list,
        lambda words: [word for word in words if word.startswith("ab")], sizes,
        notes="Has to look at every word, which is what the trie avoids.",
    ))

    return comparison


def run_all(quick: bool = False) -> list[Comparison]:
    sizes = QUICK_SIZES if quick else SIZES

    return [
        sorting_comparison(sizes),
        sorted_input_comparison(sizes),
        lookup_comparison(sizes),
        search_comparison(sizes),
        structure_comparison(sizes),
        heap_build_comparison(sizes),
        tree_shape_comparison(sizes),
        trie_comparison(sizes),
    ]


def main() -> None:
    quick = "--quick" in sys.argv
    print(f"Running benchmarks{' (quick)' if quick else ''}. This takes a minute or two.\n")

    comparisons = run_all(quick=quick)

    for comparison in comparisons:
        print(comparison.report())
        speeds = relative_speed(comparison.results)
        if len(speeds) > 1:
            fastest = min(speeds, key=lambda name: speeds[name])
            others = ", ".join(
                f"{name} {ratio:.1f}x" for name, ratio in speeds.items() if name != fastest
            )
            print(f"  At the largest size, {fastest} was fastest ({others} slower).")
        print()

    written = save(comparisons)
    print(f"Wrote {written}")

    charts = plot(comparisons)
    if charts:
        print(f"Wrote {len(charts)} chart(s) to {charts[0].parent}")
    else:
        print(
            "matplotlib is not installed, so no charts were drawn. The JSON above holds "
            "every number, and `pip install matplotlib` adds the pictures."
        )

    disagreements = [
        result
        for comparison in comparisons
        for result in comparison.results
        if not result.matches_claim
    ]
    if disagreements:
        print("\nMeasurements that disagree with the claimed complexity:")
        for result in disagreements:
            print(f"  {result.summary()}")
        print(
            "\nA disagreement is not automatically a bug. Timing is noisy, and neighbouring "
            "curves are hard to separate over a narrow range of sizes. It is a prompt to "
            "look, not a verdict."
        )
    else:
        print("\nEvery measured curve matched the complexity its code claims.")


if __name__ == "__main__":
    main()
