"""The Interactive DSA Lab visualiser.

    streamlit run app/main.py

This file is deliberately thin. Every decision it makes lives in
`app/presentation.py`, which is plain Python and is covered by the test suite. What
remains here is layout: widgets, charts and text. The reasoning is in that module's
docstring, and it comes down to keeping the untestable part small enough to check
by reading it.
"""

from __future__ import annotations

import streamlit as st

from app.presentation import (
    AVL_RULES,
    INPUT_EXPLANATIONS,
    INPUT_KINDS,
    RED_BLACK_RULES,
    available_algorithms,
    build_frames,
    frame_at,
    inspect,
    measure_algorithm,
    race,
    suggest_input,
    tree_layout,
)
from dsalab.algorithms.greedy_vs_dp import run_all_searches
from dsalab.structures.avl import AVLTree
from dsalab.structures.bst import BinarySearchTree
from dsalab.structures.red_black import RedBlackTree

st.set_page_config(page_title="Interactive DSA Lab", layout="wide")


def draw_array(frame, container) -> None:
    """Draw one frame of an array as a bar chart with the active positions marked."""
    import pandas as pd

    if frame is None:
        container.info("Nothing to draw yet.")
        return

    data = pd.DataFrame({
        "position": range(len(frame.values)),
        "value": frame.values,
        "state": [
            "active" if index in frame.highlighted else "resting"
            for index in range(len(frame.values))
        ],
    })
    container.bar_chart(data, x="position", y="value", color="state", height=260)
    container.caption(frame.note)


def sorting_tab() -> None:
    st.subheader("Watch a sort")
    st.write(
        "Every algorithm here is the same code the tests run and the benchmarks time. "
        "There is no separate animated copy that could drift out of agreement with it."
    )

    left, middle, right = st.columns(3)
    algorithm = left.selectbox("Algorithm", available_algorithms())
    kind = middle.selectbox("Input", INPUT_KINDS)
    size = right.slider("How many values", 5, 40, 16)

    st.caption(INPUT_EXPLANATIONS[kind])

    values = suggest_input(kind, size)
    frames = build_frames(values, algorithm)

    if not frames:
        st.info("That input needed no work at all.")
        return

    position = st.slider("Step", 0, len(frames) - 1, 0)
    draw_array(frames[position], st)

    counts: dict[str, int] = {}
    for frame in frames[: position + 1]:
        counts[frame.kind] = counts.get(frame.kind, 0) + 1

    columns = st.columns(max(2, len(counts)))
    for column, (name, count) in zip(columns, sorted(counts.items()), strict=False):
        column.metric(name, count)

    st.caption(f"{len(frames)} steps in total for {size} values.")


def race_tab() -> None:
    st.subheader("Race two algorithms")
    st.write(
        "Both run on **identical** input, which is the whole point: racing on two "
        "different random arrays measures the arrays as much as the algorithms."
    )

    left, middle, right, fourth = st.columns(4)
    first = left.selectbox("First", available_algorithms(), index=2)
    second = middle.selectbox("Second", available_algorithms(), index=5)
    kind = right.selectbox("Input", INPUT_KINDS, key="race-input")
    size = fourth.slider("How many values", 5, 40, 20, key="race-size")

    st.caption(INPUT_EXPLANATIONS[kind])

    contest = race(suggest_input(kind, size), first, second)
    position = st.slider("Step", 0, max(0, contest.length - 1), 0, key="race-step")

    columns = st.columns(2)
    for column, competitor in zip(columns, (contest.left, contest.right), strict=True):
        column.markdown(f"**{competitor.name}**")
        draw_array(frame_at(competitor, position), column)
        column.metric("comparisons", competitor.comparisons)
        column.metric("moves", competitor.swaps)

    st.success(contest.verdict())


def complexity_tab() -> None:
    st.subheader("Complexity detective")
    st.write(
        "Rather than trusting the docstring, this counts the comparisons each "
        "algorithm makes across growing input sizes, fits every standard growth curve "
        "to the measurements, and reports which one the data actually matches."
    )

    algorithm = st.selectbox("Algorithm", available_algorithms(), key="complexity")
    verdict = measure_algorithm(algorithm)

    st.metric("Measured growth", verdict.best.curve, f"R squared {verdict.best.r_squared:.4f}")
    st.write(verdict.summary())

    import pandas as pd

    st.dataframe(
        pd.DataFrame(
            {
                "curve": [fit.curve for fit in verdict.all_fits],
                "R squared": [round(fit.r_squared, 4) for fit in verdict.all_fits],
            }
        ),
        hide_index=True,
    )
    st.line_chart(
        pd.DataFrame({"size": verdict.sizes, "comparisons": verdict.measurements}),
        x="size",
        y="comparisons",
    )


def structures_tab() -> None:
    st.subheader("Trees, and the rules they keep")
    st.write(
        "Insert values and watch the shape. The rules beside the tree are the ones the "
        "structure states in its own code, checked live after every insertion."
    )

    left, right = st.columns(2)
    kind = left.selectbox("Structure", ["plain search tree", "AVL tree", "red black tree"])
    order = right.selectbox("Insertion order", ["sorted", "random"])
    size = st.slider("How many values", 3, 40, 15, key="tree-size")

    values = suggest_input("sorted" if order == "sorted" else "random", size)

    if kind == "plain search tree":
        tree, rules, layout_kind = BinarySearchTree(values), [], "bst"
    elif kind == "AVL tree":
        tree, rules, layout_kind = AVLTree(values), AVL_RULES, "avl"
    else:
        tree, rules, layout_kind = RedBlackTree(values), RED_BLACK_RULES, "red_black"

    st.metric("Height", tree.height(), help="Edges on the longest root to leaf path.")

    if order == "sorted" and kind == "plain search tree":
        st.warning(
            f"Sorted input has turned this into a chain of height {tree.height()}, which "
            "is a linked list wearing a tree costume. This is exactly the failure the "
            "AVL and red black trees exist to prevent: try them with the same input."
        )

    import pandas as pd

    nodes = tree_layout(tree, layout_kind)
    if nodes:
        st.scatter_chart(
            pd.DataFrame(nodes),
            x="x",
            y="y",
            color="colour" if kind == "red black tree" else None,
            size=None,
            height=300,
        )

    panel = inspect(tree, rules)
    st.markdown("**Rules**")
    for rule in panel.rules:
        if panel.status_of(rule) == "holds":
            st.markdown(f"- holds: {rule}")
        else:
            st.markdown(f"- **broken**: {rule} ({panel.broken[rule]})")

    if panel.healthy:
        st.success("Every rule the structure states about itself currently holds.")


def counterexample_tab() -> None:
    st.subheader("Where greedy goes wrong")
    st.write(
        "Every book says greedy algorithms are not always optimal and almost none hand "
        "you a failing input. This searches for them, smallest first, and shows both "
        "answers side by side."
    )

    if st.button("Search for counterexamples"):
        with st.spinner("Searching small inputs, smallest first..."):
            reports = run_all_searches()

        for report in reports:
            st.markdown(f"### {report.problem}")

            if not report.found_any:
                st.info(report.summary())
                continue

            example = report.smallest
            st.error(f"Found one after {report.tried} inputs.")
            st.code(example.inputs, language="text")

            left, right = st.columns(2)
            left.metric("Greedy", str(example.greedy_answer), f"score {example.greedy_score}")
            right.metric("Correct", str(example.correct_answer),
                         f"score {example.correct_score}")
            st.write(example.explanation)


def main() -> None:
    st.title("Interactive DSA Lab")
    st.caption(
        "Data structures and algorithms written from scratch, traced, tested and "
        "benchmarked. Everything you see here runs the same code as the test suite."
    )

    watch, contest, detective, structures, greedy = st.tabs(
        ["Watch a sort", "Race", "Complexity detective", "Trees", "Where greedy fails"]
    )

    with watch:
        sorting_tab()
    with contest:
        race_tab()
    with detective:
        complexity_tab()
    with structures:
        structures_tab()
    with greedy:
        counterexample_tab()


if __name__ == "__main__":
    main()
