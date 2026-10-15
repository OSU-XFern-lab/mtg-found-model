from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------
# Settings to edit if needed
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
OUT_DIR = SCRIPT_DIR / "outputs"

INPUTS = {
    "motus": DATA_DIR / "motus" / "outputs" / "manifests" / "mag_manifest.tsv",
    "jgi_img": DATA_DIR / "jgi_img" / "outputs" / "aligned" / "manifests" / "mag_manifest.tsv",
}

# Use genus by default. Species is often too noisy for a first-pass diversity view.
RANKS = ["domain", "phylum", "class_", "order", "family", "genus"]
DISPLAY_RANK = {"class_": "class"}

# Controls how noisy the printed tree is.
TOP_CHILDREN_PER_NODE = 8
MIN_CHILD_COUNT = 2
MAX_DEPTH = len(RANKS)

# Controls the bar plots.
TOP_LABELS_PER_RANK = 20
PLOT_SOURCE_BREAKDOWN = True

# If the same genome id appears in both manifests and you only want to count it once,
# set this to True. Default keeps source-specific rows because abundance = number of MAGs.
DROP_DUPLICATE_GENOMES = False


# -----------------------------
# Loading / cleaning
# -----------------------------
def clean_label(value: object, rank: str) -> str:
    """Standardize missing taxonomic labels."""
    if pd.isna(value):
        return f"unclassified {pretty_rank(rank)}"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "n/a", "null", "-"}:
        return f"unclassified {pretty_rank(rank)}"
    return text


def pretty_rank(rank: str) -> str:
    return DISPLAY_RANK.get(rank, rank)


def load_manifests() -> pd.DataFrame:
    frames = []
    for source, path in INPUTS.items():
        if not path.exists():
            print(f"[warn] Missing input for {source}: {path}")
            continue

        df = pd.read_csv(path, sep="\t", dtype=str)
        missing = [col for col in ["genome", *RANKS] if col not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        df = df[["genome", *RANKS]].copy()
        df.insert(1, "source", source)
        frames.append(df)

    if not frames:
        expected = "\n".join(f"  - {source}: {path}" for source, path in INPUTS.items())
        raise FileNotFoundError(f"No manifest files found. Expected:\n{expected}")

    df = pd.concat(frames, ignore_index=True)

    if DROP_DUPLICATE_GENOMES:
        df = df.drop_duplicates(subset=["genome"], keep="first")

    for rank in RANKS:
        df[rank] = df[rank].map(lambda x, r=rank: clean_label(x, r))

    return df


# -----------------------------
# Abundance tables
# -----------------------------
def write_basic_tables(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    source_summary = (
        df.groupby("source")
        .size()
        .reset_index(name="n_genomes")
        .sort_values("n_genomes", ascending=False)
    )
    source_summary["fraction"] = source_summary["n_genomes"] / len(df)
    source_summary.to_csv(OUT_DIR / "source_summary.tsv", sep="\t", index=False)

    rows = []
    for rank in RANKS:
        rank_counts = (
            df.groupby(["source", rank])
            .size()
            .reset_index(name="count")
            .rename(columns={rank: "label"})
        )
        rank_counts["rank"] = pretty_rank(rank)
        rank_counts["source_total"] = rank_counts.groupby("source")["count"].transform("sum")
        rank_counts["fraction_within_source"] = rank_counts["count"] / rank_counts["source_total"]
        rows.append(rank_counts[["source", "rank", "label", "count", "fraction_within_source"]])

        overall = (
            df.groupby(rank)
            .size()
            .reset_index(name="count")
            .rename(columns={rank: "label"})
        )
        overall.insert(0, "source", "all")
        overall["rank"] = pretty_rank(rank)
        overall["fraction_within_source"] = overall["count"] / len(df)
        rows.append(overall[["source", "rank", "label", "count", "fraction_within_source"]])

    rank_abundance = pd.concat(rows, ignore_index=True)
    rank_abundance = rank_abundance.sort_values(["rank", "source", "count"], ascending=[True, True, False])
    rank_abundance.to_csv(OUT_DIR / "rank_abundance.tsv", sep="\t", index=False)

    full_nodes = []
    total = len(df)
    for depth, rank in enumerate(RANKS, start=1):
        prefix_cols = RANKS[:depth]
        counts = df.groupby(prefix_cols).size().reset_index(name="count")
        for row in counts.itertuples(index=False):
            labels = list(row[:depth])
            count = int(row[-1])
            full_nodes.append(
                {
                    "depth": depth,
                    "rank": pretty_rank(rank),
                    "label": labels[-1],
                    "path": "; ".join(labels),
                    "count": count,
                    "fraction_total": count / total,
                }
            )
    pd.DataFrame(full_nodes).sort_values(["depth", "count"], ascending=[True, False]).to_csv(
        OUT_DIR / "tree_nodes_full.tsv", sep="\t", index=False
    )


# -----------------------------
# Pruned text tree
# -----------------------------
def prefix_counts(df: pd.DataFrame) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = defaultdict(int)
    for row in df[RANKS[:MAX_DEPTH]].itertuples(index=False, name=None):
        for depth in range(1, len(row) + 1):
            counts[tuple(row[:depth])] += 1
    return dict(counts)


def direct_children(counts: dict[tuple[str, ...], int], prefix: tuple[str, ...]) -> list[tuple[tuple[str, ...], int]]:
    child_depth = len(prefix) + 1
    children = []
    for path, count in counts.items():
        if len(path) == child_depth and path[: len(prefix)] == prefix:
            children.append((path, count))
    children.sort(key=lambda x: (-x[1], x[0][-1]))
    return children


def build_pruned_tree_lines(
    counts: dict[tuple[str, ...], int],
    total: int,
    prefix: tuple[str, ...] = (),
    indent: int = 0,
) -> list[str]:
    if len(prefix) >= MAX_DEPTH:
        return []

    children = [(path, count) for path, count in direct_children(counts, prefix) if count >= MIN_CHILD_COUNT]
    kept = children[:TOP_CHILDREN_PER_NODE]
    other = children[TOP_CHILDREN_PER_NODE:]

    lines = []
    for path, count in kept:
        rank = pretty_rank(RANKS[len(path) - 1])
        label = path[-1]
        pct = 100 * count / total
        lines.append(f"{'  ' * indent}- {rank}: {label}  [{count:,} MAGs; {pct:.1f}%]")
        lines.extend(build_pruned_tree_lines(counts, total, path, indent + 1))

    if other:
        other_count = sum(count for _, count in other)
        rank = pretty_rank(RANKS[len(prefix)])
        pct = 100 * other_count / total
        lines.append(f"{'  ' * indent}- {rank}: Other {rank}s  [{other_count:,} MAGs; {pct:.1f}%]")

    return lines


def write_tree(df: pd.DataFrame) -> None:
    counts = prefix_counts(df)
    lines = [
        "Taxonomic tree summary",
        "======================",
        f"Total MAGs: {len(df):,}",
        f"Ranks: {' > '.join(pretty_rank(r) for r in RANKS[:MAX_DEPTH])}",
        f"Pruning: top {TOP_CHILDREN_PER_NODE} children per node; minimum child count = {MIN_CHILD_COUNT}",
        "",
    ]
    lines.extend(build_pruned_tree_lines(counts, len(df)))
    (OUT_DIR / "tax_tree.txt").write_text("\n".join(lines) + "\n")


# -----------------------------
# Plots
# -----------------------------
def plot_top_rank(df: pd.DataFrame, rank: str) -> None:
    pretty = pretty_rank(rank)
    top_labels = df[rank].value_counts().head(TOP_LABELS_PER_RANK).index.tolist()
    plot_df = df[df[rank].isin(top_labels)].copy()

    if PLOT_SOURCE_BREAKDOWN:
        table = pd.crosstab(plot_df[rank], plot_df["source"])
        table["total"] = table.sum(axis=1)
        table = table.sort_values("total", ascending=True).drop(columns="total")
    else:
        table = plot_df[rank].value_counts().sort_values(ascending=True).to_frame("all")

    ax = table.plot(kind="barh", stacked=PLOT_SOURCE_BREAKDOWN, figsize=(9, max(4, 0.35 * len(table))))
    ax.set_title(f"Top {len(table)} {pretty} labels")
    ax.set_xlabel("Number of MAGs")
    ax.set_ylabel(pretty)
    ax.legend(title="source", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"top_{pretty}.png", dpi=200)
    plt.close()


def write_plots(df: pd.DataFrame) -> None:
    for rank in RANKS:
        plot_top_rank(df, rank)


def try_write_plotly_tree(df: pd.DataFrame) -> None:
    """Optional interactive treemap/sunburst. Skips cleanly if plotly is unavailable."""
    try:
        import plotly.express as px
    except ImportError:
        print("[warn] plotly is not installed; skipping interactive HTML tree.")
        return

    path_cols = RANKS[:MAX_DEPTH]
    grouped = df.groupby(path_cols).size().reset_index(name="count")

    fig = px.treemap(
        grouped,
        path=[px.Constant("all MAGs"), *path_cols],
        values="count",
        title="MAG taxonomic diversity treemap",
    )
    fig.write_html(OUT_DIR / "tax_tree_treemap.html")

    fig = px.sunburst(
        grouped,
        path=[px.Constant("all MAGs"), *path_cols],
        values="count",
        title="MAG taxonomic diversity sunburst",
    )
    fig.write_html(OUT_DIR / "tax_tree_sunburst.html")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_manifests()

    df.to_csv(OUT_DIR / "combined_manifest_taxonomy.tsv", sep="\t", index=False)
    write_basic_tables(df)
    write_tree(df)
    write_plots(df)
    try_write_plotly_tree(df)

    print(f"Loaded {len(df):,} MAGs from {df['source'].nunique()} source(s).")
    print(f"Wrote outputs to: {OUT_DIR}")
    print(f"Start with: {OUT_DIR / 'tax_tree.txt'}")
    print(f"Abundance table: {OUT_DIR / 'rank_abundance.tsv'}")


if __name__ == "__main__":
    main()
