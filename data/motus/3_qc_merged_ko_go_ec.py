from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path


def contig_from_gene_id(gene_id: str) -> str:
    text = (gene_id or "").strip()
    if not text:
        return ""

    # Prodigal headers can include metadata after ' # ...'; keep only the ID token.
    token = text.split(" #", 1)[0].strip()

    # Typical format: <contig_id>_<gene_index>
    if "_" in token:
        contig, suffix = token.rsplit("_", 1)
        if suffix.isdigit():
            return contig
    return token


def split_terms(value: str) -> list[str]:
    text = (value or "").strip()
    if not text or text == "-":
        return []
    return [term.strip() for term in text.split(",") if term.strip() and term.strip() != "-"]


def percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    rank = (len(sorted_values) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_values[low])
    frac = rank - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def resolve_project_dir() -> Path:
    # Try script location first.
    candidates: list[Path] = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent)

    # Then try cwd and its parents (for notebook/interactive runs).
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    seen: set[Path] = set()
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)

        # Case 1: already in motus directory.
        merged_direct = base / "outputs" / "annotations" / "merged_ko_go_ec.tsv"
        if merged_direct.exists():
            return base

        # Case 2: running from repo root or a sibling path.
        merged_nested = base / "data" / "motus" / "outputs" / "annotations" / "merged_ko_go_ec.tsv"
        if merged_nested.exists():
            return base / "data" / "motus"

    checked = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not locate mOTUs project directory containing outputs/annotations/merged_ko_go_ec.tsv. "
        f"Checked:\n  - {checked}"
    )


project_dir = resolve_project_dir()
merged_tsv = project_dir / "outputs" / "annotations" / "merged_ko_go_ec.tsv"
out_dir = project_dir / "outputs" / "annotations" / "qc_merged_ko_go_ec"
out_dir.mkdir(parents=True, exist_ok=True)

if not merged_tsv.exists():
    raise FileNotFoundError(f"Missing merged file: {merged_tsv}")

print(f"Reading: {merged_tsv}")

# Distribution counters
genes_per_contig: Counter[str] = Counter()
genes_per_mag: Counter[str] = Counter()

# Coverage counters
total_genes = 0
genes_with_ko = 0
genes_with_go = 0
genes_with_ec = 0

# Per-gene annotation count distributions
ko_terms_per_gene: Counter[int] = Counter()
go_terms_per_gene: Counter[int] = Counter()
ec_terms_per_gene: Counter[int] = Counter()

with open(merged_tsv, "r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")

    for i, row in enumerate(reader, start=1):
        mag = (row.get("mag") or "").strip()
        gene_id = (row.get("gene_id") or "").strip()

        if not gene_id:
            continue

        total_genes += 1

        if mag:
            genes_per_mag[mag] += 1

        contig = contig_from_gene_id(gene_id)
        if contig:
            genes_per_contig[contig] += 1

        ko_terms = split_terms(row.get("KO", ""))
        go_terms = split_terms(row.get("GO", ""))
        ec_terms = split_terms(row.get("EC", ""))

        ko_terms_per_gene[len(ko_terms)] += 1
        go_terms_per_gene[len(go_terms)] += 1
        ec_terms_per_gene[len(ec_terms)] += 1

        if ko_terms:
            genes_with_ko += 1
        if go_terms:
            genes_with_go += 1
        if ec_terms:
            genes_with_ec += 1

        if i % 200000 == 0:
            print(f"  processed {i:,} rows...")

print("\nFinished scanning merged file.")
print(f"Total genes (rows): {total_genes:,}")

if total_genes == 0:
    raise RuntimeError("merged_ko_go_ec.tsv has no data rows.")

frac_ko = genes_with_ko / total_genes
frac_go = genes_with_go / total_genes
frac_ec = genes_with_ec / total_genes

print("\nAnnotation fractions (within merged_ko_go_ec.tsv):")
print(f"  KO: {genes_with_ko:,}/{total_genes:,} = {frac_ko:.4%}")
print(f"  GO: {genes_with_go:,}/{total_genes:,} = {frac_go:.4%}")
print(f"  EC: {genes_with_ec:,}/{total_genes:,} = {frac_ec:.4%}")

print(
    "\nNote: This merged file is built from genes with at least one annotation (KO/GO/EC). "
    "So these fractions are relative to merged rows, not all predicted genes in each MAG."
)

contig_counts = sorted(genes_per_contig.values())
mag_counts = sorted(genes_per_mag.values())


def print_distribution(name: str, values: list[int]) -> None:
    if not values:
        print(f"\n{name}: no data")
        return

    n = len(values)
    mean = sum(values) / n
    print(f"\n{name}")
    print(f"  n: {n:,}")
    print(f"  min: {values[0]:,}")
    print(f"  mean: {mean:,.2f}")
    print(f"  median: {percentile(values, 0.50):,.2f}")
    print(f"  p90: {percentile(values, 0.90):,.2f}")
    print(f"  p95: {percentile(values, 0.95):,.2f}")
    print(f"  p99: {percentile(values, 0.99):,.2f}")
    print(f"  max: {values[-1]:,}")


print_distribution("Genes per contig", contig_counts)
print_distribution("Genes per MAG", mag_counts)

print("\nTop 20 MAGs by gene count:")
for mag, count in genes_per_mag.most_common(20):
    print(f"  {mag}\t{count}")

print("\nTop 20 contigs by gene count:")
for contig, count in genes_per_contig.most_common(20):
    print(f"  {contig}\t{count}")

# Write detailed outputs for notebook exploration.
with open(out_dir / "genes_per_mag.tsv", "w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(["mag", "n_genes"])
    for mag, count in genes_per_mag.most_common():
        writer.writerow([mag, count])

with open(out_dir / "genes_per_contig.tsv", "w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(["contig", "n_genes"])
    for contig, count in genes_per_contig.most_common():
        writer.writerow([contig, count])

with open(out_dir / "ko_terms_per_gene_distribution.tsv", "w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(["n_ko_terms", "n_genes"])
    for n_terms in sorted(ko_terms_per_gene):
        writer.writerow([n_terms, ko_terms_per_gene[n_terms]])

with open(out_dir / "go_terms_per_gene_distribution.tsv", "w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(["n_go_terms", "n_genes"])
    for n_terms in sorted(go_terms_per_gene):
        writer.writerow([n_terms, go_terms_per_gene[n_terms]])

with open(out_dir / "ec_terms_per_gene_distribution.tsv", "w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(["n_ec_terms", "n_genes"])
    for n_terms in sorted(ec_terms_per_gene):
        writer.writerow([n_terms, ec_terms_per_gene[n_terms]])

with open(out_dir / "annotation_fractions.tsv", "w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(["metric", "n_genes", "fraction_of_merged_rows"])
    writer.writerow(["with_KO", genes_with_ko, f"{frac_ko:.8f}"])
    writer.writerow(["with_GO", genes_with_go, f"{frac_go:.8f}"])
    writer.writerow(["with_EC", genes_with_ec, f"{frac_ec:.8f}"])

try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    plt.hist(contig_counts, bins=80)
    plt.xlabel("Genes per contig")
    plt.ylabel("Number of contigs")
    plt.title("Distribution of genes per contig")
    plt.tight_layout()
    plt.savefig(out_dir / "distribution_genes_per_contig.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.hist(mag_counts, bins=80)
    plt.xlabel("Genes per MAG")
    plt.ylabel("Number of MAGs")
    plt.title("Distribution of genes per MAG")
    plt.tight_layout()
    plt.savefig(out_dir / "distribution_genes_per_mag.png", dpi=160)
    plt.close()

    labels = ["KO", "GO", "EC"]
    values = [frac_ko, frac_go, frac_ec]
    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Fraction of genes")
    plt.title("Fraction of genes with annotations")
    plt.tight_layout()
    plt.savefig(out_dir / "annotation_fractions.png", dpi=160)
    plt.close()

    # Cap the x-axis for readability and include the heavy tail in the last bin label.
    go_max = max(go_terms_per_gene) if go_terms_per_gene else 0
    ec_max = max(ec_terms_per_gene) if ec_terms_per_gene else 0
    x_cap = max(10, min(60, max(go_max, ec_max)))

    go_plot = Counter(go_terms_per_gene)
    ec_plot = Counter(ec_terms_per_gene)
    if go_max > x_cap:
        go_plot[x_cap] += sum(v for k, v in go_terms_per_gene.items() if k > x_cap)
    if ec_max > x_cap:
        ec_plot[x_cap] += sum(v for k, v in ec_terms_per_gene.items() if k > x_cap)

    x_vals = list(range(0, x_cap + 1))
    go_y = [go_plot.get(x, 0) for x in x_vals]
    ec_y = [ec_plot.get(x, 0) for x in x_vals]

    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, go_y, label="GO terms per gene")
    plt.plot(x_vals, ec_y, label="EC terms per gene")
    plt.xlabel(f"Terms per gene (last bin = {x_cap} includes tail)")
    plt.ylabel("Number of genes")
    plt.title("Distribution of GO/EC terms per gene")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "distribution_go_ec_terms_per_gene.png", dpi=160)
    plt.close()

    print(f"\nSaved plots to: {out_dir}")
except Exception as exc:
    print(f"\nSkipping plots (matplotlib unavailable or plotting failed): {exc}")

print(f"Saved QC tables to: {out_dir}")
