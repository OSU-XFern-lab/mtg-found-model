#!/usr/bin/env python3
"""Turn biological facts into natural-sounding sentences.

This file controls wording only. ``views.py`` decides which facts a gene has.
Every random choice uses a caller-provided ``random.Random``, so the same seed
always produces the same text.
"""

from __future__ import annotations

import random
import re

# Never turn these biological abbreviations into lower-case words.
ACRONYMS = {
    "DNA", "RNA", "mRNA", "tRNA", "rRNA", "ATP", "ADP", "AMP", "GTP", "GDP", "NAD", "NADH",
    "NADP", "NADPH", "FAD", "FMN", "CoA", "ABC", "TCA", "PTS", "LPS", "EC", "MFS", "TPR",
    "HTH", "PAS", "GGDEF", "EAL", "CRISPR", "Cas", "II", "III", "IV", "V", "VI", "S1", "L1",
}

MIXED_CASE_WORD = re.compile(r"^[A-Z][a-z]*[A-Z0-9]")  # RpoC, NifH, McrA


def _is_acronym(word: str) -> bool:
    plain_word = word.strip("(),.;:")
    return (
        plain_word in ACRONYMS
        or bool(MIXED_CASE_WORD.match(plain_word))
        or plain_word.isupper()
    )


def sentence_case(text: str) -> str:
    """Upper-case the first letter without damaging DNA, RNA, NifH, and similar names."""
    text = text.strip()
    if not text:
        return text
    first = text.split(maxsplit=1)[0]
    if _is_acronym(first):
        return text
    return text[0].upper() + text[1:]


def as_sentence(text: str) -> str:
    text = sentence_case(text).rstrip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def strip_terminal_period(text: str) -> str:
    return text.rstrip().rstrip(".")


# Pfam and TIGRFAM report an uninformative 100% identity, so confidence uses e-value only.
STRONG, MODERATE, WEAK = "strong", "moderate", "weak"


def confidence(evalue: str | float | None) -> str:
    try:
        value = float(evalue)
    except (TypeError, ValueError):
        return MODERATE
    if value <= 1e-30:
        return STRONG
    if value <= 1e-5:
        return MODERATE
    return WEAK


HEDGES = {
    STRONG: ["", "", ""],
    MODERATE: ["likely ", "probable ", "predicted "],
    WEAK: ["possible ", "putative ", "weakly similar to a "],
}


def hedge(rng: random.Random, level: str) -> str:
    return rng.choice(HEDGES[level])


# ``render`` chooses one template at random. The facts placed into these templates never change.
TEMPLATES = {
    "product": [
        "{name}.",
        "This protein is annotated as {lower}.",
        "A protein identified as {lower}.",
        "Annotated as {lower}.",
        "The gene encodes {article}{lower}.",
    ],
    "function": [
        "{name}.",
        "Functions as {article}{lower}.",
        "Its function is described as {lower}.",
        "Assigned the function of {lower}.",
    ],
    "enzyme": [
        "An enzyme with {lower} activity.",
        "Catalyses the reaction of {article}{lower}.",
        "Shows {lower} activity.",
        "This enzyme acts as {article}{lower}.",
    ],
    "orthology": [
        "{name}.",
        "Belongs to an orthologous group described as {lower}.",
        "Grouped with proteins described as {lower}.",
    ],
    "category": [
        "It falls under {lower}.",
        "Its broad functional category is {lower}.",
        "Involved in {lower}.",
    ],
    "pathway": [
        "Associated with {lower}.",
        "Participates in {lower}.",
        "Acts within {lower}.",
    ],
    "domain_single": [
        "Contains {article}{lower} domain.",
        "The protein carries {article}{lower} domain.",
        "Built around {article}{lower} domain.",
    ],
    "domain_multi": [
        "A multi-domain protein composed of {chain}.",
        "Its domain architecture runs {chain}.",
        "Contains, in order along the sequence, {chain}.",
    ],
    "family": [
        "{name}",
        "{name}",
        "Protein-family evidence notes: {name}",
    ],
    "taxonomic": [
        "{anchor}, encoded by {taxon}.",
        "{anchor}, found in {taxon}.",
        "From {taxon}: {lower_anchor}.",
        "{taxon} carries this gene, {lower_anchor}.",
    ],
    "environmental": [
        "{anchor}, from {env}.",
        "{anchor}, recovered from {env}.",
        "Sampled from {env}: {lower_anchor}.",
    ],
    "unannotated": [
        "A protein of unknown function{tail}.",
        "An uncharacterised protein{tail}.",
        "A protein with no confidently assigned function{tail}.",
    ],
}


def _article(word: str) -> str:
    return "an " if word[:1].lower() in "aeiou" else "a "


def render(rng: random.Random, view: str, name: str, **extra) -> str:
    """Put ``name`` into a randomly chosen template for one view."""
    templates = TEMPLATES.get(view)
    if not templates:
        return as_sentence(name)

    # Templates add their own punctuation, so remove a period already present in the name.
    name = strip_terminal_period(name)
    first = name.split(maxsplit=1)[0] if name else ""
    lower = name if (not name or _is_acronym(first)) else name[0].lower() + name[1:]
    slots = {
        "name": name,
        "lower": lower,
        "article": _article(lower),
        "anchor": name,
        "lower_anchor": lower,
        **extra,
    }
    try:
        return as_sentence(rng.choice(templates).format(**slots))
    except KeyError:
        return as_sentence(name)


def domain_chain(rng: random.Random, domains: list[str]) -> str:
    """Describe domains in their order along the protein sequence."""
    if len(domains) == 1:
        return render(rng, "domain_single", domains[0])
    joiner = rng.choice([", then ", " followed by ", ", then "])
    chain = joiner.join(strip_terminal_period(domain) for domain in domains[:4])
    if len(domains) > 4:
        chain += f", and {len(domains) - 4} further domain{'s' if len(domains) - 4 > 1 else ''}"
    return render(rng, "domain_multi", "", chain=chain)


def truncate(parts: list[str], max_chars: int) -> list[str]:
    """Keep complete views, in order, until adding another would exceed the limit."""
    kept: list[str] = []
    characters_used = 0
    for part in parts:
        if characters_used + len(part) + 1 > max_chars and kept:
            break
        kept.append(part)
        characters_used += len(part) + 1
    return kept
