"""Names and immutable IDs shared by shard ingestion, reading, and training.

Protein representations live inside each MAG shard.  This module deliberately
contains no corpus loading or cache mutation logic.
"""

from pathlib import Path

TOKEN_ROOT = Path(__file__).resolve().parents[1] / "token"

SPECIAL = {"PAD": 0, "MASK": 1, "UCT": 2, "UNK": 3}
NUM_SPECIAL = len(SPECIAL)

# Retired after incremental-assignment validation showed that hierarchical
# MMseqs cluster membership is not reproducible by direct representative
# search.  Sequence-clustering modes `cluster97` and `clusterX` intentionally
# remain absent from the active mode registry.
DISCRETE_MODES = ("vq_kmeans", "vq_vae")
ALL_MODES = DISCRETE_MODES + ("esmc",)


def mode_to_dirname(mode: str) -> str:
    """Map a representation name to its frozen-artifact directory."""
    return mode


def mode_dir(mode: str) -> Path:
    """Return the frozen artifact directory used during new-data ingestion."""
    if mode not in ALL_MODES:
        raise ValueError("unknown representation {!r}".format(mode))
    return TOKEN_ROOT / mode_to_dirname(mode)
