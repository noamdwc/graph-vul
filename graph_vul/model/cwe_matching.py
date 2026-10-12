"""Catalog, pair targets and training-only negative selection."""
from dataclasses import asdict, dataclass
from collections.abc import Iterable
import json
from pathlib import Path
import random
import re

import torch

CATALOG_PATH = Path(__file__).resolve().parents[2] / "configs/cwe_catalog_v1.json"
NONE = "NONE"


def parse_cwe_ids(value: object) -> list[str]:
    """Normalize all real annotations; missing/unknown labels remain empty."""
    if value is None:
        return []
    values = (
        value if isinstance(value, Iterable) and not isinstance(value, str) else [value]
    )
    result = set()
    for item in values:
        text = str(item).strip().upper()
        if text.isdigit():
            text = "CWE-" + text
        result.update(re.findall(r"CWE-\d+", text))
    return sorted(result)


def load_catalog(path: str | Path = CATALOG_PATH) -> dict:
    catalog = json.loads(Path(path).read_text())
    descriptions = catalog["descriptions"]
    if not catalog["version"] or not descriptions:
        raise ValueError("Catalog needs a version and real CWE descriptions")
    for cwe, description in descriptions.items():
        if parse_cwe_ids(cwe) != [cwe] or not description.strip():
            raise ValueError(f"Invalid catalog entry: {cwe}")
    return catalog


@dataclass(frozen=True)
class MatchingConfig:
    ablation_mode: str = "pair"
    projection_dim: int = 128
    negatives_per_positive: int = 1
    loss_lambda: float = 0.1
    triplet_margin: float = 0.2

    @property
    def use_triplet_loss(self) -> bool:
        return self.ablation_mode in {"triplet_random", "triplet_hard"}

    @property
    def use_hard_negatives(self) -> bool:
        return self.ablation_mode == "triplet_hard"

    def to_dict(self) -> dict:
        return asdict(self)


def build_matching_config(
    ablation_mode="pair",
    projection_dim=128,
    negatives_per_positive=1,
    loss_lambda=0.1,
    triplet_margin=0.2,
):
    if ablation_mode not in {"pair", "graph_pair", "triplet_random", "triplet_hard"}:
        raise ValueError(f"Unknown experiment: {ablation_mode}")
    if (
        projection_dim < 1
        or negatives_per_positive < 1
        or loss_lambda < 0
        or triplet_margin <= 0
    ):
        raise ValueError("Invalid matching settings")
    return MatchingConfig(
        ablation_mode,
        projection_dim,
        negatives_per_positive,
        loss_lambda,
        triplet_margin,
    )


def pair_targets(labels, annotations, candidate_ids, device=None):
    """Unknown vulnerable rows have no pair supervision, never a NONE target."""
    targets = torch.zeros(len(labels), len(candidate_ids), device=device)
    supervised = torch.ones(len(labels), dtype=torch.bool, device=device)
    index = {cwe: i for i, cwe in enumerate(candidate_ids)}
    for row, (label, cwes) in enumerate(zip(labels, annotations)):
        if label == 0:
            targets[row, index[NONE]] = 1
        elif not cwes:
            supervised[row] = False
        else:
            missing = set(cwes) - index.keys()
            if missing:
                raise ValueError(
                    f"Annotated CWEs outside versioned catalog: {sorted(missing)}"
                )
            for cwe in cwes:
                targets[row, index[cwe]] = 1
    return targets, supervised


class CWENegativeSampler:
    """Select incorrect candidates; hard means highest current pair-head score."""

    def __init__(self, candidate_cwe_ids, seed=42):
        self.candidate_cwe_ids = list(candidate_cwe_ids)
        self.rng = random.Random(seed)

    def select(self, positive_ids, count, scores=None):
        indices = [
            i for i, cwe in enumerate(self.candidate_cwe_ids) if cwe not in positive_ids
        ]
        if scores is not None:
            # Catalog order resolves score ties deterministically.
            indices.sort(key=lambda i: (-float(scores[i]), i))
            return indices[:count]
        return self.rng.sample(indices, min(count, len(indices)))


class CWEEmbeddingStore:
    def __init__(self, cwe_embeddings_dir, embedding_dim):
        self.cwe_dir = Path(cwe_embeddings_dir)
        self.embedding_dim = embedding_dim

    def get(self, cwe_id):
        embedding = torch.load(self.cwe_dir / f"{cwe_id}.pt", weights_only=True).float()
        if (
            embedding.shape != (self.embedding_dim,)
            or not torch.isfinite(embedding).all()
        ):
            raise ValueError(f"Invalid embedding for {cwe_id}")
        return embedding

    def stack(self, cwe_ids, device):
        return torch.stack([self.get(cwe) for cwe in cwe_ids]).to(device)
