from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MatchingConfig:
    ablation_mode: str = "E0"
    projection_dim: int = 128
    negatives_per_positive: int = 1
    loss_lambda: float = 0.1
    triplet_margin: float = 0.2
    use_negative_classification: bool = False
    use_triplet_loss: bool = False
    use_hard_negatives: bool = False

    @property
    def enabled(self) -> bool:
        return self.use_negative_classification or self.use_triplet_loss

    def to_dict(self) -> dict:
        return asdict(self)


def build_matching_config(
    ablation_mode: str = "E0",
    projection_dim: int = 128,
    negatives_per_positive: int = 1,
    loss_lambda: float = 0.1,
    triplet_margin: float = 0.2,
) -> MatchingConfig:
    mode = ablation_mode.upper().strip()
    if mode not in {"E0", "E1", "E2", "E3", "E4"}:
        raise ValueError(f"Unsupported ablation mode: {ablation_mode}")
    return MatchingConfig(
        ablation_mode=mode,
        projection_dim=projection_dim,
        negatives_per_positive=negatives_per_positive,
        loss_lambda=loss_lambda,
        triplet_margin=triplet_margin,
        use_negative_classification=mode in {"E1", "E3", "E4"},
        use_triplet_loss=mode in {"E2", "E3", "E4"},
        use_hard_negatives=mode == "E4",
    )


def cwe_numeric_id(cwe_id: str) -> int | None:
    text = str(cwe_id).strip().upper()
    if text.startswith("CWE-"):
        text = text[4:]
    return int(text) if text.isdigit() else None


def cwe_family_id(cwe_id: str) -> int | None:
    value = cwe_numeric_id(cwe_id)
    if value is None:
        return None
    return value // 100


class CWENegativeSampler:
    def __init__(
        self,
        candidate_cwe_ids: list[str],
        seed: int = 42,
    ) -> None:
        filtered = sorted({str(cwe_id).strip() for cwe_id in candidate_cwe_ids if str(cwe_id).strip()})
        self.candidate_cwe_ids = filtered
        self.rng = random.Random(seed)

    def _choose_hard_negative(self, positive_cwe_id: str, candidates: list[str]) -> tuple[str, str]:
        pos_num = cwe_numeric_id(positive_cwe_id)
        pos_family = cwe_family_id(positive_cwe_id)

        family_matches = [
            candidate for candidate in candidates
            if pos_family is not None and cwe_family_id(candidate) == pos_family
        ]
        if family_matches:
            return self.rng.choice(family_matches), "same_family"

        numeric_matches = [
            candidate for candidate in candidates
            if pos_num is not None and cwe_numeric_id(candidate) is not None
        ]
        if numeric_matches:
            closest = min(
                numeric_matches,
                key=lambda candidate: abs(cwe_numeric_id(candidate) - pos_num),  # type: ignore[arg-type]
            )
            return closest, "similar_numeric"

        return self.rng.choice(candidates), "fallback_random"

    def sample(
        self,
        positive_cwe_id: str,
        use_hard_negatives: bool = False,
    ) -> tuple[str, str]:
        candidates = [candidate for candidate in self.candidate_cwe_ids if candidate != str(positive_cwe_id).strip()]
        if not candidates:
            return str(positive_cwe_id).strip(), "self_fallback"
        if use_hard_negatives:
            return self._choose_hard_negative(str(positive_cwe_id), candidates)
        return self.rng.choice(candidates), "random"


class CWEEmbeddingStore:
    def __init__(self, cwe_embeddings_dir: str | Path, embedding_dim: int) -> None:
        self.cwe_dir = Path(cwe_embeddings_dir)
        self.embedding_dim = embedding_dim
        self._cache: dict[str, torch.Tensor] = {}

    def get(self, cwe_id: str) -> torch.Tensor:
        key = str(cwe_id).strip()
        if key not in self._cache:
            path = self.cwe_dir / f"{key}.pt"
            if path.exists():
                self._cache[key] = torch.load(path, weights_only=True).float()
            else:
                self._cache[key] = torch.zeros(self.embedding_dim, dtype=torch.float32)
        return self._cache[key]

    def stack(self, cwe_ids: list[str], device: torch.device) -> torch.Tensor:
        if not cwe_ids:
            return torch.zeros(0, self.embedding_dim, device=device)
        return torch.stack([self.get(cwe_id) for cwe_id in cwe_ids], dim=0).to(device)

    def available_cwe_ids(self) -> list[str]:
        return sorted(path.stem for path in self.cwe_dir.glob("*.pt"))


def candidate_cwe_ids_from_subset(subset_csv: str | Path) -> list[str]:
    subset_path = Path(subset_csv)
    if not subset_path.exists():
        return []
    df = pd.read_csv(subset_path, usecols=["cwe_id"])
    return sorted({
        str(cwe_id).strip()
        for cwe_id in df["cwe_id"].dropna().astype(str).tolist()
        if str(cwe_id).strip()
    })


def cosine_similarity_logits(anchor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(anchor, other, dim=-1)
