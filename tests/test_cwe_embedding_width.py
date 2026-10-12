import pytest
import torch
from graph_vul.model.cwe_matching import CWEEmbeddingStore


def test_embedding_boundary_rejects_wrong_width_and_nonfinite(tmp_path):
    torch.save(torch.zeros(10), tmp_path / "CWE-119.pt")
    with pytest.raises(ValueError, match="Invalid embedding"):
        CWEEmbeddingStore(tmp_path, 768).get("CWE-119")
    torch.save(torch.full((768,), float("nan")), tmp_path / "CWE-119.pt")
    with pytest.raises(ValueError, match="Invalid embedding"):
        CWEEmbeddingStore(tmp_path, 768).get("CWE-119")
