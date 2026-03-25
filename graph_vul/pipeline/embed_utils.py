"""Shared DeBERTa embedding utilities."""

from importlib.util import find_spec
import logging

import torch
from transformers import AutoModel, AutoTokenizer

from graph_vul.config import (
    DEBERTA_EMBEDDING_DIM,
    DEBERTA_MODEL_NAME,
    UNIXCODER_MODEL_NAME,
)

logger = logging.getLogger(__name__)


def resolve_torch_device() -> torch.device:
    """Select the best available inference device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_deberta(
    model_name: str = DEBERTA_MODEL_NAME,
) -> tuple[AutoTokenizer, AutoModel]:
    """Load the DeBERTa tokenizer and model in eval mode."""
    if find_spec("sentencepiece") is None:
        raise RuntimeError(
            "DeBERTa tokenizer dependency missing: install `sentencepiece` "
            "and retry. This project's requirements include it."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = resolve_torch_device()
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    logger.info("Loaded %s on %s", model_name, device)
    return tokenizer, model


@torch.no_grad()
def embed_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    max_length: int = 128,
    batch_size: int = 16,
) -> torch.Tensor:
    """Embed a list of text strings using DeBERTa, returning [len(texts), embedding_dim]."""
    device = next(model.parameters()).device
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_embeddings = [torch.zeros(DEBERTA_EMBEDDING_DIM) for _ in batch]
        non_empty = [(idx, text) for idx, text in enumerate(batch) if text.strip()]

        if not non_empty:
            embeddings.extend(batch_embeddings)
            continue

        non_empty_indices = [idx for idx, _ in non_empty]
        non_empty_texts = [text for _, text in non_empty]
        inputs = tokenizer(
            non_empty_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        batch_output = outputs.last_hidden_state.mean(dim=1).cpu()

        for output_idx, batch_idx in enumerate(non_empty_indices):
            batch_embeddings[batch_idx] = batch_output[output_idx]

        embeddings.extend(batch_embeddings)

    return torch.stack(embeddings)


def load_unixcoder(
    model_name: str = UNIXCODER_MODEL_NAME,
) -> tuple[AutoTokenizer, AutoModel]:
    """Load the UniXcoder tokenizer and model in eval mode."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = resolve_torch_device()
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    logger.info("Loaded %s on %s", model_name, device)
    return tokenizer, model


@torch.no_grad()
def embed_mean_pool(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    max_length: int = 512,
) -> torch.Tensor:
    """Encode a single text with UniXcoder and return mean-pooled hidden states (768-dim)."""
    device = next(model.parameters()).device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    hidden = outputs.last_hidden_state  # [1, seq_len, 768]
    attention_mask = inputs["attention_mask"]  # [1, seq_len]
    mask = attention_mask.unsqueeze(-1).float()  # [1, seq_len, 1]
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # [1, 768]
    return pooled.squeeze(0).cpu()
