"""Shared DeBERTa embedding utilities."""

import torch
from transformers import AutoModel, AutoTokenizer

from graph_vul.config import DEBERTA_EMBEDDING_DIM, DEBERTA_MODEL_NAME


def load_deberta(
    model_name: str = DEBERTA_MODEL_NAME,
) -> tuple[AutoTokenizer, AutoModel]:
    """Load the DeBERTa tokenizer and model in eval mode."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def embed_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    max_length: int = 128,
) -> torch.Tensor:
    """Embed a list of text strings using DeBERTa, returning [len(texts), embedding_dim]."""
    embeddings = []
    for text in texts:
        if not text.strip():
            embeddings.append(torch.zeros(DEBERTA_EMBEDDING_DIM))
            continue

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        embeddings.append(emb.cpu())

    return torch.stack(embeddings)
