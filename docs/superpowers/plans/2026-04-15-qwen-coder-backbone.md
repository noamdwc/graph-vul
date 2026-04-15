# Qwen3-0.6B Code Backbone Swap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fine-tune code backbone from `microsoft/unixcoder-base` (768d, 12L, bidirectional) with `Qwen/Qwen3-0.6B` (1024d, 28L, causal), preserving gated-residual fusion hooks, late CWE concat, and triplet + hard-negative training.

**Architecture:** Introduce new `CODE_MODEL_NAME` / `CODE_EMBEDDING_DIM` constants alongside the existing `UNIXCODER_*` ones (the latter still powers the frozen-embedding baseline). Swap all fine-tune-path tokenizer loads and the encoder default to Qwen. In `FuncFinetuneModel`, add a one-function layer-path detector (Qwen exposes `model.layers`; RoBERTa exposes `encoder.layer`) and change the pooling from mean-over-non-pad to last-non-pad-token (native choice for causal LMs). Fix the missing pad token on the Qwen tokenizer. Bump the Colab notebook's default `fusion_layers` to `list(range(28))`.

**Tech Stack:** PyTorch 2, HuggingFace Transformers (`AutoModel`, `AutoTokenizer`), PyTorch Lightning, PyG, W&B.

**Spec:** `docs/superpowers/specs/2026-04-15-qwen-coder-backbone-design.md`.

---

## File Structure

**Created (tests only):**
- `tests/test_func_finetune_model_qwen.py` — smoke test for construction, forward shapes, and fusion-hook gradient flow with the Qwen backbone.

**Modified (code):**
- `graph_vul/config.py` — add `CODE_MODEL_NAME`, `CODE_EMBEDDING_DIM`.
- `graph_vul/model/func_finetune_model.py` — layer-path helper, pooling swap, default `model_name` → `CODE_MODEL_NAME`.
- `graph_vul/dataset/func_finetune_dataset.py` — default tokenizer + pad-token fallback.
- `graph_vul/dataset/external_datasets.py` — same two changes as the training dataset.
- `graph_vul/train_finetune.py` — tokenizer load and W&B run-name string.
- `graph_vul/eval_external.py` — tokenizer load.
- `graph_vul/export_finetune_predictions.py` — tokenizer load.
- `scripts/sweep_threshold.py` — tokenizer load.

**Modified (notebook):**
- `notebooks/colab_finetune.ipynb` — `fusion_layers` default cell: `list(range(12))` → `list(range(28))`.

**Intentionally untouched (baseline / frozen-embedding path):**
- `graph_vul/pipeline/embed_functions.py`, `graph_vul/pipeline/embed_utils.py`, `graph_vul/model/func_classifier.py`.

---

## Task 1: Add Qwen config constants

**Files:**
- Modify: `graph_vul/config.py:97-100`

- [ ] **Step 1: Add the new constants**

In `graph_vul/config.py`, directly after the `# UniXcoder` block, insert:

```python
# Backbone for fine-tuning: was UniXcoder-base (768d, 12L, bidirectional),
# now Qwen3-0.6B (1024d, 28L, decoder-only causal).
# UNIXCODER_* constants above are retained for the frozen-embedding baseline
# path (embed_functions.py, func_classifier.py); they are not used by the
# fine-tune model.
CODE_MODEL_NAME = "Qwen/Qwen3-0.6B"
CODE_EMBEDDING_DIM = 1024
```

- [ ] **Step 2: Smoke-import**

Run: `python -c "from graph_vul.config import CODE_MODEL_NAME, CODE_EMBEDDING_DIM, UNIXCODER_MODEL_NAME; print(CODE_MODEL_NAME, CODE_EMBEDDING_DIM, UNIXCODER_MODEL_NAME)"`

Expected: `Qwen/Qwen3-0.6B 1024 microsoft/unixcoder-base`

- [ ] **Step 3: Commit**

```bash
git add graph_vul/config.py
git commit -m "Add CODE_MODEL_NAME/CODE_EMBEDDING_DIM for Qwen backbone"
```

---

## Task 2: Layer-path helper + model default + pooling swap

**Files:**
- Modify: `graph_vul/model/func_finetune_model.py` (imports, new helper above the class, `_register_fusion_hooks`, `_encode_features`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_func_finetune_model_qwen.py`:

```python
"""Smoke tests for FuncFinetuneModel with the Qwen3-0.6B backbone."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from graph_vul.config import CODE_EMBEDDING_DIM, CODE_MODEL_NAME, DEBERTA_EMBEDDING_DIM
from graph_vul.model.func_finetune_model import FuncFinetuneModel


@pytest.fixture(scope="module")
def tiny_model():
    model = FuncFinetuneModel(
        model_name=CODE_MODEL_NAME,
        cwe_dim=DEBERTA_EMBEDDING_DIM,
        graph_dim=64,
        num_node_types=8,
        num_relations=4,
        fusion_layers=[0, 13, 27],  # first, middle-ish, last
        fusion_method="scalar_gated_residual",
    )
    model.eval()
    return model


def _dummy_graph_batch(batch_size: int) -> Batch:
    data_list = []
    for _ in range(batch_size):
        data_list.append(
            Data(
                node_type_ids=torch.tensor([0, 1, 2], dtype=torch.long),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                edge_type_ids=torch.tensor([0, 1], dtype=torch.long),
            )
        )
    return Batch.from_data_list(data_list)


def test_forward_shapes(tiny_model):
    B, S = 2, 16
    input_ids = torch.randint(0, 1000, (B, S), dtype=torch.long)
    attention_mask = torch.ones(B, S, dtype=torch.long)
    attention_mask[1, 10:] = 0  # second sample has padding
    cwe = torch.randn(B, DEBERTA_EMBEDDING_DIM)
    graph_batch = _dummy_graph_batch(B)

    with torch.no_grad():
        out = tiny_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cwe_embedding=cwe,
            graph_batch=graph_batch,
            return_matching_outputs=True,
        )

    assert out["logits"].shape == (B, 1)
    assert out["pooled"].shape == (B, CODE_EMBEDDING_DIM)
    assert out["function_match"].shape[0] == B
    assert out["cwe_match"].shape[0] == B


def test_last_token_pooling_uses_last_non_pad(tiny_model):
    """The pooled vector must come from the last non-pad position, not the end of the sequence."""
    B, S = 1, 16
    input_ids = torch.randint(0, 1000, (B, S), dtype=torch.long)

    # Full mask: last token pooling picks index 15
    mask_full = torch.ones(B, S, dtype=torch.long)
    # Padded mask: last non-pad is index 4
    mask_padded = torch.zeros(B, S, dtype=torch.long)
    mask_padded[0, :5] = 1
    # Make input_ids identical in the first 5 positions so the two pooled vectors differ
    # only if we are actually indexing differently.
    cwe = torch.randn(B, DEBERTA_EMBEDDING_DIM)

    with torch.no_grad():
        out_full = tiny_model(
            input_ids=input_ids, attention_mask=mask_full,
            cwe_embedding=cwe, return_matching_outputs=True,
        )
        out_padded = tiny_model(
            input_ids=input_ids, attention_mask=mask_padded,
            cwe_embedding=cwe, return_matching_outputs=True,
        )

    assert not torch.allclose(out_full["pooled"], out_padded["pooled"]), (
        "Pooled vectors must differ when attention_mask selects a different last token"
    )


def test_fusion_hook_receives_gradient(tiny_model):
    B, S = 2, 16
    input_ids = torch.randint(0, 1000, (B, S), dtype=torch.long)
    attention_mask = torch.ones(B, S, dtype=torch.long)
    cwe = torch.randn(B, DEBERTA_EMBEDDING_DIM)
    graph_batch = _dummy_graph_batch(B)

    tiny_model.train()
    logits = tiny_model(
        input_ids=input_ids, attention_mask=attention_mask,
        cwe_embedding=cwe, graph_batch=graph_batch,
    )
    logits.sum().backward()

    fused_layer_key = next(iter(tiny_model.fusion_modules))
    fusion_mod = tiny_model.fusion_modules[fused_layer_key]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fusion_mod.parameters()), (
        "Fusion module received no gradient — hook did not fire"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_func_finetune_model_qwen.py -v`

Expected: FAIL — `model_name` default is still `UNIXCODER_MODEL_NAME`, fusion hooks try `encoder.encoder.layer` (`AttributeError: 'Qwen3Model' object has no attribute 'encoder'`), and/or pooled dim mismatch.

- [ ] **Step 3: Update imports**

In `graph_vul/model/func_finetune_model.py`, replace the config import block and the model-name reference:

```python
from graph_vul.config import (
    CODE_MODEL_NAME,
    DEBERTA_EMBEDDING_DIM,
    FUNC_MLP_DROPOUT,
    FUNC_MLP_HIDDEN_DIM,
    GRAPH_DROPOUT,
    GRAPH_HIDDEN_DIM,
    GRAPH_NODE_EMBED_DIM,
    MATCHING_PROJECTION_DIM,
)
```

(Remove `UNIXCODER_MODEL_NAME` from the import; everything else stays.)

- [ ] **Step 4: Add the layer-path helper above the class**

Immediately below the existing imports and above `class FuncFinetuneModel`, insert:

```python
def _encoder_layers(encoder):
    """Return the list of transformer blocks regardless of backbone family.

    Qwen3 (decoder-only, Llama-style) exposes layers at ``encoder.model.layers``;
    RoBERTa-style models such as UniXcoder expose them at ``encoder.encoder.layer``.
    The fusion-hook contract (tuple or tensor output) is the same for both.
    """
    if hasattr(encoder, "model") and hasattr(encoder.model, "layers"):
        return encoder.model.layers
    return encoder.encoder.layer
```

- [ ] **Step 5: Swap the default `model_name`**

In the `FuncFinetuneModel.__init__` signature, change:

```python
model_name: str = UNIXCODER_MODEL_NAME,
```

to:

```python
model_name: str = CODE_MODEL_NAME,
```

- [ ] **Step 6: Use the helper in `_register_fusion_hooks`**

Replace:

```python
encoder_layers = self.encoder.encoder.layer
```

with:

```python
# Backbone-agnostic layer access: Qwen3 uses encoder.model.layers,
# RoBERTa-style models use encoder.encoder.layer.
encoder_layers = _encoder_layers(self.encoder)
```

Nothing else in the hook body changes — Llama decoder layers return `(hidden_state, ...)` tuples from their forward hook, matching the existing branch.

- [ ] **Step 7: Swap pooling in `_encode_features`**

Find the mean-pool block:

```python
# Mean pool over non-padding tokens
mask = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [B, 768]
```

Replace with:

```python
# Last-non-pad-token pooling.
# Qwen3 is a causal (left-to-right) LM, so only the final non-pad token has
# attended to the entire function. Mean-over-non-pad would mix causally-
# truncated representations and is not the idiomatic pooling choice.
last_idx = attention_mask.sum(dim=1) - 1                            # [B]
pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]  # [B, hidden]
```

- [ ] **Step 8: Run the test to verify it passes**

Run: `pytest tests/test_func_finetune_model_qwen.py -v`

Expected: all three tests PASS. The first run will download the Qwen3-0.6B weights (~1.2 GB) — allow a minute.

- [ ] **Step 9: Commit**

```bash
git add graph_vul/model/func_finetune_model.py tests/test_func_finetune_model_qwen.py
git commit -m "Swap fine-tune encoder to Qwen3-0.6B with last-token pooling"
```

---

## Task 3: Dataset tokenizer swap (+ pad-token fallback)

**Files:**
- Modify: `graph_vul/dataset/func_finetune_dataset.py:18-25,77-81`

- [ ] **Step 1: Update the config import**

Replace:

```python
from graph_vul.config import (
    DEBERTA_EMBEDDING_DIM,
    FUNC_SPLITS_DIR,
    FUNC_SUBSET_CSV,
    GRAPH_ARTIFACTS_DIR,
    PROCESSED_DIR,
    UNIXCODER_MODEL_NAME,
)
```

with:

```python
from graph_vul.config import (
    CODE_MODEL_NAME,
    DEBERTA_EMBEDDING_DIM,
    FUNC_SPLITS_DIR,
    FUNC_SUBSET_CSV,
    GRAPH_ARTIFACTS_DIR,
    PROCESSED_DIR,
)
```

- [ ] **Step 2: Replace tokenizer load + add pad-token fallback**

Replace:

```python
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(UNIXCODER_MODEL_NAME)
```

with:

```python
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(CODE_MODEL_NAME)

        # Qwen3 ships without a pad token; reuse EOS for right-padding.
        # attention_mask still distinguishes real vs pad, so last-non-pad-token
        # pooling and the loss mask stay correct.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
```

- [ ] **Step 3: Smoke-test the tokenizer path**

Run: `python -c "from graph_vul.dataset.func_finetune_dataset import FuncFinetuneDataset; from transformers import AutoTokenizer; from graph_vul.config import CODE_MODEL_NAME; tok = AutoTokenizer.from_pretrained(CODE_MODEL_NAME); tok.pad_token = tok.eos_token; out = tok(['int main(){}', 'void f(){}'], padding='max_length', max_length=32, truncation=True, return_tensors='pt'); print(out['input_ids'].shape, out['attention_mask'].sum(dim=1))"`

Expected: `torch.Size([2, 32]) tensor([..., ...])` (two positive counts, second ≤ first).

- [ ] **Step 4: Run existing dataset tests**

Run: `pytest tests/test_func_finetune_dataset.py -v`

Expected: PASS (they should not exercise a specific tokenizer vocab, but flag any failure as a signal to inspect).

- [ ] **Step 5: Commit**

```bash
git add graph_vul/dataset/func_finetune_dataset.py
git commit -m "Point FuncFinetuneDataset at CODE_MODEL_NAME with pad-token fallback"
```

---

## Task 4: External datasets tokenizer swap

**Files:**
- Modify: `graph_vul/dataset/external_datasets.py:15-25,117-121`

- [ ] **Step 1: Update the config import**

Replace `UNIXCODER_MODEL_NAME` with `CODE_MODEL_NAME` in the existing `from graph_vul.config import (...)` block at the top of the file.

- [ ] **Step 2: Replace tokenizer load + add pad-token fallback**

Replace:

```python
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(UNIXCODER_MODEL_NAME)
```

with:

```python
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(CODE_MODEL_NAME)

        # Qwen3 has no pad token by default; EOS stands in for right padding.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
```

- [ ] **Step 3: Run the external-adapter tests**

Run: `pytest tests/test_primevul_adapter.py -v`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add graph_vul/dataset/external_datasets.py
git commit -m "Point external datasets at CODE_MODEL_NAME with pad-token fallback"
```

---

## Task 5: Training-entrypoint tokenizer + run-name update

**Files:**
- Modify: `graph_vul/train_finetune.py:30-45,154-160,234-240`

- [ ] **Step 1: Update the config import**

In `graph_vul/train_finetune.py`, replace `UNIXCODER_MODEL_NAME` with `CODE_MODEL_NAME` in the `from graph_vul.config import (...)` block.

- [ ] **Step 2: Update the tokenizer load (around line 156)**

Replace:

```python
    tokenizer = AutoTokenizer.from_pretrained(UNIXCODER_MODEL_NAME)
```

with:

```python
    # Fine-tune backbone is Qwen3-0.6B; the frozen baseline path still uses
    # UNIXCODER_MODEL_NAME.
    tokenizer = AutoTokenizer.from_pretrained(CODE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
```

- [ ] **Step 3: Update the W&B run-name string (line ~236)**

Replace:

```python
    model_name = "func-graph-finetune-unixcoder-rgcn" if use_graph else "func-finetune-unixcoder"
```

with:

```python
    # Display label for W&B; reflects the active fine-tune backbone.
    model_name = "func-graph-finetune-qwen3-rgcn" if use_graph else "func-finetune-qwen3"
```

- [ ] **Step 4: Syntax check**

Run: `python -c "import graph_vul.train_finetune"`

Expected: no import error.

- [ ] **Step 5: Commit**

```bash
git add graph_vul/train_finetune.py
git commit -m "Train entrypoint: load Qwen3 tokenizer + rename W&B run"
```

---

## Task 6: Eval / export / sweep tokenizer updates

**Files:**
- Modify: `graph_vul/eval_external.py:30-40,156-160`
- Modify: `graph_vul/export_finetune_predictions.py:24-32,190-194`
- Modify: `scripts/sweep_threshold.py:30-40`

- [ ] **Step 1: `eval_external.py`**

Replace `UNIXCODER_MODEL_NAME` with `CODE_MODEL_NAME` in the config import, then replace the tokenizer load:

```python
    tokenizer = AutoTokenizer.from_pretrained(UNIXCODER_MODEL_NAME)
```

with:

```python
    tokenizer = AutoTokenizer.from_pretrained(CODE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
```

- [ ] **Step 2: `export_finetune_predictions.py`**

Same two edits as Step 1, in this file.

- [ ] **Step 3: `scripts/sweep_threshold.py`**

Replace:

```python
    tokenizer = AutoTokenizer.from_pretrained(cfg.UNIXCODER_MODEL_NAME)
```

with:

```python
    tokenizer = AutoTokenizer.from_pretrained(cfg.CODE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
```

- [ ] **Step 4: Syntax check all three**

Run: `python -c "import graph_vul.eval_external, graph_vul.export_finetune_predictions" && python -c "import importlib.util, pathlib; spec = importlib.util.spec_from_file_location('s', pathlib.Path('scripts/sweep_threshold.py')); spec.loader.exec_module(importlib.util.module_from_spec(spec))"`

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add graph_vul/eval_external.py graph_vul/export_finetune_predictions.py scripts/sweep_threshold.py
git commit -m "Eval/export/sweep: load Qwen3 tokenizer"
```

---

## Task 7: Notebook default `fusion_layers` → `list(range(28))`

**Files:**
- Modify: `notebooks/colab_finetune.ipynb`

- [ ] **Step 1: Find the fusion-layers default cell**

Run: `grep -n "fusion_layers" notebooks/colab_finetune.ipynb | head`

Identify the cell that sets the training default (typically `"fusion_layers": list(range(12))` or similar).

- [ ] **Step 2: Update the default**

In the located cell, replace `list(range(12))` with `list(range(28))`. Add or update the neighboring comment to read: `# 28 = number of decoder layers in Qwen3-0.6B; fuses graph embedding into every block.`

- [ ] **Step 3: Sanity-check the notebook JSON**

Run: `python -c "import json; json.load(open('notebooks/colab_finetune.ipynb'))" && echo OK`

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/colab_finetune.ipynb
git commit -m "Colab finetune notebook: default fusion_layers to all 28 Qwen3 blocks"
```

---

## Task 8: End-to-end smoke

- [ ] **Step 1: Run the full unit suite**

Run: `pytest tests/ -x --ignore=tests/test_fusion_export_smoke.py -q`

(Exclude the fusion-export smoke because it may pin the UniXcoder hidden dim; inspect separately if time allows.)

Expected: PASS.

- [ ] **Step 2: Construct the full model once through the training entrypoint path**

Run: `python -c "from graph_vul.config import CODE_MODEL_NAME; from graph_vul.model.func_finetune_model import FuncFinetuneModel; m = FuncFinetuneModel(model_name=CODE_MODEL_NAME, graph_dim=64, num_node_types=8, num_relations=4, fusion_layers=list(range(28))); print('params:', sum(p.numel() for p in m.parameters())/1e6, 'M'); print('fusion layers:', sorted(int(k) for k in m.fusion_modules))"`

Expected: `params: ~600+ M`, `fusion layers: [0, 1, ..., 27]`.

- [ ] **Step 3: No commit unless something needed fixing.** If Step 1 surfaced a test regression, fix it in a focused commit.

---

## Notes for the implementer

- The first time any step downloads `Qwen/Qwen3-0.6B`, it pulls ~1.2 GB to the HF cache; subsequent steps are fast.
- The three tests in Task 2 are self-contained — they don't need real data.
- Old UniXcoder checkpoints are **not** loadable into the new model (hidden dim differs). This is expected; the Lightning module will raise a clear shape-mismatch error.
- Every architectural change carries a short comment explaining *why* (causal pooling, backbone-agnostic layer access, pad-token fallback). Do not remove those comments.
