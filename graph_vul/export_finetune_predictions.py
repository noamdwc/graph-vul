"""Export per-row predictions from a fine-tuned UniXcoder checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

# Prediction export should run from local artifacts only.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from graph_vul.config import (
    FINETUNE_BATCH_SIZE,
    FINETUNE_CHECKPOINT_PATH,
    FUNC_SPLITS_DIR,
    FUNC_SUBSET_CSV,
    HELDOUT_CWE_IDS,
    PROCESSED_DIR,
    CODE_MODEL_NAME,
)
from graph_vul.analysis.cwe_matching_report import (
    candidate_cwe_matrix,
    retrieval_columns,
    write_summary_artifacts,
)
from graph_vul.dataset.func_finetune_dataset import (
    FuncFinetuneDataset,
    available_splits,
    finetune_graph_collate,
    load_split_assignments,
)
from graph_vul.model.cwe_matching import CWEEmbeddingStore
from graph_vul.model.func_finetune_lit_module import FuncFinetuneLitModule

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/analysis/func_finetune_predictions")
DEFAULT_SPLITS = ("val", "test", "unseen_val", "unseen_test")


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_split_metadata(
    split: str,
    subset_csv: str | Path = FUNC_SUBSET_CSV,
    splits_dir: str | Path = FUNC_SPLITS_DIR,
) -> pd.DataFrame:
    subset_df = pd.read_csv(subset_csv)
    split_assignments = load_split_assignments(split=split, splits_dir=splits_dir)
    merged = subset_df.merge(split_assignments, on="sample_id", how="inner", sort=False)
    if "is_unseen_cwe" not in merged.columns:
        merged["is_unseen_cwe"] = merged["cwe_id"].astype(str).isin(HELDOUT_CWE_IDS) | merged["split"].str.startswith("unseen")
    merged["is_unseen_cwe"] = merged["is_unseen_cwe"].astype(int)
    if "row_id" not in merged.columns:
        merged["row_id"] = merged["sample_id"]
    if "label" not in merged.columns and "vul" in merged.columns:
        merged["label"] = merged["vul"]
    if "char_length" not in merged.columns and "func_before" in merged.columns:
        merged["char_length"] = merged["func_before"].fillna("").astype(str).str.len()
    if "line_count" not in merged.columns and "func_before" in merged.columns:
        merged["line_count"] = merged["func_before"].fillna("").astype(str).str.count(r"\n") + 1
    return merged.reset_index(drop=True)


def export_split_predictions(
    model: FuncFinetuneLitModule,
    tokenizer: AutoTokenizer,
    split: str,
    output_dir: str | Path,
    candidate_ids: list[str],
    candidate_match: torch.Tensor,
    batch_size: int = FINETUNE_BATCH_SIZE,
    threshold: float = 0.5,
    subset_csv: str | Path = FUNC_SUBSET_CSV,
    splits_dir: str | Path = FUNC_SPLITS_DIR,
    num_workers: int = 4,
) -> Path | None:
    metadata_df = build_split_metadata(split=split, subset_csv=subset_csv, splits_dir=splits_dir)
    if metadata_df.empty:
        logger.warning("Skipping %s: no rows in metadata", split)
        return None

    use_graph = model.model.graph_dim > 0
    dataset = FuncFinetuneDataset(
        split=split,
        tokenizer=tokenizer,
        subset_csv=subset_csv,
        splits_dir=splits_dir,
        embedding_dim=model.model.cwe_projection.in_features,
        use_graph=use_graph,
    )
    if len(dataset) != len(metadata_df):
        raise ValueError(f"Metadata/dataset length mismatch for {split}: {len(metadata_df)} vs {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=finetune_graph_collate if use_graph else None,
    )

    logits_parts: list[torch.Tensor] = []
    probs_parts: list[torch.Tensor] = []
    preds_parts: list[torch.Tensor] = []
    token_lengths_parts: list[torch.Tensor] = []
    function_match_parts: list[torch.Tensor] = []

    model.eval()
    model.freeze()
    device = next(model.parameters()).device
    progress = tqdm(loader, desc=f"Exporting {split}", unit="batch")
    with torch.inference_mode():
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cwe_embedding = batch["cwe_embedding"].to(device)
            graph_batch = batch.get("graph_data")
            if graph_batch is not None:
                graph_batch = graph_batch.to(device)

            outputs = model.model(
                input_ids,
                attention_mask,
                cwe_embedding,
                graph_batch=graph_batch,
                return_matching_outputs=True,
            )
            logits = outputs["logits"].squeeze(-1)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()

            logits_parts.append(logits.detach().cpu())
            probs_parts.append(probs.detach().cpu())
            preds_parts.append(preds.detach().cpu())
            token_lengths_parts.append(batch["attention_mask"].sum(dim=1).detach().cpu())
            function_match_parts.append(outputs["function_match"].detach().cpu())

    result_df = metadata_df.copy()
    result_df["y_true"] = result_df["label"].astype(int)
    result_df["logit"] = torch.cat(logits_parts).numpy()
    result_df["prob"] = torch.cat(probs_parts).numpy()
    result_df["y_pred"] = torch.cat(preds_parts).numpy()
    result_df["score"] = result_df["prob"]
    result_df["token_length"] = torch.cat(token_lengths_parts).numpy()
    retrieval = retrieval_columns(
        function_match=torch.cat(function_match_parts).to(device),
        candidate_ids=candidate_ids,
        candidate_match=candidate_match,
        true_cwe_ids=result_df["cwe_id"].astype(str).tolist(),
    )
    for column, values in retrieval.items():
        result_df[column] = values
    result_df["ablation_mode"] = model.hparams.get("ablation_mode", "E0")

    output_path = Path(output_dir) / f"{split}_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    write_summary_artifacts(result_df, output_dir=output_dir, prefix=split)
    logger.info("Wrote %s predictions: %d rows -> %s", split, len(result_df), output_path)
    return output_path


def export_predictions(
    checkpoint_path: str | Path = FINETUNE_CHECKPOINT_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    batch_size: int = FINETUNE_BATCH_SIZE,
    threshold: float = 0.5,
    subset_csv: str | Path = FUNC_SUBSET_CSV,
    splits_dir: str | Path = FUNC_SPLITS_DIR,
    num_workers: int = 4,
) -> None:
    device = _device()
    logger.info("Loading checkpoint from %s on %s", checkpoint_path, device)
    tokenizer = AutoTokenizer.from_pretrained(CODE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FuncFinetuneLitModule.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    ablation_mode = str(model.hparams.get("ablation_mode", "E0")).lower()
    effective_output_dir = Path(output_dir) / ablation_mode
    cwe_store = CWEEmbeddingStore(PROCESSED_DIR / "cwe_embeddings", model.model.cwe_projection.in_features)
    candidate_ids, candidate_match = candidate_cwe_matrix(cwe_store, model.model, device)
    available = set(available_splits(splits_dir))

    outputs: dict[str, str] = {}
    for split in splits:
        if split not in available:
            logger.info("Skipping %s export because the split is not present in %s", split, splits_dir)
            continue
        exported = export_split_predictions(
            model=model,
            tokenizer=tokenizer,
            split=split,
            output_dir=effective_output_dir,
            candidate_ids=candidate_ids,
            candidate_match=candidate_match,
            batch_size=batch_size,
            threshold=threshold,
            subset_csv=subset_csv,
            splits_dir=splits_dir,
            num_workers=num_workers,
        )
        if exported is not None:
            outputs[split] = str(exported)

    manifest = {
        "checkpoint_path": str(checkpoint_path),
        "threshold": threshold,
        "ablation_mode": ablation_mode.upper(),
        "splits": outputs,
    }
    manifest_path = effective_output_dir / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("Wrote export manifest to %s", manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export fine-tune predictions by split")
    parser.add_argument("--checkpoint", type=Path, default=FINETUNE_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=FINETUNE_BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--subset-csv", type=Path, default=FUNC_SUBSET_CSV)
    parser.add_argument("--splits-dir", type=Path, default=FUNC_SPLITS_DIR)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--splits", nargs="*", default=list(DEFAULT_SPLITS))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    export_predictions(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        splits=tuple(args.splits),
        batch_size=args.batch_size,
        threshold=args.threshold,
        subset_csv=args.subset_csv,
        splits_dir=args.splits_dir,
        num_workers=args.num_workers,
    )
