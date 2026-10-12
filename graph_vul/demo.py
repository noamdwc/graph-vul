"""Rank CWE matches for a C function using a calibrated checkpoint."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch_geometric.data import Batch
from transformers import AutoTokenizer

from graph_vul.config import CODE_MODEL_NAME
from graph_vul.export_finetune_predictions import load_model
from graph_vul.inference import predict_batch
from graph_vul.pipeline.treesitter_graphs import function_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--function", type=Path, help="C source file; omit to read stdin"
    )
    args = parser.parse_args()
    code = args.function.read_text() if args.function else sys.stdin.read()
    if not code.strip():
        raise ValueError("Provide a nonempty C function")
    module = load_model(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(CODE_MODEL_NAME)
    batch = dict(
        tokenizer(
            [code],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
    )
    if module.model.graph_dim:
        batch["graph_data"] = Batch.from_data_list(
            [function_graph(code, module.hparams.node_vocab, module.hparams.edge_vocab)]
        )
    device = next(module.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        result = predict_batch(module, batch)
    ranked = torch.argsort(result["scores"][0, :-1], descending=True, stable=True)[:3]
    print(
        json.dumps(
            {
                "prediction": module.candidate_ids[int(result["chosen"][0])],
                "margin": float(result["margin"][0]),
                "threshold": float(module.decision_threshold),
                "none_logit": float(result["scores"][0, -1]),
                "ranked_cwes": [
                    {
                        "cwe": module.candidate_ids[i],
                        "logit": float(result["scores"][0, i]),
                    }
                    for i in ranked.tolist()
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
