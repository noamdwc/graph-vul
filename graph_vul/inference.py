"""Shared complete-catalog inference and validation-only calibration."""
import numpy as np
import torch


def decision(scores, threshold):
    """Vulnerable iff best real-CWE logit minus NONE logit exceeds threshold.

    Strict comparison makes ties NONE. Real-CWE ties use fixed catalog order.
    """
    best_scores, best_indices = scores[:, :-1].max(dim=1)
    margins = (best_scores - scores[:, -1]).double()
    vulnerable = margins > threshold
    chosen = torch.where(vulnerable, best_indices, scores.shape[1] - 1)
    return {
        "scores": scores,
        "margin": margins,
        "y_pred": vulnerable.long(),
        "chosen": chosen,
    }


def calibrate_threshold(margins, labels):
    margins = np.asarray(margins, dtype=np.float64)
    labels = np.asarray(labels)
    if (
        not len(labels)
        or set(labels.tolist()) != {0, 1}
        or not np.isfinite(margins).all()
    ):
        raise ValueError(
            "Calibration requires finite validation scores and both classes"
        )
    unique_scores, score_groups, counts = np.unique(
        margins, return_inverse=True, return_counts=True
    )
    positives_per_score = np.bincount(score_groups, weights=labels)
    total_positives = labels.sum()
    # Strict > excludes each tied score group at once. Include an all-positive boundary.
    thresholds = np.r_[np.nextafter(unique_scores[0], -np.inf), unique_scores]
    true_positives = np.r_[
        total_positives, total_positives - np.cumsum(positives_per_score)
    ]
    predicted_positives = np.r_[len(labels), len(labels) - np.cumsum(counts)]
    f1_scores = 2 * true_positives / (total_positives + predicted_positives)
    # Higher threshold breaks F1 ties, favoring fewer false positives.
    best_index = np.flatnonzero(f1_scores == f1_scores.max())[-1]
    return float(thresholds[best_index])


def score_batch(module, batch):
    features = module.model.encode_functions(
        batch["input_ids"], batch["attention_mask"], batch.get("graph_data")
    )
    candidates = torch.cat(
        [module.real_cwe_embeddings, module.model.none_embedding[None]], dim=0
    )
    scores = module.model.score_candidates(features, candidates)
    return features, candidates, scores


def predict_batch(module, batch, require_calibration=True):
    if require_calibration and not bool(module.calibrated):
        raise ValueError("Final inference requires a calibrated checkpoint")
    _, _, scores = score_batch(module, batch)
    return decision(scores, module.decision_threshold)
