import pandas as pd
from graph_vul.analysis.cwe_matching_report import metrics


def test_short_function_metrics_keep_positive_and_benign_support():
    frame = pd.DataFrame(
        {
            "y_true": [1, 0, 0],
            "y_pred": [1, 1, 0],
            "margin": [2.0, 1.0, -1.0],
            "token_length": [64, 20, 65],
            "retrieval_hit_top1": [1, None, None],
            "retrieval_hit_top3": [1, None, None],
        }
    )
    result = metrics(frame[frame["token_length"] <= 64])
    assert result["rows"] == 2
    assert result["positive_support"] == 1
    assert result["benign_support"] == 1
    assert result["false_positive_rate"] == 1.0
    assert result["retrieval_top1"] == 1.0
