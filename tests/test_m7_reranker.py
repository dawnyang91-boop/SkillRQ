from skillrq.m7 import prepare_m7_data
from skillrq.m7.evaluate import evaluate_reranked_predictions
from skillrq.utils.io import read_jsonl, write_jsonl


def test_prepare_m7_data_builds_positive_and_hard_negative_examples(tmp_path):
    m4_root = tmp_path / "m4"
    output_root = tmp_path / "m7"
    _write_m4_fixture(m4_root)

    stats = prepare_m7_data(m4_data_root=m4_root, output_root=output_root, negatives_per_positive=1)

    examples = list(read_jsonl(output_root / "rerank_examples.jsonl"))
    pools = list(read_jsonl(output_root / "query_candidate_pools.jsonl"))
    labels = [row["label"] for row in examples]
    assert stats["queries"] == 1
    assert stats["positives"] == 2
    assert stats["negatives"] == 2
    assert labels.count(1) == 2
    assert labels.count(0) == 2
    positive_examples = [row for row in examples if row["label"] == 1]
    assert positive_examples[0]["stage_label"] == "FIRST"
    assert positive_examples[1]["stage_label"] == "FINAL"
    assert set(pools[0]["candidate_pool_ids"]) >= {"tool_a", "tool_b", "tool_c"}


def test_evaluate_m7_predictions_reports_retrieval_and_sequence_metrics(tmp_path):
    prediction_path = tmp_path / "reranked_predictions.jsonl"
    output_path = tmp_path / "metrics.json"
    write_jsonl(
        prediction_path,
        [
            {
                "query_id": "q1",
                "gold_ids": ["tool_a", "tool_b"],
                "sequence_ids": ["tool_a", "tool_b"],
                "predicted_tool_order": ["tool_a", "tool_b", "tool_c"],
                "reranked_capabilities": [
                    {"candidate_id": "tool_a"},
                    {"candidate_id": "tool_b"},
                    {"candidate_id": "tool_c"},
                ],
            }
        ],
    )

    metrics = evaluate_reranked_predictions(prediction_path=prediction_path, output_path=output_path, top_ks=(1, 2))

    assert metrics["evaluated_queries"] == 1
    assert metrics["recall@1"] == 0.5
    assert metrics["recall@2"] == 1.0
    assert metrics["completeness@2"] == 1.0
    assert metrics["first_tool_accuracy"] == 1.0
    assert metrics["transition_accuracy"] == 1.0
    assert metrics["kendall_tau"] == 1.0


def _write_m4_fixture(root):
    write_jsonl(
        root / "candidates.jsonl",
        [
            {
                "candidate_id": "tool_a",
                "name": "Auth",
                "text": "Auth token API",
                "source_dataset": "fixture",
                "semantic_id": "A/Auth/START/IO",
                "code_path": ["A", "Auth", "START", "IO"],
                "role_hint": "START",
            },
            {
                "candidate_id": "tool_b",
                "name": "Create",
                "text": "Create schedule API",
                "source_dataset": "fixture",
                "semantic_id": "A/Create/FINALIZE/IO",
                "code_path": ["A", "Create", "FINALIZE", "IO"],
                "role_hint": "FINALIZE",
            },
            {
                "candidate_id": "tool_c",
                "name": "Delete",
                "text": "Delete schedule API",
                "source_dataset": "fixture",
                "semantic_id": "A/Delete/FINALIZE/IO",
                "code_path": ["A", "Delete", "FINALIZE", "IO"],
                "role_hint": "FINALIZE",
            },
            {
                "candidate_id": "tool_d",
                "name": "Weather",
                "text": "Weather API",
                "source_dataset": "fixture",
                "semantic_id": "B/Weather/SUPPORT/IO",
                "code_path": ["B", "Weather", "SUPPORT", "IO"],
                "role_hint": "SUPPORT",
            },
        ],
    )
    write_jsonl(
        root / "queries.jsonl",
        [
            {
                "query_id": "q1",
                "query": "create a schedule after authentication",
                "split": "train",
                "source_dataset": "fixture",
                "gold_ids": ["tool_a", "tool_b"],
                "sequence_ids": ["tool_a", "tool_b"],
            }
        ],
    )
