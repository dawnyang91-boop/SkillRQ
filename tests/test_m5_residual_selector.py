from skillrq.m5 import prepare_m5_data
from skillrq.m5.evaluate import evaluate_m5_predictions
from skillrq.utils.io import read_jsonl, write_jsonl


def test_prepare_m5_data_builds_residual_steps(tmp_path):
    m4_root = tmp_path / "m4"
    output_root = tmp_path / "m5"
    _write_m4_fixture(m4_root)

    stats = prepare_m5_data(m4_data_root=m4_root, output_root=output_root, max_steps=3)

    examples = list(read_jsonl(output_root / "residual_examples.jsonl"))
    plans = list(read_jsonl(output_root / "query_residual_plans.jsonl"))
    assert stats["queries"] == 1
    assert stats["residual_examples"] == 2
    assert len(plans[0]["residual_plan"]) == 2
    assert examples[0]["target_ids"] == ["tool_a", "tool_b"]
    assert examples[0]["coverage_gain"] == 2
    assert examples[0]["normalized_coverage_gain"] == 2 / 3
    assert examples[1]["target_ids"] == ["tool_c"]
    assert examples[1]["covered_before"] == ["tool_a", "tool_b"]


def test_evaluate_m5_predictions_reports_coverage_metrics(tmp_path):
    prediction_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "metrics.json"
    write_jsonl(
        prediction_path,
        [
            {
                "query_id": "q1",
                "gold_ids": ["tool_a", "tool_b"],
                "residual_code_paths": [
                    {
                        "step_index": 0,
                        "semantic_id": "A/B/C/D",
                        "retrieved_capabilities": [{"candidate_id": "tool_a"}],
                    },
                    {
                        "step_index": 1,
                        "semantic_id": "E/F/G/H",
                        "retrieved_capabilities": [{"candidate_id": "tool_b"}, {"candidate_id": "tool_a"}],
                    },
                ],
            }
        ],
    )

    metrics = evaluate_m5_predictions(prediction_path=prediction_path, output_path=output_path, top_ks=(1, 2, 3))

    assert metrics["evaluated_queries"] == 1
    assert metrics["step_0_coverage_gain"] == 0.5
    assert metrics["step_1_coverage_gain"] == 0.5
    assert metrics["recall@1"] == 0.5
    assert metrics["recall@2"] == 1.0
    assert metrics["completeness@2"] == 1.0
    assert metrics["candidate_redundancy_ratio"] == 1 / 3


def _write_m4_fixture(root):
    write_jsonl(
        root / "candidates.jsonl",
        [
            {
                "candidate_id": "tool_a",
                "semantic_id": "A/B/C/D",
                "code_path": ["A", "B", "C", "D"],
                "role_hint": "START",
            },
            {
                "candidate_id": "tool_b",
                "semantic_id": "A/B/C/D",
                "code_path": ["A", "B", "C", "D"],
                "role_hint": "START",
            },
            {
                "candidate_id": "tool_c",
                "semantic_id": "A/B/E/F",
                "code_path": ["A", "B", "E", "F"],
                "role_hint": "SUPPORT",
            },
        ],
    )
    write_jsonl(
        root / "queries.jsonl",
        [
            {
                "query_id": "q1",
                "query": "use three tools",
                "split": "train",
                "source_dataset": "fixture",
                "gold_ids": ["tool_a", "tool_b", "tool_c"],
            }
        ],
    )
