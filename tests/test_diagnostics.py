from skillrq.diagnostics import run_diagnostics
from skillrq.utils.io import read_jsonl, write_jsonl


def test_run_diagnostics_outputs_upper_bounds_and_attribution_reports(tmp_path):
    _write_fixture(tmp_path)

    summary = run_diagnostics(
        project_root=tmp_path,
        target="capability",
        output_root=tmp_path / "reports" / "diagnostics" / "capability",
        top_ks=(1, 2),
        include_joint_predictions=False,
    )

    output_root = tmp_path / "reports" / "diagnostics" / "capability"
    assert summary["candidate_pool"]["candidate_pool_upper_bounds"]["m4"]["oracle_recall@2"] == 1.0
    assert (output_root / "candidate_pool_upper_bounds.json").exists()
    assert (output_root / "codebook_diagnostics.json").exists()
    assert (output_root / "multi_positive_diagnostics.json").exists()
    assert (output_root / "negative_sampling_diagnostics.json").exists()
    assert (output_root / "sequence_chain_diagnostics.json").exists()
    cases = list(read_jsonl(output_root / "candidate_pool_failure_cases.jsonl"))
    assert cases


def _write_fixture(root):
    m4_root = root / "data" / "processed" / "m4" / "capability"
    m7_root = root / "data" / "processed" / "m7" / "capability"
    m4_pred_root = root / "runs" / "m4_query_to_code" / "predictions" / "capability"
    m5_pred_root = root / "runs" / "m5_residual_selector" / "predictions" / "capability"
    m7_pred_root = root / "runs" / "m7_reranker" / "predictions" / "capability"
    candidates = [
        {
            "candidate_id": "tool_a",
            "name": "Auth",
            "text": "Auth token API",
            "semantic_id": "A/Auth/START/IO",
            "code_path": ["A", "Auth", "START", "IO"],
            "role_hint": "START",
            "labels": {"l1": "auth", "l2": "auth", "l3": "START", "l4": "io"},
            "metadata": {"category": "auth"},
            "source_dataset": "fixture",
        },
        {
            "candidate_id": "tool_b",
            "name": "Create",
            "text": "Create schedule API",
            "semantic_id": "A/Create/FINALIZE/IO",
            "code_path": ["A", "Create", "FINALIZE", "IO"],
            "role_hint": "FINALIZE",
            "labels": {"l1": "schedule", "l2": "create", "l3": "FINALIZE", "l4": "io"},
            "metadata": {"category": "schedule"},
            "source_dataset": "fixture",
        },
        {
            "candidate_id": "tool_c",
            "name": "Delete",
            "text": "Delete schedule API",
            "semantic_id": "A/Delete/FINALIZE/IO",
            "code_path": ["A", "Delete", "FINALIZE", "IO"],
            "role_hint": "FINALIZE",
            "labels": {"l1": "schedule", "l2": "delete", "l3": "FINALIZE", "l4": "io"},
            "metadata": {"category": "schedule"},
            "source_dataset": "fixture",
        },
    ]
    queries = [
        {
            "query_id": "q1",
            "query": "create schedule after auth",
            "split": "test",
            "source_dataset": "fixture",
            "gold_ids": ["tool_a", "tool_b"],
            "sequence_ids": ["tool_a", "tool_b"],
            "gold_code_paths": [
                {"candidate_id": "tool_a", "semantic_id": "A/Auth/START/IO"},
                {"candidate_id": "tool_b", "semantic_id": "A/Create/FINALIZE/IO"},
            ],
        }
    ]
    write_jsonl(m4_root / "candidates.jsonl", candidates)
    write_jsonl(m4_root / "queries.jsonl", queries)
    write_jsonl(
        m7_root / "rerank_examples.jsonl",
        [
            {
                "query_id": "q1",
                "query": "create schedule after auth",
                "candidate_id": "tool_a",
                "candidate_name": "Auth",
                "candidate_text": "Auth token API",
                "semantic_id": "A/Auth/START/IO",
                "code_path": ["A", "Auth", "START", "IO"],
                "role_label": "START",
                "stage_label": "FIRST",
                "label": 1,
                "features": {"code_match_score": 1.0, "matched_levels": 1.0, "text_overlap_score": 0.5},
            },
            {
                "query_id": "q1",
                "query": "create schedule after auth",
                "candidate_id": "tool_c",
                "candidate_name": "Delete",
                "candidate_text": "Delete schedule API",
                "semantic_id": "A/Delete/FINALIZE/IO",
                "code_path": ["A", "Delete", "FINALIZE", "IO"],
                "role_label": "FINALIZE",
                "stage_label": "UNKNOWN",
                "label": 0,
                "features": {"code_match_score": 0.5, "matched_levels": 0.5, "text_overlap_score": 0.2},
            },
        ],
    )
    write_jsonl(
        m4_pred_root / "predictions.jsonl",
        [
            {
                "query_id": "q1",
                "query": "create schedule after auth",
                "gold_ids": ["tool_a", "tool_b"],
                "retrieved_capabilities": [{"candidate_id": "tool_c"}, {"candidate_id": "tool_a"}, {"candidate_id": "tool_b"}],
            }
        ],
    )
    write_jsonl(
        m5_pred_root / "predictions.jsonl",
        [
            {
                "query_id": "q1",
                "query": "create schedule after auth",
                "gold_ids": ["tool_a", "tool_b"],
                "residual_code_paths": [
                    {"retrieved_capabilities": [{"candidate_id": "tool_a"}]},
                    {"retrieved_capabilities": [{"candidate_id": "tool_c"}]},
                ],
            }
        ],
    )
    write_jsonl(
        m7_pred_root / "reranked_predictions.jsonl",
        [
            {
                "query_id": "q1",
                "query": "create schedule after auth",
                "gold_ids": ["tool_a", "tool_b"],
                "sequence_ids": ["tool_a", "tool_b"],
                "predicted_tool_order": ["tool_a", "tool_c"],
                "reranked_capabilities": [{"candidate_id": "tool_a"}, {"candidate_id": "tool_c"}],
            }
        ],
    )
