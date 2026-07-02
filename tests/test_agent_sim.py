from skillrq.agent_sim import evaluate_tool_call_plans, simulate_mock_tool_calls
from skillrq.utils.io import read_jsonl, write_jsonl


def test_mock_simulation_and_evaluation(tmp_path):
    prompt_path = tmp_path / "prompt_records.jsonl"
    sim_root = tmp_path / "sim"
    metrics_path = tmp_path / "metrics.json"
    write_jsonl(
        prompt_path,
        [
            {
                "query_id": "q1",
                "query": "Find and book a hotel.",
                "gold_ids": ["tool_search", "tool_book"],
                "sequence_ids": ["tool_search", "tool_book"],
                "capability_plan": [
                    {
                        "step": 1,
                        "role": "START",
                        "operation": "hotel_search",
                        "candidate_tools": [{"candidate_id": "tool_search", "name": "search_hotels"}],
                        "why_needed": "find candidate hotels",
                    },
                    {
                        "step": 2,
                        "role": "FINALIZE",
                        "operation": "booking",
                        "candidate_tools": [{"candidate_id": "tool_book", "name": "book_hotel"}],
                        "why_needed": "book selected hotel",
                    },
                ],
            }
        ],
    )

    summary = simulate_mock_tool_calls(prompt_record_path=prompt_path, output_root=sim_root, max_calls=4)
    plans = list(read_jsonl(sim_root / "tool_call_plans.jsonl"))
    metrics = evaluate_tool_call_plans(plan_path=sim_root / "tool_call_plans.jsonl", output_path=metrics_path, top_ks=(1, 2))

    assert summary["queries"] == 1
    assert [call["candidate_id"] for call in plans[0]["tool_calls"]] == ["tool_search", "tool_book"]
    assert metrics["prompt_grounding_rate"] == 1.0
    assert metrics["tool_set_recall@2"] == 1.0
    assert metrics["first_tool_accuracy"] == 1.0
    assert metrics["transition_accuracy"] == 1.0
