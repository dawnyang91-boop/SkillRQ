from skillrq.prompting import build_code_guided_prompts
from skillrq.utils.io import read_jsonl, write_jsonl


def test_build_code_guided_prompts_uses_m5_steps_and_m7_tools(tmp_path):
    m5_path = tmp_path / "m5.jsonl"
    m7_path = tmp_path / "m7.jsonl"
    output_root = tmp_path / "prompts"
    write_jsonl(
        m5_path,
        [
            {
                "query_id": "q1",
                "query": "Find and book a hotel near the museum.",
                "gold_ids": ["tool_hotel"],
                "code_plan": [
                    {
                        "step_index": 0,
                        "codes": ["travel", "hotel_search", "START", "location_date_price"],
                        "role": "START",
                        "purpose": "find hotels under location and date constraints",
                        "expected_coverage_gain": 0.5,
                        "retrieved_capabilities": [{"candidate_id": "tool_hotel"}],
                    }
                ],
            }
        ],
    )
    write_jsonl(
        m7_path,
        [
            {
                "query_id": "q1",
                "query": "Find and book a hotel near the museum.",
                "reranked_capabilities": [
                    {
                        "candidate_id": "tool_hotel",
                        "name": "search_hotels",
                        "final_score": 0.9,
                        "prompt_usefulness_score": 0.8,
                        "code_path_consistency_score": 1.0,
                        "schema_compatibility_score": 0.7,
                        "coverage_gain_score": 0.5,
                        "code_explanation": "search hotels by location and date",
                    }
                ],
            }
        ],
    )

    summary = build_code_guided_prompts(
        prediction_path=m7_path,
        m5_prediction_path=m5_path,
        output_root=output_root,
        top_tools_per_step=2,
    )

    rows = list(read_jsonl(output_root / "prompt_records.jsonl"))
    assert summary["queries"] == 1
    assert rows[0]["capability_plan"][0]["candidate_tools"][0]["name"] == "search_hotels"
    assert "Capability Plan:" in rows[0]["prompt"]
    assert "hotel_search" in rows[0]["prompt"]
    assert "search_hotels" in rows[0]["prompt"]
