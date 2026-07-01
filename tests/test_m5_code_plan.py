from skillrq.m5.planning import prepare_code_path_planning_data
from skillrq.utils.io import read_jsonl, write_jsonl


def test_prepare_code_path_planning_data_builds_role_schema_plan(tmp_path):
    m4_root = tmp_path / "m4" / "capability"
    output_root = tmp_path / "m5_code_plan" / "capability"
    _write_fixture(m4_root)

    stats = prepare_code_path_planning_data(
        m4_data_root=m4_root,
        output_root=output_root,
        max_steps=4,
    )

    plans = list(read_jsonl(output_root / "query_code_plans.jsonl"))
    examples = list(read_jsonl(output_root / "code_plan_examples.jsonl"))

    assert stats["queries"] == 1
    assert plans[0]["code_plan"][0]["role"] == "START"
    assert "purpose" in plans[0]["code_plan"][0]
    assert plans[0]["code_plan"][0]["operation"] == "hotel"
    assert any(row["stop_label"] == 1 for row in examples)
    assert examples[0]["covered_roles"] == []
    assert "predicted_paths" in examples[0]["planner_state"]


def _write_fixture(root):
    write_jsonl(
        root / "candidates.jsonl",
        [
            {
                "candidate_id": "tool_hotel",
                "semantic_id": "L1-travel/L2-hotel/L3-start/L4-location_date_price",
                "code_path": ["L1-travel", "L2-hotel", "L3-start", "L4-location_date_price"],
                "role_hint": "START",
                "name": "HotelSearch",
                "text": "search hotels by location date price",
            },
            {
                "candidate_id": "tool_distance",
                "semantic_id": "L1-map/L2-distance/L3-check/L4-location_distance",
                "code_path": ["L1-map", "L2-distance", "L3-check", "L4-location_distance"],
                "role_hint": "CHECK",
                "name": "DistanceCheck",
                "text": "check distance by location",
            },
        ],
    )
    write_jsonl(
        root / "queries.jsonl",
        [
            {
                "query_id": "q1",
                "query": "Find a hotel and verify distance.",
                "split": "train",
                "source_dataset": "fixture",
                "gold_ids": ["tool_hotel", "tool_distance"],
                "gold_code_paths": [
                    {
                        "candidate_id": "tool_hotel",
                        "semantic_id": "L1-travel/L2-hotel/L3-start/L4-location_date_price",
                        "codes": ["L1-travel", "L2-hotel", "L3-start", "L4-location_date_price"],
                        "role_hint": "START",
                    },
                    {
                        "candidate_id": "tool_distance",
                        "semantic_id": "L1-map/L2-distance/L3-check/L4-location_distance",
                        "codes": ["L1-map", "L2-distance", "L3-check", "L4-location_distance"],
                        "role_hint": "CHECK",
                    },
                ],
            }
        ],
    )
