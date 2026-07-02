from skillrq.m5.planning import _build_candidate_retrieval_index, _load_m4_predictions, _retrieve_for_path, prepare_code_path_planning_data
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


def test_load_m4_predictions_keeps_retrieved_capabilities(tmp_path):
    prediction_path = tmp_path / "m4_predictions.jsonl"
    write_jsonl(
        prediction_path,
        [
            {
                "query_id": "q1",
                "predicted_code_paths": [{"semantic_id": "p1", "codes": ["a", "b", "c", "d"]}],
                "retrieved_capabilities": [{"candidate_id": "tool_a"}],
            }
        ],
    )

    rows = _load_m4_predictions(prediction_path)

    assert rows["q1"]["predicted_code_paths"][0]["semantic_id"] == "p1"
    assert rows["q1"]["retrieved_capabilities"][0]["candidate_id"] == "tool_a"


def test_retrieve_for_path_reuses_m4_candidates_and_deduplicates_used_ids():
    candidates = [
        {
            "candidate_id": "tool_code",
            "semantic_id": "L1-travel/L2-hotel/L3-start/L4-location_date_price",
            "code_path": ["L1-travel", "L2-hotel", "L3-start", "L4-location_date_price"],
            "name": "HotelSearch",
            "text": "search hotels",
        },
        {
            "candidate_id": "tool_m4",
            "semantic_id": "L1-map/L2-distance/L3-check/L4-location_distance",
            "code_path": ["L1-map", "L2-distance", "L3-check", "L4-location_distance"],
            "name": "DistanceCheck",
            "text": "check distance",
        },
        {
            "candidate_id": "tool_l1_only",
            "semantic_id": "L1-travel/L2-flight/L3-support/L4-location",
            "code_path": ["L1-travel", "L2-flight", "L3-support", "L4-location"],
            "name": "FlightSearch",
            "text": "search flights",
        },
    ]
    candidate_index = _build_candidate_retrieval_index(candidates)
    m4_retrieved = [
        {
            "candidate_id": "tool_m4",
            "name": "DistanceCheck",
            "code_match_score": 0.9,
        }
    ]

    rows = _retrieve_for_path(
        {"semantic_id": "p1", "codes": ["L1-travel", "L2-hotel", "L3-start", "L4-location_date_price"], "probability": 0.5},
        candidate_index,
        limit=5,
        m4_retrieved=m4_retrieved,
        used_candidate_ids=set(),
    )

    assert rows[0]["candidate_id"] == "tool_m4"
    assert rows[0]["m4_candidate_prior"] > 0
    assert rows[0]["retrieval_source"] in {"m4_prior", "m4_prior+code_bucket"}
    assert all(row["candidate_id"] != "tool_l1_only" for row in rows)

    rows_after_used = _retrieve_for_path(
        {"semantic_id": "p1", "codes": ["L1-travel", "L2-hotel", "L3-start", "L4-location_date_price"], "probability": 0.5},
        candidate_index,
        limit=5,
        m4_retrieved=m4_retrieved,
        used_candidate_ids={"tool_m4"},
    )
    assert all(row["candidate_id"] != "tool_m4" for row in rows_after_used)


def test_retrieve_for_path_exact_first_can_prioritize_exact_code_match_over_m4_prior():
    candidates = [
        {
            "candidate_id": "tool_exact",
            "semantic_id": "L1-travel/L2-hotel/L3-start/L4-location_date_price",
            "code_path": ["L1-travel", "L2-hotel", "L3-start", "L4-location_date_price"],
            "name": "HotelSearch",
            "text": "search hotels",
        },
        {
            "candidate_id": "tool_m4",
            "semantic_id": "L1-map/L2-distance/L3-check/L4-location_distance",
            "code_path": ["L1-map", "L2-distance", "L3-check", "L4-location_distance"],
            "name": "DistanceCheck",
            "text": "check distance",
        },
    ]
    candidate_index = _build_candidate_retrieval_index(candidates)
    path = {"semantic_id": "p1", "codes": ["L1-travel", "L2-hotel", "L3-start", "L4-location_date_price"], "probability": 0.5}
    m4_retrieved = [{"candidate_id": "tool_m4", "name": "DistanceCheck", "code_match_score": 1.0}]

    phase2_rows = _retrieve_for_path(path, candidate_index, limit=5, m4_retrieved=m4_retrieved)
    exact_first_rows = _retrieve_for_path(
        path,
        candidate_index,
        limit=5,
        m4_retrieved=m4_retrieved,
        exact_first_retrieval=True,
    )

    assert phase2_rows[0]["candidate_id"] == "tool_m4"
    assert exact_first_rows[0]["candidate_id"] == "tool_exact"


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
