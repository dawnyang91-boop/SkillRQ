from skillrq.m4.soft import _build_path_catalog, _build_query_examples, verbalize_code_path


def test_soft_m4_path_catalog_and_query_examples_are_multi_path():
    candidates = [
        {
            "candidate_id": "tool_a",
            "semantic_id": "L1-travel/L2-hotel/L3-start/L4-location_date_price",
            "code_path": ["L1-travel", "L2-hotel", "L3-start", "L4-location_date_price"],
            "labels": {"l1": "Travel", "l2": "Hotel Search", "l3": "START", "l4": "location date price"},
            "role_hint": "START",
            "name": "HotelSearch",
        },
        {
            "candidate_id": "tool_b",
            "semantic_id": "L1-map/L2-route/L3-support/L4-location_distance",
            "code_path": ["L1-map", "L2-route", "L3-support", "L4-location_distance"],
            "labels": {"l1": "Map", "l2": "Route Planning", "l3": "SUPPORT", "l4": "location distance"},
            "role_hint": "SUPPORT",
            "name": "RoutePlanner",
        },
    ]
    queries = [
        {
            "query_id": "q1",
            "query": "Find a hotel and check distance.",
            "split": "train",
            "gold_ids": ["tool_a", "tool_b"],
            "gold_code_paths": [
                {
                    "semantic_id": "L1-travel/L2-hotel/L3-start/L4-location_date_price",
                    "codes": ["L1-travel", "L2-hotel", "L3-start", "L4-location_date_price"],
                },
                {
                    "semantic_id": "L1-map/L2-route/L3-support/L4-location_distance",
                    "codes": ["L1-map", "L2-route", "L3-support", "L4-location_distance"],
                },
            ],
        }
    ]

    catalog = _build_path_catalog(candidates, queries)
    examples = _build_query_examples(queries, catalog)

    assert len(catalog) == 2
    assert len(examples) == 1
    assert len(examples[0]["gold_path_ids"]) == 2
    assert "Domain: Travel" in catalog[1]["verbalization"]
    assert "Role: START" in catalog[1]["verbalization"]


def test_verbalize_code_path_uses_labels_when_available():
    text = verbalize_code_path(
        {
            "codes": ["L1-travel", "L2-hotel", "L3-start", "L4-location_date_price"],
            "labels": {"l1": "Travel", "l2": "Hotel Search", "l3": "START", "l4": "location/date/price"},
        }
    )

    assert "Domain: Travel" in text
    assert "Operation: Hotel Search" in text
    assert "IO Constraint: location/date/price" in text
