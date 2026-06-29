from skillrq.m4.sequence_split import build_sequence_eval_view
from skillrq.utils.io import read_jsonl, write_jsonl


def test_build_sequence_eval_view_splits_sequence_train_queries(tmp_path):
    data_root = tmp_path / "m4" / "capability"
    output_root = tmp_path / "m4_sequence_eval" / "capability"
    _write_fixture(data_root)

    stats = build_sequence_eval_view(
        m4_data_root=data_root,
        output_root=output_root,
        sequence_dev_size=1,
        sequence_test_size=1,
        seed=1,
    )

    queries = list(read_jsonl(output_root / "queries.jsonl"))
    pairs = list(read_jsonl(output_root / "train_pairs.jsonl"))
    split_by_query_id = {row["query_id"]: row["split"] for row in queries}
    pair_split_by_query_id = {row["query_id"]: row["split"] for row in pairs}

    assert stats["sequence_dev_queries"] == 1
    assert stats["sequence_test_queries"] == 1
    assert stats["sequence_train_queries_before_split"] == 2
    assert sorted(split_by_query_id.values()).count("sequence_dev") == 1
    assert sorted(split_by_query_id.values()).count("sequence_test") == 1
    assert pair_split_by_query_id["q_seq_1"] == split_by_query_id["q_seq_1"]
    assert pair_split_by_query_id["q_seq_2"] == split_by_query_id["q_seq_2"]
    assert split_by_query_id["q_plain"] == "train"


def _write_fixture(root):
    write_jsonl(root / "candidates.jsonl", [{"candidate_id": "tool_a"}])
    queries = [
        {
            "query_id": "q_seq_1",
            "query": "sequence one",
            "source_dataset": "fixture",
            "source_split": "train",
            "split": "train",
            "gold_ids": ["tool_a"],
            "gold_code_paths": [{"candidate_id": "tool_a", "semantic_id": "A", "codes": ["A", "B", "C", "D"]}],
            "sequence_ids": ["tool_a"],
        },
        {
            "query_id": "q_seq_2",
            "query": "sequence two",
            "source_dataset": "fixture",
            "source_split": "train",
            "split": "train",
            "gold_ids": ["tool_a"],
            "gold_code_paths": [{"candidate_id": "tool_a", "semantic_id": "A", "codes": ["A", "B", "C", "D"]}],
            "sequence_ids": ["tool_a"],
        },
        {
            "query_id": "q_plain",
            "query": "plain",
            "source_dataset": "fixture",
            "source_split": "train",
            "split": "train",
            "gold_ids": ["tool_a"],
            "gold_code_paths": [{"candidate_id": "tool_a", "semantic_id": "A", "codes": ["A", "B", "C", "D"]}],
            "sequence_ids": [],
        },
    ]
    write_jsonl(root / "queries.jsonl", queries)
    write_jsonl(
        root / "train_pairs.jsonl",
        [
            {"query_id": row["query_id"], "query": row["query"], "split": row["split"], "candidate_id": "tool_a"}
            for row in queries
        ],
    )
