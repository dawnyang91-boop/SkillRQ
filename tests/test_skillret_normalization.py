import json
from pathlib import Path

from skillrq.config.schema import PathsConfig
from skillrq.data import build_skillret_processed_data
from skillrq.utils.io import read_jsonl


def test_build_skillret_processed_data(tmp_path):
    raw_root = tmp_path / "raw"
    _write_skillret_fixture(raw_root)

    config_values = dict(PathsConfig.defaults())
    config_values["raw_root"] = str(raw_root)
    paths = PathsConfig.from_mapping(config_values, project_root=tmp_path)

    stats = build_skillret_processed_data(paths)

    assert stats["skills"] == 2
    assert stats["queries"] == 11
    assert stats["qrels"] == 12
    assert stats["multi_skill_queries"] == 1
    assert stats["splits"] == {"train": 9, "dev": 1, "test": 1}

    skills = list(read_jsonl(paths.processed_root / "skills.jsonl"))
    queries = list(read_jsonl(paths.processed_root / "queries.jsonl"))
    qrels = list(read_jsonl(paths.processed_root / "qrels.jsonl"))
    dev_split = list(read_jsonl(paths.processed_root / "splits" / "dev.jsonl"))

    assert skills[0]["skill_id"] == "s1"
    assert skills[0]["source_dataset"] == "skillret"
    assert any(query["query_id"] == "q-train-000010" for query in queries)
    assert {qrel["query_id"] for qrel in qrels} >= {"q-train-000001", "q-test-000001"}
    assert dev_split == [
        {
            "query_id": "q-train-000010",
            "source_dataset": "skillret",
            "source_split": "train",
            "split": "dev",
        }
    ]


def _write_skillret_fixture(raw_root: Path):
    train = raw_root / "skillret" / "train"
    test = raw_root / "skillret" / "test"
    train.mkdir(parents=True)
    test.mkdir(parents=True)

    skills = [
        {
            "id": "s1",
            "name": "alpha",
            "namespace": "@example/alpha",
            "description": "Alpha skill",
            "body": "Use alpha.",
            "skill_md": "# Alpha",
            "domain": "documents",
            "major": "Software Engineering",
            "sub": "Development",
            "primary_action": "parse",
            "primary_object": "file",
        },
        {
            "id": "s2",
            "name": "beta",
            "namespace": "@example/beta",
            "description": "Beta skill",
            "body": "Use beta.",
            "skill_md": "# Beta",
            "domain": "validation",
            "major": "Data",
            "sub": "Quality",
            "primary_action": "check",
            "primary_object": "number",
        },
    ]
    _write_jsonl(train / "skills.jsonl", skills)
    _write_jsonl(test / "skills.jsonl", skills)

    train_queries = [
        {
            "id": f"q-train-{index:06d}",
            "original_id": f"q-train-{index:06d}",
            "query": f"Train query {index}",
            "skill_ids": ["s1"],
            "skill_names": ["alpha"],
            "k": 1,
            "generator_model": "fixture",
        }
        for index in range(1, 11)
    ]
    train_queries[0]["skill_ids"] = ["s1", "s2"]
    train_queries[0]["skill_names"] = ["alpha", "beta"]
    train_queries[0]["k"] = 2
    _write_jsonl(train / "queries.jsonl", train_queries)

    test_queries = [
        {
            "id": "q-test-000001",
            "original_id": "q-test-000001",
            "query": "Test query",
            "skill_ids": ["s2"],
            "skill_names": ["beta"],
            "k": 1,
            "generator_model": "fixture",
        }
    ]
    _write_jsonl(test / "queries.jsonl", test_queries)

    train_qrels = [
        {"query_id": "q-train-000001", "skill_id": "s1", "relevance": 1},
        {"query_id": "q-train-000001", "skill_id": "s2", "relevance": 1},
    ]
    train_qrels.extend(
        {"query_id": f"q-train-{index:06d}", "skill_id": "s1", "relevance": 1}
        for index in range(2, 11)
    )
    _write_jsonl(train / "qrels.jsonl", train_qrels)
    _write_jsonl(test / "qrels.jsonl", [{"query_id": "q-test-000001", "skill_id": "s2", "relevance": 1}])


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
