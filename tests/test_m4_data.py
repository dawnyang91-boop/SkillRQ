from pathlib import Path

from skillrq.config.schema import PathsConfig
from skillrq.m4 import prepare_m4_data
from skillrq.utils.io import read_jsonl, write_jsonl


def test_prepare_m4_capability_data(tmp_path):
    paths = _paths(tmp_path)
    _write_capability_fixture(paths)

    stats = prepare_m4_data(paths, target="capability", datasets=["api_bank"])

    rows = list(read_jsonl(paths.processed_root / "m4" / "capability" / "train_pairs.jsonl"))
    assert stats["candidates"] == 1
    assert stats["queries"] == 1
    assert stats["train_pairs"] == 1
    assert rows[0]["code_path"] == ["L1-weather", "L2-get_weather", "L3-start", "L4-io"]


def _paths(project_root: Path) -> PathsConfig:
    values = dict(PathsConfig.defaults())
    values["raw_root"] = str(project_root / "raw")
    values["processed_root"] = "processed"
    values["capability_raw_root"] = str(project_root / "capability_raw")
    values["capability_processed_root"] = "processed/capability"
    values["report_root"] = "reports"
    return PathsConfig.from_mapping(values, project_root=project_root)


def _write_capability_fixture(paths: PathsConfig) -> None:
    write_jsonl(
        paths.capability_processed_root / "code_assignments.jsonl",
        [
            {
                "object_id": "api_bank::get_weather",
                "source_dataset": "api_bank",
                "name": "GetWeather",
                "semantic_id": "L1-weather/L2-get_weather/L3-start/L4-io",
                "code_path": ["L1-weather", "L2-get_weather", "L3-start", "L4-io"],
                "l3_label": "START",
                "code_explanation": "weather code",
            }
        ],
    )
    write_jsonl(
        paths.capability_processed_root / "capabilities.jsonl",
        [
            {
                "capability_id": "api_bank::get_weather",
                "source_dataset": "api_bank",
                "source_capability_id": "1",
                "capability_type": "api",
                "name": "GetWeather",
                "description": "Get weather by city.",
                "category": "weather",
                "domain": "weather",
                "provider": "API-Bank",
                "tool_name": "GetWeather",
                "api_name": "GetWeather",
                "required_parameters": [{"name": "city", "type": "str"}],
                "optional_parameters": [],
                "output_schema": {"temperature": {"type": "number"}},
            }
        ],
    )
    write_jsonl(
        paths.capability_processed_root / "capability_queries.jsonl",
        [
            {
                "query_id": "q1",
                "query": "weather in london",
                "source_dataset": "api_bank",
                "source_split": "samples",
                "gold_capability_ids": ["api_bank::get_weather"],
                "tool_call_sequence": ["api_bank::get_weather"],
            }
        ],
    )
