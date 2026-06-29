from pathlib import Path

from skillrq.codebook import build_m3_codebooks
from skillrq.config.schema import PathsConfig
from skillrq.utils.io import read_jsonl, write_jsonl


def test_build_m3_code_assignments_for_capability_and_skill(tmp_path):
    paths = _paths(tmp_path)
    _write_fixture(paths)

    summary = build_m3_codebooks(paths, datasets=["api_bank", "skillret"])

    capability_rows = list(read_jsonl(paths.capability_processed_root / "code_assignments.jsonl"))
    skill_rows = list(read_jsonl(paths.processed_root / "skill" / "code_assignments.jsonl"))

    assert summary["capability_assignments"] == 1
    assert summary["skill_assignments"] == 1
    assert len(capability_rows[0]["code_path"]) == 4
    assert capability_rows[0]["semantic_id"].count("/") == 3
    assert capability_rows[0]["l3_label"] == "START"
    assert skill_rows[0]["source_dataset"] == "skillret"
    assert (paths.report_root / "code_cards" / "api_bank_code_cards.md").exists()
    assert (paths.report_root / "code_cards" / "skillret_code_cards.md").exists()


def _paths(project_root: Path) -> PathsConfig:
    values = dict(PathsConfig.defaults())
    values["raw_root"] = str(project_root / "raw")
    values["processed_root"] = "processed"
    values["capability_raw_root"] = str(project_root / "capability_raw")
    values["capability_processed_root"] = "processed/capability"
    values["report_root"] = "reports"
    return PathsConfig.from_mapping(values, project_root=project_root)


def _write_fixture(paths: PathsConfig) -> None:
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
                "category": "query",
                "domain": "weather",
                "provider": "API-Bank",
                "tool_name": "GetWeather",
                "api_name": "GetWeather",
                "required_parameters": [{"name": "city", "type": "str"}],
                "optional_parameters": [],
                "output_schema": {"temperature": {"type": "number"}},
                "method": "GET",
            }
        ],
    )
    write_jsonl(
        paths.capability_processed_root / "capability_sequences.jsonl",
        [
            {
                "query_id": "q1",
                "step_index": 0,
                "capability_id": "api_bank::get_weather",
                "source_dataset": "api_bank",
            }
        ],
    )
    write_jsonl(
        paths.processed_root / "skills.jsonl",
        [
            {
                "skill_id": "skill/weather",
                "source_dataset": "skillret",
                "source_skill_id": "skill/weather",
                "source_split": "train",
                "name": "weather-check",
                "description": "Check weather data.",
                "body": "Use examples to verify weather reports.",
                "domain_label": "weather",
                "primary_action": "check",
            }
        ],
    )
