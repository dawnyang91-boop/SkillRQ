import csv
import json
from pathlib import Path

from skillrq.capability import build_capability_processed_data
from skillrq.config.schema import PathsConfig
from skillrq.utils.io import read_jsonl


def test_build_capability_data_from_api_bank_fixture(tmp_path):
    raw_root = tmp_path / "raw"
    _write_api_bank_fixture(raw_root)
    paths = _paths(tmp_path, raw_root)

    stats = build_capability_processed_data(paths, dataset="api_bank")

    assert stats["capabilities"] == 2
    assert stats["queries"] == 1
    assert stats["qrels"] == 2
    assert stats["sequences"] == 2
    assert stats["min_unique_tools_per_query"] == 2
    assert stats["max_tool_calls_per_trajectory"] == 2

    capabilities = list(read_jsonl(paths.capability_processed_root / "capabilities.jsonl"))
    queries = list(read_jsonl(paths.capability_processed_root / "capability_queries.jsonl"))
    sequences = list(read_jsonl(paths.capability_processed_root / "capability_sequences.jsonl"))

    assert {row["api_name"] for row in capabilities} == {"GetUserToken", "AddAgenda"}
    assert queries[0]["query"] == "Add a meeting tomorrow."
    assert len(queries[0]["gold_capability_ids"]) == 2
    assert sequences[0]["arguments"] == {"username": "foo", "password": "bar"}


def _paths(project_root: Path, raw_root: Path) -> PathsConfig:
    values = dict(PathsConfig.defaults())
    values["raw_root"] = str(project_root / "legacy_raw")
    values["capability_raw_root"] = str(raw_root)
    values["capability_processed_root"] = "capability_out"
    return PathsConfig.from_mapping(values, project_root=project_root)


def _write_api_bank_fixture(raw_root: Path):
    root = raw_root / "DAMO-ConvAI" / "api-bank"
    (root / "data").mkdir(parents=True)
    sample_dir = root / "lv1-lv2-samples" / "level-1-given-desc"
    sample_dir.mkdir(parents=True)

    with (root / "data" / "all_apis.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "类型", "应用场景", "API名称", "参数", "路径", "类名", "input_parameters", "expressions", "api_info"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "1",
                "类型": "auth",
                "应用场景": "account",
                "API名称": "获取用户Token",
                "路径": "apis/get_user_token.py",
                "类名": "GetUserToken",
                "api_info": "description = \"Get token\"\ninput_parameters = {'username': {'type': 'str', 'description': 'user'}, 'password': {'type': 'str', 'description': 'pass'}}\noutput_parameters = {'token': {'type': 'str', 'description': 'token'}}",
            }
        )
        writer.writerow(
            {
                "id": "2",
                "类型": "calendar",
                "应用场景": "agenda",
                "API名称": "添加日程",
                "路径": "apis/add_agenda.py",
                "类名": "AddAgenda",
                "api_info": "description = \"Add agenda\"\ninput_parameters = {'token': {'type': 'str', 'description': 'token'}}\noutput_parameters = {'status': {'type': 'str', 'description': 'status'}}",
            }
        )

    messages = [
        {"role": "User", "text": "Add a meeting tomorrow."},
        {"role": "API", "api_name": "GetUserToken", "param_dict": {"username": "foo", "password": "bar"}, "result": {"exception": None}},
        {"role": "API", "api_name": "AddAgenda", "param_dict": {"token": "tok"}, "result": {"exception": None}},
        {"role": "AI", "text": "Done."},
    ]
    with (sample_dir / "AddAgenda-level-1-1.jsonl").open("w", encoding="utf-8") as handle:
        for message in messages:
            handle.write(json.dumps(message, ensure_ascii=False))
            handle.write("\n")

