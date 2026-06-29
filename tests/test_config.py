from pathlib import Path

from skillrq.config import PathsConfig, load_paths_config


def test_paths_config_defaults_resolve_relative_paths(tmp_path):
    config = PathsConfig.from_mapping(PathsConfig.defaults(), project_root=tmp_path)

    assert config.raw_root == Path("/Users/sihan/code/skill-rec/data/raw")
    assert config.processed_root == tmp_path / "data/processed"
    assert config.run_root == tmp_path / "runs"
    assert config.capability_raw_root == tmp_path / "data/raw"
    assert config.capability_processed_root == tmp_path / "data/processed/capability"


def test_load_paths_config_reads_flat_yaml(tmp_path, monkeypatch):
    config_file = tmp_path / "paths.yaml"
    config_file.write_text(
        "\n".join(
            [
                "raw_root: /tmp/raw",
                "project_data_root: local_data",
                "processed_root: local_data/processed",
                "index_root: local_data/indexes",
                "cache_root: local_data/cache",
                "run_root: local_runs",
                "report_root: local_reports",
                "capability_raw_root: local_raw",
                "capability_processed_root: local_capability",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    config = load_paths_config(config_file)

    assert config.raw_root == Path("/tmp/raw")
    assert config.project_data_root == tmp_path / "local_data"
    assert config.report_root == tmp_path / "local_reports"
    assert config.capability_raw_root == tmp_path / "local_raw"
    assert config.capability_processed_root == tmp_path / "local_capability"
