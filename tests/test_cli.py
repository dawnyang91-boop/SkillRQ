import pytest

from skillrq.cli import main


def test_help_command_runs(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "SkillRQ experiment toolkit" in captured.out
