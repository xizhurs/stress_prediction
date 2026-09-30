from __future__ import annotations

from stress_prediction.cli import main


def test_root_help_is_side_effect_free(capsys: object) -> None:
    assert main([]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "train-lgb" in output


def test_train_help_is_side_effect_free(capsys: object) -> None:
    try:
        main(["train-lgb", "--help"])
    except SystemExit as error:
        assert error.code == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "--validation-start" in output
