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


def test_recall_target_defaults_match_the_training_default() -> None:
    from stress_prediction.cli import build_parser
    from stress_prediction.training.lightgbm import DEFAULT_RECALL_TARGET

    parser = build_parser()
    for command in ("train-lgb", "eval-timestral", "eval-chronos2"):
        assert parser.parse_args([command]).recall_target == DEFAULT_RECALL_TARGET
    arguments = parser.parse_args(["compare-models", "--run", "a=b"])
    assert arguments.recall_target == DEFAULT_RECALL_TARGET
