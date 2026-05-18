import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from gec_tagger_train.eval_m2 import (
    M2Score,
    format_m2_predictions,
    parse_m2scorer_output,
    run_m2scorer,
)


def test_format_predictions_emits_m2_block() -> None:
    src_tokens = ["he", "go", "home"]
    pred_tokens = ["he", "goes", "home"]
    block = format_m2_predictions(src_tokens=src_tokens, pred_tokens=pred_tokens)
    assert block.startswith("S he go home")
    assert "A 1 2|||" in block


def test_no_change_emits_noop_edit() -> None:
    block = format_m2_predictions(src_tokens=["he", "is", "ok"], pred_tokens=["he", "is", "ok"])
    assert "A -1 -1|||noop|||" in block


def test_parse_m2scorer_canonical_output() -> None:
    sample = "Precision   : 0.6500\nRecall      : 0.4123\nF_0.5       : 0.5876\n"
    score = parse_m2scorer_output(sample)
    assert isinstance(score, M2Score)
    assert score.precision == pytest.approx(0.65)
    assert score.recall == pytest.approx(0.4123)
    assert score.f0_5 == pytest.approx(0.5876)


def test_parse_m2scorer_with_counters() -> None:
    sample = (
        "TP          : 42\nFP          : 8\nFN          : 50\n"
        "Precision   : 0.84\nRecall      : 0.4565\nF0.5        : 0.732\n"
    )
    score = parse_m2scorer_output(sample)
    assert score.tp == 42
    assert score.fp == 8
    assert score.fn == 50


def test_parse_m2scorer_raises_on_missing_fields() -> None:
    with pytest.raises(ValueError):
        parse_m2scorer_output("Precision : 0.5\n")


def test_run_m2scorer_invokes_subprocess(tmp_path: Path) -> None:
    bin_path = tmp_path / "m2scorer"
    bin_path.write_text("#!/bin/sh\n")
    bin_path.chmod(0o755)
    gold = tmp_path / "gold.m2"
    gold.write_text("S he\nA -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n\n")
    fake = subprocess.CompletedProcess(
        args=["m2scorer"],
        returncode=0,
        stdout="Precision : 0.7\nRecall : 0.5\nF0.5 : 0.65\n",
        stderr="",
    )
    with patch("gec_tagger_train.eval_m2.subprocess.run", return_value=fake) as runner:
        score = run_m2scorer(
            m2scorer_bin=bin_path,
            gold_m2=gold,
            hyp_lines=["He goes home."],
        )
    assert runner.call_count == 1
    assert score.f0_5 == pytest.approx(0.65)


def test_run_m2scorer_raises_when_bin_missing(tmp_path: Path) -> None:
    gold = tmp_path / "gold.m2"
    gold.write_text("S x\n\n")
    with pytest.raises(FileNotFoundError):
        run_m2scorer(
            m2scorer_bin=tmp_path / "missing-scorer",
            gold_m2=gold,
            hyp_lines=["x"],
        )
