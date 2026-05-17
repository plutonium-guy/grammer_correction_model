from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


def format_m2_predictions(*, src_tokens: list[str], pred_tokens: list[str]) -> str:
    """Emit a single-sentence M² block for the m2scorer tool.

    Edits are derived from a token-level diff. Replacement spans are emitted
    as `A start end|||R|||replacement|||REQUIRED|||-NONE-|||0`. A perfect
    match emits the canonical `noop` edit (`A -1 -1`).
    """
    lines: list[str] = [f"S {' '.join(src_tokens)}"]
    matcher = SequenceMatcher(a=src_tokens, b=pred_tokens, autojunk=False)
    any_edit = False
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        any_edit = True
        replacement = " ".join(pred_tokens[j1:j2])
        kind = {"replace": "R", "delete": "U", "insert": "M"}[op]
        end = i1 if op == "insert" else i2
        lines.append(
            f"A {i1} {end}|||{kind}|||{replacement}|||REQUIRED|||-NONE-|||0"
        )
    if not any_edit:
        lines.append("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class M2Score:
    """Scores parsed from m2scorer stdout."""

    precision: float
    recall: float
    f0_5: float
    tp: int
    fp: int
    fn: int


_M2_LINE = re.compile(
    r"^(Precision|Recall|F_?0\.5|TP|FP|FN)\s*:\s*([0-9.eE+-]+)\s*$",
    re.MULTILINE,
)


def parse_m2scorer_output(stdout: str) -> M2Score:
    """Parse the `Precision : … Recall : … F0.5 : …` block printed by m2scorer.

    Tolerant of the slight format drift between m2scorer forks
    (`F0.5` vs `F_0.5`). TP/FP/FN may be absent on some builds; default to 0.
    """
    found: dict[str, float] = {}
    for match in _M2_LINE.finditer(stdout):
        key = match.group(1).replace("_", "").lower()
        found[key] = float(match.group(2))
    if "precision" not in found or "recall" not in found or "f0.5" not in found:
        raise ValueError(
            f"m2scorer output missing precision/recall/f0.5 lines: {stdout!r}"
        )
    return M2Score(
        precision=found["precision"],
        recall=found["recall"],
        f0_5=found["f0.5"],
        tp=int(found.get("tp", 0)),
        fp=int(found.get("fp", 0)),
        fn=int(found.get("fn", 0)),
    )


def run_m2scorer(
    *,
    m2scorer_bin: Path,
    gold_m2: Path,
    hyp_lines: Iterable[str],
    hyp_path: Path | None = None,
) -> M2Score:
    """Invoke m2scorer as a subprocess and return parsed scores.

    `m2scorer_bin` is the path to the `m2scorer` script from the upstream
    repo (https://github.com/nusnlp/m2scorer). `hyp_lines` is the model's
    output, one corrected sentence per line; we write it to `hyp_path` (or
    a temp file when `None`). `gold_m2` is the reference M² file.

    Raises `FileNotFoundError` if the scorer or gold file is missing,
    `subprocess.CalledProcessError` if the scorer exits non-zero,
    `ValueError` if the output can't be parsed.
    """
    if not m2scorer_bin.is_file():
        raise FileNotFoundError(f"m2scorer not found at {m2scorer_bin}")
    if not gold_m2.is_file():
        raise FileNotFoundError(f"gold M² not found at {gold_m2}")

    if hyp_path is None:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".hyp", delete=False, encoding="utf-8"
        )
        try:
            for line in hyp_lines:
                tmp.write(line.rstrip("\n") + "\n")
            tmp.close()
            return _invoke_m2scorer(m2scorer_bin, Path(tmp.name), gold_m2)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
    else:
        hyp_path.parent.mkdir(parents=True, exist_ok=True)
        with hyp_path.open("w", encoding="utf-8") as f:
            for line in hyp_lines:
                f.write(line.rstrip("\n") + "\n")
        return _invoke_m2scorer(m2scorer_bin, hyp_path, gold_m2)


def _invoke_m2scorer(bin_path: Path, hyp: Path, gold: Path) -> M2Score:
    proc = subprocess.run(
        [str(bin_path), str(hyp), str(gold)],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_m2scorer_output(proc.stdout)
