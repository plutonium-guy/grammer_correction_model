from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.base import Source
from data_pipeline.types import Pair, Split

# Edit type: (start, end, replacement, annotator).
M2Edit = tuple[int, int, str, int]


class BEA2019Source(Source):
    name = "bea2019"

    def __init__(self, config: PipelineConfig, split: Split) -> None:
        super().__init__(config)
        self.split = split

    def _m2_file(self) -> Path:
        fname = {
            Split.TRAIN: "train.m2",
            Split.DEV: "dev.m2",
            Split.TEST: "test.m2",
        }[self.split]
        return self.raw_path / fname

    def iter_pairs(self) -> Iterator[Pair]:
        path = self._m2_file()
        if not path.exists():
            raise FileNotFoundError(
                f"expected {path} (download BEA-2019 to {self.raw_path})"
            )
        # BEA-2019 is single-annotator. Reduce per-annotator edits to a flat
        # list (any annotator id maps to the same set of edits in practice).
        for src_toks, by_annotator in _parse_m2(path):
            edits = _flatten_annotators(by_annotator)
            tgt_toks = _apply_edits(src_toks, edits)
            yield Pair(
                src=" ".join(src_toks),
                tgt=" ".join(tgt_toks),
                source=self.name,
                split=self.split,
            )


def _parse_m2(path: Path) -> Iterator[tuple[list[str], dict[int, list[tuple[int, int, str]]]]]:
    """Yield `(src_toks, edits_by_annotator)` per record.

    Each `A` line in M² format carries an annotator id in field index 5. Multi-
    annotator corpora (CoNLL-2014) have two annotators per sentence; we keep
    them separate so callers can either flatten (BEA-2019 / single-annotator)
    or yield one pair per annotator (CoNLL-2014 eval).

    `start == -1` (noop) edits are dropped silently.
    """
    src_toks: list[str] | None = None
    edits_by_ann: dict[int, list[tuple[int, int, str]]] = {}
    with path.open() as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line.startswith("S "):
                if src_toks is not None:
                    yield src_toks, edits_by_ann
                src_toks = line[2:].split(" ")
                edits_by_ann = {}
            elif line.startswith("A "):
                # A start end|||type|||replacement|||REQUIRED|||-NONE-|||annotator
                parts = line[2:].split("|||")
                span = parts[0].split(" ")
                start, end = int(span[0]), int(span[1])
                if start == -1:
                    continue
                replacement = parts[2]
                annotator = int(parts[5]) if len(parts) > 5 else 0
                edits_by_ann.setdefault(annotator, []).append(
                    (start, end, replacement)
                )
            elif line == "":
                if src_toks is not None:
                    yield src_toks, edits_by_ann
                    src_toks, edits_by_ann = None, {}
        if src_toks is not None:
            yield src_toks, edits_by_ann


def _flatten_annotators(
    by_annotator: dict[int, list[tuple[int, int, str]]],
) -> list[tuple[int, int, str]]:
    """Pick a single annotator's edits to apply.

    For single-annotator corpora the choice is trivial. For multi-annotator
    corpora the caller should iterate `by_annotator` directly; this helper
    just picks the lowest annotator id present so the legacy BEA-2019 path
    stays deterministic.
    """
    if not by_annotator:
        return []
    first_ann = min(by_annotator)
    return by_annotator[first_ann]


def _apply_edits(
    src_toks: list[str], edits: list[tuple[int, int, str]]
) -> list[str]:
    if not edits:
        return list(src_toks)
    # Apply edits in right-to-left order to keep offsets stable.
    out = list(src_toks)
    for start, end, replacement in sorted(edits, key=lambda e: -e[0]):
        repl = replacement.split(" ") if replacement else []
        out[start:end] = repl
    return out
