from __future__ import annotations

from collections.abc import Iterator

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.base import Source
from data_pipeline.sources.bea2019 import _apply_edits, _parse_m2
from data_pipeline.types import Pair, Split


class CoNLL2014Source(Source):
    """CoNLL-2014 GEC eval set.

    The official corpus has two annotators per sentence; we yield one
    `Pair` per annotator (preserving both gold corrections) and tag each
    pair's `meta["annotator"]` so downstream evaluators can group or pick.
    Skips sentences where an annotator produced zero edits if at least one
    other annotator produced a non-empty correction, to avoid penalising
    the model for predicting a real edit that one annotator missed.
    """

    name = "conll2014"

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)

    def iter_pairs(self) -> Iterator[Pair]:
        path = self.raw_path / "official-2014.0.m2"
        if not path.exists():
            raise FileNotFoundError(
                f"expected {path} (download CoNLL-2014 to {self.raw_path})"
            )
        for src_toks, by_annotator in _parse_m2(path):
            if not by_annotator:
                # Sentence has no edits from any annotator: emit a single
                # identity pair so downstream stats reflect the true split
                # size.
                yield Pair(
                    src=" ".join(src_toks),
                    tgt=" ".join(src_toks),
                    source=self.name,
                    split=Split.TEST,
                    meta={"annotator": 0},
                )
                continue
            for annotator_id in sorted(by_annotator):
                edits = by_annotator[annotator_id]
                tgt_toks = _apply_edits(src_toks, edits)
                yield Pair(
                    src=" ".join(src_toks),
                    tgt=" ".join(tgt_toks),
                    source=self.name,
                    split=Split.TEST,
                    meta={"annotator": annotator_id},
                )
