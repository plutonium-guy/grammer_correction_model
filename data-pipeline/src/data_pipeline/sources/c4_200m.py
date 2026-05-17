from __future__ import annotations

import os
from collections.abc import Iterator

from datasets import load_dataset

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.base import Source
from data_pipeline.types import Pair, Split

# Original `liweili/c4_200m` is a script-based dataset and `datasets>=4`
# refuses to load it (`RuntimeError: Dataset scripts are no longer
# supported`). Operators must point this loader at a parquet mirror via
# C4_200M_DATASET_ID. Sensible mirrors at time of writing:
#   - "nbroad/c4_200m_gec_train_clean"     (filtered, parquet)
#   - "Yaxin/C4_200M_Subset"                (small subset, parquet)
# The mirror's column names may differ, so input/output column names are
# also operator-tunable.
_DEFAULT_DATASET_ID = "nbroad/c4_200m_gec_train_clean"
_DEFAULT_INPUT_COL = "input"
_DEFAULT_OUTPUT_COL = "output"


class C4200MSource(Source):
    """Streaming loader for a parquet-backed C4_200M mirror.

    Override `C4_200M_DATASET_ID`, `C4_200M_INPUT_COL`, and
    `C4_200M_OUTPUT_COL` to point at a different mirror with different
    column names. Streams to avoid downloading the full set; capped at
    `max_rows` for development and small-scale runs.
    """

    name = "c4_200m"

    def __init__(self, config: PipelineConfig, max_rows: int = 2_000_000) -> None:
        super().__init__(config)
        self.max_rows = max_rows
        self.dataset_id = os.environ.get("C4_200M_DATASET_ID", _DEFAULT_DATASET_ID)
        self.input_col = os.environ.get("C4_200M_INPUT_COL", _DEFAULT_INPUT_COL)
        self.output_col = os.environ.get("C4_200M_OUTPUT_COL", _DEFAULT_OUTPUT_COL)

    def iter_pairs(self) -> Iterator[Pair]:
        ds = load_dataset(self.dataset_id, split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= self.max_rows:
                break
            src = row.get(self.input_col)
            tgt = row.get(self.output_col)
            if not isinstance(src, str) or not isinstance(tgt, str):
                continue
            yield Pair(
                src=src,
                tgt=tgt,
                source=self.name,
                split=Split.TRAIN,
            )
            count += 1
