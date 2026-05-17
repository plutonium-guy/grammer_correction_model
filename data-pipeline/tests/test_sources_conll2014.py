import shutil
from pathlib import Path

from data_pipeline.config import PipelineConfig
from data_pipeline.sources.conll2014 import CoNLL2014Source
from data_pipeline.types import Split


def test_loads_conll_m2(tmp_path: Path, fixtures_dir: Path) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    src_dir = cfg.raw_dir / "conll2014"
    src_dir.mkdir()
    shutil.copy(fixtures_dir / "tiny.m2", src_dir / "official-2014.0.m2")

    src = CoNLL2014Source(config=cfg)
    pairs = list(src.iter_pairs())
    assert len(pairs) == 3
    assert all(p.split == Split.TEST for p in pairs)
    assert all(p.source == "conll2014" for p in pairs)


def test_multi_annotator_yields_one_pair_per_annotator(
    tmp_path: Path, fixtures_dir: Path
) -> None:
    cfg = PipelineConfig(root=tmp_path).ensure_dirs()
    src_dir = cfg.raw_dir / "conll2014"
    src_dir.mkdir()
    shutil.copy(fixtures_dir / "multi_annotator.m2", src_dir / "official-2014.0.m2")

    src = CoNLL2014Source(config=cfg)
    pairs = list(src.iter_pairs())
    # Sentence 1 has two annotators -> 2 pairs.
    # Sentence 2 has one annotator -> 1 pair.
    # Sentence 3 has no edits -> 1 identity pair.
    assert len(pairs) == 4

    # First sentence yields one pair per annotator with distinct targets.
    s1 = [p for p in pairs if p.src == "He go to school ."]
    assert len(s1) == 2
    targets = sorted(p.tgt for p in s1)
    assert targets == ["He goes to school .", "He went to school ."]
    annotators = sorted(p.meta["annotator"] for p in s1)
    assert annotators == [0, 1]

    # Identity pair for the no-edit sentence.
    s3 = [p for p in pairs if p.src == "Hello world ."]
    assert len(s3) == 1
    assert s3[0].tgt == "Hello world ."
