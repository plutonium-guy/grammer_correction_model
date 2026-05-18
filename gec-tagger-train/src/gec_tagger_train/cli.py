from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import typer
from transformers import AutoTokenizer

from gec_tagger_train.config import StageConfig
from gec_tagger_train.dataset import GECTaggerDataset
from gec_tagger_train.tag_vocab import TagVocab, build_tag_vocab

app = typer.Typer(no_args_is_help=True)


@app.command("build-vocab")
def build_vocab(
    jsonl: Path = typer.Option(..., exists=True, help="gec_tagger.jsonl input"),
    out: Path = typer.Option(..., help="Output tags.json path"),
    min_count: int = typer.Option(5, help="Minimum frequency to retain a tag"),
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    vocab = build_tag_vocab(jsonl=jsonl, min_count=min_count, out=out)
    typer.echo(f"wrote {len(vocab)} tags to {out}")


@app.command("train")
def train(
    stage: int = typer.Option(..., min=1, max=3),
    jsonl: Path = typer.Option(..., exists=True),
    tags: Path = typer.Option(..., exists=True),
    out: Path = typer.Option(...),
    max_steps: int = typer.Option(-1),
    num_epochs: int = typer.Option(1),
    per_device_batch_size: int = typer.Option(16),
    learning_rate: float = typer.Option(1e-5),
    max_length: int = typer.Option(128),
) -> None:
    from gec_tagger_train.trainer import run_training

    out.mkdir(parents=True, exist_ok=True)
    vocab = TagVocab.load(tags)
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
    ds = GECTaggerDataset(jsonl=jsonl, tokenizer=tok, vocab=vocab, max_length=max_length)
    cfg = StageConfig(
        name=f"stage{stage}",
        train_jsonl=jsonl,
        output_dir=out,
        max_length=max_length,
        per_device_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        max_steps=max_steps,
        warmup_ratio=0.1 if max_steps < 0 else 0.0,
    )
    metrics = run_training(cfg=cfg, dataset=ds, vocab=vocab, tokenizer=tok)
    typer.echo(f"trained stage {stage}: {metrics}")


@app.command("export")
def export(
    checkpoint: Path = typer.Option(..., exists=True),
    tags: Path = typer.Option(..., exists=True, help="tags.json from build-vocab"),
    out: Path = typer.Option(
        ...,
        help="Output ONNX file. tags.json + tokenizer co-located in the same dir.",
    ),
    max_length: int = typer.Option(128),
    opset: int = typer.Option(17),
) -> None:
    """Export checkpoint to ONNX and co-locate `tags.json` + tokenizer files in the same dir."""
    import shutil

    from gec_tagger_train.export_onnx import export_tagger_to_onnx
    from gec_tagger_train.model import DebertaTagger

    state = _load_state_dict(checkpoint)
    num_tags = int(state["classifier.weight"].shape[0])
    model = DebertaTagger(num_tags=num_tags)
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith("encoder.")]
    if missing:
        raise RuntimeError(
            f"checkpoint missing non-encoder keys: {missing}; cannot export"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    export_tagger_to_onnx(model=model, out_path=out, max_length=max_length, opset=opset)

    # Co-locate tags.json so gec-engine can resolve logit ids back to tag strings.
    shutil.copy(tags, out.parent / "tags.json")

    # Co-locate the DeBERTa tokenizer for the Rust runtime.
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)
    tok.save_pretrained(str(out.parent))

    typer.echo(f"exported ONNX to {out}; tags.json + tokenizer in {out.parent}")


def _iter_dev_pairs(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (src, ref) tuples from a JSONL file with {src, ref} records."""
    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield rec["src"], rec["ref"]


@app.command("eval")
def eval_cmd(
    checkpoint: Path = typer.Option(..., exists=True),
    tags: Path = typer.Option(..., exists=True),
    dev: Path = typer.Option(..., exists=True, help="JSONL with {src, ref} per row"),
    max_length: int = typer.Option(128),
    max_iter: int = typer.Option(
        3,
        min=1,
        max=10,
        help="Iterative decoding passes. GECToR convention is 3.",
    ),
) -> None:
    """Run a tagger checkpoint on a dev JSONL and print ERRANT F0.5."""
    from gec_tagger_train.decode import apply_tags_iterative
    from gec_tagger_train.eval_errant import compute_errant_f05
    from gec_tagger_train.model import DebertaTagger

    vocab = TagVocab.load(tags)
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)

    state = _load_state_dict(checkpoint)
    num_tags = int(state["classifier.weight"].shape[0])
    model = DebertaTagger(num_tags=num_tags)
    model.load_state_dict(state, strict=False)
    model.eval()

    def tag_words(words: list[str]) -> list[str]:
        if not words:
            return []
        enc = tok(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
            )["logits"]
        preds = logits.argmax(dim=-1)[0].tolist()
        word_ids = enc.word_ids(0)
        # First-subword strategy.
        seen: set[int] = set()
        word_tag_ids: list[int] = [vocab.id_of("$KEEP")] * len(words)
        for pos, w in enumerate(word_ids):
            if w is None or w in seen:
                continue
            seen.add(w)
            if w < len(word_tag_ids):
                word_tag_ids[w] = preds[pos]
        return [
            vocab.tags[i] if 0 <= i < len(vocab.tags) else "$KEEP"
            for i in word_tag_ids
        ]

    srcs: list[str] = []
    refs: list[str] = []
    hyps: list[str] = []
    for src, ref in _iter_dev_pairs(dev):
        words = src.split()
        if not words:
            continue
        hyp_words = apply_tags_iterative(words, tag_words, max_iter=max_iter)
        srcs.append(src)
        refs.append(ref)
        hyps.append(" ".join(hyp_words))

    score = compute_errant_f05(src=srcs, ref=refs, hyp=hyps)
    typer.echo(json.dumps(score))


def _load_state_dict(checkpoint: Path) -> dict[str, Any]:
    bin_path = checkpoint / "pytorch_model.bin"
    safe_path = checkpoint / "model.safetensors"
    if safe_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safe_path))
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu", weights_only=True)  # type: ignore[no-any-return]
    raise FileNotFoundError(f"no model weights found under {checkpoint}")


if __name__ == "__main__":
    app()
