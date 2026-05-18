#!/usr/bin/env python
"""Evaluate an mlx-lm LoRA adapter on a {src, ref} JSONL dev set.

Generates a corrected sentence per row via the merged base+adapter using
mlx-lm in-process (one model load), then scores precision/recall/F0.5
against the references using ERRANT.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mlx_lm import generate, load

from gec_tagger_train.eval_errant import compute_errant_f05

SYSTEM_PROMPT = "Correct the grammar of the user text. Preserve meaning."


def build_prompt(tokenizer, text: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--dev", required=True, type=Path)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    model, tokenizer = load(args.model, adapter_path=str(args.adapter))

    srcs: list[str] = []
    refs: list[str] = []
    hyps: list[str] = []

    started = time.time()
    n = 0
    with args.dev.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            rec = json.loads(line)
            src = rec["src"]
            ref = rec["ref"]
            prompt = build_prompt(tokenizer, src)
            hyp = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens)
            srcs.append(src)
            refs.append(ref)
            hyps.append(hyp.strip())
            n += 1
            if args.limit and n >= args.limit:
                break

    elapsed = time.time() - started
    score = compute_errant_f05(src=srcs, ref=refs, hyp=hyps)
    out = {**score, "n": n, "elapsed_s": round(elapsed, 1)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
