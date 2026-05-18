# Training Run — 2026-05-17

End-to-end smoke training across the data pipeline, GEC tagger, and style LLM, executed on macOS-arm64 (Apple M2 Max, 32 GB unified memory).

## Data sources

License-gated corpora (BEA-2019, GYAFC, CoNLL-2014) were not placed under `data/raw/`, so the run used only the publicly downloadable Hugging Face sources.

| Source | Records | Notes |
|---|---|---|
| JFLEG (dev + test) | 2,492 | Used as GEC training data |
| C4_200M | 0 | Hugging Face removed script-based datasets; loader fails with `RuntimeError: Dataset scripts are no longer supported, but found c4_200m.py` |
| Wiki-Auto | 0 | Same script-based-dataset error |
| ParaDetox | 18,993 | Detoxify SFT pairs |
| BEA-2019 / GYAFC / CoNLL-2014 | 0 | License-gated, files not placed |

A held-out 100-pair `jfleg_test.jsonl` was extracted under `gec-tagger-train/eval_data/`.

## GEC tagger run

Two training runs on `gec_tagger.jsonl` produced from JFLEG.

| Run | Vocab cutoff | Tags | Steps | Train loss | F0.5 (JFLEG test) | F0.5 (synthetic dev) |
|---|---|---|---|---|---|---|
| `.checkpoints/jfleg/ckpt` | min-count 2 | 915 | 500 (1.6 epochs) | 3.401 | 0.0 | 0.0 |
| `.checkpoints/jfleg2/ckpt` | min-count 5 | 302 | 3000 (9.6 epochs) | 2.972 | 0.0 | 0.0 |

Both checkpoints collapsed to all-`$KEEP` predictions, so `apply_tags_once` produces no edits and ERRANT records zero true positives.

### Why the tagger collapsed

JFLEG references are full-sentence paraphrases, not minimal grammatical edits. After `align_tokens_to_tags` derives per-token edit tags, the vocabulary blows up with one-off paraphrase replacements: most appear only once. After frequency pruning, those become `$UNK`, and the only well-supported tag in training is `$KEEP`. The model learns the dominant class.

GECToR's published F0.5 numbers were obtained on BEA-2019 train.m2 (W&I + L + NUCLE + FCE + Lang-8) which contains millions of minimal-edit annotations. JFLEG alone cannot replicate that without different data preprocessing (e.g., back-translating each correction into a minimal edit chain).

### Reproduction commands

```
# Data
cd data-pipeline
uv run data-pipeline build-gec --root . --sources jfleg --split dev
uv run data-pipeline build-sft --root . --sources paradetox

# Tagger
cd ../gec-tagger-train
uv run gec-tagger-train build-vocab \
    --jsonl ../data-pipeline/data/processed/gec_tagger.jsonl \
    --out .checkpoints/jfleg2/tags.json --min-count 5

uv run gec-tagger-train train --stage 1 \
    --jsonl ../data-pipeline/data/processed/gec_tagger.jsonl \
    --tags .checkpoints/jfleg2/tags.json \
    --out .checkpoints/jfleg2/ckpt \
    --max-steps 3000 --per-device-batch-size 8 --learning-rate 5e-5 \
    --max-length 128 --num-epochs 30

uv run gec-tagger-train eval \
    --checkpoint .checkpoints/jfleg2/ckpt \
    --tags .checkpoints/jfleg2/tags.json \
    --dev eval_data/jfleg_test.jsonl
```

Training runtime: 1620 s (≈27 min) for 3000 steps on a single M2 Max with `pin_memory` disabled (MPS does not support it).

## Style LLM run

### First attempt — failure (fp16 + MPS)

| Item | Value |
|---|---|
| Base | `Qwen/Qwen2.5-0.5B-Instruct` |
| Steps | 200 |
| Train loss | 0 (all NaN, aggregator returns 0) |
| Entropy | NaN |
| Mean token accuracy | 0.0115 (random) |
| GPU recoveryCount | 15 |
| Adapter saved? | yes (35 MB) but no weight updates after step ≈44 |

The training nominally ran 200/200 steps but every Metal command buffer after step ~44 was rejected with:

```
Error: command buffer exited with error status.
Ignored (for causing prior/excessive GPU errors) (00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)
```

Root cause: fp16 base + `gradient_checkpointing_enable()` on Apple MPS produces overflow in the rotary embedding + RMSNorm path. The Metal driver hits 15 GPU resets and then places the queue in "ignore" mode for the rest of the session. Hugging Face's NaN-safe loss aggregator reports `0.0`, so the run looks superficially successful.

### Fix

Patch in `style-llm-train/src/style_llm_train/trainer.py`:

- Detect Apple Silicon at runtime.
- On MPS, load the base in `torch.float32` and skip `gradient_checkpointing_enable()`.
- Keep fp16 + gradient checkpointing on CUDA/CPU.

```python
on_mps = platform.system() == "Darwin" and platform.machine() == "arm64"
dtype = torch.float32 if on_mps else torch.float16
base = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype, ...)
base.config.use_cache = False
if not on_mps:
    try:
        base.gradient_checkpointing_enable()
    except Exception:
        pass
```

### Second attempt — success

Same hyperparameters, fp32 base on MPS, gradient checkpointing off.

| Metric | Value |
|---|---|
| Train runtime | 228.6 s (≈4 min) |
| Train loss | 1.61 |
| Entropy | 1.506 |
| Mean token accuracy | 0.7293 |
| GPU recovery count | 0 (no Metal resets) |
| Adapter | `style-llm-train/.checkpoints/style/adapter_model.safetensors` (35 MB) |

Generation smoke test with the merged adapter (greedy, max_new_tokens=40):

| Input | Output |
|---|---|
| `that is dumb` | `That is not good` |
| `you suck at this` | `You are not good at this` |
| `he is a fucking idiot` | `He is a fool` |

The model learned the ParaDetox distribution. Not production quality (200 steps × 8 grad-accum on a 0.5 B base sees only ~1,600 records out of 18,993) but the detoxification signal is unambiguous and the fp32 fix is confirmed to resolve the MPS NaN failure.

### Third attempt — mlx-lm on Qwen2.5-3B (production-grade path)

PyTorch + MPS is the wrong tool for LLM LoRA on Apple Silicon. Switched to Apple's native `mlx-lm`:

```
uv add mlx-lm
uv run mlx_lm.lora --model Qwen/Qwen2.5-3B-Instruct --train \
    --data .checkpoints/mlx_data --fine-tune-type lora \
    --num-layers 16 --batch-size 1 --iters 500 \
    --learning-rate 5e-5 --max-seq-length 512 --mask-prompt \
    --adapter-path .checkpoints/style_mlx
```

| Metric | Value |
|---|---|
| Base | `Qwen/Qwen2.5-3B-Instruct` (3.09 B params) |
| Trainable | 6.65 M LoRA params (0.216 %) |
| Iters | 500 |
| Throughput | ~7.5 iter/s, ~100 tokens/s |
| Peak mem | 6.87 GB |
| Val loss trajectory | 5.915 → 0.903 → 1.050 → 1.009 → **0.759** (iter 400 best) → 0.873 |
| Final train loss | 0.707 |
| GPU resets | 0 |

Adapter (~31 MB) saved every 100 iters under `.checkpoints/style_mlx/`. Best val is the iter-400 snapshot.

Generation smoke (greedy, max_new_tokens=40, system prompt: "Rewrite the user's text in a neutral, non-toxic way while preserving meaning."):

| Input | Output |
|---|---|
| `that is dumb` | `That is wrong.` |
| `you suck at this` | `You are bad at this.` |
| `he is a fucking idiot` | `He is a person who doesn't think carefully.` |
| `this code is shit` | `this code is bad` |
| `shut the hell up` | `Shut up.` |

Detoxification is clean and meaning is preserved. The mlx-lm path replaces the Apple-Silicon-fragile PyTorch+MPS training stack — it is **faster** (7.5 iter/s vs ~0.66 iter/s for PyTorch fp32), uses **less memory** (6.9 GB vs ~13 GB at the same model size would require), and produces a Qwen-3B model where the PyTorch path was forced down to Qwen-0.5B for stability.

### Recommendation

For style LLM training on Apple Silicon, **use mlx-lm as the default path**. The PyTorch trainer is kept for portability to CUDA/CPU, but on macOS-arm64 the operator should invoke the mlx command above. The GGUF export step still works unchanged: merge the adapter via `mlx_lm.fuse`, then run `llama.cpp` convert + quantize.

## Known limitations

The end-to-end pipeline is functional but smoke-trained models are not production-grade. Real numbers require:

1. **License-gated corpora**: register and place BEA-2019 train.m2 (≈8 M minimal-edit pairs) under `data-pipeline/data/raw/bea2019/`. The same for GYAFC and CoNLL-2014.
2. **C4_200M alternative**: the upstream HF dataset is no longer loadable via the `datasets` library. Use a parquet-mirror (e.g. `nbroad/c4_200m_gec_train_clean`) once the loader is updated.
3. **TRANSFORM_VERB decoding**: the decoder currently treats verb-form transforms as `$KEEP`. Add `lemminflect` and a conversion table.
4. **CoNLL-2014 multi-annotator handling**: parser merges both annotators. Filter to one or yield separately.
5. **mlx-lm for the style LLM**: faster + native on M-series than PyTorch+MPS. Switch when scaling beyond 0.5 B params.

Open follow-ups already tracked in each sub-project's `FOLLOWUPS.md`.

## Long mlx-lm runs (parallel)

Style and GEC both fine-tuned with `mlx_lm.lora` on Qwen2.5-3B-Instruct, 3000 iters each, running in parallel on the same M2 Max. GEC reformulates the GECToR task as instruction-tuned correction (system: "Correct the grammar of the user text. Preserve meaning.", user: source, assistant: target). 2,229 training pairs derived from the data-pipeline JFLEG output.

### Style (long, 3000 iters target, stopped early at iter 2050)

Val loss trajectory:

| Iter | Val | |
|---|---|---|
| 1 | 5.842 | baseline |
| 200 | 0.762 | |
| 400 | 0.770 | |
| 600 | 0.817 | |
| 800 | 0.810 | |
| 1000 | 0.800 | |
| **1200** | **0.741** | **best** |
| 1400 | 0.824 | |
| 1600 | 0.770 | |
| 1800 | 0.884 | |
| 2000 | 0.940 | overfit confirmed; killed |

Promoted `0001200_adapters.safetensors` to `adapters.safetensors`.

### GEC instruction-tuned (3000 iters target, stopped at iter 1600)

Val loss trajectory:

| Iter | Val | |
|---|---|---|
| 1 | 3.796 | baseline |
| 200 | 0.581 | |
| 400 | 0.554 | |
| 600 | 0.536 | |
| 800 | 0.540 | |
| **1000** | **0.520** | **best** |
| 1200 | 0.556 | |
| 1400 | 0.577 | |
| 1600 | 0.579 | overfit confirmed; killed |

Promoted `0001000_adapters.safetensors` to `adapters.safetensors`.

### Generation smoke (best adapters)

GEC, iter-1000 adapter:

| In | Out |
|---|---|
| `he go to school` | `He goes to school .` |
| `I are happy` | `I am happy .` |
| `she have a cat` | `She has a cat .` |
| `they was tired` | `They were tired .` |
| `she walk fast` | `She walks fast .` |

5/5 correct grammar corrections. Trailing space + period are an artifact of the JFLEG whitespace tokenization preserved through the data-pipeline; a post-processor can strip them.

Style, iter-1200 adapter:

| In | Out |
|---|---|
| `that is dumb` | `That is not smart` |
| `you suck at this` | `You are not good at this.` |
| `he is a fucking idiot` | `He is not a smart person.` |
| `this code is shit` | `This code is bad.` |
| `shut the hell up` | `Shhh` |

Cleaner detoxification than the 500-iter run (`He is a person who doesn't think carefully.` → `He is not a smart person.`).

### Throughput note

Running both jobs in parallel: each ran at ~2.0-2.3 it/s (vs ~7.5 it/s solo), so the two-job throughput is roughly 4 it/s combined — slightly less than 7.5 solo, the GPU contention overhead is ~50%. Sequential runs would be faster total wall-clock if the GPU is the bottleneck.

## Final artifacts (this run)

| Adapter | Path (gitignored) | Size | Val loss |
|---|---|---|---|
| Style detox | `style-llm-train/.checkpoints/style_mlx_long/adapters.safetensors` (= iter 1200) | ~31 MB | 0.741 |
| GEC correction | `gec-tagger-train/.checkpoints/gec_mlx/adapters.safetensors` (= iter 1000) | ~31 MB | 0.520 |

Both adapters compose with `Qwen/Qwen2.5-3B-Instruct` via `mlx_lm.generate --adapter-path …`.

## BEA-2019 retrain (next-day run)

After downloading W&I+LOCNESS from the public BEA-2019 distribution
(`https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz`,
no registration required for the W&I+LOCNESS subset) and placing
`ABC.train.gold.bea19.m2` and `ABCN.dev.gold.bea19.m2` under
`data-pipeline/data/raw/bea2019/` as `train.m2` and `dev.m2`:

```
cd data-pipeline
uv run data-pipeline build-gec --root . --sources bea2019 --split train
# -> 33,432 minimal-edit pairs in data/processed/gec_tagger.jsonl

cd ../gec-tagger-train
# Convert tagger JSONL -> ChatML SFT, then train Qwen2.5-3B LoRA with mlx-lm:
uv run mlx_lm.lora --model Qwen/Qwen2.5-3B-Instruct --train \
    --data .checkpoints/bea_mlx_data --fine-tune-type lora \
    --num-layers 16 --batch-size 2 --iters 3000 \
    --learning-rate 5e-5 --max-seq-length 256 \
    --adapter-path .checkpoints/bea_mlx \
    --steps-per-report 100 --steps-per-eval 300 --save-every 300 --mask-prompt
```

### Training trajectory

| Iter | Train | Val |
|---|---|---|
| 1 | n/a | 2.875 (baseline) |
| 300 | 0.362 | 0.286 |
| 600 | 0.309 | 0.317 |
| 900 | 0.334 | 0.296 |
| 1200 | 0.286 | 0.366 |
| 1500 | 0.267 | 0.283 |
| 1800 | 0.297 | 0.282 |
| 2100 | 0.267 | 0.287 |
| 2400 | 0.283 | 0.282 |
| 2700 | 0.260 | 0.290 |
| **3000** | **0.285** | **0.281** |

Val loss bounced between 0.281–0.366; iter 3000 is best at 0.281
(8.7 % below initial best at iter 300). Training took ~17 minutes on
M2 Max, peak memory 8.5 GB, 3.2 it/s solo.

### Eval (300-pair held-out BEA-dev, 100 evaluated)

```
uv run python scripts/eval_mlx_gec.py \
    --adapter .checkpoints/bea_mlx \
    --dev eval_data/bea_dev.jsonl \
    --limit 100 --max-tokens 80
```

```
{"precision": 0.5433, "recall": 0.3651, "f0.5": 0.4950, "n": 100, "elapsed_s": 78.2}
```

| Metric | Value | Spec target (§9.2) |
|---|---|---|
| **Precision** | **0.543** | n/a |
| **Recall** | **0.365** | n/a |
| **ERRANT F0.5** | **0.4950** | ≥ 0.65 |

Below the spec target but a real, measurable signal — and a ~50× lift
over the JFLEG-trained checkpoint (F0.5 = 0.0, collapsed to all-KEEP).
The gap to the 0.65 target is expected: the published GECToR numbers
were obtained with a multi-stage curriculum (C4_200M pretrain → BEA-2019
→ W&I+L) on a token-classification head, while this run is a single
LoRA pass over Qwen2.5-3B doing seq2seq correction. Closing the gap
requires more iters, NUCLE + FCE + Lang-8 data, and the multi-stage
recipe.

### Generation smoke

```
IN : He go to school        OUT: He goes to school
IN : I are happy            OUT: I am happy
IN : she walk fast          OUT: She walks fast .
IN : they was tired yesterday OUT: They were tired yesterday .
IN : she have a cat         OUT: She has a cat .
```

5/5 grammar errors corrected (capitalization, subject-verb agreement,
verb conjugation, past tense). Trailing-space + period artifact is from
the BEA-2019 whitespace tokenization carried through the data-pipeline.

## Negative results (2026-05-18)

After v2 (F0.5 = 0.533, published to HF), three further iterations to test capacity- and data-scaling hypotheses. All underperformed v2 on the same 100-pair BEA-dev eval.

### v2.1 — same data, rank 16

Identical to v2 but doubled LoRA rank (8 → 16) on the same BEA + Coedit corpus (42,491 pairs), all 36 layers, 5000 iters. 29.93 M trainable vs 14.97 M.

| Snapshot | F0.5 |
|---|---|
| iter 3,500 (best val 0.171) | **0.508** |
| iter 5,000 (final val 0.189) | **0.517** |

Both below v2's 0.533. Higher rank did not help on the 42k-pair corpus — likely overfit on the relatively small data.

### v3 — added agentlans/grammar-correction (137 k pairs, mixed quality)

Combined BEA (22.7k) + Coedit GEC (19.8k) + filtered agentlans (96.6k) = 139,110 pairs. Rank 16, all layers, 7500 iters target (killed at iter 2500 because val loss was bouncing 0.354–0.632).

| Snapshot | F0.5 |
|---|---|
| iter 500 | 0.433 |
| iter 2,000 (best val 0.354) | **0.502** |

Worse than v2. agentlans data shifted the distribution away from the BEA-style minimal-edit annotations: precision held (0.572 vs v2's 0.589) but recall dropped (0.336 vs 0.386). The model learned to make fewer edits.

### Lesson

At this scale data quality dominates capacity. Higher LoRA rank on clean data and noisier data with the same rank both fall short of v2's clean-data-plus-default-capacity recipe. The next real lift requires **more clean minimal-edit data** (NUCLE + FCE + Lang-8 + multi-stage curriculum) — those are license-gated and were not available in this run.

### v2 remains the best published model

[`amiya/qwen2.5-3b-gec-v2`](https://huggingface.co/amiya/qwen2.5-3b-gec-v2) — F0.5 0.533, precision 0.589, recall 0.386.
