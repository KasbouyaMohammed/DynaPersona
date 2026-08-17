# DynaPersona: Context-Gated Mixture of LoRA Experts for Multi-Style Dialogue Adaptation

Companion code, configuration, and reproduction scripts for the DynaPersona paper.

The method conditions low-rank adaptation on dialogue context. A gating network reads persona,
emotion, and conversational-style signals and routes each input across K LoRA experts whose combined
rank equals a single static LoRA, so the comparison is at a matched LoRA adapter-rank budget. The
total trainable parameters are **not** identical: the gate and persona projection add a small
overhead beyond the matched adapter budget (Static LoRA 8,716,288 trainable; DynaPersona 8,918,660
trainable). On multi-style dialogue (EmpatheticDialogues + PersonaChat) the mixture improves test
perplexity over a matched-rank static LoRA and over a routed-LoRA (SLIM-style) baseline at the same
adapter-rank budget.

## Headline results (three seeds, best-validation checkpoint)

| Model | Adapter params | Total trainable | Test perplexity | vs static |
|---|---|---|---|---|
| Static LoRA | 8,716,288 | 8,716,288 | 14.278 | — |
| SLIM routed LoRA | 8,716,288 | 8,918,273 | 14.145 | +0.93% (sd 0.26) |
| DynaPersona-MoE (K=4) | 8,716,288 | 8,918,660 | 13.941 | +2.36% (sd 0.12) |

Adapter-rank budget is matched (8,716,288 in all three); total trainable differs because of the gate
and persona-projection overhead (202,372 parameters for DynaPersona, about 2.3 percent of the adapter
size). Every per-seed paired bootstrap interval excludes zero (a within-seed quantity over test
examples; between-seed variation is summarized by the three-seed mean and standard deviation).
Backbone: Qwen2.5-1.5B-Instruct (frozen).

## Contents

- `dynapersona_full_run.py` — single-style pipeline (EmpatheticDialogues): data prep, static LoRA,
  and the scalar-gate control, with all measurement functions (perplexity, generation, latency,
  memory, bootstrap).
- `dynapersona_moe_run.py` — multi-style pipeline (EmpatheticDialogues + PersonaChat): matched
  static LoRA, the SLIM-style routed baseline, and DynaPersona-MoE. Imports `dynapersona_full_run.py`.
- `count_params.py` — module-wise trainable-parameter inventory from tensor shapes (no GPU needed).
- `regenerate_tables.py` — regenerates every results table from the per-seed JSON files.
- `make_figs.py` — regenerates the paper figures (Fig. 3 uses hatch patterns so the two
  corpus-defined styles are distinguishable in grayscale).
- `requirements.txt` — pinned package versions.
- `results/` — per-seed metrics (`results_seed*_moe.json`, `results_seed*.json`) and aggregates.
  These files store the computed bootstrap summaries (mean difference, 95 percent interval, resample
  fraction), not per-example negative log-likelihood arrays; the intervals are reproduced by
  re-running the evaluation code.
- `figures/` — the figures used in the paper.
- `checkpoints/` — adapter weights are not stored in the repository. The validation-selected
  DynaPersona-MoE checkpoints for seeds 42, 123, and 7 are available from the corresponding author on
  reasonable request (see `checkpoints/README.txt`).

## Environment

Python 3.12, one CUDA GPU. Install:

```
pip install -r requirements.txt
```

The runs were produced on a single NVIDIA RTX PRO 6000 (Blackwell) with bfloat16.

## Data

Downloaded automatically, no manual steps:
- EmpatheticDialogues from the original CSV distribution.
- PersonaChat (truecased) from the Hugging Face Hub (`bavard/personachat_truecased`).

The multi-style benchmark blends 40,000 training pairs from each source (80,000 total), balanced by
style, with fixed shuffling (seed 0), giving 80,000 training, 6,900 validation, and 6,901 test pairs.
Preprocessing is in `build_multistyle_data` in `dynapersona_moe_run.py`.

Split construction: EmpatheticDialogues uses its official train/validation/test partitions.
PersonaChat uses its official training split for training, and its official validation split is
divided by row index into a validation half (rows 0 to 3,899) and a test half (rows 3,900 to 7,800).
Because PersonaChat conversations are stored as contiguous blocks of turns, this index split keeps
the two halves conversation-disjoint except at the boundary: one conversation of 1,000 (conv_id 499)
crosses the midpoint and appears in both halves, affecting eight examples (one in the validation
half, seven in the test half). This is a known limitation of the benchmark construction.

Style and corpus are confounded in this benchmark: empathetic examples come from EmpatheticDialogues
and persona examples from PersonaChat, so the reported routing differences show source- and
style-dependent allocation and do not by themselves separate conversational-style specialization from
dataset-specific specialization.

## Reproduce

Set an output root (checkpoints, data, results are written under it):

```
export PROJECT_ROOT=/path/to/DynaPersona_FullRun
```

Multi-style comparison (static, SLIM, MoE; three seeds; latency and memory):

```
python dynapersona_moe_run.py
```

Single-style control (scalar gate ties static LoRA on EmpatheticDialogues):

```
python dynapersona_full_run.py
```

Configuration (rank, K, epochs, seeds, batch size, learning rate) is at the top of each script and
overridable by environment variable (`K_EXPERTS`, `LB_LAMBDA`, `RESULTS_TAG`). Training uses three
epochs, batch size 8, gradient accumulation 4, learning rate 5e-4, bfloat16, response-only loss
masking, and validation-selected checkpoints. Seeds: 42, 123, 7.

## Verify the numbers

```
python count_params.py                 # module-wise parameter inventory
python regenerate_tables.py results    # every results table from the raw per-seed JSON
```

`regenerate_tables.py` reads only the released JSON files, so any table in the paper can be checked
against the raw outputs.

## License

Released for review and reproduction. Base model and datasets retain their respective licenses.
