# DynaPersona: Context-Gated Mixture of LoRA Experts for Multi-Source Dialogue Adaptation

Companion code, configuration, and reproduction scripts for the DynaPersona paper.

DynaPersona conditions low-rank adaptation on dialogue context and routes each input across K LoRA experts whose combined rank equals a single static LoRA. The comparison is therefore made at a matched LoRA adapter-rank budget. Total trainable parameters are **not** identical because the gate and persona projection add a small overhead (Static LoRA 8,716,288 trainable; DynaPersona 8,918,660 trainable).

The evaluation combines EmpatheticDialogues and PersonaChat. The implemented gate receives persona and emotion encodings plus a three-dimensional style slot. In the reported run, PersonaChat examples carry the source label `persona`, while the style encoder vocabulary is `empathetic`, `informative`, and `casual`; unknown labels use the first slot. Consequently, both benchmark sources use the same style one-hot slot in this experiment. Routing differences between the two source groups must therefore be interpreted as **source/context-dependent allocation driven by the remaining context signals**, not as evidence that the style one-hot itself separates the two corpora.

## Headline results (three seeds, best-validation checkpoint)

| Model | Adapter params | Total trainable | Test perplexity | vs static |
|---|---:|---:|---:|---:|
| Static LoRA | 8,716,288 | 8,716,288 | 14.278 | — |
| SLIM routed LoRA | 8,716,288 | 8,918,273 | 14.145 | +0.93% (sd 0.26) |
| DynaPersona-MoE (K=4) | 8,716,288 | 8,918,660 | 13.941 | +2.36% (sd 0.12) |

Adapter-rank budget is matched (8,716,288 in all three); total trainable differs because of routing/persona-projection overhead. Every per-seed paired-bootstrap interval for DynaPersona versus static LoRA excludes zero; this is a within-seed quantity over test examples. Between-seed variation is summarized separately by the three-seed mean and standard deviation.

Backbone: Qwen2.5-1.5B-Instruct (frozen).

## Contents

- `dynapersona_full_run.py` — EmpatheticDialogues single-source pipeline and scalar-gate control.
- `dynapersona_moe_run.py` — two-source pipeline (EmpatheticDialogues + PersonaChat), matched static LoRA, routed-LoRA baseline, and DynaPersona-MoE.
- `count_params.py` — module-wise trainable-parameter inventory from tensor shapes.
- `regenerate_tables.py` — regenerates the computational result tables from released JSON metrics.
- `make_figs.py` — regenerates the figures as PNG and SVG; Fig. 3 uses hatch patterns for grayscale accessibility.
- `requirements.txt` — package versions used by the experiment scripts.
- `results/` — per-seed metrics and aggregate summaries.
- `figures/` — review-ready vector figures; PNG versions can be regenerated with `make_figs.py`.
- `checkpoints/` — checkpoint-availability note. Validation-selected DynaPersona-MoE checkpoints for seeds 42, 123, and 7 are available from the corresponding author on reasonable request.

## Environment

Python 3.12, one CUDA GPU.

```bash
pip install -r requirements.txt
```

The reported runs used a single NVIDIA RTX PRO 6000 (Blackwell) with bfloat16.

## Data

The scripts obtain:
- EmpatheticDialogues from its original CSV distribution.
- PersonaChat (truecased) from the Hugging Face Hub (`bavard/personachat_truecased`).

The two-source benchmark contains 40,000 training pairs from each corpus (80,000 total), shuffled with seed 0, with 6,900 validation pairs and 6,901 test pairs. Preprocessing is implemented in `build_multistyle_data` in `dynapersona_moe_run.py`.

EmpatheticDialogues uses its official train/validation/test partitions. PersonaChat uses its official training split for training; its official validation split is divided by row index into validation (rows 0–3,899) and test (rows 3,900–7,800). Because PersonaChat conversations are stored as contiguous blocks of turns, one boundary conversation of 1,000 (`conv_id 499`) crosses the midpoint, affecting eight examples (one validation, seven test). This is a benchmark-construction limitation.

The two source groups differ simultaneously in corpus, persona conditioning, and emotion conditioning. The evaluation therefore does not isolate which individual context signal causes the observed routing allocation.

## Reproduce

Set an output root:

```bash
export PROJECT_ROOT=/path/to/DynaPersona_FullRun
```

Two-source comparison (static, routed baseline, MoE; three seeds):

```bash
python dynapersona_moe_run.py
```

Single-source scalar-gate control:

```bash
python dynapersona_full_run.py
```

Training uses three epochs, batch size 8, gradient accumulation 4, learning rate 5e-4, bfloat16, response-only loss masking, and validation-selected checkpoints. Seeds are 42, 123, and 7. Configuration is defined at the top of the scripts; `K_EXPERTS`, `LB_LAMBDA`, and `RESULTS_TAG` are available as environment overrides for the MoE pipeline.

## Released results

The multi-source per-seed JSON files contain the parameter inventories, training/validation summaries, perplexities, generation metrics, profiling measurements, routing summaries, and paired-bootstrap summaries used by the manuscript. Generated-text examples are not included in the review package because they are not used to compute the reported tables.

The single-source JSON files contain the perplexity and paired-bootstrap summaries used for the scalar-gate control. Per-example negative log-likelihood arrays are not stored in this repository; the evaluation code computes them when evaluation is rerun.

## Verify the numbers

```bash
python count_params.py
python regenerate_tables.py results
```

The table-regeneration script reads the released JSON metrics. The aggregate files provide the cross-seed perplexity and improvement summaries.

## License

Released for review and reproduction. Base model and datasets retain their respective licenses.
