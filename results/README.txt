Released computational result files.

Multi-style files:
  results_seed42_moe.json
  results_seed123_moe.json
  results_seed7_moe.json

These contain the per-seed parameter inventories, training/validation summaries, perplexities, generation metrics, profiling measurements, routing summaries, and paired-bootstrap summaries used in the manuscript. Generated-text examples are omitted from this review package because they are not used to compute the reported tables.

Single-source control files:
  results_seed42.json
  results_seed123.json
  results_seed7.json

These contain the per-seed perplexities and paired-bootstrap summaries used for the scalar-gate control.

Aggregate files:
  aggregate_moe.json
  aggregate.json

Per-example negative log-likelihood arrays are not stored in this repository. The evaluation code computes them during evaluation and uses them to obtain the released paired-bootstrap summaries.
