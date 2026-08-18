"""Regenerate the computational results tables from the released per-seed JSON files.

    python regenerate_tables.py path/to/results

Expects the two-source benchmark files results_seed{42,123,7}_moe.json produced by
`dynapersona_moe_run.py`, and optionally the single-source control files
results_seed{42,123,7}.json from `dynapersona_full_run.py`.
Values are read from the released result files.

Note: the historical JSON key `expert_usage_by_style` is retained for compatibility with
the original run output. It groups by the stored source labels (`empathetic` and `persona`);
those labels map to the same encoded style slot in the reported experiment, so the table is
reported as source-grouped routing rather than as a distinct style-vector effect.
"""
import sys, json, glob, os
from statistics import mean, pstdev


def load(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        if "smoke" in f:
            continue
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    return out


def col(rows, key):
    return [r[key] for r in rows if key in r]


def fmt_ci(b):
    lo, hi = b["ci95"]
    return f"diff {b['mean_diff']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  P(better)={b['p_dyna_better']:.2f}"


def main(res_dir):
    ms = load(os.path.join(res_dir, "results_seed*_moe.json"))
    if not ms:
        print("No two-source benchmark results found in", res_dir); return
    seeds = [r["seed"] for r in ms]

    print("=" * 78)
    print("TABLE 1  Trainable-parameter inventory (two-source benchmark, seed", seeds[0], ")")
    pis = ms[0].get("param_inventory_static", {})
    pim = ms[0].get("param_inventory_moe", {})
    print("  static LoRA total :", f"{pis.get('total_trainable', 0):,}", pis.get("by_module"))
    print("  MoE total         :", f"{pim.get('total_trainable', 0):,}", pim.get("by_module"))

    print("=" * 78)
    print("TABLE 2  Test perplexity (best-validation checkpoint)")
    print(f"  {'model':<16}" + "".join(f"seed{s:<8}" for s in seeds) + "mean")
    for key, name in (("ppl_static", "Static LoRA"), ("ppl_slim", "SLIM routed"),
                      ("ppl_moe", "DynaPersona-MoE")):
        vals = col(ms, key)
        if vals:
            print(f"  {name:<16}" + "".join(f"{v:<12.3f}" for v in vals) + f"{mean(vals):.3f}")
    mvs = col(ms, "improvement_ppl_pct")
    svs = col(ms, "improvement_slim_vs_static_pct")
    if mvs:
        print(f"  MoE vs static : {mean(mvs):+.2f}%  (sd {pstdev(mvs):.2f})")
    if svs:
        print(f"  SLIM vs static: {mean(svs):+.2f}%  (sd {pstdev(svs):.2f})")

    print("-" * 78)
    print("  Paired bootstrap (per seed):")
    for r in ms:
        print(f"   seed {r['seed']}:")
        for bk, label in (("bootstrap", "MoE vs static"),
                          ("bootstrap_slim_vs_static", "SLIM vs static"),
                          ("bootstrap_moe_vs_slim", "MoE vs SLIM")):
            if bk in r:
                print(f"     {label:<15} {fmt_ci(r[bk])}")

    print("=" * 78)
    print("TABLE 3  Generation metrics (mean over seeds)")
    metrics = ["bleu", "rougeL", "bertscore_f1", "distinct1", "distinct2"]
    print(f"  {'model':<16}" + "".join(f"{m:<14}" for m in metrics))
    for gkey, name in (("gen_static", "Static LoRA"), ("gen_slim", "SLIM routed"),
                       ("gen_moe", "DynaPersona-MoE")):
        gens = col(ms, gkey)
        if gens:
            row = [mean([g[m] for g in gens if m in g]) for m in metrics]
            print(f"  {name:<16}" + "".join(f"{v:<14.4f}" for v in row))

    print("=" * 78)
    print("TABLE 4  Efficiency (mean over seeds; 64 new tokens, 40-token prompt)")
    fields = ["latency_ms_mean", "prefill_ms", "decode_only_throughput_tok_s", "weight_gb", "peak_alloc_gb"]
    print(f"  {'model':<16}" + "".join(f"{x:<24}" for x in ["latency_ms", "prefill_ms", "decode_tok/s", "weight_gb", "peak_gb"]))
    for pkey, name in (("profile_static", "Static LoRA"), ("profile_slim", "SLIM routed"),
                       ("profile_moe", "DynaPersona-MoE")):
        profs = col(ms, pkey)
        if profs:
            row = [mean([p[f] for p in profs if f in p]) for f in fields]
            print(f"  {name:<16}" + "".join(f"{v:<24.2f}" for v in row))

    print("=" * 78)
    print("TABLE 5  Expert usage by source group (per seed)")
    for r in ms:
        eu = r.get("expert_usage_by_style", {})
        print(f"  seed {r['seed']}:", {k: [round(x, 3) for x in v] for k, v in eu.items()})

    ss = load(os.path.join(res_dir, "results_seed*[0-9].json"))
    if ss:
        print("=" * 78)
        print("TABLE 6  Single-source control (EmpatheticDialogues; scalar gate)")
        st = col(ss, "ppl_static"); dy = col(ss, "ppl_dynapersona")
        if st and dy:
            print(f"  static mean {mean(st):.3f}   scalar-gate mean {mean(dy):.3f}   "
                  f"change {mean(col(ss, 'improvement_ppl_pct')):+.2f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
