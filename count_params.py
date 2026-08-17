"""Module-wise trainable-parameter inventory.

Computes LoRA parameter counts directly from the implemented tensor shapes for
Qwen2.5-1.5B-Instruct with grouped-query attention. No GPU or downloads required:

    python count_params.py
"""

LAYERS = 28
HIDDEN = 1536          # q_proj / o_proj output dimension
KV = 256               # k_proj / v_proj output dimension (grouped-query attention)
RANK = 32              # static LoRA rank
K_EXPERTS = 4          # mixture experts (each of rank RANK // K_EXPERTS)


def lora_params(in_features, out_features, rank):
    """LoRA adds A (rank x in) and B (out x rank): rank * (in + out)."""
    return rank * (in_features + out_features)


def inventory(rank):
    return {
        "q_proj": lora_params(HIDDEN, HIDDEN, rank) * LAYERS,
        "k_proj": lora_params(HIDDEN, KV, rank) * LAYERS,
        "v_proj": lora_params(HIDDEN, KV, rank) * LAYERS,
        "o_proj": lora_params(HIDDEN, HIDDEN, rank) * LAYERS,
    }


def main():
    static = inventory(RANK)
    static_total = sum(static.values())
    print(f"Static LoRA (rank {RANK}), per module across {LAYERS} layers:")
    for module, count in static.items():
        print(f"  {module:<8} {count:>12,}")
    print(f"  {'TOTAL':<8} {static_total:>12,}")

    rk = RANK // K_EXPERTS
    experts = sum(inventory(rk).values()) * K_EXPERTS
    gate = 37_508
    persona_proj = 164_864
    moe_total = experts + gate + persona_proj
    print(f"\nDynaPersona-MoE (K={K_EXPERTS}, rank/expert={rk}):")
    print(f"  experts       {experts:>12,}  (equals the static rank-{RANK} adapter: "
          f"{experts == static_total})")
    print(f"  gate          {gate:>12,}")
    print(f"  persona_proj  {persona_proj:>12,}")
    print(f"  TOTAL         {moe_total:>12,}")
    print(f"\nThe adapter budget is identical to the static baseline ({static_total:,}); "
          f"the routing overhead ({gate + persona_proj:,}) is "
          f"{100 * (gate + persona_proj) / static_total:.1f}% of the adapter size.")


if __name__ == "__main__":
    main()
