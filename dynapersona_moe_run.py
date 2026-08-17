"""
DynaPersona-MoE + baselines, multi-style dialogue (EmpatheticDialogues + PersonaChat).

Per seed, all systems use the same rank-32 LoRA adapter-rank budget:
  1. static LoRA            (no routing)
  2. SLIM-style routed LoRA (single LoRA + context sigmoid route between LoRA and identity)
  3. DynaPersona-MoE        (K experts + context softmax gate + load balancing)
Each model is evaluated for perplexity, generation metrics, and latency/memory,
and best-validation checkpoints are saved under the configured project root.

Reuses data/encoders/measurements from dynapersona_full_run.py.
Env: SMOKE=1 (fast check), K_EXPERTS (default 4), LB_LAMBDA (default 0.01),
     RESULTS_TAG (default "_moe").
"""
import os, json, time, math, gc, pickle
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

import dynapersona_full_run as base
from dynapersona_full_run import (CFG, DEVICE, DTYPE, PR, DATA_DIR, RES_DIR, CKPT_DIR,
                                  set_seed, log, PersonaEncoder, EmotionEncoder, StyleEncoder,
                                  ChatDataset, make_collate, build_prompt, perplexity,
                                  gen_metrics, DynaModel, train_model, bootstrap_ppl_diff, SMOKE)

K_EXPERTS = int(os.environ.get("K_EXPERTS", "4"))
LB_LAMBDA = float(os.environ.get("LB_LAMBDA", "0.01"))
MOE_TAG = os.environ.get("RESULTS_TAG", "_moe")


def _load_personachat(max_pairs_per_split):
    from datasets import load_dataset
    ds = load_dataset("bavard/personachat_truecased")
    log(f"PersonaChat columns: {ds['train'].column_names}")

    def to_pairs(split, limit):
        out = []
        for ex in split:
            personality = ex.get("personality") or []
            history = ex.get("history") or []
            cands = ex.get("candidates") or []
            resp = cands[-1] if cands else None
            if not history or not resp:
                continue
            persona = " ".join(personality) if isinstance(personality, list) else str(personality)
            context = history[-1] if isinstance(history, list) else str(history)
            if not context.strip() or not resp.strip():
                continue
            out.append(dict(context=context.strip(), response=resp.strip(),
                            emotion="neutral", persona=persona.strip() or "A friendly conversational partner",
                            style="persona", conv_id=f"pc_{len(out)}", turn_idx=len(history) // 2))
            if len(out) >= limit:
                break
        return out

    val = ds["validation"]; n = len(val)
    return (to_pairs(ds["train"], max_pairs_per_split),
            to_pairs(val.select(range(0, n // 2)), max_pairs_per_split // 6),
            to_pairs(val.select(range(n // 2, n)), max_pairs_per_split // 6))


def build_multistyle_data(smoke=False):
    tag = "_smoke" if smoke else ""
    if os.path.exists(f"{DATA_DIR}/ms_train{tag}.pkl"):
        log("Multistyle data already present.")
        return
    base.build_data()
    ed_tr = base.load_split("train"); ed_va = base.load_split("val"); ed_te = base.load_split("test")
    cap_tr = 4000 if smoke else 40000
    cap_ev = 400 if smoke else 3000
    ed_tr, ed_va, ed_te = ed_tr[:cap_tr], ed_va[:cap_ev], ed_te[:cap_ev]
    pc_tr, pc_va, pc_te = _load_personachat(cap_tr)
    log(f"ED: {len(ed_tr)}/{len(ed_va)}/{len(ed_te)}  PersonaChat: {len(pc_tr)}/{len(pc_va)}/{len(pc_te)}")
    import random
    for name, a, b in (("train", ed_tr, pc_tr), ("val", ed_va, pc_va), ("test", ed_te, pc_te)):
        blend = a + b
        random.Random(0).shuffle(blend)
        with open(f"{DATA_DIR}/ms_{name}{tag}.pkl", "wb") as f:
            pickle.dump(blend, f)
        log(f"ms_{name}: {len(blend)} pairs (styles: empathetic + persona)")


def load_ms(name, smoke=False):
    tag = "_smoke" if smoke else ""
    with open(f"{DATA_DIR}/ms_{name}{tag}.pkl", "rb") as f:
        return pickle.load(f)


class MoEHolder:
    def __init__(self):
        self.weights = None
        self.alpha = None


class MoELoRALinear(nn.Module):
    """Frozen base Linear + K low-rank experts, combined per-example by holder.weights."""
    def __init__(self, base_linear, K, r_k, alpha, holder):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.A = nn.ParameterList([nn.Parameter(torch.empty(r_k, in_f)) for _ in range(K)])
        self.B = nn.ParameterList([nn.Parameter(torch.zeros(out_f, r_k)) for _ in range(K)])
        for a in self.A:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.K, self.scaling, self.holder = K, alpha / r_k, holder

    def forward(self, x, *args, **kwargs):
        out = self.base(x, *args, **kwargs)
        w = self.holder.weights
        xin = x.to(self.A[0].dtype)
        acc = None
        for k in range(self.K):
            delta = F.linear(F.linear(xin, self.A[k]), self.B[k]) * self.scaling
            if w is not None:
                delta = w[:, k].view(-1, 1, 1).to(delta.dtype) * delta
            acc = delta if acc is None else acc + delta
        return out + acc.to(out.dtype)


TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def inject_moe(model, K, r_k, alpha, holder):
    n = 0
    for name, module in list(model.named_modules()):
        for t in TARGETS:
            if name.endswith(t) and isinstance(module, nn.Linear):
                parent = model.get_submodule(name.rsplit(".", 1)[0])
                setattr(parent, t, MoELoRALinear(module, K, r_k, alpha, holder).to(DEVICE))
                n += 1
    log(f"Injected {n} routed-LoRA layers (K={K}, rank/expert={r_k}).")
    return n


class MoEModel(nn.Module):
    """mode='moe' (K experts, softmax gate + load balancing),
       mode='slim' (single LoRA, sigmoid route between LoRA and identity),
       mode='random' (K experts, frozen random gate -- control)."""
    def __init__(self, seed, K=K_EXPERTS, mode="moe"):
        super().__init__()
        set_seed(seed)
        self.gating = True
        self.mode = mode
        if mode == "slim":
            K = 1
        self.K = K
        self.tokenizer = AutoTokenizer.from_pretrained(CFG["base_model"], trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            CFG["base_model"], torch_dtype=DTYPE, trust_remote_code=True).to(DEVICE)
        self.model.config.use_cache = False
        for p in self.model.parameters():
            p.requires_grad = False
        r_k = CFG["lora_r"] // K
        self.holder = MoEHolder()
        inject_moe(self.model, K, r_k, alpha=2.0 * r_k, holder=self.holder)
        for m in self.model.modules():
            if isinstance(m, MoELoRALinear):
                m.A.float(); m.B.float()
        self.persona = PersonaEncoder(256).to(DEVICE)
        self.emotion = EmotionEncoder()
        self.style = StyleEncoder()
        self.gate = nn.Sequential(nn.Linear(256 + 29 + 3, 128), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(128, K)).to(DEVICE)
        if mode == "random":
            for p in self.gate.parameters():
                p.requires_grad = False
        self.uniform = False

    def context_weights(self, batch):
        p = self.persona(batch["persona"])
        e = self.emotion.encode(batch["emotion"], DEVICE)
        s = self.style.encode(batch["style"], DEVICE)
        logits = self.gate(torch.cat([p, e, s], -1))
        if self.mode == "slim":
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1)

    def set_alpha_for_batch(self, batch, bsz):
        if self.uniform:
            self.holder.weights = torch.full((bsz, self.K), 1.0 / self.K, device=DEVICE)
            return self.holder.weights
        w = self.context_weights(batch)
        self.holder.weights = w
        return w

    def forward(self, batch):
        ids = batch["input_ids"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
        lab = batch["labels"].to(DEVICE)
        w = self.set_alpha_for_batch(batch, ids.size(0))
        out = self.model(input_ids=ids, attention_mask=am, labels=lab)
        out.alpha = w
        return out


def moe_param_inventory(model):
    by = dict(experts=0, gate=0, persona_proj=0)
    total_trainable = total_all = 0
    for n, p in model.named_parameters():
        total_all += p.numel()
        if p.requires_grad:
            total_trainable += p.numel()
            if ".A." in n or ".B." in n:
                by["experts"] += p.numel()
            elif "gate" in n:
                by["gate"] += p.numel()
            elif "persona" in n and "encoder" not in n:
                by["persona_proj"] += p.numel()
    return dict(by_module=by, total_trainable=total_trainable, total_all=total_all,
                trainable_fraction=total_trainable / total_all, K=model.K, mode=model.mode)


def train_moe(model, tr, va, seed):
    tok = model.tokenizer
    collate = make_collate(tok.pad_token_id)
    tl = DataLoader(ChatDataset(tr, tok, CFG["max_length"]), batch_size=CFG["batch_size"],
                    shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate)
    vl = DataLoader(ChatDataset(va, tok, CFG["max_length"]), batch_size=CFG["batch_size"],
                    shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)
    expert_params = [p for n, p in model.model.named_parameters() if p.requires_grad]
    ctx_params = [p for p in (list(model.gate.parameters()) + list(model.persona.proj.parameters()))
                  if p.requires_grad]
    opt = torch.optim.AdamW([{"params": expert_params, "lr": CFG["lr"]},
                             {"params": ctx_params, "lr": CFG["lr"]}],
                            betas=(0.9, 0.999), weight_decay=CFG["weight_decay"])
    total = len(tl) * CFG["num_epochs"] // CFG["grad_accum"]
    sched = get_linear_schedule_with_warmup(opt, int(total * CFG["warmup_ratio"]), total)
    hist = dict(train_loss=[], val_loss=[], usage=[])
    best, best_state, stp = math.inf, None, 0
    for epoch in range(1, CFG["num_epochs"] + 1):
        model.uniform = (epoch == 1)
        model.model.train()
        run, seen = 0.0, 0
        usage = torch.zeros(model.K, device=DEVICE); ub = 0
        opt.zero_grad()
        t0 = time.time()
        for bi, batch in enumerate(tl):
            out = model(batch)
            lb = torch.tensor(0.0, device=DEVICE)
            if not model.uniform and out.alpha is not None and model.mode == "moe":
                imp = out.alpha.mean(0)
                lb = model.K * (imp * imp).sum()
                usage += imp.detach(); ub += 1
            loss = (out.loss + LB_LAMBDA * lb) / CFG["grad_accum"]
            loss.backward()
            if (bi + 1) % CFG["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(); stp += 1
            run += out.loss.item(); seen += 1
            if (bi + 1) % 200 == 0:
                u = (usage / max(ub, 1)).tolist() if ub else [1.0 / model.K] * model.K
                log(f"[{model.mode} s{seed}] ep{epoch} step{stp} loss {run/seen:.4f} "
                    f"usage {[round(x,2) for x in u]} lr {sched.get_last_lr()[0]:.2e}")
        hist["train_loss"].append(run / max(seen, 1))
        model.model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for batch in vl:
                vloss += model(batch).loss.item(); vn += 1
        vloss /= max(vn, 1); hist["val_loss"].append(vloss)
        if ub:
            hist["usage"].append((usage / ub).tolist())
        log(f"[{model.mode} s{seed}] epoch {epoch} done: train {hist['train_loss'][-1]:.4f} "
            f"val {vloss:.4f} time {(time.time()-t0)/60:.1f}m")
        if vloss < best:
            best = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        try:
            os.makedirs(CKPT_DIR, exist_ok=True)
            torch.save(best_state, f"{CKPT_DIR}/{model.mode}_ms_seed{seed}.pt")
        except Exception as e:
            log("ckpt save failed:", e)
        log(f"[{model.mode} s{seed}] restored best-val checkpoint (val {best:.4f})")
    model.uniform = False
    hist["best_val_loss"] = best; hist["best_val_ppl"] = math.exp(best)
    return hist


@torch.no_grad()
def expert_usage_by_style(model, data, tok, n=2000):
    model.model.eval()
    from collections import defaultdict
    by = defaultdict(lambda: np.zeros(model.K)); cnt = defaultdict(int)
    for it in data[:n]:
        b = dict(persona=[it["persona"]], emotion=[it["emotion"]], style=[it.get("style", "empathetic")])
        w = model.context_weights(b)[0].cpu().numpy()
        by[it["style"]] += w; cnt[it["style"]] += 1
    return {s: (by[s] / max(cnt[s], 1)).tolist() for s in by}


def _profile(model, tok):
    try:
        return base.profile_latency_memory(model, tok)
    except Exception as e:
        return {"error": str(e)}


def run_seed(seed, smoke=False):
    out = f"{RES_DIR}/results_seed{seed}{MOE_TAG}{'_smoke' if smoke else ''}.json"
    if os.path.exists(out) and not smoke:
        try:
            prev = json.load(open(out))
            if all(k in prev for k in ("ppl_static", "ppl_slim", "ppl_moe")):
                log(f"seed {seed} extended already complete, skipping."); return prev
        except Exception:
            pass
    R = dict(seed=seed, K=K_EXPERTS, smoke=smoke)
    tr = load_ms("train", smoke); va = load_ms("val", smoke); te = load_ms("test", smoke)
    R["data_sizes"] = dict(train=len(tr), val=len(va), test=len(te))

    static = DynaModel(gating=False, seed=seed); tok = static.tokenizer
    R["param_inventory_static"] = base.module_param_inventory(static)
    R["train_static"] = train_model(static, tr, va, "static_ms", seed)
    sp = perplexity(static, te, tok); R["ppl_static"] = sp["ppl"]
    R["gen_static"] = gen_metrics(static, te, tok, CFG["gen_eval_n"])
    R["profile_static"] = _profile(static, tok)
    json.dump(R, open(out, "w"), indent=2)
    del static; gc.collect(); torch.cuda.empty_cache()

    slim = MoEModel(seed, mode="slim")
    R["param_inventory_slim"] = moe_param_inventory(slim)
    R["train_slim"] = train_moe(slim, tr, va, seed)
    lp = perplexity(slim, te, tok); R["ppl_slim"] = lp["ppl"]
    R["gen_slim"] = gen_metrics(slim, te, tok, CFG["gen_eval_n"])
    R["profile_slim"] = _profile(slim, tok)
    R["improvement_slim_vs_static_pct"] = (R["ppl_static"] - R["ppl_slim"]) / R["ppl_static"] * 100
    R["bootstrap_slim_vs_static"] = bootstrap_ppl_diff(sp["per_example"], lp["per_example"], CFG["bootstrap_n"])
    json.dump(R, open(out, "w"), indent=2)
    del slim; gc.collect(); torch.cuda.empty_cache()

    moe = MoEModel(seed, mode="moe")
    R["param_inventory_moe"] = moe_param_inventory(moe)
    R["train_moe"] = train_moe(moe, tr, va, seed)
    mp = perplexity(moe, te, tok); R["ppl_moe"] = mp["ppl"]
    R["gen_moe"] = gen_metrics(moe, te, tok, CFG["gen_eval_n"])
    R["profile_moe"] = _profile(moe, tok)
    R["expert_usage_by_style"] = expert_usage_by_style(moe, te, tok)
    R["bootstrap"] = bootstrap_ppl_diff(sp["per_example"], mp["per_example"], CFG["bootstrap_n"])
    R["bootstrap_moe_vs_slim"] = bootstrap_ppl_diff(lp["per_example"], mp["per_example"], CFG["bootstrap_n"])
    R["improvement_ppl_pct"] = (R["ppl_static"] - R["ppl_moe"]) / R["ppl_static"] * 100
    json.dump(R, open(out, "w"), indent=2)
    log(f"seed {seed} DONE -> static {R['ppl_static']:.3f} | slim {R['ppl_slim']:.3f} "
        f"| moe {R['ppl_moe']:.3f} || moe vs static {R['improvement_ppl_pct']:+.2f}% "
        f"| slim vs static {R['improvement_slim_vs_static_pct']:+.2f}%")
    log(f"  latency ms (static/slim/moe): {R['profile_static'].get('latency_ms_mean')}/"
        f"{R['profile_slim'].get('latency_ms_mean')}/{R['profile_moe'].get('latency_ms_mean')}")
    del moe; gc.collect(); torch.cuda.empty_cache()
    return R


def main():
    log(f"Extended run. GPU {torch.cuda.get_device_name(0)}  SMOKE={SMOKE}  K={K_EXPERTS}")
    build_multistyle_data(smoke=SMOKE)
    res = [run_seed(s, smoke=SMOKE) for s in CFG["seeds"]]
    if len(res) > 1:
        def col(k):
            return [r[k] for r in res if k in r]
        agg = dict(ppl_static=col("ppl_static"), ppl_slim=col("ppl_slim"), ppl_moe=col("ppl_moe"),
                   moe_vs_static=col("improvement_ppl_pct"),
                   slim_vs_static=col("improvement_slim_vs_static_pct"))
        for k in ("moe_vs_static", "slim_vs_static"):
            v = agg[k]
            agg[k + "_mean"] = float(np.mean(v)); agg[k + "_std"] = float(np.std(v))
        json.dump(agg, open(f"{RES_DIR}/aggregate{MOE_TAG}.json", "w"), indent=2)
        log(f"AGGREGATE: {agg}")
    log("Extended run ALL DONE.")


if __name__ == "__main__":
    main()
