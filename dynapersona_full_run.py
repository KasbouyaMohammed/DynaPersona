"""
DynaPersona full experimental run (self-contained).

What this does, end to end, on one GPU:
  1. Rebuilds the EmpatheticDialogues data with the exact schema the model expects.
  2. Implements alpha-gated LoRA: alpha(c) = sigma(W[persona;emotion;style]+b)
     multiplies the LoRA delta, and gradients flow to the gating network.
  3. Trains a matched static-LoRA baseline under identical seed/data/budget/decoding.
  4. Trains DynaPersona (alpha-gated) with the 3-stage curriculum.
  5. Measures the following and writes ONE results JSON per seed:
       - module-wise trainable parameter inventory
       - test perplexity (response-only) for base / static / dynapersona
       - generation metrics: BLEU, ROUGE-L, Distinct-1/2, (optional) BERTScore
       - latency mean/std and throughput (decoding-only and end-to-end), fixed protocol
       - memory breakdown (weights / peak allocated / reserved / KV-cache estimate)
       - ablations (no-gate, no-persona, no-emotion, no-style) as relative ppl change
       - long-conversation degradation (ppl by turn index)
       - bootstrap CIs on the static->dynapersona perplexity difference
       - alpha statistics overall and by emotion

Everything is saved to PROJECT_ROOT so interrupted sessions do not lose completed outputs.
Set SMOKE=True for a fast correctness check before the full run.

Run in Colab:
    from google.colab import drive; drive.mount('/content/drive')
    !pip -q install transformers==4.44.2 datasets==2.16.0 peft==0.11.1 accelerate==0.33.0 \
        evaluate==0.4.1 rouge-score sacrebleu bert-score sentencepiece
    !python /content/drive/MyDrive/DynaPersona_FullRun/dynapersona_full_run.py
For resilience run it detached:
    !nohup python .../dynapersona_full_run.py > /content/drive/MyDrive/DynaPersona_FullRun/run.log 2>&1 &
    then watch:  !tail -n 40 /content/drive/MyDrive/DynaPersona_FullRun/run.log
"""

import os, json, time, math, random, gc, pickle, argparse
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SMOKE = os.environ.get("SMOKE", "0") == "1"

CFG = dict(
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    project_root=os.environ.get("PROJECT_ROOT", "/content/drive/MyDrive/DynaPersona_FullRun"),
    max_length=512,
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    batch_size=8,
    grad_accum=4,
    lr=5e-4,
    gate_lr_mult=1.0,
    gate_identity_reg=0.1,
    warmup_ratio=0.03,
    num_epochs=3,
    weight_decay=0.01,
    seeds=[42, 123, 7],
    gen_eval_n=800,
    latency_iters=60,
    latency_warmup=10,
    latency_seq_len=64,
    latency_new_tokens=64,
    bootstrap_n=2000,
    use_bertscore=True,
)

if SMOKE:
    CFG.update(dict(num_epochs=2, seeds=[42], gen_eval_n=60,
                    latency_iters=15, bootstrap_n=300, use_bertscore=False))
    SMOKE_TRAIN_N, SMOKE_EVAL_N = 4000, 600
else:
    SMOKE_TRAIN_N = SMOKE_EVAL_N = None

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
PR = CFG["project_root"]
DATA_DIR = f"{PR}/data"
CKPT_DIR = f"{PR}/checkpoints"
RES_DIR = f"{PR}/results"
for d in (PR, DATA_DIR, CKPT_DIR, RES_DIR):
    os.makedirs(d, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


ED_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"


def _parse_ed_csv(path):
    """Parse an EmpatheticDialogues CSV into per-conversation rows.
    Columns: conv_id,utterance_idx,context,prompt,speaker_idx,utterance,selfeval,tags.
    Commas inside text are encoded as _comma_ in the distribution.
    """
    import csv
    convs = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                convs[row[0]].append(dict(
                    utterance_idx=int(row[1]), emotion=row[2],
                    speaker_idx=int(row[4]), utterance=row[5].replace("_comma_", ",")))
            except ValueError:
                continue
    return convs


def build_data():
    if os.path.exists(f"{DATA_DIR}/train_full.pkl") and os.path.exists(f"{DATA_DIR}/test_full.pkl"):
        log("Data already present, skipping rebuild.")
        return
    import tarfile, urllib.request
    tgz = f"{DATA_DIR}/ed.tar.gz"
    if not os.path.exists(tgz):
        log("Downloading EmpatheticDialogues CSVs...")
        urllib.request.urlretrieve(ED_URL, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(DATA_DIR)
    root = f"{DATA_DIR}/empatheticdialogues"

    def proc(csv_name):
        convs = _parse_ed_csv(f"{root}/{csv_name}")
        out = []
        for cid, items in convs.items():
            items.sort(key=lambda x: x["utterance_idx"])
            for i in range(len(items) - 1):
                cur, nxt = items[i], items[i + 1]
                if cur["speaker_idx"] != nxt["speaker_idx"]:
                    out.append(dict(
                        context=cur["utterance"], response=nxt["utterance"],
                        emotion=cur["emotion"],
                        persona=f"An empathetic listener responding to someone experiencing {cur['emotion']}",
                        style="empathetic", conv_id=cid, turn_idx=i))
        return out

    for name, csv_name in (("train", "train.csv"), ("val", "valid.csv"), ("test", "test.csv")):
        data = proc(csv_name)
        with open(f"{DATA_DIR}/{name}_full.pkl", "wb") as f:
            pickle.dump(data, f)
        log(f"{name}: {len(data)} pairs")


def load_split(name, limit=None):
    with open(f"{DATA_DIR}/{name}_full.pkl", "rb") as f:
        data = pickle.load(f)
    if limit:
        data = data[:limit]
    return data


from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

EMOTIONS_27 = ['admiration','amusement','anger','annoyance','approval','caring','confusion',
    'curiosity','desire','disappointment','disapproval','disgust','embarrassment','excitement',
    'fear','gratitude','grief','joy','love','nervousness','optimism','pride','realization',
    'relief','remorse','sadness','surprise']
VA = {'joy':(0.8,0.6),'anger':(-0.8,0.7),'sadness':(-0.7,-0.4),'fear':(-0.6,0.8),
    'surprise':(0.0,0.7),'disgust':(-0.7,0.5),'admiration':(0.7,0.3),'gratitude':(0.8,0.2),
    'love':(0.9,0.5)}
ED_VA = {'afraid':(-0.6,0.7),'angry':(-0.8,0.7),'annoyed':(-0.5,0.4),'anticipating':(0.3,0.5),
    'anxious':(-0.5,0.7),'apprehensive':(-0.4,0.5),'ashamed':(-0.6,0.3),'caring':(0.6,0.3),
    'confident':(0.6,0.4),'content':(0.6,-0.2),'devastated':(-0.9,0.5),'disappointed':(-0.6,0.1),
    'disgusted':(-0.7,0.5),'embarrassed':(-0.5,0.4),'excited':(0.7,0.8),'faithful':(0.5,0.1),
    'furious':(-0.9,0.8),'grateful':(0.8,0.2),'guilty':(-0.6,0.3),'hopeful':(0.6,0.4),
    'impressed':(0.6,0.4),'jealous':(-0.5,0.5),'joyful':(0.8,0.6),'lonely':(-0.7,-0.3),
    'nostalgic':(0.1,0.1),'prepared':(0.4,0.3),'proud':(0.7,0.5),'sad':(-0.7,-0.4),
    'sentimental':(0.2,0.1),'surprised':(0.0,0.7),'terrified':(-0.7,0.9),'trusting':(0.5,0.2)}
STYLES = ['empathetic', 'informative', 'casual']


class PersonaEncoder(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.proj = nn.Sequential(nn.Linear(384, hidden), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(hidden, hidden), nn.LayerNorm(hidden))

    def forward(self, texts):
        inp = self.tok(texts, return_tensors="pt", truncation=True, max_length=128, padding=True)
        inp = {k: v.to(self.encoder.device) for k, v in inp.items()}
        with torch.no_grad():
            emb = self.encoder(**inp).last_hidden_state.mean(1)
        return self.proj(emb.float())


class EmotionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.idx = {e: i for i, e in enumerate(EMOTIONS_27)}

    def encode(self, labels, device):
        v = torch.zeros(len(labels), 29, device=device)
        for i, e in enumerate(labels):
            e = str(e).lower().strip()
            if e in self.idx:
                v[i, self.idx[e]] = 1.0
            va = VA.get(e) or ED_VA.get(e)
            if va:
                v[i, 27], v[i, 28] = va
        return v


class StyleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.idx = {s: i for i, s in enumerate(STYLES)}

    def encode(self, labels, device):
        v = torch.zeros(len(labels), 3, device=device)
        for i, s in enumerate(labels):
            v[i, self.idx.get(str(s).lower().strip(), 0)] = 1.0
        return v


class Gate(nn.Module):
    def __init__(self, din=256 + 29 + 3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 128), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, p, e, s):
        return 2.0 * self.net(torch.cat([p, e, s], dim=-1))


class AlphaHolder:
    def __init__(self):
        self.alpha = None


def _alpha_lora_forward(self, x, *args, **kwargs):
    result = self.base_layer(x, *args, **kwargs)
    holder = self._alpha_holder
    torch_result_dtype = result.dtype
    for name in self.active_adapters:
        if name not in self.lora_A:
            continue
        A, B = self.lora_A[name], self.lora_B[name]
        drop, scaling = self.lora_dropout[name], self.scaling[name]
        xin = drop(x).to(A.weight.dtype)
        delta = B(A(xin)) * scaling
        a = holder.alpha
        if a is not None:
            delta = a.to(delta.dtype) * delta
        result = result + delta.to(torch_result_dtype)
    return result


def install_alpha_gate(peft_model, holder):
    from peft.tuners.lora import Linear as LoraLinear
    n = 0
    for m in peft_model.modules():
        if isinstance(m, LoraLinear):
            m._alpha_holder = holder
            m.forward = _alpha_lora_forward.__get__(m, m.__class__)
            n += 1
    log(f"Installed alpha-gated forward on {n} LoRA layers.")
    return n


class DynaModel(nn.Module):
    def __init__(self, gating: bool, seed: int):
        super().__init__()
        set_seed(seed)
        self.gating = gating
        self.tokenizer = AutoTokenizer.from_pretrained(CFG["base_model"], trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            CFG["base_model"], torch_dtype=DTYPE, trust_remote_code=True).to(DEVICE)
        base.config.use_cache = False
        lcfg = LoraConfig(r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"],
                          target_modules=CFG["target_modules"], lora_dropout=CFG["lora_dropout"],
                          bias="none", task_type=TaskType.CAUSAL_LM)
        self.model = get_peft_model(base, lcfg)
        self.holder = AlphaHolder()
        if gating:
            install_alpha_gate(self.model, self.holder)
            self.persona = PersonaEncoder(256).to(DEVICE)
            self.emotion = EmotionEncoder()
            self.style = StyleEncoder()
            self.gate = Gate().to(DEVICE)
        self.fixed_alpha = None

    def context_alpha(self, batch):
        p = self.persona(batch["persona"])
        e = self.emotion.encode(batch["emotion"], DEVICE)
        s = self.style.encode(batch["style"], DEVICE)
        return self.gate(p, e, s)

    def set_alpha_for_batch(self, batch, bsz):
        if not self.gating:
            self.holder.alpha = None
            return None
        if self.fixed_alpha is not None:
            self.holder.alpha = torch.full((bsz, 1, 1), float(self.fixed_alpha), device=DEVICE)
            return self.holder.alpha
        a = self.context_alpha(batch)
        self.holder.alpha = a.view(bsz, 1, 1)
        return a

    def forward(self, batch):
        ids = batch["input_ids"].to(DEVICE)
        am = batch["attention_mask"].to(DEVICE)
        lab = batch["labels"].to(DEVICE)
        a = self.set_alpha_for_batch(batch, ids.size(0))
        out = self.model(input_ids=ids, attention_mask=am, labels=lab)
        out.alpha = a
        return out


SYS = "You are an empathetic conversational agent."


def build_prompt(context):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{context}<|im_end|>\n<|im_start|>assistant\n")


class ChatDataset(Dataset):
    def __init__(self, data, tok, max_len):
        self.data, self.tok, self.max_len = data, tok, max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        it = self.data[i]
        prompt = build_prompt(it["context"])
        full = prompt + it["response"] + "<|im_end|>"
        pid = self.tok(prompt, add_special_tokens=False)["input_ids"]
        fid = self.tok(full, add_special_tokens=False, truncation=True, max_length=self.max_len)["input_ids"]
        labels = list(fid)
        for j in range(min(len(pid), len(labels))):
            labels[j] = -100
        return dict(input_ids=fid, labels=labels,
                    persona=it["persona"], emotion=it["emotion"],
                    style=it.get("style", "empathetic"),
                    turn_idx=it.get("turn_idx", 0))


def make_collate(pad_id):
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        ids, ams, labs = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_id] * n)
            ams.append([1] * len(b["input_ids"]) + [0] * n)
            labs.append(b["labels"] + [-100] * n)
        return dict(
            input_ids=torch.tensor(ids, dtype=torch.long),
            attention_mask=torch.tensor(ams, dtype=torch.long),
            labels=torch.tensor(labs, dtype=torch.long),
            persona=[b["persona"] for b in batch],
            emotion=[b["emotion"] for b in batch],
            style=[b["style"] for b in batch],
            turn_idx=[b["turn_idx"] for b in batch],
        )
    return collate


from transformers import get_linear_schedule_with_warmup


def train_model(model, train_data, val_data, tag, seed):
    tok = model.tokenizer
    collate = make_collate(tok.pad_token_id)
    tl = DataLoader(ChatDataset(train_data, tok, CFG["max_length"]), batch_size=CFG["batch_size"],
                    shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate)
    vl = DataLoader(ChatDataset(val_data, tok, CFG["max_length"]), batch_size=CFG["batch_size"],
                    shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)

    params = [{"params": [p for p in model.model.parameters() if p.requires_grad], "lr": CFG["lr"]}]
    if model.gating:
        gate_params = list(model.gate.parameters()) + list(model.persona.proj.parameters())
        params.append({"params": gate_params, "lr": CFG["lr"] * CFG["gate_lr_mult"]})
    opt = torch.optim.AdamW(params, betas=(0.9, 0.999), eps=1e-8, weight_decay=CFG["weight_decay"])
    total = len(tl) * CFG["num_epochs"] // CFG["grad_accum"]
    sched = get_linear_schedule_with_warmup(opt, int(total * CFG["warmup_ratio"]), total)

    hist = dict(train_loss=[], val_loss=[], alpha_mean=[], alpha_std=[], gate_grad=[])
    best = math.inf
    best_state = None
    step = 0
    for epoch in range(1, CFG["num_epochs"] + 1):
        if model.gating:
            model.fixed_alpha = 1.0 if epoch == 1 else None
        model.model.train()
        run_loss, alphas, seen = 0.0, [], 0
        opt.zero_grad()
        t0 = time.time()
        for bi, batch in enumerate(tl):
            out = model(batch)
            reg = 0.0
            if model.gating and model.fixed_alpha is None and out.alpha is not None:
                reg = CFG["gate_identity_reg"] * ((out.alpha - 1.0) ** 2).mean()
            loss = (out.loss + reg) / CFG["grad_accum"]
            loss.backward()
            if out.alpha is not None:
                alphas.append(out.alpha.mean().item())
            if (bi + 1) % CFG["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                if model.gating and model.fixed_alpha is None:
                    g = [p.grad.norm().item() for p in model.gate.parameters() if p.grad is not None]
                    if g:
                        hist["gate_grad"].append(float(np.mean(g)))
                opt.step(); sched.step(); opt.zero_grad(); step += 1
            run_loss += out.loss.item(); seen += 1
            if (bi + 1) % 200 == 0:
                am = np.mean(alphas[-200:]) if alphas else float("nan")
                log(f"[{tag} s{seed}] ep{epoch} step{step} loss {run_loss/seen:.4f} "
                    f"alpha {am:.3f} lr {sched.get_last_lr()[0]:.2e}")
        hist["train_loss"].append(run_loss / max(seen, 1))
        if alphas:
            hist["alpha_mean"].append(float(np.mean(alphas)))
            hist["alpha_std"].append(float(np.std(alphas)))

        model.model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for batch in vl:
                vloss += model(batch).loss.item(); vn += 1
        vloss /= max(vn, 1)
        hist["val_loss"].append(vloss)
        log(f"[{tag} s{seed}] epoch {epoch} done: train {hist['train_loss'][-1]:.4f} "
            f"val {vloss:.4f} time {(time.time()-t0)/60:.1f}m")
        if vloss < best:
            best = vloss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.named_parameters() if v.requires_grad}
            save_adapter(model, f"{CKPT_DIR}/{tag}_seed{seed}")
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        log(f"[{tag} s{seed}] restored best-val checkpoint (val {best:.4f}) for evaluation")
    hist["best_val_loss"] = best
    hist["best_val_ppl"] = math.exp(best)
    return hist


def save_adapter(model, path):
    os.makedirs(path, exist_ok=True)
    model.model.save_pretrained(path)
    if model.gating:
        torch.save({"gate": model.gate.state_dict(),
                    "persona_proj": model.persona.proj.state_dict()}, f"{path}/context.pt")


@torch.no_grad()
def perplexity(model, data, tok, by_turn=False):
    """Response-only perplexity via token-weighted mean NLL. Optionally bucket by turn index."""
    collate = make_collate(tok.pad_token_id)
    dl = DataLoader(ChatDataset(data, tok, CFG["max_length"]), batch_size=CFG["batch_size"],
                    shuffle=False, num_workers=2, collate_fn=collate)
    model.model.eval()
    tot_nll, tot_tok = 0.0, 0
    per_example = []
    turn_nll, turn_tok = defaultdict(float), defaultdict(int)
    for batch in dl:
        ids = batch["input_ids"].to(DEVICE); am = batch["attention_mask"].to(DEVICE)
        lab = batch["labels"].to(DEVICE)
        model.set_alpha_for_batch(batch, ids.size(0))
        logits = model.model(input_ids=ids, attention_mask=am).logits
        sl = logits[:, :-1, :]; slab = lab[:, 1:]
        nll = F.cross_entropy(sl.reshape(-1, sl.size(-1)).float(), slab.reshape(-1),
                              ignore_index=-100, reduction="none").view(slab.size())
        mask = (slab != -100)
        ex_nll = nll.sum(1); ex_tok = mask.sum(1).clamp(min=1)
        for k in range(ids.size(0)):
            per_example.append((ex_nll[k].item(), ex_tok[k].item()))
            if by_turn:
                t = int(batch["turn_idx"][k])
                turn_nll[t] += ex_nll[k].item(); turn_tok[t] += ex_tok[k].item()
        tot_nll += ex_nll.sum().item(); tot_tok += ex_tok.sum().item()
    ppl = math.exp(tot_nll / max(tot_tok, 1))
    res = dict(ppl=ppl, per_example=per_example)
    if by_turn:
        res["by_turn"] = {t: math.exp(turn_nll[t] / max(turn_tok[t], 1))
                          for t in sorted(turn_nll) if turn_tok[t] > 0}
    return res


def module_param_inventory(model):
    inv = defaultdict(int); total_trainable = 0; total_all = 0
    for n, p in model.named_parameters():
        total_all += p.numel()
        if p.requires_grad:
            total_trainable += p.numel()
            if ".lora_" in n:
                for mod in CFG["target_modules"]:
                    if mod in n:
                        inv[mod] += p.numel()
            elif "gate" in n:
                inv["gate"] += p.numel()
            elif "persona" in n and "encoder" not in n:
                inv["persona_proj"] += p.numel()
            else:
                inv["other"] += p.numel()
    return dict(by_module=dict(inv), total_trainable=total_trainable,
                total_all=total_all, trainable_fraction=total_trainable / total_all)


@torch.no_grad()
def gen_metrics(model, data, tok, n):
    import evaluate
    sub = data[:n]
    model.model.eval()
    model.model.config.use_cache = True
    preds, refs = [], []
    for it in sub:
        prompt = build_prompt(it["context"])
        enc = tok(prompt, return_tensors="pt").to(DEVICE)
        if model.gating:
            b = dict(persona=[it["persona"]], emotion=[it["emotion"]], style=[it.get("style", "empathetic")])
            model.set_alpha_for_batch(b, 1)
        else:
            model.holder.alpha = None
        gen = model.model.generate(**enc, max_new_tokens=48, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
        txt = tok.decode(gen[0][enc["input_ids"].size(1):], skip_special_tokens=True).strip()
        preds.append(txt if txt else "."); refs.append(it["response"])
    model.model.config.use_cache = False

    def distinct(seqs, n_):
        grams, tot = set(), 0
        for s in seqs:
            toks = s.split()
            for i in range(len(toks) - n_ + 1):
                grams.add(tuple(toks[i:i + n_])); tot += 1
        return len(grams) / max(tot, 1)

    out = dict(distinct1=distinct(preds, 1), distinct2=distinct(preds, 2), n=len(preds))
    try:
        bleu = evaluate.load("sacrebleu")
        out["bleu"] = bleu.compute(predictions=preds, references=[[r] for r in refs])["score"]
    except Exception as e:
        out["bleu_error"] = str(e)
    try:
        rouge = evaluate.load("rouge")
        out["rougeL"] = rouge.compute(predictions=preds, references=refs)["rougeL"]
    except Exception as e:
        out["rouge_error"] = str(e)
    if CFG["use_bertscore"]:
        try:
            bs = evaluate.load("bertscore")
            r = bs.compute(predictions=preds, references=refs, lang="en",
                           model_type="microsoft/deberta-xlarge-mnli", batch_size=16)
            out["bertscore_f1"] = float(np.mean(r["f1"]))
        except Exception as e:
            out["bertscore_error"] = str(e)
    out["samples"] = [dict(context=sub[i]["context"], ref=refs[i], pred=preds[i])
                      for i in range(min(8, len(preds)))]
    return out


@torch.no_grad()
def profile_latency_memory(model, tok):
    model.model.eval()
    model.model.config.use_cache = True
    ctx = "I have been feeling quite overwhelmed at work lately and I am not sure what to do."
    prompt = build_prompt(ctx)
    enc = tok(prompt, return_tensors="pt").to(DEVICE)
    if model.gating:
        model.set_alpha_for_batch(dict(persona=["An empathetic listener responding to someone experiencing anxious"],
                                       emotion=["anxious"], style=["empathetic"]), 1)
    else:
        model.holder.alpha = None
    gkw = dict(max_new_tokens=CFG["latency_new_tokens"], min_new_tokens=CFG["latency_new_tokens"],
               do_sample=False, pad_token_id=tok.pad_token_id)
    for _ in range(CFG["latency_warmup"]):
        model.model.generate(**enc, **gkw)
    torch.cuda.synchronize()
    e2e = []
    for _ in range(CFG["latency_iters"]):
        t = time.perf_counter()
        model.model.generate(**enc, **gkw)
        torch.cuda.synchronize()
        e2e.append(time.perf_counter() - t)
    e2e = np.array(e2e)
    pf = []
    for _ in range(CFG["latency_iters"]):
        t = time.perf_counter()
        model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        torch.cuda.synchronize()
        pf.append(time.perf_counter() - t)
    prefill = float(np.median(pf))
    new_tokens = CFG["latency_new_tokens"]
    decode_time = max(e2e.mean() - prefill, 1e-6)
    torch.cuda.reset_peak_memory_stats()
    _ = model.model.generate(**enc, **gkw); torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9
    weights = sum(p.numel() * p.element_size() for p in model.model.parameters()) / 1e9
    model.model.config.use_cache = False
    return dict(
        latency_ms_mean=float(e2e.mean() * 1000), latency_ms_std=float(e2e.std() * 1000),
        prefill_ms=float(prefill * 1000),
        e2e_throughput_tok_s=float(new_tokens / e2e.mean()),
        decode_only_throughput_tok_s=float(new_tokens / decode_time),
        new_tokens=new_tokens, prompt_tokens=int(enc["input_ids"].size(1)),
        peak_alloc_gb=float(peak), reserved_gb=float(reserved), weight_gb=float(weights),
    )


def bootstrap_ppl_diff(static_pe, dyna_pe, n_boot):
    """Paired bootstrap over test examples on the difference in corpus perplexity."""
    s = np.array(static_pe); d = np.array(dyna_pe)
    m = min(len(s), len(d)); s, d = s[:m], d[:m]
    idx = np.arange(m); diffs = []
    for _ in range(n_boot):
        b = np.random.choice(idx, m, replace=True)
        ps = math.exp(s[b, 0].sum() / max(s[b, 1].sum(), 1))
        pd = math.exp(d[b, 0].sum() / max(d[b, 1].sum(), 1))
        diffs.append(ps - pd)
    diffs = np.array(diffs)
    return dict(mean_diff=float(diffs.mean()),
                ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                p_dyna_better=float((diffs > 0).mean()))


@torch.no_grad()
def alpha_by_emotion(model, data, tok, n=1500):
    if not model.gating:
        return {}
    model.model.eval()
    by = defaultdict(list)
    for it in data[:n]:
        b = dict(persona=[it["persona"]], emotion=[it["emotion"]], style=[it.get("style", "empathetic")])
        a = model.context_alpha(b).item()
        by[it["emotion"]].append(a)
    allv = [v for vs in by.values() for v in vs]
    return dict(overall_mean=float(np.mean(allv)), overall_std=float(np.std(allv)),
                by_emotion={k: float(np.mean(v)) for k, v in sorted(by.items())})


@torch.no_grad()
def ablations(model, data, tok):
    if not model.gating:
        return {}
    base = perplexity(model, data, tok)["ppl"]
    results = {"full": base}
    orig = model.context_alpha

    def patched(zero):
        def f(batch):
            p = model.persona(batch["persona"]) if "persona" not in zero else torch.zeros(len(batch["persona"]), 256, device=DEVICE)
            e = model.emotion.encode(batch["emotion"], DEVICE) if "emotion" not in zero else torch.zeros(len(batch["emotion"]), 29, device=DEVICE)
            s = model.style.encode(batch["style"], DEVICE) if "style" not in zero else torch.zeros(len(batch["style"]), 3, device=DEVICE)
            return model.gate(p, e, s)
        return f

    for name, zero in (("no_persona", {"persona"}), ("no_emotion", {"emotion"}), ("no_style", {"style"})):
        model.context_alpha = patched(zero)
        results[name] = perplexity(model, data, tok)["ppl"]
    model.context_alpha = orig

    def force_one(batch):
        return torch.ones(len(batch["persona"]), 1, device=DEVICE)
    model.context_alpha = force_one
    results["no_gate_alpha1"] = perplexity(model, data, tok)["ppl"]
    model.context_alpha = orig

    rel = {k: (results[k] - base) / base * 100 for k in results if k != "full"}
    return dict(ppl=results, relative_pct_vs_full=rel)


def run_seed(seed):
    out_path = f"{RES_DIR}/results_seed{seed}{'_smoke' if SMOKE else ''}.json"
    if os.path.exists(out_path) and not SMOKE:
        try:
            prev = json.load(open(out_path))
            if "improvement_ppl_pct" in prev:
                log(f"seed {seed} already complete, skipping."); return prev
        except Exception:
            pass
        log(f"seed {seed} has a partial result, re-running it.")
    R = dict(seed=seed, config={k: CFG[k] for k in
             ("base_model","lora_r","lora_alpha","lora_dropout","target_modules",
              "batch_size","grad_accum","lr","num_epochs","max_length")}, smoke=SMOKE)

    train_data = load_split("train", SMOKE_TRAIN_N)
    val_data = load_split("val", SMOKE_EVAL_N)
    test_data = load_split("test", SMOKE_EVAL_N)
    R["data_sizes"] = dict(train=len(train_data), val=len(val_data), test=len(test_data))

    base_only = DynaModel(gating=False, seed=seed)
    tok = base_only.tokenizer
    R["ppl_base_pretrained"] = perplexity(base_only, test_data, tok)["ppl"]
    del base_only; gc.collect(); torch.cuda.empty_cache()

    static = DynaModel(gating=False, seed=seed)
    R["param_inventory_static"] = module_param_inventory(static)
    R["train_static"] = train_model(static, train_data, val_data, "static", seed)
    static_ppl = perplexity(static, test_data, tok, by_turn=True)
    R["ppl_static"] = static_ppl["ppl"]; R["static_by_turn"] = static_ppl["by_turn"]
    R["gen_static"] = gen_metrics(static, test_data, tok, CFG["gen_eval_n"])
    R["profile_static"] = profile_latency_memory(static, tok)
    json.dump(R, open(out_path, "w"), indent=2)
    del static; gc.collect(); torch.cuda.empty_cache()

    dyna = DynaModel(gating=True, seed=seed)
    R["param_inventory_dynapersona"] = module_param_inventory(dyna)
    R["train_dynapersona"] = train_model(dyna, train_data, val_data, "dynapersona", seed)
    dyna_ppl = perplexity(dyna, test_data, tok, by_turn=True)
    R["ppl_dynapersona"] = dyna_ppl["ppl"]; R["dyna_by_turn"] = dyna_ppl["by_turn"]
    R["gen_dynapersona"] = gen_metrics(dyna, test_data, tok, CFG["gen_eval_n"])
    R["profile_dynapersona"] = profile_latency_memory(dyna, tok)
    R["alpha_stats"] = alpha_by_emotion(dyna, test_data, tok)
    R["ablations"] = ablations(dyna, test_data, tok)
    R["bootstrap"] = bootstrap_ppl_diff(static_ppl["per_example"], dyna_ppl["per_example"], CFG["bootstrap_n"])

    R["improvement_ppl_pct"] = (R["ppl_static"] - R["ppl_dynapersona"]) / R["ppl_static"] * 100
    json.dump(R, open(out_path, "w"), indent=2)
    th = R["train_dynapersona"]
    gg = th.get("gate_grad", [])
    log(f"seed {seed} DONE -> {out_path}")
    log(f"  static ppl {R['ppl_static']:.3f} | dynapersona ppl {R['ppl_dynapersona']:.3f} "
        f"| improvement {R['improvement_ppl_pct']:+.2f}% | alpha "
        f"{R['alpha_stats'].get('overall_mean', float('nan')):.3f}"
        f"+-{R['alpha_stats'].get('overall_std', float('nan')):.3f}")
    log(f"  GATE LEARNING: per-epoch alpha_mean {['%.3f'%x for x in th.get('alpha_mean', [])]} "
        f"alpha_std {['%.3f'%x for x in th.get('alpha_std', [])]} "
        f"| gate_grad(mean over dynamic steps) {np.mean(gg) if gg else float('nan'):.2e} "
        f"| nonzero_grad_steps {len(gg)}")
    del dyna; gc.collect(); torch.cuda.empty_cache()
    return R


def main():
    log(f"GPU: {torch.cuda.get_device_name(0)}  SMOKE={SMOKE}")
    build_data()
    all_res = []
    for seed in CFG["seeds"]:
        all_res.append(run_seed(seed))
    if len(all_res) > 1:
        agg = {}
        for key in ("ppl_static", "ppl_dynapersona", "improvement_ppl_pct"):
            vals = [r[key] for r in all_res]
            agg[key] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)), values=vals)
        json.dump(agg, open(f"{RES_DIR}/aggregate.json", "w"), indent=2)
        log(f"AGGREGATE: {agg}")
    log("ALL DONE.")


if __name__ == "__main__":
    main()
