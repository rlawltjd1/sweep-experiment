#!/usr/bin/env python
"""
Week 2 sweep — 학교 서버용 standalone 스크립트.
- 체크포인트/resume: 이미 끝난 (vector, layer, alpha)는 건너뛰고 이어서 돈다.
- 1주차 산출물(probe.pkl, b0_baseline.json, fn_subset_*.npy, 평가셋)을 그대로 사용.
- transformers 버전에 따라 layer 출력이 tuple/tensor 둘 다 가능 → hook이 둘 다 처리.

사용 예:
  python week2_sweep.py --batch 64 --all-layers
  python week2_sweep.py --eval v1 --alphas 1,2          # 일부만
"""
import os, json, re, time, argparse
from collections import Counter
import numpy as np, pandas as pd, torch
import joblib
from tqdm.auto import tqdm

# ─────────────────────────── 설정 ───────────────────────────
P = argparse.ArgumentParser()
P.add_argument("--model", default="meta-llama/Llama-3.2-3B")
P.add_argument("--out",   default="results")
P.add_argument("--data",  default="data/eval")
P.add_argument("--src",   default="src/eval")
P.add_argument("--cell-c", default="cell_c_test_final.csv")
P.add_argument("--cell-b", default="cell_bbb_domain_v10_256_revised.csv")
P.add_argument("--batch", type=int, default=32)
P.add_argument("--max-len", type=int, default=128)
P.add_argument("--alphas", default="0.5,1,2,4")
P.add_argument("--all-layers", action="store_true", help="0~28 전체 (없으면 우선순위 17개)")
P.add_argument("--eval", choices=["both","latent","tg"], default="both")
P.add_argument("--no-fine", action="store_true")
P.add_argument("--seed", type=int, default=42)
args = P.parse_args()

PRIORITY_LAYERS = [4,5,9,10,11,13,14,18,19,20,21,22,23,24,25,26,27]
ALPHAS = [float(a) for a in args.alphas.split(",")]
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.out, exist_ok=True)
print(f"[cfg] device={device} batch={args.batch} alphas={ALPHAS} all_layers={args.all_layers}")

# ─────────────────────────── 모델 ───────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
model.config.output_hidden_states = False
N_LAYERS, HIDDEN = model.config.num_hidden_layers, model.config.hidden_size
LAYERS = list(range(N_LAYERS+1)) if args.all_layers else PRIORITY_LAYERS
print(f"[model] layers={N_LAYERS} hidden={HIDDEN} sweep_layers={len(LAYERS)}")

# ─────────────────────────── 1주차 자산 ───────────────────────────
import sys; sys.path.insert(0, args.src)
from metrics import evaluate, evaluate_by_group, HATE

pb = joblib.load(os.path.join(args.out, "probe.pkl")); scaler, clf = pb["scaler"], pb["clf"]
b0 = json.load(open(os.path.join(args.out, "b0_baseline.json")))

b0_latent = b0["results"]["eval_latent_v2"]
b0_tg = b0["results"]["eval_toxigen_v1"]

def load_eval(path, group=False):
    df = pd.read_csv(path)
    if df["label"].dtype == object:
        df["label"] = df["label"].astype(str).str.strip().str.lower().map({"hate":1,"non-hate":0})
    df["label"] = df["label"].astype(int)
    g = df["target_group"].tolist() if (group and "target_group" in df) else None
    return df["text"].tolist(), df["label"].to_numpy(), g

ev_texts, ev_labels, _         = load_eval(os.path.join(args.data,"eval_latent_v2.csv"))
tg_texts, tg_labels, tg_groups = load_eval(os.path.join(args.data,"eval_toxigen_v1.csv"), group=True)
fn_latent = np.load(os.path.join(args.out, "fn_subset_eval_latent_v2.npy"))
fn_tg = np.load(os.path.join(args.out, "fn_subset_eval_toxigen_v1.npy"))

print(f"[data] eval_latent_v2={len(ev_labels)} (FN {len(fn_latent)}) | toxigen={len(tg_labels)} (FN {len(fn_tg)})")
# ─────────────────────────── 공용 함수 ───────────────────────────
class SteeringHook:
    def __init__(self, vec, alpha):
        self.v = torch.tensor(vec, dtype=torch.float16, device=device); self.a = float(alpha); self.h=None
    def _fn(self, m, i, o):
        if isinstance(o, tuple):
            o[0][:, -1, :] = o[0][:, -1, :] + self.a * self.v; return o
        o[:, -1, :] = o[:, -1, :] + self.a * self.v; return o
    def attach(self, layer): self.h = layer.register_forward_hook(self._fn); return self
    def detach(self):
        if self.h is not None: self.h.remove(); self.h=None

def norm(t): return re.sub(r"\s+"," ", re.sub(r"[^a-z0-9 ]","", str(t).lower())).strip()

def prebatch(texts):
    texts = [str(t) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i])); inv = np.argsort(order)
    batches = []
    for s in range(0, len(order), args.batch):
        idx = order[s:s+args.batch]
        batches.append(tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                           truncation=True, max_length=args.max_len))
    return batches, inv

@torch.inference_mode()
def hidden_from_batches(batches, inv):
    out = []
    for enc in batches:
        enc = {k: v.to(device) for k, v in enc.items()}
        h = model.model(**enc, use_cache=False, output_hidden_states=False).last_hidden_state[:, -1, :]
        out.append(h.float().cpu().numpy())
    return np.concatenate(out, 0)[inv]

def recovery_rate(pred, fn_idx):
    return 0.0 if len(fn_idx)==0 else float((pred[fn_idx]==HATE).mean())

@torch.inference_mode()
def extract_all_layers(texts):
    out=[]
    for s in range(0, len(texts), args.batch):
        enc = tok([str(t) for t in texts[s:s+args.batch]], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_len).to(device)
        hs = model.model(**enc, use_cache=False, output_hidden_states=True).hidden_states
        out.append(np.stack([h[:, -1, :].float().cpu().numpy() for h in hs], axis=1))
    return np.concatenate(out, 0)

def unit_per_layer(d): return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)

# ─────────────────────────── 벡터 (있으면 로드, 없으면 생성) ───────────────────────────
def build_vectors():
    paths = {n: os.path.join(args.out, f"{n}.npy") for n in ["v_AB","v_AC","v_random","v_harm"]}
    if all(os.path.exists(p) for p in paths.values()):
        print("[vec] 기존 벡터 로드"); return {n: np.load(p) for n,p in paths.items()}
    print("[vec] 벡터 생성")
    dfc = pd.read_csv(args.cell_c); dfb = pd.read_csv(args.cell_b)
    tri = (dfc[["idx","cell_a","cell_c_modified"]].rename(columns={"cell_a":"A","cell_c_modified":"C"})
           .merge(dfb[["text_clean","generated_text"]].rename(columns={"text_clean":"A","generated_text":"B"}), on="A", how="inner")
           .dropna(subset=["A","B","C"]))
    tri = tri[(tri["B"].astype(str).str.len()>0)&(tri["C"].astype(str).str.len()>0)]
    hA, hB, hC = extract_all_layers(tri["A"].tolist()), extract_all_layers(tri["B"].tolist()), extract_all_layers(tri["C"].tolist())
    v_AB = unit_per_layer(hA.mean(0)-hB.mean(0)); v_AC = unit_per_layer(hA.mean(0)-hC.mean(0))
    rng = np.random.RandomState(args.seed); v_random = unit_per_layer(rng.randn(N_LAYERS+1, HIDDEN))
    # v_harm (last-token 재추출)
    from datasets import load_dataset
    guard = set(map(norm, tri["A"])) | set(map(norm, tri["C"]))
    _sp = os.path.join(args.data, "sanity_hatexplain.csv")   # sanity held-out도 v_harm에서 제외
    if os.path.exists(_sp):
        guard |= set(map(norm, pd.read_csv(_sp)["text"]))
    try: hx = load_dataset("Hate-speech-CNERG/hatexplain", split="train", trust_remote_code=True)
    except TypeError: hx = load_dataset("Hate-speech-CNERG/hatexplain", split="train")
    def maj(L): c=Counter(L); top,n=c.most_common(1)[0]; return -1 if list(c.values()).count(n)>1 else top
    harm, safe = [], []
    for ex in hx:
        t = " ".join(ex["post_tokens"])
        if norm(t) in guard: continue
        l = maj(ex["annotators"]["label"])     # 0 hate / 1 normal / 2 offensive
        if   l == 0: harm.append(t)
        elif l == 1: safe.append(t)
    r2=np.random.RandomState(args.seed); r2.shuffle(harm); r2.shuffle(safe); harm,safe=harm[:300],safe[:300]
    v_harm = unit_per_layer(extract_all_layers(harm).mean(0) - extract_all_layers(safe).mean(0))
    V = {"v_AB":v_AB,"v_AC":v_AC,"v_random":v_random,"v_harm":v_harm}
    for n,v in V.items(): np.save(paths[n], v)
    return V

VECTORS = build_vectors()

# ─────────────────────────── 체크포인트 sweep ───────────────────────────
def sweep_one(vec, L, a, batches, inv, labels, fn_idx):
    hk = SteeringHook(vec[L], a).attach(model.model.layers[L])
    try: X = hidden_from_batches(batches, inv)
    finally: hk.detach()
    pred = clf.predict(scaler.transform(X)); m = evaluate(pred, labels)
    return m, recovery_rate(pred, fn_idx), pred

def done_keys(path):
    if not os.path.exists(path): return set()
    d = pd.read_csv(path)
    return {(r.vector, int(r.layer), float(r.alpha)) for r in d.itertuples()}

def append_row(path, row):
    hdr = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=hdr, index=False)

def run(tag, texts, labels, fn_idx, b0m, vectors, layers, alphas, fine=False):
    path = os.path.join(args.out, f"sweep_{'fine' if fine else 'coarse'}_{tag}.csv")
    done = done_keys(path)
    batches, inv = prebatch(texts)
    todo = [(n,L,a) for n in vectors for L in layers for a in alphas if (n,int(L),float(a)) not in done]
    print(f"[{tag}{'/fine' if fine else ''}] todo {len(todo)} / done {len(done)} -> {path}")
    for n,L,a in tqdm(todo, desc=f"sweep[{tag}{'/F' if fine else ''}]"):
        m, fnrec, _ = sweep_one(vectors[n], L, a, batches, inv, labels, fn_idx)
        append_row(path, dict(vector=n, layer=L, alpha=a, macro_f1=m["macro_f1"],
                              hate_recall=m["hate_recall"], fn_recovery=fnrec,
                              d_f1=m["macro_f1"]-b0m["macro_f1"]))
    return path

# coarse
targets = []
if args.eval in ("both", "latent"):
    targets.append(("eval_latent_v2", ev_texts, ev_labels, fn_latent, b0_latent))
if args.eval in ("both", "tg"):
    targets.append(("eval_toxigen_v1", tg_texts, tg_labels, fn_tg, b0_tg))

t0 = time.time()
for tag, txt, lab, fn, b0m in targets:
    cpath = run(tag, txt, lab, fn, b0m, VECTORS, LAYERS, ALPHAS, fine=False)
    # fine: v_AB, v_AC best 주변
    if not args.no_fine:
        dfc = pd.read_csv(cpath)
        fine_vecs = {}; fine_layers=set(); fine_alphas=set()
        for n in ["v_AB","v_AC"]:
            b = dfc[dfc.vector==n].loc[dfc[dfc.vector==n].macro_f1.idxmax()]
            L0,a0 = int(b.layer), float(b.alpha)
            for L in range(L0-2, L0+3):
                if 0<=L<=N_LAYERS: fine_layers.add(L)
            for d in (-0.5,-0.25,0,0.25,0.5):
                if a0+d>0: fine_alphas.add(round(a0+d,2))
            fine_alphas.add(-a0)
            fine_vecs[n]=VECTORS[n]
        run(tag, txt, lab, fn, b0m, fine_vecs, sorted(fine_layers), sorted(fine_alphas), fine=True)
print(f"[done] {time.time()-t0:.0f}s")

# ─────────────────────────── 메인 표 ───────────────────────────
def main_table(tag, b0m):
    cp = os.path.join(args.out, f"sweep_coarse_{tag}.csv")
    fp = os.path.join(args.out, f"sweep_fine_{tag}.csv")
    dfc = pd.read_csv(cp); dff = pd.read_csv(fp) if os.path.exists(fp) else None
    label = {"v_random":"B1 Random","v_harm":"B2 v_harm","v_AB":"E1 v_AB (target)","v_AC":"E2 v_AC (cue)"}
    rows = [dict(Setup="B0 No steering", layer="—", alpha="—",
                 macro_f1=round(b0m["macro_f1"],4), hate_recall=round(b0m["hate_recall"],4),
                 fn_recovery="—", d_f1=0.0)]
    for n in ["v_random","v_harm","v_AB","v_AC"]:
        pool = pd.concat([dfc[dfc.vector==n]] + ([dff[dff.vector==n]] if dff is not None else []), ignore_index=True)
        b = pool.loc[pool.macro_f1.idxmax()]
        rows.append(dict(Setup=label[n], layer=int(b.layer), alpha=b.alpha,
                         macro_f1=round(b.macro_f1,4), hate_recall=round(b.hate_recall,4),
                         fn_recovery=round(b.fn_recovery,4), d_f1=round(b.d_f1,4)))
    t = pd.DataFrame(rows); t.to_csv(os.path.join(args.out, f"main_table_{tag}.csv"), index=False)
    print(f"\n===== MAIN TABLE [{tag}] =====\n{t.to_string(index=False)}")

for tag, _, _, _, b0m in targets:
    main_table(tag, b0m)
print("[all done] 그래프는 결과 csv로 노트북에서 그리세요.")