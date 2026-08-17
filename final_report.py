import os, re, glob, json

results = []

def parse_run(logdir):
    ovr = os.path.join(logdir, ".hydra/overrides.yaml")
    console = os.path.join(logdir, "console.log")
    if not os.path.exists(ovr) or not os.path.exists(console):
        return None
    with open(ovr) as f:
        t = f.read()
    model = "TT" if "size12M_tt" in t else "BL"
    m = re.search(r"task=(\S+)", t)
    task = m.group(1).replace("dmc_","").replace("_"," ") if m else "?"
    m = re.search(r"seed=(\d+)", t)
    seed = int(m.group(1)) if m else 0
    m = re.search(r"steps=(\d+)", t)
    steps = int(m.group(1)) if m else 0
    with open(console) as f:
        lines = f.readlines()
    if len(lines) < 5: return None
    scores = [float(re.search(r"episode/score ([\d.]+)", l).group(1)) for l in lines if "episode/score" in l]
    dyns = [float(re.search(r"train/loss/dyn ([\d.]+)", l).group(1)) for l in lines if "train/loss/dyn" in l]
    reps = [float(re.search(r"train/loss/rep ([\d.]+)", l).group(1)) for l in lines if "train/loss/rep" in l]
    gates = [float(re.search(r"tt_global_gate ([\d.]+)", l).group(1)) for l in lines if "tt_global_gate" in l]
    if not scores: return None
    return {"model":model,"task":task,"seed":seed,"steps":steps,"steps_real":len(scores)*64,
            "eval":scores[-1],"best":max(scores),"median":sorted(scores)[len(scores)//2],
            "dyn":dyns[-1] if dyns else 0,"rep":reps[-1] if reps else 0,
            "gate":gates[-1] if gates else None,"n":len(scores)}

all_dirs = []
for base in ["logdir/2026-06-20/","logdir/2026-06-21/","logdir/2026-06-22/"]:
    if os.path.exists(base): all_dirs.extend(glob.glob(base+"*/"))
all_dirs.append("logdir_tt/")
for d in all_dirs:
    r = parse_run(d)
    if r: results.append(r)

# Dedup by (model, task, seed) — keep highest steps
dedup = {}
for r in results:
    key = (r["model"], r["task"], r["seed"])
    if key not in dedup or r["steps_real"] > dedup[key]["steps_real"]:
        dedup[key] = r

print("=" * 85)
print(f"{'Task':<18} {'Seed':>4} {'Step':>6} {'BL':>8} {'TT':>8} {'Delta':>8} {'Gate':>6} {'dyn/rep':>8} {'best_BL':>8} {'best_TT':>8}")
print("=" * 85)
for task in ["cartpole swingup", "cheetah run", "walker walk"]:
    for seed in [42, 123, 456]:
        bl = dedup.get(("BL", task, seed))
        tt = dedup.get(("TT", task, seed))
        if not bl and not tt: continue
        step = max((bl or {}).get("steps",0), (tt or {}).get("steps",0))
        bl_e = f'{bl["eval"]:.0f}' if bl else "MISSING"
        tt_e = f'{tt["eval"]:.0f}' if tt else "MISSING"
        bl_b = f'{bl["best"]:.0f}' if bl else ""
        tt_b = f'{tt["best"]:.0f}' if tt else ""
        delta = f'{((tt["eval"]-bl["eval"])/bl["eval"]*100):+.1f}%' if bl and tt and bl["eval"]>0 else "-"
        gate = f'{tt["gate"]:.2f}' if tt and tt["gate"] else "-"
        dr = f'{max(bl["dyn"] if bl else 0,tt["dyn"] if tt else 0):.1f}/{max(bl["rep"] if bl else 0,tt["rep"] if tt else 0):.1f}'
        print(f'{task:<18} {seed:>4} {step:>6} {bl_e:>8} {tt_e:>8} {delta:>8} {gate:>6} {dr:>8} {bl_b:>8} {tt_b:>8}')

# Walker 300k analysis
print("\n=== WALKER 300K (available only) ===")
for seed in [42, 123, 456]:
    bl = dedup.get(("BL", "walker walk", seed))
    tt = dedup.get(("TT", "walker walk", seed))
    bl_s = f'{bl["steps"]}k eval={bl["eval"]:.0f}' if bl else "MISSING"
    tt_s = f'{tt["steps"]}k eval={tt["eval"]:.0f}' if tt else "MISSING"
    print(f"  s{seed}: BL={bl_s}  TT={tt_s}")

# Gate stats
gates_all = [r["gate"] for r in dedup.values() if r["gate"]]
print(f"\n=== GATE ===")
print(f"  Range: {min(gates_all):.2f}-{max(gates_all):.2f}, Mean: {sum(gates_all)/len(gates_all):.2f}")

# KL comparison
print(f"\n=== KL STABILITY ===")
for task in ["cartpole swingup", "cheetah run", "walker walk"]:
    bl_dr = [f'{r["dyn"]:.1f}/{r["rep"]:.1f}' for k,r in dedup.items() if r["task"]==task and r["model"]=="BL"]
    tt_dr = [f'{r["dyn"]:.1f}/{r["rep"]:.1f}' for k,r in dedup.items() if r["task"]==task and r["model"]=="TT"]
    print(f"  {task:<18}: BL=[{', '.join(bl_dr)}]  TT=[{', '.join(tt_dr)}]")

