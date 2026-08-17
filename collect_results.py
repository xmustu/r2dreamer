import os, re, glob

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
    if len(lines) < 5:
        return None
    scores = []
    dyns, reps, gates = [], [], []
    for l in lines:
        m = re.search(r"episode/score ([\d.]+)", l)
        if m: scores.append(float(m.group(1)))
        m = re.search(r"train/loss/dyn ([\d.]+)", l)
        if m: dyns.append(float(m.group(1)))
        m = re.search(r"train/loss/rep ([\d.]+)", l)
        if m: reps.append(float(m.group(1)))
        m = re.search(r"tt_global_gate ([\d.]+)", l)
        if m: gates.append(float(m.group(1)))
    if not scores:
        return None
    return {
        "model": model, "task": task, "seed": seed, "steps": steps,
        "eval": scores[-1], "best": max(scores),
        "dyn": dyns[-1] if dyns else 0, "rep": reps[-1] if reps else 0,
        "gate": gates[-1] if gates else None, "n": len(scores),
    }

all_dirs = []
for base in ["logdir/2026-06-20/", "logdir/2026-06-21/", "logdir/2026-06-22/"]:
    if os.path.exists(base):
        all_dirs.extend(glob.glob(base + "*/"))
all_dirs.append("logdir_tt/")

all_results = []
for d in all_dirs:
    r = parse_run(d)
    if r:
        all_results.append(r)

print("all_results count:", len(all_results))
for r in sorted(all_results, key=lambda x: (x["task"], x["seed"], x["model"])):
    print("  {:<18} s{} {:>6} {:>6} {}".format(r["task"], r["seed"], r["model"], int(r["eval"]), "gate="+str(r["gate"]) if r["gate"] else ""))

# Dedup by (model, task, seed) — keep highest step count
dedup = {}
for r in all_results:
    key = (r["model"], r["task"], r["seed"])
    if key not in dedup or r["steps"] > dedup[key]["steps"]:
        dedup[key] = r

print("\n=== DEDUPED TABLE ===")
header = "{:<18} {:>4} {:>6} {:>7} {:>7} {:>8} {:>5} {:>8}".format(
    "Task", "Seed", "Steps", "BL", "TT", "Delta", "Gate", "dyn/rep")
print(header)
print("=" * len(header))
for task in ["cartpole swingup", "cheetah run", "walker walk"]:
    for seed in [42, 123, 456]:
        bl = dedup.get(("BL", task, seed))
        tt = dedup.get(("TT", task, seed))
        if not bl and not tt:
            continue
        steps = max((bl or {}).get("steps", 0), (tt or {}).get("steps", 0))
        bl_e = "{:.0f}".format(bl["eval"]) if bl else "N/A"
        tt_e = "{:.0f}".format(tt["eval"]) if tt else "N/A"
        if bl and tt and bl["eval"] > 0:
            pct = (tt["eval"] - bl["eval"]) / bl["eval"] * 100
            delta = "{:+.1f}%".format(pct)
        else:
            delta = "-"
        gate = "{:.2f}".format(tt["gate"]) if tt and tt["gate"] else "-"
        dr = "{:.1f}/{:.1f}".format(
            max(bl["dyn"] if bl else 0, tt["dyn"] if tt else 0),
            max(bl["rep"] if bl else 0, tt["rep"] if tt else 0))
        print("{:<18} {:>4} {:>6} {:>7} {:>7} {:>8} {:>5} {:>8}".format(
            task, seed, steps, bl_e, tt_e, delta, gate, dr))

print("\n=== CONVERGED WM (dyn <= 1.3) ===")
converged = [r for r in dedup.values() if r["dyn"] <= 1.3]
pairs = {}
for r in converged:
    key = (r["task"], r["seed"])
    pairs.setdefault(key, {})[r["model"]] = r["eval"]
diffs = []
for key, vals in sorted(pairs.items()):
    if "BL" in vals and "TT" in vals:
        d = vals["TT"] - vals["BL"]
        diffs.append(d)
        pct = d / vals["BL"] * 100
        print("  {:<18} s{}: BL={:.0f} TT={:.0f}  Delta={:+.0f} ({:+.1f}%)".format(
            key[0], key[1], vals["BL"], vals["TT"], d, pct))
if diffs:
    mean_d = sum(diffs) / len(diffs)
    mean_pct = sum(d / max(abs(vals["BL"]), 0.1) for key, vals in pairs.items() if "BL" in vals and "TT" in vals for d in [vals["TT"] - vals["BL"]]) / len(diffs) * 100
    wins = sum(1 for d in diffs if d > 0)
    print("  Mean Delta: {:+.1f}  Win rate: {}/{}".format(mean_d, wins, len(diffs)))

print("\n=== WALKER 300K vs 100K COMPARISON ===")
for seed in [42, 123, 456]:
    bl100 = next((r for r in all_results if r["model"]=="BL" and r["task"]=="walker walk" and r["seed"]==seed and r["steps"]<=100000), None)
    bl300 = next((r for r in all_results if r["model"]=="BL" and r["task"]=="walker walk" and r["seed"]==seed and r["steps"]>=300000), None)
    tt100 = next((r for r in all_results if r["model"]=="TT" and r["task"]=="walker walk" and r["seed"]==seed and r["steps"]<=100000), None)
    tt300 = next((r for r in all_results if r["model"]=="TT" and r["task"]=="walker walk" and r["seed"]==seed and r["steps"]>=300000), None)
    line = "  s{}  100k: BL={}  TT={}  |  300k: BL={}  TT={}".format(
        seed,
        "{:.0f}".format(bl100["eval"]) if bl100 else "N/A",
        "{:.0f}".format(tt100["eval"]) if tt100 else "N/A",
        "{:.0f}".format(bl300["eval"]) if bl300 else "N/A",
        "{:.0f}".format(tt300["eval"]) if tt300 else "N/A")
    print(line)

# Gate summary
print("\n=== GATE VALUES (all TT runs) ===")
gate_vals = [r["gate"] for r in dedup.values() if r["gate"] is not None]
if gate_vals:
    print("  Range: {:.2f} - {:.2f}, Mean: {:.2f}".format(
        min(gate_vals), max(gate_vals), sum(gate_vals)/len(gate_vals)))
    print("  All: {}".format(", ".join("{:.2f}".format(g) for g in sorted(gate_vals))))
