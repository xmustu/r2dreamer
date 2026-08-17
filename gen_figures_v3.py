import json, os, re, glob
import numpy as np
np.seterr(invalid='ignore')  # suppress NaN warnings from empty slices
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ===== Load data =====
with open("/home/zhengkai/r2dreamer/poster_data.json") as f:
    all_runs = json.load(f)

dedup = {}
for r in all_runs:
    key = (r["model"], r["task"], r["seed"])
    if key not in dedup or len(r["curve"]) > len(dedup[key]["curve"]):
        dedup[key] = r

walker_300k = {}
for k, r in list(dedup.items()):
    if r["task"] == "_walker_walk" and r["steps"] > 100000:
        walker_300k[k] = r
        del dedup[k]

style = {
    "BL": {"color": "#6B7280", "fill": "#D1D5DB", "lw": 2.5, "label": "Baseline RSSM"},
    "TT": {"color": "#2563EB", "fill": "#93C5FD", "lw": 2.5, "label": "Two-Timescale RSSM"},
}

TASK_LABELS = {
    "_cartpole_swingup": "Cartpole Swingup",
    "_cheetah_run": "Cheetah Run",
    "_walker_walk": "Walker Walk",
}

def compute_mean_std_curve(runs, step_bins):
    """Bin multiple runs' curves, return (centers, mean, std) or (None,None,None)."""
    centers = (step_bins[:-1] + step_bins[1:]) / 2
    binned = []
    for key, r in runs.items():
        pts = sorted(r["curve"], key=lambda x: x["step"])
        steps = np.array([p["step"] for p in pts])
        evals = np.array([p["eval"] for p in pts])
        _, uniq = np.unique(steps, return_index=True)
        steps, evals = steps[uniq], evals[uniq]
        if len(steps) < 2:
            continue
        binned.append(np.interp(centers, steps, evals))
    if len(binned) < 1:
        return None, None, None
    binned = np.array(binned)
    return centers, np.nanmean(binned, axis=0), np.nanstd(binned, axis=0)

# ===== Figure 1: Main Learning Curves (mean + shaded std) =====
fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))

for idx, task in enumerate(["_cartpole_swingup", "_cheetah_run", "_walker_walk"]):
    ax = axes[idx]
    bins = np.arange(0, 105000, 2000)

    for mk in ["BL", "TT"]:
        runs = {k: v for k, v in dedup.items() if v["task"] == task and k[0] == mk}
        if len(runs) == 0:
            continue
        ctr, mn, sd = compute_mean_std_curve(runs, bins)
        if ctr is None:
            continue
        s = style[mk]
        ax.plot(ctr, mn, color=s["color"], lw=s["lw"], label=s["label"])
        ax.fill_between(ctr, mn - sd, mn + sd, color=s["fill"], alpha=0.35)

        # Final annotation: mean of last eval across seeds
        final_vals = [r["curve"][-1]["eval"] for r in runs.values()]
        ax.annotate(f'{np.mean(final_vals):.0f}',
                    xy=(ctr[-1], mn[-1]), xytext=(10, 3),
                    textcoords="offset points", fontsize=10,
                    color=s["color"], fontweight="bold")

    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel("Episode Score", fontsize=12)
    ax.set_title(TASK_LABELS[task], fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.tick_params(labelsize=10)
    ax.set_xlim(0, 102000)

    if idx == 0:
        ax.legend(handles=[
            Line2D([0],[0], color=style["BL"]["color"], lw=2.5, label=style["BL"]["label"]),
            Line2D([0],[0], color=style["TT"]["color"], lw=2.5, label=style["TT"]["label"]),
        ], loc="lower right", fontsize=11, framealpha=0.9)

plt.tight_layout(pad=2.5)
plt.savefig("/home/zhengkai/r2dreamer/fig1_learning_curves.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig 1 saved")

# ===== Figure 2: Early Training Zoom =====
fig, ax = plt.subplots(figsize=(9, 5.5))
bins = np.arange(0, 55000, 1500)
task = "_cartpole_swingup"

for mk in ["BL", "TT"]:
    runs = {k: v for k, v in dedup.items() if v["task"] == task and k[0] == mk}
    if len(runs) == 0: continue
    ctr, mn, sd = compute_mean_std_curve(runs, bins)
    if ctr is None: continue
    s = style[mk]
    ax.plot(ctr, mn, color=s["color"], lw=3.0, label=s["label"])
    ax.fill_between(ctr, mn - sd, mn + sd, color=s["fill"], alpha=0.3)
    final_vals = [r["curve"][-1]["eval"] for r in runs.values()]
    ax.annotate(f'{np.mean(final_vals):.0f}',
                xy=(ctr[-1], mn[-1]), xytext=(10, 5),
                textcoords="offset points", fontsize=13,
                color=s["color"], fontweight="bold")

ax.set_xlabel("Environment Steps", fontsize=13)
ax.set_ylabel("Episode Score", fontsize=13)
ax.set_title("Early Training (Cartpole Swingup, N=3 seeds)", fontsize=15, fontweight="bold")
ax.legend(fontsize=12, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle=":")
ax.tick_params(labelsize=11)
ax.set_xlim(0, 52000)
ax.annotate("TT RSSM consistently\noutperforms baseline",
            xy=(12000, 170), fontsize=12, fontstyle="italic", color="#1D4ED8",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#DBEAFE", edgecolor="#2563EB", alpha=0.9))
plt.tight_layout()
plt.savefig("/home/zhengkai/r2dreamer/fig2_early_training.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig 2 saved")

# ===== Figure 3: KL Stability =====
console_kl = {}
for base in ["logdir/2026-06-20/", "logdir/2026-06-21/"]:
    for d in glob.glob(os.path.join("/home/zhengkai/r2dreamer", base, "*/")):
        ovr = os.path.join(d, ".hydra/overrides.yaml")
        console = os.path.join(d, "console.log")
        if not os.path.exists(ovr) or not os.path.exists(console): continue
        with open(ovr) as f: t = f.read()
        model = "TT" if "size12M_tt" in t else "BL"
        m = re.search(r"task=(\S+)", t); task = m.group(1) if m else "?"
        m = re.search(r"seed=(\d+)", t); seed = int(m.group(1)) if m else 0
        key = (model, task, seed)
        dyns, reps = [], []
        with open(console) as f:
            for line in f:
                m = re.search(r"train/loss/dyn ([\d.]+)", line)
                if m: dyns.append(float(m.group(1)))
                m = re.search(r"train/loss/rep ([\d.]+)", line)
                if m: reps.append(float(m.group(1)))
        if dyns and reps:
            console_kl[key] = {"dyn": np.mean(dyns[-20:]), "rep": np.mean(reps[-20:])}

# TT cartpole s42
d = "/home/zhengkai/r2dreamer/logdir_tt/"
console = os.path.join(d, "console.log")
if os.path.exists(console):
    dyns, reps = [], []
    with open(console) as f:
        for line in f:
            m = re.search(r"train/loss/dyn ([\d.]+)", line)
            if m: dyns.append(float(m.group(1)))
            m = re.search(r"train/loss/rep ([\d.]+)", line)
            if m: reps.append(float(m.group(1)))
    if dyns and reps:
        console_kl[("TT", "_cartpole_swingup", 42)] = {"dyn": np.mean(dyns[-20:]), "rep": np.mean(reps[-20:])}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
tasks_plot = ["_cartpole_swingup", "_cheetah_run", "_walker_walk"]
task_labels = ["Cartpole", "Cheetah", "Walker"]
x = np.arange(len(tasks_plot))
width = 0.35

for idx, metric in enumerate(["dyn", "rep"]):
    ax = axes[idx]
    bl_vals, tt_vals = [], []
    for t in tasks_plot:
        bl_k = [v[metric] for k,v in console_kl.items() if k[1]==t and k[0]=="BL"]
        tt_k = [v[metric] for k,v in console_kl.items() if k[1]==t and k[0]=="TT"]
        bl_vals.append(np.mean(bl_k) if bl_k else 0)
        tt_vals.append(np.mean(tt_k) if tt_k else 0)

    bars1 = ax.bar(x - width/2, bl_vals, width, color="#6B7280", label="Baseline", edgecolor="white", lw=0.5)
    bars2 = ax.bar(x + width/2, tt_vals, width, color="#2563EB", label="TT RSSM", edgecolor="white", lw=0.5)

    for bar, val in zip(bars1, bl_vals):
        if val > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                    f'{val:.2f}', ha="center", va="bottom", fontsize=10, color="#6B7280", fontweight="bold")
    for bar, val in zip(bars2, tt_vals):
        if val > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                    f'{val:.2f}', ha="center", va="bottom", fontsize=10, color="#2563EB", fontweight="bold")

    ml = "Dynamics Loss" if metric == "dyn" else "Representation Loss"
    ax.set_title(f"KL Stability — {ml}", fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(task_labels, fontsize=12)
    ax.set_ylabel("Loss Value", fontsize=12)
    ymax = max(max(bl_vals), max(tt_vals))
    ax.set_ylim(0, ymax * 1.45 if ymax > 0 else 2)
    ax.grid(True, axis="y", alpha=0.2, linestyle=":")
    if idx == 1:
        ax.legend(fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.savefig("/home/zhengkai/r2dreamer/fig3_kl_stability.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig 3 saved")

# ===== Figure 4: Walker 300k =====
fig, ax = plt.subplots(figsize=(9, 5.5))
for key, r in sorted(walker_300k.items()):
    s = style[r["model"]]
    pts = sorted(r["curve"], key=lambda x: x["step"])
    steps = np.array([p["step"] for p in pts])
    evals = np.array([p["eval"] for p in pts])
    ax.plot(steps, evals, color=s["color"], lw=2.0, alpha=0.85,
            label=s["label"] + f" (s{r['seed']})")
    ax.annotate(f'{evals[-1]:.0f}', xy=(steps[-1], evals[-1]),
                xytext=(8, 5), textcoords="offset points", fontsize=10,
                color=s["color"], fontweight="bold")
ax.set_xlabel("Environment Steps", fontsize=13)
ax.set_ylabel("Episode Score", fontsize=13)
ax.set_title("Walker Walk — 300k Extension", fontsize=15, fontweight="bold")
ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle=":")
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig("/home/zhengkai/r2dreamer/fig4_walker_300k.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig 4 saved")

# ===== Figure 5: Gate by task (mean ± std) =====
curves_by_task = {"cartpole": [], "cheetah": [], "walker": []}
for base in ["logdir/2026-06-20/", "logdir/2026-06-21/"]:
    for d in glob.glob(os.path.join("/home/zhengkai/r2dreamer", base, "*/")):
        console = os.path.join(d, "console.log")
        ovr = os.path.join(d, ".hydra/overrides.yaml")
        if not os.path.exists(console) or not os.path.exists(ovr): continue
        with open(ovr) as f:
            t = f.read()
        if "size12M_tt" not in t: continue
        m = re.search(r"task=dmc_(\w+)", t)
        tk = m.group(1).split("_")[0] if m else "?"
        if tk not in curves_by_task: continue
        steps, gates = [], []
        with open(console) as f:
            for line in f:
                m = re.search(r"\[(\d+)\] train/tt_global_gate ([\d.]+)", line)
                if m:
                    steps.append(int(m.group(1)))
                    gates.append(float(m.group(2)))
        if steps and len(steps) > 5:
            curves_by_task[tk].append({"steps": np.array(steps), "gates": np.array(gates)})

fig, ax = plt.subplots(figsize=(9, 4.5))
colors = {"cartpole": "#2563EB", "cheetah": "#059669", "walker": "#D97706"}
labels = {"cartpole": "Cartpole (3 runs)", "cheetah": "Cheetah (3 runs)", "walker": "Walker (3 runs)"}

for tk in ["cartpole", "cheetah", "walker"]:
    curves = curves_by_task[tk]
    if not curves: continue
    max_s = max(c["steps"].max() for c in curves)
    common = np.arange(0, min(max_s, 100000), 5000)
    interp = np.array([np.interp(common, c["steps"], c["gates"]) for c in curves])
    mn = np.nanmean(interp, axis=0)
    sd = np.nanstd(interp, axis=0)
    ax.plot(common, mn, color=colors[tk], lw=2.0, label=labels[tk])
    ax.fill_between(common, mn - sd, mn + sd, color=colors[tk], alpha=0.2)

ax.axhline(y=0.01, color="#DC2626", ls="--", lw=1, alpha=0.4)
ax.axhline(y=0.50, color="#DC2626", ls="--", lw=1, alpha=0.4)
ax.text(ax.get_xlim()[1]*0.78, 0.52, "Upper bound 0.50", fontsize=9, color="#DC2626", alpha=0.6, fontstyle="italic")
ax.text(ax.get_xlim()[1]*0.78, 0.07, "Lower bound 0.01", fontsize=9, color="#DC2626", alpha=0.6, fontstyle="italic")
ax.set_xlabel("Environment Steps", fontsize=13)
ax.set_ylabel("Global Gate Value", fontsize=13)
ax.set_title("Global Gate by Task (Mean ± Std)", fontsize=15, fontweight="bold")
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle=":")
ax.set_ylim(-0.05, 0.70)
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig("/home/zhengkai/r2dreamer/fig5_gate_behavior.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig 5 saved")

print("\n=== Done ===")
