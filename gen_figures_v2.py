import json, os, re, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ===== Load data =====
with open("/home/zhengkai/r2dreamer/poster_data.json") as f:
    all_runs = json.load(f)

# Dedup: for each (model, task, seed), keep the run with most datapoints
dedup = {}
for r in all_runs:
    key = (r["model"], r["task"], r["seed"])
    if key not in dedup or len(r["curve"]) > len(dedup[key]["curve"]):
        dedup[key] = r

# Separate 300k walker runs
walker_300k = {}
for k, r in dedup.items():
    if r["task"] == "_walker_walk" and r["seed"] in [123, 456]:
        walker_300k[k] = r

# Remove 300k from dedup for main figure (keep only 100k)
dedup_100k = {k: v for k, v in dedup.items() if v["steps"] <= 100000}

style = {"BL": {"color": "#6B7280", "ls": "-", "lw": 2.0, "label": "Baseline RSSM"},
         "TT": {"color": "#2563EB", "ls": "-", "lw": 2.0, "label": "Two-Timescale RSSM"}}

N_SKIP = 5  # plot every 5th point to reduce density

def smooth_curve(steps, evals, window=5):
    """Moving average smoothing."""
    if len(evals) < window * 2:
        return steps, evals
    smoothed = np.convolve(evals, np.ones(window)/window, mode='valid')
    # Adjust steps to match smoothed length
    offset = (len(evals) - len(smoothed)) // 2
    return steps[offset:offset+len(smoothed)], smoothed

def subsample_curve(steps, evals, skip=N_SKIP):
    """Take every Nth point."""
    return steps[::skip], evals[::skip]

def plot_task_curves(ax, task, title, task_data, show_legend=False):
    """Plot learning curves for a task, subsampled + smoothed."""
    for key, r in sorted(task_data.items()):
        if r["task"] != task:
            continue
        s = style[r["model"]]
        pts = sorted(r["curve"], key=lambda x: x["step"])
        steps = np.array([p["step"] for p in pts])
        evals = np.array([p["eval"] for p in pts])
        # Subsample + smooth
        steps, evals = subsample_curve(steps, evals, skip=N_SKIP)
        if len(steps) > 10:
            steps, evals = smooth_curve(steps, evals, window=3)
        ax.plot(steps, evals, color=s["color"], ls=s["ls"], lw=s["lw"], alpha=0.85)
        # Annotate final score
        last_eval = r["curve"][-1]["eval"]
        ax.annotate(f'{last_eval:.0f}',
                     xy=(r["curve"][-1]["step"], last_eval),
                     xytext=(8, 3), textcoords="offset points", fontsize=9,
                     color=s["color"], fontweight="bold")

    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel("Episode Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.tick_params(labelsize=10)
    ax.set_xlim(0, 105000)

    if show_legend:
        legend_elements = [
            Line2D([0], [0], color=style["BL"]["color"], lw=2.5, label=style["BL"]["label"]),
            Line2D([0], [0], color=style["TT"]["color"], lw=2.5, label=style["TT"]["label"]),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=11, framealpha=0.9)

# ===== Figure 1: Main Learning Curves (3-panel, clean) =====
fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))

plot_task_curves(axes[0], "_cartpole_swingup", "Cartpole Swingup", dedup_100k, show_legend=True)
plot_task_curves(axes[1], "_cheetah_run", "Cheetah Run", dedup_100k)
plot_task_curves(axes[2], "_walker_walk", "Walker Walk", dedup_100k)

plt.tight_layout(pad=2.5)
plt.savefig("/home/zhengkai/r2dreamer/fig1_learning_curves.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig 1 saved: learning curves (subsampled + smoothed)")

# ===== Figure 2: Early Training Zoom (Cartpole s42, clean) =====
fig, ax = plt.subplots(figsize=(9, 5.5))
for key, r in sorted(dedup_100k.items()):
    if r["task"] != "_cartpole_swingup" or r["seed"] != 42:
        continue
    s = style[r["model"]]
    pts = sorted(r["curve"], key=lambda x: x["step"])
    steps = np.array([p["step"] for p in pts])
    evals = np.array([p["eval"] for p in pts])
    steps, evals = subsample_curve(steps, evals, skip=3)
    steps, evals = smooth_curve(steps, evals, window=3)
    ax.plot(steps, evals, color=s["color"], ls=s["ls"], lw=3.0, label=s["label"])
    last = r["curve"][-1]
    ax.annotate(f'{last["eval"]:.0f}',
                xy=(last["step"], last["eval"]),
                xytext=(8, 5), textcoords="offset points", fontsize=12,
                color=s["color"], fontweight="bold")

ax.set_xlabel("Environment Steps", fontsize=13)
ax.set_ylabel("Episode Score", fontsize=13)
ax.set_title("Early Training Comparison (Cartpole, Seed 42)", fontsize=15, fontweight="bold")
ax.legend(fontsize=12, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.25, linestyle=":")
ax.tick_params(labelsize=11)
ax.set_xlim(0, 55000)

# Highlight annotation
ax.annotate("TT RSSM +24% at 50k steps",
            xy=(25000, 250), fontsize=12, fontstyle="italic", color="#1D4ED8",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#DBEAFE",
                     edgecolor="#2563EB", alpha=0.9))

plt.tight_layout()
plt.savefig("/home/zhengkai/r2dreamer/fig2_early_training.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig 2 saved: early training zoom (clean)")

# ===== Figure 3: KL Stability (unchanged) =====
console_kl = {}
for base in ["logdir/2026-06-20/", "logdir/2026-06-21/"]:
    for d in glob.glob(os.path.join("/home/zhengkai/r2dreamer", base, "*/")):
        ovr = os.path.join(d, ".hydra/overrides.yaml")
        console = os.path.join(d, "console.log")
        if not os.path.exists(ovr) or not os.path.exists(console):
            continue
        with open(ovr) as f:
            t = f.read()
        model = "TT" if "size12M_tt" in t else "BL"
        m = re.search(r"task=(\S+)", t)
        task = m.group(1) if m else "?"
        m = re.search(r"seed=(\d+)", t)
        seed = int(m.group(1)) if m else 0
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

# TT cartpole s42 from logdir_tt
for d in ["/home/zhengkai/r2dreamer/logdir_tt/"]:
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
            console_kl[("TT", "_cartpole_swingup", 42)] = {
                "dyn": np.mean(dyns[-20:]), "rep": np.mean(reps[-20:])}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
tasks_plot = ["_cartpole_swingup", "_cheetah_run", "_walker_walk"]
task_labels = ["Cartpole", "Cheetah", "Walker"]
x = np.arange(len(tasks_plot))
width = 0.35

for idx, metric in enumerate(["dyn", "rep"]):
    ax = axes[idx]
    bl_vals = []
    tt_vals = []
    for task_key in tasks_plot:
        bl_k = [v for k, v in console_kl.items() if k[1] == task_key and k[0] == "BL"]
        tt_k = [v for k, v in console_kl.items() if k[1] == task_key and k[0] == "TT"]
        bl_vals.append(np.mean([v[metric] for v in bl_k]) if bl_k else 0)
        tt_vals.append(np.mean([v[metric] for v in tt_k]) if tt_k else 0)

    bars1 = ax.bar(x - width/2, bl_vals, width, color="#6B7280", label="Baseline", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, tt_vals, width, color="#2563EB", label="TT RSSM", edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars1, bl_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.2f}', ha="center", va="bottom", fontsize=10, color="#6B7280", fontweight="bold")
    for bar, val in zip(bars2, tt_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.2f}', ha="center", va="bottom", fontsize=10, color="#2563EB", fontweight="bold")

    metric_label = "Dynamics Loss" if metric == "dyn" else "Representation Loss"
    ax.set_title(f"KL Stability — {metric_label}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=12)
    ax.set_ylabel("Loss Value", fontsize=12)
    ymax = max(max(bl_vals), max(tt_vals)) * 1.45
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    if idx == 1:
        ax.legend(fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.savefig("/home/zhengkai/r2dreamer/fig3_kl_stability.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig 3 saved: KL stability")

# ===== Figure 4: Walker 300k (cleaned) =====
fig, ax = plt.subplots(figsize=(9, 5.5))

# 300k curves
for key, r in sorted(walker_300k.items()):
    s = style[r["model"]]
    pts = sorted(r["curve"], key=lambda x: x["step"])
    steps = np.array([p["step"] for p in pts])
    evals = np.array([p["eval"] for p in pts])
    steps, evals = subsample_curve(steps, evals, skip=8)
    steps, evals = smooth_curve(steps, evals, window=5)
    suffix = f" (s{r['seed']})"
    label = s["label"] + suffix
    ax.plot(steps, evals, color=s["color"], ls=s["ls"], lw=2.5, label=label, alpha=0.9)
    last = r["curve"][-1]
    ax.annotate(f'{last["eval"]:.0f}',
                xy=(last["step"], last["eval"]),
                xytext=(8, 5), textcoords="offset points", fontsize=10,
                color=s["color"], fontweight="bold")

# 100k faint curves for reference
for key, r in sorted(dedup_100k.items()):
    if r["task"] != "_walker_walk":
        continue
    s = style[r["model"]]
    pts = sorted(r["curve"], key=lambda x: x["step"])
    steps = np.array([p["step"] for p in pts])
    evals = np.array([p["eval"] for p in pts])
    steps, evals = subsample_curve(steps, evals, skip=N_SKIP)
    ax.plot(steps, evals, color=s["color"], ls=":", lw=1.0, alpha=0.3)

ax.axvline(x=100000, color="gray", ls="--", lw=1, alpha=0.5)
ax.text(102000, ax.get_ylim()[1]*0.92, "Extended to 300k", fontsize=10, color="gray", alpha=0.7, fontstyle="italic")

ax.set_xlabel("Environment Steps", fontsize=13)
ax.set_ylabel("Episode Score", fontsize=13)
ax.set_title("Walker Walk — 300k Extension", fontsize=15, fontweight="bold")
ax.legend(fontsize=11, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.25, linestyle=":")
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig("/home/zhengkai/r2dreamer/fig4_walker_300k.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig 4 saved: walker 300k (cleaned)")

# ===== Figure 5: Gate behavior (cleaned) =====
fig, ax = plt.subplots(figsize=(9, 4.5))
for base in ["logdir/2026-06-20/", "logdir/2026-06-21/"]:
    for d in glob.glob(os.path.join("/home/zhengkai/r2dreamer", base, "*/")):
        console = os.path.join(d, "console.log")
        ovr = os.path.join(d, ".hydra/overrides.yaml")
        if not os.path.exists(console) or not os.path.exists(ovr):
            continue
        with open(ovr) as f:
            t = f.read()
        if "size12M_tt" not in t:
            continue
        m = re.search(r"task=(\S+)", t)
        task = m.group(1).replace("dmc_","").replace("_"," ") if m else "?"
        m = re.search(r"seed=(\d+)", t)
        seed = int(m.group(1)) if m else 0
        steps, gates = [], []
        with open(console) as f:
            for line in f:
                m = re.search(r"\[(\d+)\] train/tt_global_gate ([\d.]+)", line)
                if m:
                    steps.append(int(m.group(1)))
                    gates.append(float(m.group(2)))
        if steps:
            label = f"{task} s{seed}"
            steps_a = np.array(steps)
            gates_a = np.array(gates)
            steps_a, gates_a = subsample_curve(steps_a, gates_a, skip=3)
            ax.plot(steps_a, gates_a, lw=1.5, label=label, alpha=0.8)

# Gate bounds
ax.axhline(y=0.01, color="#DC2626", ls="--", lw=1, alpha=0.4)
ax.axhline(y=0.50, color="#DC2626", ls="--", lw=1, alpha=0.4)
ax.text(ax.get_xlim()[1]*0.78, 0.52, "Upper bound 0.50", fontsize=9, color="#DC2626", alpha=0.6, fontstyle="italic")
ax.text(ax.get_xlim()[1]*0.78, 0.07, "Lower bound 0.01", fontsize=9, color="#DC2626", alpha=0.6, fontstyle="italic")

ax.set_xlabel("Environment Steps", fontsize=13)
ax.set_ylabel("Global Gate Value", fontsize=13)
ax.set_title("Learned Global Gate — All TT Runs", fontsize=15, fontweight="bold")
ax.legend(fontsize=9, ncol=2, loc="upper right", framealpha=0.9)
ax.grid(True, alpha=0.25, linestyle=":")
ax.set_ylim(-0.05, 0.70)
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig("/home/zhengkai/r2dreamer/fig5_gate_behavior.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig 5 saved: gate behavior (cleaned)")

print("\n=== All figures regenerated with subsampling + smoothing ===")
print("  fig1_learning_curves.png  - every 5th point + moving avg")
print("  fig2_early_training.png   - every 3rd point + moving avg")
print("  fig3_kl_stability.png     - bar chart (unchanged)")
print("  fig4_walker_300k.png      - every 8th point + moving avg")
print("  fig5_gate_behavior.png    - every 3rd point")
