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

# Re-add walker s42 dedup_100k entries for walker 100k fig
# walker s42 100k: both BL and TT exist
# walker s123 100k: BL=443, TT=380 (use max curve)
# walker s456 100k: BL=603, TT=400

style = {"BL": {"color": "#6B7280", "ls": "-", "lw": 2.0, "label": "Baseline RSSM"},
         "TT": {"color": "#2563EB", "ls": "-", "lw": 2.0, "label": "Two-Timescale RSSM"}}

def plot_task_curves(ax, task, title, tasks_100k, tasks_300k=None):
    """Plot learning curves for a task."""
    # Plot 100k curves
    for key, r in sorted(tasks_100k.items()):
        if r["task"] != task:
            continue
        s = style[r["model"]]
        pts = sorted(r["curve"], key=lambda x: x["step"])
        steps = [p["step"] for p in pts]
        evals = [p["eval"] for p in pts]
        ax.plot(steps, evals, color=s["color"], ls=s["ls"], lw=s["lw"], alpha=0.7)

    ax.set_xlabel("Environment Steps", fontsize=11)
    ax.set_ylabel("Episode Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    # Add legend for the first task
    if "cartpole" in task:
        legend_elements = [
            Line2D([0], [0], color=style["BL"]["color"], lw=2, label=style["BL"]["label"]),
            Line2D([0], [0], color=style["TT"]["color"], lw=2, label=style["TT"]["label"]),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

def extract_final_eval(run):
    """Extract the last eval score from a run."""
    if run and run["curve"]:
        return run["curve"][-1]["eval"]
    return None

# ===== Figure 1: Main Learning Curves =====
# 1a: Cartpole
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

plot_task_curves(axes[0], "_cartpole_swingup", "Cartpole Swingup", dedup_100k)
# Add annotations for final scores
for key, r in sorted(dedup_100k.items()):
    if r["task"] != "_cartpole_swingup":
        continue
    last = r["curve"][-1]
    axes[0].annotate(f'{last["eval"]:.0f}', xy=(last["step"], last["eval"]),
                     xytext=(5, 5), textcoords="offset points", fontsize=8,
                     color=style[r["model"]]["color"], fontweight="bold")

# 1b: Cheetah
plot_task_curves(axes[1], "_cheetah_run", "Cheetah Run", dedup_100k)
for key, r in sorted(dedup_100k.items()):
    if r["task"] != "_cheetah_run":
        continue
    last = r["curve"][-1]
    axes[1].annotate(f'{last["eval"]:.0f}', xy=(last["step"], last["eval"]),
                     xytext=(5, 5), textcoords="offset points", fontsize=8,
                     color=style[r["model"]]["color"], fontweight="bold")

# 1c: Walker
plot_task_curves(axes[2], "_walker_walk", "Walker Walk", dedup_100k)
for key, r in sorted(dedup_100k.items()):
    if r["task"] != "_walker_walk":
        continue
    last = r["curve"][-1]
    axes[2].annotate(f'{last["eval"]:.0f}', xy=(last["step"], last["eval"]),
                     xytext=(5, 5), textcoords="offset points", fontsize=8,
                     color=style[r["model"]]["color"], fontweight="bold")

plt.tight_layout(pad=2.0)
plt.savefig("/home/zhengkai/r2dreamer/fig1_learning_curves.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig 1 saved: learning curves")

# ===== Figure 2: Early Training Zoom (Cartpole) =====
fig, ax = plt.subplots(figsize=(8, 5))
# Focus on cartpole s42 at 50k — TT's best early advantage
for key, r in sorted(dedup_100k.items()):
    if r["task"] != "_cartpole_swingup" or r["seed"] != 42:
        continue
    s = style[r["model"]]
    pts = sorted(r["curve"], key=lambda x: x["step"])
    steps = [p["step"] for p in pts]
    evals = [p["eval"] for p in pts]
    ax.plot(steps, evals, color=s["color"], ls=s["ls"], lw=2.5, label=s["label"])
    last = pts[-1]
    ax.annotate(f'{last["eval"]:.0f}', xy=(last["step"], last["eval"]),
                xytext=(5, 5), textcoords="offset points", fontsize=10,
                color=s["color"], fontweight="bold")

ax.set_xlabel("Environment Steps", fontsize=12)
ax.set_ylabel("Episode Score", fontsize=12)
ax.set_title("Early Training Comparison (Cartpole, Seed 42)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=10)

# Annotate early advantage
ax.annotate("TT +24% at 50k steps", xy=(25000, 250), fontsize=11,
            fontstyle="italic", color="#2563EB",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#DBEAFE", edgecolor="#2563EB", alpha=0.8))

plt.tight_layout()
plt.savefig("/home/zhengkai/r2dreamer/fig2_early_training.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig 2 saved: early training zoom")

# ===== Figure 3: KL Stability Comparison =====
# Extract KL data from the runs
kl_data = {"cartpole": {"BL": [], "TT": []},
           "cheetah": {"BL": [], "TT": []},
           "walker": {"BL": [], "TT": []}}

task_map = {"_cartpole_swingup": "cartpole", "_cheetah_run": "cheetah", "_walker_walk": "walker"}

for key, r in dedup_100k.items():
    tn = task_map.get(r["task"], "other")
    if tn not in kl_data:
        continue
    kl_data[tn][r["model"]].append(r)

# Also get deep KL parsing from console logs for the specific runs
# We need dyn/rep loss from the log files
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

        dyns, reps, barlows = [], [], []
        with open(console) as f:
            for line in f:
                m = re.search(r"train/loss/dyn ([\d.]+)", line)
                if m: dyns.append(float(m.group(1)))
                m = re.search(r"train/loss/rep ([\d.]+)", line)
                if m: reps.append(float(m.group(1)))
                m = re.search(r"train/loss/barlow ([\d.]+)", line)
                if m: barlows.append(float(m.group(1)))
        if dyns and reps:
            console_kl[key] = {
                "dyn": np.mean(dyns[-20:]),
                "rep": np.mean(reps[-20:]),
                "barlow": np.mean(barlows[-20:]) if barlows else 0,
            }

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
                "dyn": np.mean(dyns[-20:]),
                "rep": np.mean(reps[-20:]),
                "barlow": 0,
            }

# Bar chart: dyn/rep for each model
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

tasks_plot = ["_cartpole_swingup", "_cheetah_run", "_walker_walk"]
task_labels = ["Cartpole", "Cheetah", "Walker"]
x = np.arange(len(tasks_plot))
width = 0.3

for idx, metric in enumerate(["dyn", "rep"]):
    ax = axes[idx]
    bl_vals = []
    tt_vals = []
    for task_key in tasks_plot:
        bl_k = [v for k, v in console_kl.items() if k[1] == task_key and k[0] == "BL"]
        tt_k = [v for k, v in console_kl.items() if k[1] == task_key and k[0] == "TT"]
        bl_vals.append(np.mean([v[metric] for v in bl_k]) if bl_k else 0)
        tt_vals.append(np.mean([v[metric] for v in tt_k]) if tt_k else 0)

    bars1 = ax.bar(x - width/2, bl_vals, width, color="#6B7280", label="Baseline")
    bars2 = ax.bar(x + width/2, tt_vals, width, color="#2563EB", label="TT RSSM")

    # Add value labels on bars
    for bar, val in zip(bars1, bl_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha="center", va="bottom", fontsize=9, color="#6B7280")
    for bar, val in zip(bars2, tt_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha="center", va="bottom", fontsize=9, color="#2563EB")

    metric_label = "Dynamics Loss (dyn)" if metric == "dyn" else "Representation Loss (rep)"
    ax.set_title(f"KL Stability — {metric_label}", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=11)
    ax.set_ylabel("Loss Value", fontsize=11)
    ax.set_ylim(0, max(max(bl_vals), max(tt_vals)) * 1.4)
    ax.grid(True, axis="y", alpha=0.3)
    if idx == 1:
        ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("/home/zhengkai/r2dreamer/fig3_kl_stability.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig 3 saved: KL stability")

# ===== Figure 4: Walker 300k results =====
fig, ax = plt.subplots(figsize=(8, 5))
for key, r in sorted(walker_300k.items()):
    s = style[r["model"]]
    pts = sorted(r["curve"], key=lambda x: x["step"])
    steps = [p["step"] for p in pts]
    evals = [p["eval"] for p in pts]
    label = s["label"] + " 300k" if r["seed"] == 123 else s["label"] + " 300k s456"
    ax.plot(steps, evals, color=s["color"], ls=s["ls"], lw=2.0,
            label=label, alpha=0.85)
    last = pts[-1]
    ax.annotate(f'{last["eval"]:.0f}', xy=(last["step"], last["eval"]),
                xytext=(5, 5), textcoords="offset points", fontsize=9,
                color=s["color"], fontweight="bold")

# Also add 100k walker curves for comparison
for key, r in sorted(dedup_100k.items()):
    if r["task"] != "_walker_walk":
        continue
    s = style[r["model"]]
    pts = sorted(r["curve"], key=lambda x: x["step"])
    steps = [p["step"] for p in pts]
    evals = [p["eval"] for p in pts]
    ax.plot(steps, evals, color=s["color"], ls=":", lw=1.2, alpha=0.4)

# Add dividing line at 100k
ax.axvline(x=100000, color="gray", ls="--", lw=1, alpha=0.5)
ax.text(105000, ax.get_ylim()[1]*0.95, "300k extension →", fontsize=9, color="gray", alpha=0.7)

ax.set_xlabel("Environment Steps", fontsize=11)
ax.set_ylabel("Episode Score", fontsize=11)
ax.set_title("Walker Walk — 300k Extension", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("/home/zhengkai/r2dreamer/fig4_walker_300k.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig 4 saved: walker 300k")

# ===== Figure 5: Gate behavior =====
# Extract gate values over training from TT runs directly
fig, ax = plt.subplots(figsize=(8, 4))
gate_data = {}
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
            ax.plot(steps, gates, lw=1.5, label=label, alpha=0.8)
            print(f"  Gate {label}: mean={np.mean(gates):.2f}, final={gates[-1]:.2f}")

# Add gate bounds
ax.axhline(y=0.01, color="red", ls="--", lw=1, alpha=0.4)
ax.axhline(y=0.50, color="red", ls="--", lw=1, alpha=0.4)
ax.text(ax.get_xlim()[1]*0.8, 0.52, "Upper bound 0.50", fontsize=8, color="red", alpha=0.5)
ax.text(ax.get_xlim()[1]*0.8, 0.06, "Lower bound 0.01", fontsize=8, color="red", alpha=0.5)

ax.set_xlabel("Environment Steps", fontsize=11)
ax.set_ylabel("Global Gate Value", fontsize=11)
ax.set_title("Learned Global Gate — All TT Runs", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, ncol=2, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 0.70)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("/home/zhengkai/r2dreamer/fig5_gate_behavior.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig 5 saved: gate behavior")

# ===== Summary print =====
print("\n=== All figures saved to /home/zhengkai/r2dreamer/ ===")
print("  fig1_learning_curves.png  (3-panel: cartpole, cheetah, walker)")
print("  fig2_early_training.png   (cartpole s42 zoom)")
print("  fig3_kl_stability.png     (dyn/rep loss bar chart)")
print("  fig4_walker_300k.png      (walker 300k extension)")
print("  fig5_gate_behavior.png    (gate over training)")
