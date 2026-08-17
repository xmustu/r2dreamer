import json, os, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style = {"BL": {"color": "#6B7280", "lw": 2.0, "label": "Baseline RSSM"},
         "TT": {"color": "#2563EB", "lw": 2.0, "label": "Two-Timescale RSSM"}}

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

dedup_100k = {k: v for k, v in dedup.items() if v["steps"] <= 100000}

# Figure 4: Walker 300k
fig, ax = plt.subplots(figsize=(9, 5.5))
for key, r in sorted(walker_300k.items()):
    s = style[r["model"]]
    pts = sorted(r["curve"], key=lambda x: x["step"])
    steps = np.array([p["step"] for p in pts])
    evals = np.array([p["eval"] for p in pts])
    ax.plot(steps, evals, color=s["color"], lw=2.0, alpha=0.85,
            label=s["label"] + " (s%d)" % r["seed"])
    ax.annotate("%.0f" % evals[-1], xy=(steps[-1], evals[-1]),
                xytext=(8, 5), textcoords="offset points", fontsize=10,
                color=s["color"], fontweight="bold")

for key, r in sorted(dedup_100k.items()):
    if r["task"] != "_walker_walk": continue
    s = style[r["model"]]
    pts = sorted(r["curve"], key=lambda x: x["step"])
    steps = np.array([p["step"] for p in pts])
    evals = np.array([p["eval"] for p in pts])
    ax.plot(steps, evals, color=s["color"], ls=":", lw=1.0, alpha=0.3)

ax.set_xlabel("Environment Steps", fontsize=13)
ax.set_ylabel("Episode Score", fontsize=13)
ax.set_title("Walker Walk - 300k Extension", fontsize=15, fontweight="bold")
ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.25, linestyle=":")
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig("/home/zhengkai/r2dreamer/fig4_walker_300k.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig 4 saved")

# Figure 5: Gate behavior
fig, ax = plt.subplots(figsize=(9, 4.5))
for base in ["logdir/2026-06-20/", "logdir/2026-06-21/"]:
    for d in glob.glob(os.path.join("/home/zhengkai/r2dreamer", base, "*/")):
        console = os.path.join(d, "console.log")
        ovr = os.path.join(d, ".hydra/overrides.yaml")
        if not os.path.exists(console) or not os.path.exists(ovr): continue
        with open(ovr) as f: t = f.read()
        if "size12M_tt" not in t: continue
        m = re.search(r"task=(\S+)", t)
        task = m.group(1).replace("dmc_","").replace("_"," ") if m else "?"
        m = re.search(r"seed=(\d+)", t)
        seed = int(m.group(1)) if m else 0
        steps, gates = [], []
        with open(console) as f:
            for line in f:
                m = re.search(r"\[(\d+)\] train/tt_global_gate ([\d.]+)", line)
                if m: steps.append(int(m.group(1))); gates.append(float(m.group(2)))
        if steps:
            label = "%s s%d" % (task, seed)
            ax.plot(steps, gates, lw=1.5, label=label, alpha=0.8)

ax.axhline(y=0.01, color="#DC2626", ls="--", lw=1, alpha=0.4)
ax.axhline(y=0.50, color="#DC2626", ls="--", lw=1, alpha=0.4)
ax.text(ax.get_xlim()[1]*0.78, 0.52, "Upper bound 0.50", fontsize=9, color="#DC2626", alpha=0.6, fontstyle="italic")
ax.text(ax.get_xlim()[1]*0.78, 0.07, "Lower bound 0.01", fontsize=9, color="#DC2626", alpha=0.6, fontstyle="italic")
ax.set_xlabel("Environment Steps", fontsize=13)
ax.set_ylabel("Global Gate Value", fontsize=13)
ax.set_title("Learned Global Gate - All TT Runs", fontsize=15, fontweight="bold")
ax.legend(fontsize=9, ncol=2, loc="upper right", framealpha=0.9)
ax.grid(True, alpha=0.25, linestyle=":")
ax.set_ylim(-0.05, 0.70)
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig("/home/zhengkai/r2dreamer/fig5_gate_behavior.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig 5 saved")
