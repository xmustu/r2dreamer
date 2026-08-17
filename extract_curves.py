import json, os

with open("/home/zhengkai/r2dreamer/poster_data.json") as f:
    all_runs = json.load(f)

# Dedup
dedup = {}
for r in all_runs:
    key = (r["model"], r["task"], r["seed"])
    if key not in dedup or len(r["curve"]) > len(dedup[key]["curve"]):
        dedup[key] = r

# Walker 300k is already included in dedup (keep as is)

for task_label, task_key in [("Cartpole", "_cartpole_swingup"), ("Cheetah", "_cheetah_run")]:
    print("\n=== %s Swingup ===" % task_label)
    print("Model,Seed,Step,EvalScore")
    for mk in ["BL", "TT"]:
        runs = {k: v for k, v in dedup.items() if v["task"] == task_key and k[0] == mk}
        for key, r in sorted(runs.items(), key=lambda x: x[0][1]):
            for pt in sorted(r["curve"], key=lambda x: x["step"]):
                print("%s,%d,%d,%.1f" % (mk, r["seed"], pt["step"], pt["eval"]))

print("\n=== Walker Walk ===")
print("Model,Seed,Step,EvalScore,Steps(steps)")
for k, r in sorted(dedup.items(), key=lambda x: (x[1]["seed"], x[0][0])):
    if r["task"] != "_walker_walk":
        continue
    note = "300k" if r["steps"] > 100000 else "100k"
    for pt in sorted(r["curve"], key=lambda x: x["step"]):
        print("%s,%d,%d,%.1f,%s" % (r["model"], r["seed"], pt["step"], pt["eval"], note))

# Mean ± std binned data (for the shaded region in fig1)
import numpy as np
print("\n\n=== Binned Mean ± Std (for fig1 shaded regions) ===")
print("Task,Model,BinCenter,Mean,Std,N_seeds")
for task_key, task_label in [("_cartpole_swingup","Cartpole"), ("_cheetah_run","Cheetah"), ("_walker_walk","Walker")]:
    bins = np.arange(0, 105000, 2000)
    centers = (bins[:-1] + bins[1:]) / 2
    for mk in ["BL", "TT"]:
        runs = {k: v for k, v in dedup.items() if v["task"] == task_key and k[0] == mk}
        if len(runs) < 1:
            continue
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
            continue
        arr = np.array(binned)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        for i in range(len(centers)):
            print("%s,%s,%.0f,%.1f,%.1f,%d" % (task_label, mk, centers[i], mean[i], std[i], len(binned)))
