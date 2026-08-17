import os, re, glob

print("# Gate Data: tt_global_gate over training for all TT runs")
print("# Format: task, seed, step, gate_value")
print()

for base in ["logdir/2026-06-20/", "logdir/2026-06-21/"]:
    for d in sorted(glob.glob(os.path.join("/home/zhengkai/r2dreamer", base, "*/"))):
        console = os.path.join(d, "console.log")
        ovr = os.path.join(d, ".hydra/overrides.yaml")
        if not os.path.exists(console) or not os.path.exists(ovr):
            continue
        with open(ovr) as f:
            t = f.read()
        if "size12M_tt" not in t:
            continue
        m = re.search(r"task=dmc_(\w+)", t)
        task = m.group(1) if m else "?"
        m = re.search(r"seed=(\d+)", t)
        seed = int(m.group(1)) if m else 0
        m = re.search(r"steps=(\d+)", t)
        steps_total = int(m.group(1)) if m else 0
        label = "%s s%d (%dk)" % (task, seed, steps_total//1000)

        # Record variance for identifying duplicates
        pairs = []
        with open(console) as f:
            for line in f:
                m = re.search(r"\[(\d+)\] train/tt_global_gate ([\d.]+)", line)
                if m:
                    pairs.append((int(m.group(1)), float(m.group(2))))

        if len(pairs) < 3:
            continue

        print("\n# %s" % label)
        print("task=%s seed=%d steps=%d points=%d" % (task, seed, steps_total, len(pairs)))
        for step, val in pairs:
            print("%s,%d,%d,%.3f" % (task, seed, step, val))

# Also check logdir_tt for cartpole s42
d = "/home/zhengkai/r2dreamer/logdir_tt/"
console = os.path.join(d, "console.log")
ovr = os.path.join(d, ".hydra/overrides.yaml")
if os.path.exists(console) and os.path.exists(ovr):
    with open(ovr) as f:
        t = f.read()
    if "size12M_tt" in t or "tt_enabled" in t:
        pairs = []
        with open(console) as f:
            for line in f:
                m = re.search(r"\[(\d+)\] train/tt_global_gate ([\d.]+)", line)
                if m:
                    pairs.append((int(m.group(1)), float(m.group(2))))
        if pairs:
            print("\n# cartpole_swingup s42 (50k)")
            for step, val in pairs:
                print("cartpole_swingup,42,%d,%.3f" % (step, val))
