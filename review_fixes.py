"""Round 1 reviewer fixes: partial corr, WM vs VP, lagged pred, bootstrap, checkpoint selection."""
import json, numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, rankdata

LOG_BASE = '/home/zhengkai/r2dreamer/logdir/v2'
EXPERIMENTS = {
    'dv3_walker': f'{LOG_BASE}/size12M_dv3_walker_walk_s42',
    'dv3_cheetah': f'{LOG_BASE}/size12M_dv3_cheetah_run_s42',
    'dv3_cartpole': f'{LOG_BASE}/size12M_dv3_cartpole_swingup_s42',
    'sf_walker': f'{LOG_BASE}/size12M_sf_walker_walk_s42',
    'sf_cheetah': f'{LOG_BASE}/size12M_sf_cheetah_run_s42',
    'sf_center_walker': f'{LOG_BASE}/size12M_sf_center_1M_walker_s42',
    'sf_center_cheetah': f'{LOG_BASE}/size12M_sf_center_1M_cheetah_s42',
    'sf_center_cartpole': f'{LOG_BASE}/size12M_sf_center_1M_cartpole_s42',
}
WM = ['train/loss/dyn','train/loss/rep','train/loss/image','train/dyn_entropy','train/rep_entropy']
VP = ['train/val','train/ret','train/loss/value','train/loss/policy','train/adv','train/adv_std']
RW = ['train/loss/rew']
ALL_M = WM + VP + RW

def load_data(logdir):
    mp = Path(logdir) / 'metrics.jsonl'
    if not mp.exists(): return [],[]
    recs = [json.loads(l) for l in open(mp)]
    evals = {r['step']: r['episode/eval_score'] for r in recs if 'episode/eval_score' in r}
    trains = {r['step']: r for r in recs if 'train/opt/updates' in r}
    aligned = []
    for es, escore in sorted(evals.items()):
        prev = [s for s in trains if s < es]
        if not prev: continue
        td = trains[max(prev)]
        row = {'eval_step': es, 'eval_score': escore}
        for m in ALL_M:
            if m in td: row[m] = td[m]
        total = sum(td.get(k,0) for k in WM+RW+VP[:3])
        row['total_loss'] = total
        row['step'] = es
        aligned.append(row)
    return aligned, recs

def partial_corr(data, metric, control='step'):
    n = len(data)
    if n < 10: return np.nan, np.nan
    x = np.array([d[metric] for d in data])
    y = np.array([d['eval_score'] for d in data])
    z = np.array([d[control] for d in data])
    v = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if v.sum() < 10: return np.nan, np.nan
    x,y,z = x[v],y[v],z[v]
    rx,ry,rz = rankdata(x),rankdata(y),rankdata(z)
    r_xz,_ = pearsonr(rx,rz); r_yz,_ = pearsonr(ry,rz); r_xy,_ = pearsonr(rx,ry)
    num = r_xy - r_xz * r_yz
    den = np.sqrt((1-r_xz**2)*(1-r_yz**2))
    if abs(den) < 1e-10: return np.nan, np.nan
    return num/den, r_xy

def lagged_pred(data, metric, lag=2):
    n = len(data)
    if n < lag+5: return np.nan,np.nan
    cm = np.array([data[i][metric] for i in range(n-lag)])
    cr = np.array([data[i]['eval_score'] for i in range(n-lag)])
    fr = np.array([data[i+lag]['eval_score'] for i in range(n-lag)])
    rc = fr - cr
    v = np.isfinite(cm) & np.isfinite(rc)
    if v.sum() < 5: return np.nan,np.nan
    return spearmanr(cm[v], rc[v])

def block_bootstrap(data, metric, nb=1000, bs=5):
    n = len(data)
    if n < 10: return np.nan,np.nan,np.nan
    x = np.array([d[metric] for d in data])
    y = np.array([d['eval_score'] for d in data])
    v = np.isfinite(x) & np.isfinite(y)
    x,y = x[v],y[v]; n = len(x)
    obs,_ = spearmanr(x,y)
    nblk = max(1, n//bs)
    rhos = []
    for _ in range(nb):
        bis = np.random.choice(nblk, size=nblk, replace=True)
        idx = []
        for bi in bis:
            s = bi*bs; e = min(s+bs, n)
            idx.extend(range(s,e))
        idx = np.array(idx[:n])
        bx,by = x[idx],y[idx]
        if np.std(bx)<1e-10 or np.std(by)<1e-10: continue
        r,_ = spearmanr(bx,by)
        rhos.append(r)
    if not rhos: return obs,np.nan,np.nan
    rhos = np.array(rhos)
    return obs, np.percentile(rhos,2.5), np.percentile(rhos,97.5)

print("="*70)
print("ROUND 1 REVIEW FIXES")
print("="*70)

all_data = {}
for name, logdir in EXPERIMENTS.items():
    data, _ = load_data(logdir)
    if data: all_data[name] = data

print("\nFIX 2: Partial Correlation (control training step)")
print("-"*50)
for name in ['dv3_walker','dv3_cheetah','sf_cheetah']:
    if name not in all_data: continue
    d = all_data[name]
    print("\n  " + name + ":")
    for m in WM[:4]:
        vals = [x.get(m,np.nan) for x in d]
        if np.sum(np.isfinite(vals)) < 5: continue
        raw_r,_ = spearmanr([x[m] for x in d if np.isfinite(x.get(m,np.nan))],
                            [x['eval_score'] for x in d if np.isfinite(x.get(m,np.nan))])
        pr,_ = partial_corr(d, m, 'step')
        if not np.isnan(pr):
            print(f"    {m.replace('train/',''):25s} raw={raw_r:+.3f} partial={pr:+.3f}")

print("\n\nFIX 3: World-Model vs Value/Policy Separation")
print("-"*50)
for name, data in all_data.items():
    wm_vals = []
    vp_vals = []
    for m in WM:
        x = [d[m] for d in data if np.isfinite(d.get(m,np.nan))]
        y = [d['eval_score'] for d in data if np.isfinite(d.get(m,np.nan))]
        if len(x) >= 5:
            wm_vals.append(abs(spearmanr(x,y)[0]))
    for m in VP:
        x = [d[m] for d in data if np.isfinite(d.get(m,np.nan))]
        y = [d['eval_score'] for d in data if np.isfinite(d.get(m,np.nan))]
        if len(x) >= 5:
            vp_vals.append(abs(spearmanr(x,y)[0]))
    wm_best = max(wm_vals) if wm_vals else 0
    vp_best = max(vp_vals) if vp_vals else 0
    print(f"  {name:<30s} WM={wm_best:.3f} VP={vp_best:.3f} best={'WM' if wm_best>vp_best else 'VP'}")

print("\n\nFIX 5: Lagged Prediction (predict future return delta)")
print("-"*50)
for name in ['dv3_walker','dv3_cheetah','sf_cheetah']:
    if name not in all_data: continue
    d = all_data[name]
    print("\n  " + name + " (lag=2):")
    for m in ['train/loss/image','train/loss/dyn','train/dyn_entropy','train/val']:
        if m not in d[0]:
            print(f"    {m.replace('train/',''):25s} N/A (metric missing)")
            continue
        rho,p = lagged_pred(d, m, lag=2)
        if not np.isnan(rho):
            print(f"    {m.replace('train/',''):25s} rho={rho:+.3f} p={p:.4f}")

print("\n\nFIX 6: Block Bootstrap CIs")
print("-"*50)
for name in ['dv3_walker','dv3_cheetah','sf_cheetah']:
    if name not in all_data: continue
    d = all_data[name]
    print("\n  " + name + ":")
    for m in ['train/loss/image','train/loss/dyn','train/val']:
        if m not in d[0]:
            print(f"    {m.replace('train/',''):25s} N/A")
            continue
        obs,cl,ch = block_bootstrap(d, m)
        if not np.isnan(obs):
            sig = '**' if cl*ch>0 else ''
            print(f"    {m.replace('train/',''):25s} rho={obs:+.3f} [{cl:+.3f}, {ch:+.3f}] {sig}")

print("\n\nFIX 8: Checkpoint Selection Experiment")
print("-"*50)
for name in ['dv3_walker','dv3_cheetah']:
    if name not in all_data: continue
    d = all_data[name]
    best = max(x['eval_score'] for x in d)
    print("\n  " + name + ": best final return=" + str(int(best)))
    for m in ['train/loss/image','train/val','total_loss','train/loss/dyn']:
        vals = [(i, d[i].get(m,np.nan), d[i]['eval_score']) for i in range(len(d))]
        valid = [(i,v,s) for i,v,s in vals if np.isfinite(v)]
        if len(valid)<3: continue
        lb = 'loss' in m or 'total' in m
        best_idx = min(valid, key=lambda x:x[1]) if lb else max(valid, key=lambda x:x[1])
        reg = best - best_idx[2]
        print(f"    {m.replace('train/',''):25s} selected_score={best_idx[2]:.0f} regret={reg:.0f} ({reg/best*100:.1f} pct)")

print("\n"+"="*70)
print("FIXES COMPLETE")
