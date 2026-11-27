"""Does the training rate model (one Laplace per channel) charge what the real
coder charges? If it overcharges values far from the median, training
shrinks them too much (drift).
usage: python exp_rate.py IMG N LAM [--beta]"""
import sys
from pathlib import Path

import numpy as np
import torch

from bench import cached_fit, load, to_t
from gsimg import ac, sq
from gsimg.codec import shape_feat

path, N, lam = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
beta = "--beta" in sys.argv
t = to_t(load(path, 512))
m = cached_fit(Path(path).stem, t, N, 6000, beta)
q = sq.train(m, t, lam)
q.hard = True
names = ["log a", "b", "log c", "Y", "Co", "Cg", "log k"]
with torch.no_grad():
    step = q.logstep.exp()
    y = torch.round(q.lat / step)
    b = q.logb.exp()
    p = sq._lap_cdf(y + 0.5, q.mu, b) - sq._lap_cdf(y - 0.5, q.mu, b)
    lap = -torch.log2(p.clamp_min(1e-9)).sum(0).cpu().numpy()
    sym = y.long().cpu().numpy()
print(f"{Path(path).stem} N={N} lam={lam}: {sym.shape[0]} blobs")
print(f"  {'channel':7} {'step':>7} {'laplace':>8} {'entropy':>8} {'coded':>8}   (bytes)")
tot = np.zeros(3)
for j in range(sym.shape[1]):
    col = sym[:, j]
    _, cnt = np.unique(col, return_counts=True)
    pr = cnt / cnt.sum()
    ent = -(cnt * np.log2(pr)).sum()
    enc = ac.Encoder()
    ac.put_ints(enc, col)
    coded = len(enc.finish()) * 8
    row = np.array([lap[j], ent, coded]) / 8
    tot += row
    print(f"  {names[j]:7} {step[j].item():7.4f} {row[0]:8.0f} {row[1]:8.0f} {row[2]:8.0f}")
print(f"  {'total':7} {'':7} {tot[0]:8.0f} {tot[1]:8.0f} {tot[2]:8.0f}")
