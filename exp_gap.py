"""Experiment: where is the fit -> file loss, and which training variant closes it.
usage: python exp_gap.py IMG N  (uses the cached fit from bench.py)"""
import sys
import time
from pathlib import Path

import torch

from bench import bytes_for, cached_fit, curve, load, to_t
from gsimg import sq
from gsimg.codec import render_decoded
from gsimg.model import psnr

path, N = sys.argv[1], int(sys.argv[2])
name = Path(path).stem
im = load(path, 512)
t = to_t(im)
m = cached_fit(name, t, N, 6000)
avif = curve(im, "AVIF")
print(f"{name} N={N} float {psnr(m().detach(), t):.2f}")

VARIANTS = {
    "base": dict(),
    "k.01": dict(k=0.01),
    "k.004": dict(k=0.004),
    "lam.3": dict(lam=0.3),
    "lam3": dict(lam=3.0),
}
pick = sys.argv[3].split(",") if len(sys.argv) > 3 else list(VARIANTS)
for k in pick:
    kw = dict(lam=1.0)
    kw.update(VARIANTS[k])
    t0 = time.time()
    q = sq.train(m, t, **kw)
    buf = sq.encode(q)
    with torch.no_grad():
        p = psnr(render_decoded(sq.decode(buf)), t)
    ab = bytes_for(avif, p)
    save = f"{100 * (1 - len(buf) / ab):+.0f}%" if ab else "n/a"
    print(f"  {k:12} bytes {len(buf):6}  psnr {p:.2f}  AVIF same-psnr bytes {ab}  saved vs AVIF {save}  ({time.time() - t0:.0f}s)", flush=True)
