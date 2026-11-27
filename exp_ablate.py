"""Where does the float -> file quality go? Train the coder once, then put
each coded group (positions, shape, colour, k, base) back to its float
value and see how much quality comes back.
usage: python exp_ablate.py IMG N LAM [--beta]   (uses bench.py's cached fit)"""
import sys
from pathlib import Path

import torch

from bench import cached_fit, load, psnr8, to_t
from gsimg import sq
from gsimg.codec import feat_to_L
from gsimg.model import KMAX, KMIN, psnr, render

path, N, lam = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
beta = "--beta" in sys.argv
name = Path(path).stem
t = to_t(load(path, 512))
m = cached_fit(name, t, N, 6000, beta)
with torch.no_grad():
    f0 = m()
print(f"{name} N={N} lam={lam} beta={beta}: fit float {psnr(f0, t):.2f}  fit 8-bit {psnr8(f0, t):.2f}")

q = sq.train(m, t, lam)
q.hard = True


@torch.no_grad()
def mixed(free=()):
    """Render with the groups in `free` left unquantised (trained float values)."""
    xy_q, L_q, rgb_q, base_q, bits, k_q = q.parts()
    v = q.lat
    xy = q.xy if "pos" in free else xy_q
    L = feat_to_L(v[:, :3]) if "shape" in free else L_q
    rgb = sq.ycc2rgb(v[:, 3:6]) if "colour" in free else rgb_q
    base = q.base if "base" in free else base_q
    k = None
    if q.beta:
        k = v[:, 6].exp().clamp(KMIN, KMAX) if "k" in free else k_q
    img = render(xy, L, rgb, base, q.H, q.W, q.R, k=k)
    return psnr8(img, t), bits.item() / (q.H * q.W)


groups = ["pos", "shape", "colour", "base"] + (["k"] if q.beta else [])
full, bpp = mixed()
allf, _ = mixed(groups)
print(f"  file (all quantised) {full:.2f} dB at {bpp:.3f} attr bpp, kept {q.xy.shape[0]}")
print(f"  trained, nothing quantised {allf:.2f} dB  (fit -> trained drift {psnr8(f0, t) - allf:+.2f})")
for g in groups:
    p, _ = mixed([g])
    print(f"  only {g:6} unquantised: {p:.2f} dB ({p - full:+.2f})")
for g in groups:
    p, _ = mixed([x for x in groups if x != g])
    print(f"  only {g:6} quantised:   {p:.2f} dB ({p - allf:+.2f} vs none)")
