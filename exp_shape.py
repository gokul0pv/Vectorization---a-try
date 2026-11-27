"""Experiment: let the optimiser invent the blob shape instead of using a formula.

  gauss   : w = exp(-r^2/2)                          (fixed formula, reference)
  profile : w = f(r), f = 64 learned points, shared by all blobs (radial, any curve)
  stamp1  : w = S(u1, u2), S = one learned 16x16 picture, shared (any 2D shape)
  stampK  : K learned pictures; each blob picks one (soft during training,
            hard pick at the end; the file would store the index)
(u1, u2) are the blob's own stretched/rotated coordinates, as for the Gaussian.
Everything starts as a Gaussian, so only learning can make it different.
usage: python exp_shape.py IMG N ITERS [variants]"""
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from bench import load, psnr8, to_t
from gsimg.model import GaussImage, psnr

U = 3.5          # shape extent in sigma units (fits the R = 20 patch: smax = R / 3.5)
NP = 64          # profile points
S = 16           # stamp resolution


def splat_any(out, xy, L, rgb, H, W, R, wfn):
    """Like model._splat, but the weight comes from wfn(u1, u2, slice)."""
    dev = xy.device
    d = torch.arange(-R, R + 1, device=dev)
    dy, dx = torch.meshgrid(d, d, indexing="ij")
    dx, dy = dx.reshape(1, -1), dy.reshape(1, -1)
    chunk = max(1, (1 << 22) // (2 * R + 1) ** 2)
    for s in range(0, xy.shape[0], chunk):
        sl = slice(s, s + chunk)
        p, l, c = xy[sl], L[sl], rgb[sl]
        cx, cy = torch.floor(p[:, 0:1]).long(), torch.floor(p[:, 1:2]).long()
        px, py = cx + dx, cy + dy
        u1 = (px.to(p.dtype) + 0.5 - p[:, 0:1]) / l[:, 0:1]
        u2 = ((py.to(p.dtype) + 0.5 - p[:, 1:2]) - l[:, 1:2] * u1) / l[:, 2:3]
        w = wfn(u1, u2, sl)
        w = w * ((px >= 0) & (px < W) & (py >= 0) & (py < H))
        idx = (py.clamp(0, H - 1) * W + px.clamp(0, W - 1)).reshape(-1)
        out = out.index_add(0, idx, (w.unsqueeze(-1) * c.unsqueeze(1)).reshape(-1, 3))
    return out


def gauss_grid(n):
    t = torch.linspace(-U, U, n)
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    return torch.exp(-0.5 * (xx * xx + yy * yy))


class ShapeImage(GaussImage):
    def __init__(self, target, N, R, kind, K=1):
        super().__init__(target, N, R)
        self.kind, self.nk, self.hard, self.ste = kind, K, False, False
        dev = target.device
        if kind == "profile":
            r = torch.linspace(0, U, NP)
            self.prof = torch.nn.Parameter(torch.exp(-0.5 * r * r).to(dev))
        elif kind == "stamp":
            g = gauss_grid(S)[None].repeat(K, 1, 1)
            g = g * (1 + 0.05 * torch.randn_like(g))   # break symmetry between stamps
            self.stamps = torch.nn.Parameter(g[:, None].to(dev))   # K,1,S,S
            if K > 1:
                self.pick = torch.nn.Parameter(0.01 * torch.randn(N, K, device=dev))

    def shape_params(self):
        if self.kind == "profile":
            return [self.prof]
        if self.kind == "stamp":
            return [self.stamps] + ([self.pick] if self.nk > 1 else [])
        return []

    def wfn(self, u1, u2, sl):
        if self.kind == "gauss":
            return torch.exp(-0.5 * (u1 * u1 + u2 * u2))
        if self.kind == "profile":
            r = (u1 * u1 + u2 * u2 + 1e-12).sqrt()
            n, P = r.shape
            g = torch.stack([r / U * 2 - 1, torch.zeros_like(r)], -1).reshape(1, n * P, 1, 2)
            return F.grid_sample(self.prof.view(1, 1, 1, NP), g, align_corners=True,
                                 padding_mode="zeros").reshape(n, P)
        # stamp: bilinear lookup in the blob's own coordinates, zero outside
        n, P = u1.shape
        grid = torch.stack([u1, u2], -1) / U                 # n,P,2 in [-1,1]
        if self.nk == 1:
            g = grid.reshape(1, n * P, 1, 2)
            return F.grid_sample(self.stamps, g, align_corners=True, padding_mode="zeros").reshape(n, P)
        g = grid.reshape(1, n * P, 1, 2)
        allw = F.grid_sample(self.stamps.transpose(0, 1), g, align_corners=True,
                             padding_mode="zeros").reshape(self.nk, n, P)
        soft = F.softmax(self.pick[sl] / self.tau, 1)
        hard = F.one_hot(self.pick[sl].argmax(1), self.nk).to(soft.dtype)
        if self.hard:
            sel = hard
        elif self.ste:   # picture uses ONE stamp per blob; gradient flows through the soft pick
            sel = hard + soft - soft.detach()
        else:
            sel = soft
        return torch.einsum("kn p,nk->np".replace(" ", ""), allw, sel)

    def forward(self):
        out = torch.zeros(self.H * self.W, 3, device=self.xy.device)
        out = splat_any(out, self.xy, self.L(), self.rgb, self.H, self.W, self.R, self.wfn)
        base = F.interpolate(self.base, size=(self.H, self.W), mode="bicubic", align_corners=False)
        return out.reshape(self.H, self.W, 3) + base[0].permute(1, 2, 0)


def fit_shape(target, N, iters, kind, K=1, R=20, lr=(0.05, 0.05, 0.02, 0.005), slr=0.001, plr=0.05, ste_from=0.6):
    torch.manual_seed(0)
    m = ShapeImage(target, N, R, kind, K)
    m.tau = 1.0
    groups = [{"params": [m.xy], "lr": lr[0]}, {"params": [m.sraw], "lr": lr[1]},
              {"params": [m.rgb], "lr": lr[2]}, {"params": [m.base], "lr": lr[3]}]
    if kind != "gauss":
        groups.append({"params": m.shape_params()[:1], "lr": slr})
    if K > 1:
        groups.append({"params": [m.pick], "lr": plr})
    opt = torch.optim.Adam(groups)
    lr0 = [g["lr"] for g in opt.param_groups]
    for it in range(iters):
        f = 0.5 * (1 + math.cos(math.pi * it / iters))
        for g, l0 in zip(opt.param_groups, lr0):
            g["lr"] = l0 * (0.05 + 0.95 * f)
        if kind != "gauss" and it < 0.3 * iters:     # blobs settle first, as for Beta
            for g in opt.param_groups[4:]:
                g["lr"] = 0.0
        m.tau = max(0.1, 1.0 - it / (0.6 * iters))
        m.ste = it >= ste_from * iters
        img = m()
        loss = F.mse_loss(img, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it and it % 250 == 0 and it < 0.7 * iters:
            m.relocate(opt, ((img.detach() - target) ** 2).sum(-1), 0.03)
    return m


if __name__ == "__main__":
    path, N, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    pick = sys.argv[4].split(",") if len(sys.argv) > 4 else ["gauss", "profile", "stamp1", "stamp8"]
    t = to_t(load(path, 512))
    res = []
    for v in pick:
        kind, K = ("stamp", int(v[5:])) if v.startswith("stamp") else (v, 1)
        t0 = time.time()
        m = fit_shape(t, N, iters, kind, K)
        with torch.no_grad():
            p = psnr8(m(), t)
            extra = ""
            if K > 1:
                m.hard = True
                ph = psnr8(m(), t)
                use = torch.bincount(m.pick.argmax(1), minlength=K).tolist()
                extra = f"  hard pick {ph:.2f}  use {use}"
        res.append(f"{v} {p:.2f}")
        print(f"{Path(path).stem} N={N} {v:8} {p:.2f} dB{extra}  ({time.time() - t0:.0f}s)", flush=True)
        if kind == "stamp":
            torch.save(m.stamps.detach().cpu(), f"out/stamps_{Path(path).stem}_{v}.pt")
        if kind == "profile":
            torch.save(m.prof.detach().cpu(), f"out/profile_{Path(path).stem}.pt")
