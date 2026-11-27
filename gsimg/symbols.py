"""Layered picture: base grid + boxes (gsimg/rects.py) + symbol layer + free blobs.

Symbol layer: shapes that repeat (letters, icons) are drawn by ONE
prototype group of blobs, stored once; every copy is (prototype, position).
Free blobs draw the rest. All are trained together, so each prototype
learns from every copy of itself. Positions of copies are continuous
(fractional px), unlike pixel-based symbol dictionaries.
"""
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model import GaussImage, render, run_steps
from .rects import find_rects, init_colours, render_rects


def find_symbols(a, thr=24, tol=6.0, max_side=60):
    """Small shapes that differ from their local background, grouped into
    prototypes (same box size, mean abs diff < tol levels). Returns a list of
    dicts: w, h, mask (h,w), delta (h,w,3, colour minus background, 0..1),
    origins [(x, y), ...]; only shapes that occur at least twice."""
    bg = cv2.medianBlur(a.astype(np.uint8), 15).astype(np.float32)
    fg = (np.abs(a - bg).max(2) > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    groups = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < 3 or w > max_side or h > max_side:
            continue
        m = lab[y:y + h, x:x + w] == i
        d = (a[y:y + h, x:x + w] - bg[y:y + h, x:x + w]) * m[..., None] / 255.0
        for g in groups:
            if g["w"] == w and g["h"] == h and np.abs(g["d0"] - d).mean() * 255 < tol:
                g["origins"].append((x, y))
                g["ds"].append(d)
                g["mask"] |= m
                break
        else:
            groups.append(dict(w=w, h=h, d0=d, ds=[d], mask=m.copy(), origins=[(x, y)]))
    out = []
    for g in groups:
        if len(g["origins"]) >= 2:
            out.append(dict(w=g["w"], h=g["h"], mask=g["mask"], delta=np.mean(g["ds"], 0),
                            origins=g["origins"]))
    return out


class SymbolImage(GaussImage):
    def __init__(self, target, n_free, syms, R=20, per_px=0.5, mmin=3, mmax=40, boxes=None):
        super().__init__(target, n_free, R)
        dev = target.device
        self.has_boxes = boxes is not None and len(boxes) > 0
        if self.has_boxes:
            self.boxes = torch.as_tensor(np.asarray(boxes), dtype=torch.long, device=dev)
            under = F.interpolate(self.base, size=(self.H, self.W), mode="bicubic", align_corners=False)[0].permute(1, 2, 0)
            self.brgb = torch.nn.Parameter(init_colours(np.asarray(boxes), target - under.detach()))
        pxy, pcol, owner = [], [], []
        g = torch.Generator().manual_seed(0)
        for k, s in enumerate(syms):
            ys, xs = np.nonzero(s["mask"])
            mag = np.abs(s["delta"][ys, xs]).max(1) + 1e-3
            M = int(np.clip(round(per_px * len(xs)), mmin, mmax))
            idx = torch.multinomial(torch.tensor(mag / mag.sum()), M, replacement=M > len(xs), generator=g).numpy()
            pxy.append(np.stack([xs[idx], ys[idx]], 1) + 0.5)
            pcol.append(s["delta"][ys[idx], xs[idx]] * 0.6)
            owner += [k] * M
        pxy = np.concatenate(pxy) if pxy else np.zeros((0, 2))
        pcol = np.concatenate(pcol) if pcol else np.zeros((0, 3))
        self.pxy = torch.nn.Parameter(torch.tensor(pxy, dtype=torch.float32, device=dev))
        self.prgb = torch.nn.Parameter(torch.tensor(pcol, dtype=torch.float32, device=dev))
        Mt = self.pxy.shape[0]
        s0 = 0.45
        t = (s0 - 0.35) / (self.smax - 0.35)
        raw = math.log(t / (1 - t))
        self.psraw = torch.nn.Parameter(torch.tensor([raw, 0.0, raw], device=dev).repeat(Mt, 1))
        owner = torch.tensor(owner, device=dev)
        # every copy = (prototype, origin); expand to (copy, blob) pairs
        ci, bi, oo = [], [], []
        for k, s in enumerate(syms):
            blobs = torch.nonzero(owner == k)[:, 0]
            for (x, y) in s["origins"]:
                oo.append((x, y))
                ci += [len(oo) - 1] * len(blobs)
                bi.append(blobs)
        self.off = torch.nn.Parameter(torch.tensor(oo, dtype=torch.float32, device=dev).reshape(-1, 2))
        self.copy_of = torch.tensor(ci, dtype=torch.long, device=dev)
        self.blob_of = torch.cat(bi) if bi else torch.zeros(0, dtype=torch.long, device=dev)
        self.owner = owner                       # prototype of each prototype blob
        self.copy_proto = torch.tensor([k for k, s in enumerate(syms) for _ in s["origins"]], dtype=torch.long, device=dev)
        self.n_protos = len(syms)
        self.n_proto_blobs, self.n_copies = Mt, len(oo)

    def proto_L(self):
        s = torch.sigmoid(self.psraw)
        a = 0.35 + (self.smax - 0.35) * s[:, 0]
        c = 0.35 + (self.smax - 0.35) * s[:, 2]
        b = self.smax * torch.tanh(self.psraw[:, 1])
        return torch.stack([a, b, c], 1)

    def forward(self):
        pL = self.proto_L()
        xy = torch.cat([self.xy, self.pxy[self.blob_of] + self.off[self.copy_of]])
        L = torch.cat([self.L(), pL[self.blob_of]])
        rgb = torch.cat([self.rgb, self.prgb[self.blob_of]])
        img = render(xy, L, rgb, self.base, self.H, self.W, self.R)
        if self.has_boxes:
            img = img + render_rects(self.boxes, self.brgb, self.H, self.W)
        return img


def fit_symbols(target, n_free, syms, iters=6000, lr=(0.05, 0.05, 0.02, 0.005), boxes=None):
    torch.manual_seed(0)
    m = SymbolImage(target, n_free, syms, boxes=boxes)
    groups = [{"params": [m.xy], "lr": lr[0]}, {"params": [m.sraw], "lr": lr[1]},
              {"params": [m.rgb], "lr": lr[2]}, {"params": [m.base], "lr": lr[3]},
              {"params": [m.pxy], "lr": 0.02}, {"params": [m.psraw], "lr": lr[1]},
              {"params": [m.prgb], "lr": lr[2]}, {"params": [m.off], "lr": 0.005}]
    if m.has_boxes:
        groups.append({"params": [m.brgb], "lr": 0.002})
    lr0 = [g["lr"] for g in groups]
    opt_ref = {}

    def lr_at(it):
        f = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * it / iters))
        return [l0 * f for l0 in lr0]

    def hook(it, img):
        if it and it % 250 == 0 and it < 0.7 * iters:
            m.relocate(opt_ref["opt"], ((img - target) ** 2).sum(-1), 0.03)

    run_steps(m, target, groups, iters, lr_at, hook, None, opt_ref)
    return m


def fit_auto(target_img, target, N, iters=6000, log=None, use_symbols=True, use_boxes=True):
    """Find boxes and repeated shapes in the 8-bit picture (H,W,3 uint8);
    fit them + free blobs with N stored blobs in total. None if neither
    is found (then a plain fit is the same thing)."""
    a = np.asarray(target_img).astype(np.float32)
    syms = find_symbols(a) if use_symbols else []
    boxes = find_rects(a) if use_boxes else None
    if not syms and (boxes is None or not len(boxes)):
        return None
    probe = SymbolImage(target, 1, syms)
    n_free = max(N - probe.n_proto_blobs, 1)
    if log:
        log(f"  layers: {0 if boxes is None else len(boxes)} boxes, {len(syms)} prototypes, "
            f"{probe.n_copies} copies, {probe.n_proto_blobs} prototype blobs + {n_free} free")
    return fit_symbols(target, n_free, syms, iters, boxes=boxes)
