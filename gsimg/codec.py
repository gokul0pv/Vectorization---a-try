"""Shape dedupe + quantisation + file format (.gsi).

Similar shapes are merged: every Gaussian's shape (log a, b, log c) is
replaced by codebook entries shared by all Gaussians. Residual codebooks:
shape = cb1[i1] + cb2[i2] + ...  (each stage 256 entries = 1 byte), so a
few bytes pick from 65k+ effective shapes. Colours likewise. A
quantisation-aware fine-tune then repairs what the merge cost.
"""
import lzma
import struct

import numpy as np
import torch
import torch.nn.functional as F

from .model import base_shape, psnr, render

MAGIC = b"GSI2"
XYQ = 4      # position step = 1/4 px
BAND = 8     # rows of positions are coded in bands of 8 px
K = 256      # entries per codebook stage


def kmeans(x, K, iters=30):
    g = torch.Generator(device="cpu").manual_seed(0)
    cb = x[torch.randperm(x.shape[0], generator=g)[:K].to(x.device)].clone()
    for _ in range(iters):
        idx = torch.cdist(x, cb).argmin(1)
        s = torch.zeros_like(cb).index_add_(0, idx, x)
        n = torch.bincount(idx, minlength=K).float()
        empty = n == 0
        cb = torch.where(empty[:, None], x[torch.randint(0, x.shape[0], (K,), generator=g).to(x.device)],
                         s / n.clamp_min(1)[:, None])
    return cb, torch.cdist(x, cb).argmin(1)


def rvq(x, stages):
    cbs, idxs, r = [], [], x
    for _ in range(stages):
        cb, i = kmeans(r, K)
        cbs.append(cb)
        idxs.append(i)
        r = r - cb[i]
    return cbs, idxs


def _ste(x, q):
    return x + (q - x).detach()


def shape_feat(L):
    return torch.stack([L[:, 0].log(), L[:, 1], L[:, 2].log()], 1)


def feat_to_L(f):
    return torch.stack([f[:, 0].exp(), f[:, 1], f[:, 2].exp()], 1)


class Quantized(torch.nn.Module):
    """Straight-through VQ: each Gaussian keeps a free latent, the render
    sees only its nearest codebook value; indices are re-picked every step
    until `frozen`, so Gaussians can hop to a better shared shape."""

    def __init__(self, m, ss=2, cs=2):
        super().__init__()
        self.H, self.W, self.R = m.H, m.W, m.R
        sf, cf = shape_feat(m.L()).detach(), m.rgb.detach()
        with torch.no_grad():
            scbs, sidx = rvq(sf, ss)
            ccbs, cidx = rvq(cf, cs)
        P = torch.nn.Parameter
        self.xy = P(m.xy.detach().clone())
        self.sf, self.cf = P(sf.clone()), P(cf.clone())
        self.scb = torch.nn.ParameterList([P(c) for c in scbs])
        self.ccb = torch.nn.ParameterList([P(c) for c in ccbs])
        self.base = P(m.base.detach().clone())
        self.sidx, self.cidx = sidx, cidx
        self.frozen = False
        self.commit = 0.0

    def _vq(self, latent, cbs, idxs):
        cbs = [_ste(c, c.half().float()) for c in cbs]
        if not self.frozen:
            with torch.no_grad():
                r, idxs = latent.detach(), []
                for c in cbs:
                    i = torch.cdist(r, c.detach()).argmin(1)
                    idxs.append(i)
                    r = r - c.detach()[i]
        v = sum(c[i] for c, i in zip(cbs, idxs))
        self.commit = self.commit + F.mse_loss(latent, v.detach())
        return v + latent - latent.detach(), idxs

    def parts(self):
        self.commit = 0.0
        xy = _ste(self.xy, torch.round(self.xy * XYQ) / XYQ)
        s, self.sidx = self._vq(self.sf, self.scb, self.sidx)
        c, self.cidx = self._vq(self.cf, self.ccb, self.cidx)
        base = _ste(self.base, torch.round((self.base.clamp(-0.5, 1.5) + 0.5) * 127.5) / 127.5 - 0.5)
        return xy, feat_to_L(s), c, base

    def forward(self):
        xy, L, rgb, base = self.parts()
        return render(xy, L, rgb, base, self.H, self.W, self.R)


def quantize(m, target, ss=1, cs=1, iters=1500, log=None):
    import math
    q = Quantized(m, ss, cs)
    if log:
        with torch.no_grad():
            log(f"  after shape merge: psnr {psnr(q(), target):.2f}")
    # Index hopping (frozen=False) was tested: no gain, unstable. Keep indices.
    q.frozen = True
    opt = torch.optim.Adam([
        {"params": [q.xy], "lr": 0.01},
        {"params": list(q.scb) + list(q.ccb), "lr": 0.003},
        {"params": [q.base], "lr": 0.001},
    ])
    lr0 = [g["lr"] for g in opt.param_groups]
    best, best_state = float("inf"), None
    for it in range(iters):
        f = 0.5 * (1 + math.cos(math.pi * it / iters))
        for g, l0 in zip(opt.param_groups, lr0):
            g["lr"] = l0 * (0.02 + 0.98 * f)
        loss = F.mse_loss(q(), target)
        if loss.item() < best:
            best = loss.item()
            best_state = {k: v.detach().clone() for k, v in q.state_dict().items()}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    q.load_state_dict(best_state)
    if log:
        with torch.no_grad():
            log(f"  after fine-tune:   psnr {psnr(q(), target):.2f}")
    return q


def _planes(a, dtype):
    """Split integers into byte planes (hi bytes compress far better apart)."""
    b = np.ascontiguousarray(a.astype(dtype)).view(np.uint8).reshape(-1, np.dtype(dtype).itemsize)
    return b.T.copy().tobytes()


def _unplanes(buf, n, dtype):
    k = np.dtype(dtype).itemsize
    return np.frombuffer(buf, np.uint8).reshape(k, n).T.copy().view(dtype).reshape(-1)


def _pack(chunks):
    return struct.pack(f"<{len(chunks)}I", *map(len, chunks)) + b"".join(chunks)


def _unpack(buf, n):
    lens = struct.unpack(f"<{n}I", buf[:4 * n])
    out, o = [], 4 * n
    for l in lens:
        out.append(buf[o:o + l])
        o += l
    return out


def encode(q):
    with torch.no_grad():
        N = q.xy.shape[0]
        ss, cs = len(q.scb), len(q.ccb)
        qx = torch.round(q.xy[:, 0] * XYQ).long().cpu().numpy()
        qy = torch.round(q.xy[:, 1] * XYQ).long().cpu().numpy()
        qx = np.clip(qx, -XYQ * q.R, XYQ * (q.W + q.R)) + XYQ * q.R   # >= 0
        qy = np.clip(qy, -XYQ * q.R, XYQ * (q.H + q.R)) + XYQ * q.R
        bh = BAND * XYQ
        band, yin = qy // bh, qy % bh
        order = np.lexsort((qx, band))
        qx, band, yin = qx[order], band[order], yin[order]
        nb = (XYQ * (q.H + 2 * q.R)) // bh + 1
        counts = np.bincount(band, minlength=nb)
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        dx = qx.copy()
        dx[1:] = np.diff(qx)
        dx[starts[counts > 0]] = qx[starts[counts > 0]]   # each band restarts at 0
        chunks = [
            np.round((q.base.detach().clamp(-0.5, 1.5).cpu().numpy() + 0.5) * 127.5).astype(np.uint8).tobytes(),
            _planes(counts, np.uint16), _planes(dx, np.uint16), yin.astype(np.uint8).tobytes(),
        ]
        for cb, idx in list(zip(q.scb, q.sidx)) + list(zip(q.ccb, q.cidx)):
            chunks.append(cb.detach().cpu().numpy().astype(np.float16).tobytes())
            chunks.append(idx.cpu().numpy()[order].astype(np.uint8).tobytes())
        head = MAGIC + struct.pack("<HHBIBB", q.H, q.W, q.R, N, ss, cs)
        return head + lzma.compress(_pack(chunks), preset=9 | lzma.PRESET_EXTREME)


def decode(buf, device="cuda"):
    assert buf[:4] == MAGIC
    H, W, R, N, ss, cs = struct.unpack("<HHBIBB", buf[4:15])
    ch = _unpack(lzma.decompress(buf[15:]), 4 + 2 * (ss + cs))
    bh, bw = base_shape(H, W)
    base = np.frombuffer(ch[0], np.uint8).reshape(1, 3, bh, bw).astype(np.float32) / 127.5 - 0.5
    nb = (XYQ * (H + 2 * R)) // (BAND * XYQ) + 1
    counts = _unplanes(ch[1], nb, np.uint16).astype(np.int64)
    dx = _unplanes(ch[2], N, np.uint16).astype(np.int64)
    yin = np.frombuffer(ch[3], np.uint8).astype(np.int64)
    band = np.repeat(np.arange(nb), counts)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    qx = np.empty(N, np.int64)
    for s, c in zip(starts, counts):
        qx[s:s + c] = np.cumsum(dx[s:s + c])
    qy = band * BAND * XYQ + yin
    xy = np.stack([qx, qy], 1).astype(np.float32) / XYQ - R
    t = lambda a: torch.as_tensor(a, device=device)

    def stages(i0, n):
        v = 0
        for k in range(n):
            cb = np.frombuffer(ch[i0 + 2 * k], np.float16).reshape(K, 3).astype(np.float32)
            idx = np.frombuffer(ch[i0 + 2 * k + 1], np.uint8).astype(np.int64)
            v = v + cb[idx]
        return t(v)

    L = feat_to_L(stages(4, ss))
    rgb = stages(4 + 2 * ss, cs)
    return dict(xy=t(xy), L=L, rgb=rgb, base=t(base), H=H, W=W, R=R)


def render_decoded(d, scale=1.0):
    """Resolution-free: render at any scale from the same file."""
    s = scale
    H, W = round(d["H"] * s), round(d["W"] * s)
    img = render(d["xy"] * s, d["L"] * s, d["rgb"], d["base"],
                 H, W, int(np.ceil(d["R"] * s)), d.get("cut", 0.0), k=d.get("k"))
    if d.get("boxes") is not None:
        from .rects import render_rects
        # boxes are pixel-aligned; at other scales their corners are rounded to whole pixels
        b = torch.round(d["boxes"].double() * s).long()
        img = img + render_rects(b, d["brgb"], H, W)
    return img
