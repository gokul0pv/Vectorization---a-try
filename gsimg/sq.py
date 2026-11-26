"""Learned scalar quantisation + entropy coding (.gsq).

Each Gaussian's 6 attributes (shape log a, b, log c; colour Y, Co, Cg)
are snapped to a grid whose step per channel is LEARNED. Training
minimises  bits + lam * 255^2 * MSE ; bits come from a Laplace model per
channel, so Gaussians drift to cheap common values (soft shape dedupe)
and weak ones collapse to colour 0 and are dropped. Positions are coded
in 8 px bands; everything goes through the adaptive arithmetic coder.
"""
import math
import struct

import numpy as np
import torch
import torch.nn.functional as F

from . import ac
from .codec import feat_to_L, shape_feat
from .model import CUT, base_shape, psnr, render

MAGIC = b"GSQ2"   # (GSQ3 = per-blob patch radius rule; off, see model.CUT)
BAND = 8
EMAX = 7          # finest position step 1/128 px
K_POS = 0.006     # position step ~ K_POS * sigma_min / colour amplitude


def pos_exp(sym, steps, k):
    """Per-Gaussian position precision exponent e (step = 2^-e px), from the
    quantised shape + colour only, so the decoder can compute it too.
    Sharp, strong blobs get fine positions; soft, faint ones coarse."""
    v = sym.astype(np.float64) * steps.astype(np.float64)
    a, b, c = np.exp(v[:, 0]), v[:, 1], np.exp(v[:, 2])
    y, co, cg = v[:, 3], v[:, 4], v[:, 5]
    amp = np.maximum(np.abs(np.stack([y + co - cg, y + cg, y - co - cg], 1)).max(1), 1e-3)
    s11, s12, s22 = a * a, a * b, b * b + c * c
    h = (s11 + s22) / 2
    smin = np.sqrt(np.maximum(h - np.sqrt(np.maximum(h * h - (a * c) ** 2, 0)), 1e-12))
    return np.clip(np.round(-np.log2(np.clip(k * smin / amp, 2.0 ** -EMAX, 1.0))), 0, EMAX).astype(np.int64)


def rgb2ycc(c):
    r, g, b = c.unbind(-1)
    return torch.stack([(r + 2 * g + b) / 4, (r - b) / 2, (-r + 2 * g - b) / 4], -1)


def ycc2rgb(c):
    y, co, cg = c.unbind(-1)
    return torch.stack([y + co - cg, y + cg, y - co - cg], -1)


def _ste(x, q):
    return x + (q - x).detach()


def _lap_cdf(x, mu, b):
    d = x - mu
    return 0.5 + 0.5 * torch.sign(d) * (1 - torch.exp(-d.abs() / b))


class SQ(torch.nn.Module):
    def __init__(self, m, k=K_POS, steps=(0.08, 0.08, 0.08, 0.008, 0.012, 0.012)):
        super().__init__()
        self.H, self.W, self.R, self.k = m.H, m.W, m.R, float(np.float32(k))
        P = torch.nn.Parameter
        lat = torch.cat([shape_feat(m.L()), rgb2ycc(m.rgb)], 1).detach()
        st = torch.tensor(steps, device=lat.device)
        self.xy = P(m.xy.detach().clone())
        self.lat = P(lat)
        self.logstep = P(st.log())
        y = lat / st
        mu = y.median(0).values
        self.mu = P(mu)
        self.logb = P((y - mu).abs().mean(0).clamp_min(0.3).log())
        self.base = P(m.base.detach().clone())
        self.hard = False

    def symbols(self):
        return torch.round(self.lat / self.logstep.exp())

    def parts(self):
        step = self.logstep.exp()
        y = self.lat / step
        if self.hard:
            yq = _ste(y, torch.round(y))
        else:
            yq = y + torch.rand_like(y) - 0.5
        v = yq * step
        b = self.logb.exp()
        p = _lap_cdf(yq + 0.5, self.mu, b) - _lap_cdf(yq - 0.5, self.mu, b)
        bits = -torch.log2(p.clamp_min(1e-9)).sum()
        with torch.no_grad():
            e = pos_exp(torch.round(y).cpu().numpy(), step.detach().cpu().numpy(), self.k)
            sc = torch.as_tensor(2.0 ** e, dtype=self.xy.dtype, device=self.xy.device)[:, None]
        xy = _ste(self.xy, torch.round(self.xy * sc) / sc)
        base = _ste(self.base, torch.round((self.base.clamp(-0.5, 1.5) + 0.5) * 127.5) / 127.5 - 0.5)
        return xy, feat_to_L(v[:, :3]), ycc2rgb(v[:, 3:]), base, bits

    def forward(self):
        xy, L, rgb, base, bits = self.parts()
        return render(xy, L, rgb, base, self.H, self.W, self.R), bits

    @torch.no_grad()
    def prune(self):
        """Drop Gaussians whose quantised colour is exactly zero (invisible)."""
        keep = (self.symbols()[:, 3:] != 0).any(1)
        for name in ("xy", "lat"):
            p = getattr(self, name)
            setattr(self, name, torch.nn.Parameter(p[keep].clone()))
        return int((~keep).sum())


def _run(q, target, lam, iters, groups, log, tag):
    opt = torch.optim.Adam(groups, fused=True)
    lr0 = [g["lr"] for g in opt.param_groups]
    npx = q.H * q.W
    best, best_state = float("inf"), None
    for it in range(iters):
        f = 0.5 * (1 + math.cos(math.pi * it / iters))
        for g, l0 in zip(opt.param_groups, lr0):
            g["lr"] = l0 * (0.02 + 0.98 * f)
        img, bits = q()
        mse = F.mse_loss(img, target)
        loss = bits / npx + lam * 255 ** 2 * mse
        if q.hard and loss.item() < best:
            best = loss.item()
            best_state = {k: v.detach().clone() for k, v in q.state_dict().items()}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log and (it % 500 == 0 or it == iters - 1):
            log(f"  {tag} it {it:5d}  psnr {psnr(img.detach(), target):.2f}  attr bpp {bits.item() / npx:.3f}")
    if best_state:
        q.load_state_dict(best_state)


def train(m, target, lam=0.2, iters_a=1000, iters_b=600, k=K_POS, log=None):
    q = SQ(m, k)
    # A: noisy relaxation, learn the steps + rate model + values together
    q.hard = False
    _run(q, target, lam, iters_a, [
        {"params": [q.xy], "lr": 0.005},
        {"params": [q.lat], "lr": 0.003},
        {"params": [q.logstep, q.mu, q.logb], "lr": 0.01},
        {"params": [q.base], "lr": 0.001},
    ], log, "A")
    n = q.prune()
    if log:
        log(f"  pruned {n} invisible Gaussians -> {q.xy.shape[0]}")
    # B: hard rounding, steps frozen
    q.hard = True
    _run(q, target, lam, iters_b, [
        {"params": [q.xy], "lr": 0.003},
        {"params": [q.lat], "lr": 0.001},
        {"params": [q.mu, q.logb], "lr": 0.005},
        {"params": [q.base], "lr": 0.0005},
    ], log, "B")
    n = q.prune()
    if log:
        with torch.no_grad():
            q.hard = True
            img, bits = q()
            log(f"  final: {q.xy.shape[0]} Gaussians (pruned {n}), psnr {psnr(img, target):.2f}")
    return q


def encode(q):
    with torch.no_grad():
        N, R, W, H = q.xy.shape[0], q.R, q.W, q.H
        steps = q.logstep.exp().float().cpu().numpy()
        sym = q.symbols().long().cpu().numpy()
        e = pos_exp(sym, steps, q.k)
        u = np.round((q.xy.double().cpu().numpy() + R) * (2.0 ** e)[:, None]).astype(np.int64)
        u = np.maximum(u, 0)
        u[:, 0] = np.minimum(u[:, 0], ((W + 2 * R) << e) - 1)
        u[:, 1] = np.minimum(u[:, 1], ((H + 2 * R) << e) - 1)
        coarse = u >> e[:, None]                      # whole pixels
        frac = u - (coarse << e[:, None])             # e extra bits per axis
        cx, cy = coarse[:, 0], coarse[:, 1]
        band, yin = cy // BAND, cy % BAND
        order = np.lexsort((cx, band))
        cx, band, yin, sym, e, frac = cx[order], band[order], yin[order], sym[order], e[order], frac[order]
        nb = (H + 2 * R) // BAND + 1
        counts = np.bincount(band, minlength=nb)
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        dx = cx.copy()
        dx[1:] = np.diff(cx)
        dx[starts[counts > 0]] = cx[starts[counts > 0]]
        base = np.round((q.base.clamp(-0.5, 1.5).cpu().numpy() + 0.5) * 127.5).astype(np.int64).reshape(-1)
        enc = ac.Encoder()
        heads = [ac.put_ints(enc, s) for s in
                 [base, counts, dx, yin] + [sym[:, j] for j in range(6)]]
        for (fx, fy), ei in zip(frac.tolist(), e.tolist()):
            enc.raw(fx, ei)
            enc.raw(fy, ei)
        body = enc.finish()
        head = MAGIC + struct.pack("<HHBIf", H, W, R, N, q.k) + steps.astype(np.float32).tobytes()
        head += b"".join(struct.pack("<Bih", m, lo, a) for m, lo, a in heads)
        return head + body


def decode(buf, device="cuda"):
    assert buf[:4] in (MAGIC, b"GSQ2")
    cut = CUT if buf[:4] == MAGIC else 0.0
    H, W, R, N, k = struct.unpack("<HHBIf", buf[4:17])
    steps = np.frombuffer(buf[17:41], np.float32)
    o = 41
    heads = []
    for _ in range(10):
        heads.append(struct.unpack("<Bih", buf[o:o + 7]))
        o += 7
    dec = ac.Decoder(buf[o:])
    bh, bw = base_shape(H, W)
    nb = (H + 2 * R) // BAND + 1
    base = ac.get_ints(dec, 3 * bh * bw, heads[0])
    counts = ac.get_ints(dec, nb, heads[1])
    dx = ac.get_ints(dec, N, heads[2])
    yin = ac.get_ints(dec, N, heads[3])
    sym = np.stack([ac.get_ints(dec, N, heads[4 + j]) for j in range(6)], 1)
    e = pos_exp(sym, steps, np.float32(k))
    frac = np.array([(dec.raw(ei), dec.raw(ei)) for ei in e.tolist()], np.int64).reshape(N, 2)
    band = np.repeat(np.arange(nb), counts)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    cx = np.empty(N, np.int64)
    for s_, c in zip(starts, counts):
        cx[s_:s_ + c] = np.cumsum(dx[s_:s_ + c])
    cy = band * BAND + yin
    u = (np.stack([cx, cy], 1) << e[:, None]) + frac
    xy = (u / (2.0 ** e)[:, None] - R).astype(np.float32)
    t = lambda a: torch.as_tensor(a, device=device)
    v = t(sym.astype(np.float32)) * t(steps.copy())
    base = t(base.reshape(1, 3, bh, bw).astype(np.float32) / 127.5 - 0.5)
    return dict(xy=t(xy), L=feat_to_L(v[:, :3]), rgb=ycc2rgb(v[:, 3:]), base=base, H=H, W=W, R=R, cut=cut)
