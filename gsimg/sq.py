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
from .model import CUT, KMAX, KMIN, base_shape, psnr, render

# GSQ2: 6 channels, base step 1/127.5.  (GSQ3 = per-blob patch radius rule; off, see model.CUT)
# GSQ4: + flags byte after the header (bit 0: Beta blobs, 7th channel log k;
#       bit 1: adaptive shape precision, followed by float32 aref + uint8 G),
#       base step 1/510 (the 7-bit base cost up to 2.7 dB on flat screenshots).
#       With bit 1 the colour channels are coded first, then each shape
#       channel split by precision group g = -G..G (own model per group).
#       bit 2: symbol layer (gsimg/symbols.py), followed by uint16 P + uint32 C
#       + uint8 copy_e: P prototypes (blob counts, blob attributes coded like
#       free blobs, positions with the same adaptive precision as free blobs:
#       whole px + pos_exp fraction bits) and C copies (prototype id +
#       position at 2^-copy_e px, sorted in reading order).
#       Fixed 1/16 px prototype + 1/8 px copy positions capped text at 41 dB.
MAGIC_OLD, MAGIC = b"GSQ2", b"GSQ4"
BASE_SCALE = {MAGIC_OLD: 127.5, MAGIC: 510.0}
BASE_S = BASE_SCALE[MAGIC]
K_STEP0 = 0.1
SHAPE_AREF = 0.05  # adaptive shape precision (None = off): shape step * 2^-g, see shape_g
SHAPE_G = 2
MIXED = False    # stage A: rate from noisy values, picture from rounded (STE)
BAND = 8
COPY_E = 6        # copy positions: 2^-6 = 1/64 px
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


def shape_g(csym, csteps, aref, G):
    """Shape precision group per blob from the QUANTISED colour (float64, so
    the C++/Rust decoders get the same integers): strong blobs (large colour
    amplitude) get a finer shape step, faint ones a coarser one.
    g = clip(round_half_even(log2(amp / aref)), -G, G); shape step * 2^-g."""
    v = csym.astype(np.float64) * csteps.astype(np.float64)
    y, co, cg = v[:, 0], v[:, 1], v[:, 2]
    amp = np.maximum(np.abs(np.stack([y + co - cg, y + cg, y - co - cg], 1)).max(1), 1e-4)
    return np.clip(np.round(np.log2(amp / np.float64(aref))), -G, G).astype(np.int64)


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
        self.beta = bool(getattr(m, "beta", False))
        self.sym = hasattr(m, "pxy")
        assert not (self.sym and self.beta), "symbol layer is Gaussian-only for now"
        self.nf = m.xy.shape[0]
        L, rgb = m.L(), m.rgb
        if self.sym:
            L, rgb = torch.cat([L, m.proto_L()]), torch.cat([rgb, m.prgb])
            self.pxy = P(m.pxy.detach().clone())
            self.off = P(m.off.detach().clone())
            self.blob_of, self.copy_of = m.blob_of.clone(), m.copy_of.clone()
            self.owner, self.copy_proto, self.n_protos = m.owner.clone(), m.copy_proto.clone(), m.n_protos
        lat = [shape_feat(L), rgb2ycc(rgb)]
        if self.beta:
            lat.append(m.K().log()[:, None])
            steps = tuple(steps) + (K_STEP0,)
        lat = torch.cat(lat, 1).detach()
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
        self.aref = None if SHAPE_AREF is None else float(np.float32(SHAPE_AREF))
        self.G = SHAPE_G

    def blob_scale(self):
        """Per-blob step multiplier (N, C): 1 for colour and k; for the shape
        channels 2^-g from the QUANTISED colour, so the decoder knows it
        before it reads the shape (colour is coded first)."""
        m = torch.ones_like(self.lat)
        if self.aref is None:
            return m
        with torch.no_grad():
            st = self.logstep.exp().detach().float().cpu().numpy()
            csym = torch.round(self.lat[:, 3:6] / torch.as_tensor(st[3:6], device=self.lat.device)).cpu().numpy()
            g = shape_g(csym, st[3:6], self.aref, self.G)
            m[:, :3] = torch.as_tensor(2.0 ** -g, dtype=m.dtype, device=m.device)[:, None]
        return m

    def symbols(self):
        return torch.round(self.lat / (self.logstep.exp() * self.blob_scale()))

    def parts(self):
        sc_b = self.blob_scale()
        step = self.logstep.exp() * sc_b
        y = self.lat / step
        if self.hard:
            yq = _ste(y, torch.round(y))
            v = yq * step
        else:
            yq = y + torch.rand_like(y) - 0.5
            v = (_ste(y, torch.round(y)) if MIXED else yq) * step
        b = self.logb.exp()
        mu, b = self.mu / sc_b, b / sc_b    # same spread in value units
        p = _lap_cdf(yq + 0.5, mu, b) - _lap_cdf(yq - 0.5, mu, b)
        bits = -torch.log2(p.clamp_min(1e-9)).sum()
        nf = self.nf
        with torch.no_grad():
            e = pos_exp((torch.round(y[:nf]) * sc_b[:nf]).cpu().numpy(), self.logstep.exp().detach().cpu().numpy(), self.k)
            sc = torch.as_tensor(2.0 ** e, dtype=self.xy.dtype, device=self.xy.device)[:, None]
        xy = _ste(self.xy, torch.round(self.xy * sc) / sc)
        base = _ste(self.base, torch.round((self.base.clamp(-0.5, 1.5) + 0.5) * BASE_S) / BASE_S - 0.5)
        k = v[:, 6].exp().clamp(KMIN, KMAX) if self.beta else None
        L, rgb = feat_to_L(v[:, :3]), ycc2rgb(v[:, 3:6])
        if self.sym:   # expand prototypes into their copies
            with torch.no_grad():
                ep = pos_exp((torch.round(y[nf:]) * sc_b[nf:]).cpu().numpy(),
                             self.logstep.exp().detach().cpu().numpy(), self.k)
                psc = torch.as_tensor(2.0 ** ep, dtype=self.pxy.dtype, device=self.pxy.device)[:, None]
            pxy = _ste(self.pxy, torch.round(self.pxy * psc) / psc)
            cq = 2.0 ** COPY_E
            off = _ste(self.off, torch.round(self.off * cq) / cq)
            bo = self.blob_of + nf
            xy = torch.cat([xy, pxy[self.blob_of] + off[self.copy_of]])
            L, rgb = torch.cat([L[:nf], L[bo]]), torch.cat([rgb[:nf], rgb[bo]])
        return xy, L, rgb, base, bits, k

    def forward(self):
        xy, L, rgb, base, bits, k = self.parts()
        return render(xy, L, rgb, base, self.H, self.W, self.R, k=k), bits

    @torch.no_grad()
    def prune(self):
        """Drop Gaussians whose quantised colour is exactly zero (invisible)."""
        keep = (self.symbols()[:, 3:6] != 0).any(1)
        nf = self.nf
        self.xy = torch.nn.Parameter(self.xy[keep[:nf]].clone())
        self.lat = torch.nn.Parameter(self.lat[keep].clone())
        if self.sym:
            pk = keep[nf:]
            new = torch.cumsum(pk.long(), 0) - 1
            sel = pk[self.blob_of]
            self.blob_of, self.copy_of = new[self.blob_of[sel]], self.copy_of[sel]
            self.pxy = torch.nn.Parameter(self.pxy[pk].clone())
            self.owner = self.owner[pk]
        self.nf = int(keep[:nf].sum())
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


def _sym_groups(q, lr):
    return [{"params": [q.pxy], "lr": lr}, {"params": [q.off], "lr": lr / 2}] if q.sym else []


def train(m, target, lam=0.2, iters_a=1000, iters_b=600, k=K_POS, log=None):
    q = SQ(m, k)
    # A: noisy relaxation, learn the steps + rate model + values together
    q.hard = False
    _run(q, target, lam, iters_a, [
        {"params": [q.xy], "lr": 0.005},
        {"params": [q.lat], "lr": 0.003},
        {"params": [q.logstep, q.mu, q.logb], "lr": 0.01},
        {"params": [q.base], "lr": 0.001},
    ] + _sym_groups(q, 0.005), log, "A")
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
    ] + _sym_groups(q, 0.003), log, "B")
    n = q.prune()
    if log:
        with torch.no_grad():
            q.hard = True
            img, bits = q()
            log(f"  final: {q.xy.shape[0]} Gaussians (pruned {n}), psnr {psnr(img, target):.2f}")
    return q


def _attr_cols(sym, steps, aref, G):
    """Attribute columns in coding order (see GSQ4 notes at the top)."""
    if aref is None:
        return [sym[:, j] for j in range(sym.shape[1])]
    g = shape_g(sym[:, 3:6], steps[3:6], aref, G)
    cols = [sym[:, j] for j in (3, 4, 5)]
    cols += [sym[g == v, j] for j in (0, 1, 2) for v in range(-G, G + 1)]
    return cols + [sym[:, j] for j in range(6, sym.shape[1])]


def _n_attr_cols(nch, aref, G):
    return nch if aref is None else nch + 6 * G


def _read_attrs(dec, n, heads, hi, steps, aref, G, nch):
    """Inverse of _attr_cols. Returns (sym (n, nch), shape multiplier (n,), next head)."""
    sym = np.zeros((n, nch), np.int64)
    if aref is None:
        for j in range(nch):
            sym[:, j] = ac.get_ints(dec, n, heads[hi]); hi += 1
        return sym, np.ones(n), hi
    for j in (3, 4, 5):
        sym[:, j] = ac.get_ints(dec, n, heads[hi]); hi += 1
    g = shape_g(sym[:, 3:6], steps[3:6], aref, G)
    for j in (0, 1, 2):
        for v in range(-G, G + 1):
            sel = g == v
            sym[sel, j] = ac.get_ints(dec, int(sel.sum()), heads[hi]); hi += 1
    for j in range(6, nch):
        sym[:, j] = ac.get_ints(dec, n, heads[hi]); hi += 1
    return sym, 2.0 ** -g, hi


def _copy_cols(proto, oq):
    """Copies in reading order: prototype id, row delta, x (delta within a row)."""
    order = np.lexsort((oq[:, 0], oq[:, 1]))
    proto, oq = proto[order], oq[order]
    dy = np.diff(oq[:, 1], prepend=0)
    dx = oq[:, 0].copy()
    same = np.r_[False, dy[1:] == 0]
    dx[same] = (oq[1:, 0] - oq[:-1, 0])[same[1:]]
    return [proto, dy, dx]


def encode(q):
    with torch.no_grad():
        N, R, W, H = q.nf, q.R, q.W, q.H
        steps = q.logstep.exp().float().cpu().numpy()
        sym_all = q.symbols().long().cpu().numpy()
        mult_all = q.blob_scale().cpu().numpy().astype(np.float64)
        sym, mult = sym_all[:N], mult_all[:N, 0]
        symv = sym.astype(np.float64)
        symv[:, :3] *= mult[:, None]
        e = pos_exp(symv, steps, q.k)
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
        adaptive = q.aref is not None
        nb = (H + 2 * R) // BAND + 1
        counts = np.bincount(band, minlength=nb)
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        dx = cx.copy()
        dx[1:] = np.diff(cx)
        dx[starts[counts > 0]] = cx[starts[counts > 0]]
        base = np.round((q.base.clamp(-0.5, 1.5).cpu().numpy() + 0.5) * BASE_S).astype(np.int64).reshape(-1)
        enc = ac.Encoder()
        heads = [ac.put_ints(enc, c) for c in [base, counts, dx, yin] + _attr_cols(sym, steps, q.aref, q.G)]
        for (fx, fy), ei in zip(frac.tolist(), e.tolist()):
            enc.raw(fx, ei)
            enc.raw(fy, ei)
        flags = int(q.beta) | (2 if adaptive else 0) | (4 if q.sym else 0)
        extra = b""
        if q.sym:
            P = q.n_protos
            owner = q.owner.cpu().numpy()                       # sorted by prototype
            psym = sym_all[N:]
            pv = psym.astype(np.float64)
            pv[:, :3] *= mult_all[N:, :1]
            ep = pos_exp(pv, steps, q.k)
            pu = np.round(q.pxy.double().cpu().numpy() * (2.0 ** ep)[:, None]).astype(np.int64)
            pcoarse = pu >> ep[:, None]                      # floor, also for negative
            pfrac = pu - (pcoarse << ep[:, None])
            oq = np.round(q.off.double().cpu().numpy() * 2.0 ** COPY_E).astype(np.int64)
            cp = q.copy_proto.cpu().numpy()
            cols = [np.bincount(owner, minlength=P)] + _attr_cols(psym, steps, q.aref, q.G)
            cols += [pcoarse[:, 0], pcoarse[:, 1]] + _copy_cols(cp, oq)
            heads += [ac.put_ints(enc, c) for c in cols]
            for (fx, fy), ei in zip(pfrac.tolist(), ep.tolist()):
                enc.raw(fx, ei)
                enc.raw(fy, ei)
            extra = struct.pack("<HIB", P, len(cp), COPY_E)
        body = enc.finish()
        head = MAGIC + struct.pack("<HHBIfB", H, W, R, N, q.k, flags)
        if adaptive:
            head += struct.pack("<fB", q.aref, q.G)
        head += extra + steps.astype(np.float32).tobytes()
        head += b"".join(struct.pack("<Bih", m, lo, a) for m, lo, a in heads)
        return head + body


def decode(buf, device="cuda"):
    mg = bytes(buf[:4])
    assert mg in BASE_SCALE, mg
    cut, beta, symb, aref, G = 0.0, False, False, None, 0
    H, W, R, N, k = struct.unpack("<HHBIf", buf[4:17])
    o = 17
    if mg == MAGIC:
        fl = buf[o]
        beta, symb = bool(fl & 1), bool(fl & 4)
        o += 1
        if fl & 2:
            aref, G = struct.unpack("<fB", buf[o:o + 5])
            o += 5
    P = C = 0
    if symb:
        P, C, copy_e = struct.unpack("<HIB", buf[o:o + 7])
        o += 7
    nch = 7 if beta else 6
    na = _n_attr_cols(nch, aref, G)
    nheads = 4 + na + ((1 + na + 2 + 3) if symb else 0)
    steps = np.frombuffer(buf[o:o + 4 * nch], np.float32)
    o += 4 * nch
    heads = []
    for _ in range(nheads):
        heads.append(struct.unpack("<Bih", buf[o:o + 7]))
        o += 7
    dec = ac.Decoder(buf[o:])
    bh, bw = base_shape(H, W)
    nb = (H + 2 * R) // BAND + 1
    base = ac.get_ints(dec, 3 * bh * bw, heads[0])
    counts = ac.get_ints(dec, nb, heads[1])
    dx = ac.get_ints(dec, N, heads[2])
    yin = ac.get_ints(dec, N, heads[3])
    sym, mult, hi = _read_attrs(dec, N, heads, 4, steps, aref, G, nch)
    symv = sym.astype(np.float64)
    symv[:, :3] *= mult[:, None]
    e = pos_exp(symv, steps, np.float32(k))
    frac = np.array([(dec.raw(ei), dec.raw(ei)) for ei in e.tolist()], np.int64).reshape(N, 2)
    band = np.repeat(np.arange(nb), counts)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    cx = np.empty(N, np.int64)
    for s_, c in zip(starts, counts):
        cx[s_:s_ + c] = np.cumsum(dx[s_:s_ + c])
    cy = band * BAND + yin
    u = (np.stack([cx, cy], 1) << e[:, None]) + frac
    xy = (u / (2.0 ** e)[:, None] - R).astype(np.float32)
    if symb:
        pcount = ac.get_ints(dec, P, heads[hi]); hi += 1
        npb = int(pcount.sum())
        psym, pmult, hi = _read_attrs(dec, npb, heads, hi, steps, aref, G, nch)
        pv = psym.astype(np.float64)
        pv[:, :3] *= pmult[:, None]
        ep = pos_exp(pv, steps, np.float32(k))
        px = ac.get_ints(dec, npb, heads[hi]); hi += 1
        py = ac.get_ints(dec, npb, heads[hi]); hi += 1
        cproto = ac.get_ints(dec, C, heads[hi]); hi += 1
        dy = ac.get_ints(dec, C, heads[hi]); hi += 1
        cxq = ac.get_ints(dec, C, heads[hi]); hi += 1
        pfrac = np.array([(dec.raw(ei), dec.raw(ei)) for ei in ep.tolist()], np.int64).reshape(npb, 2)
        oy = np.cumsum(dy)
        ox = cxq.copy()
        for i in range(1, C):
            if dy[i] == 0:
                ox[i] = ox[i - 1] + cxq[i]
        # expand: every copy gets its prototype's blobs
        first = np.r_[0, np.cumsum(pcount)[:-1]]
        idx = np.concatenate([np.arange(first[p], first[p] + pcount[p]) for p in cproto] + [np.zeros(0, np.int64)])
        rep = np.repeat(np.arange(C), pcount[cproto])
        pxy = ((np.stack([px, py], 1) << ep[:, None]) + pfrac) / (2.0 ** ep)[:, None]
        cxy = np.stack([ox, oy], 1) / 2.0 ** copy_e
        xy = np.concatenate([xy, (pxy[idx] + cxy[rep]).astype(np.float32)])
        symv = np.concatenate([symv, pv[idx]])
    t = lambda a: torch.as_tensor(a, device=device)
    v = t(symv.astype(np.float32)) * t(steps.copy())
    base = t(base.reshape(1, 3, bh, bw).astype(np.float32) / BASE_SCALE[mg] - 0.5)
    kb = v[:, 6].exp().clamp(KMIN, KMAX) if beta else None
    return dict(xy=t(xy), L=feat_to_L(v[:, :3]), rgb=ycc2rgb(v[:, 3:6]), base=base, H=H, W=W, R=R, cut=cut, k=kb)
