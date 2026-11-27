"""Image as a sum of 2D Gaussians plus a tiny colour base grid.

pixel(p) = base(p) + sum_i rgb_i * exp(-0.5 * d^T Sigma_i^-1 d),  d = p - xy_i
Sigma_i = L_i L_i^T,  L_i = [[a, 0], [b, c]]   (stored as (a, b, c))
Each Gaussian only touches a (2R+1)^2 patch around its centre.

Beta blobs (k given): the profile exp(-q/2) becomes (1 - q/(2k))_+^k, q = d^T Sigma^-1 d.
k -> infinity is exactly the Gaussian; small k is a flat disc with a hard edge
(idea from Deformable Beta Splatting, SIGGRAPH 2025). One number per blob.
"""
import math

import torch
import torch.nn.functional as F


import os

# Fast GPU kernels, best first: CUDA C++ DLL (gsq_cuda/build.bat), Triton
# (triton-windows package), else plain PyTorch.
# GSQ_KERNEL=cuda|triton|torch forces one (for testing).
KERNELS = {}
try:
    from . import cuda_splat as _cu
    KERNELS["cuda"] = _cu.splat
except (ImportError, OSError):
    pass
try:
    from . import tri as _tri
    KERNELS["triton"] = _tri.splat
except ImportError:
    pass
KERNEL = os.environ.get("GSQ_KERNEL") or next(iter(KERNELS), "torch")
TRITON = bool(KERNELS)  # kept for older callers

KMIN, KMAX = 0.25, 12.0   # Beta sharpness range
K0 = 8.0                   # start close to a Gaussian
CUT = 0.0  # 0 = off (tested: most blobs need the full patch, grouping was 3x slower). Was 4.0: each blob is drawn within CUT * its largest sigma (<= R)


def patch_radius(L, R, cut=CUT):
    """Per-blob patch radius: min(R, ceil(cut * sigma_max)). Part of the
    format: encoder and decoders (Python, Rust) use the same rule."""
    a, b, c = L.unbind(1)
    s11, s22 = a * a, b * b + c * c
    h = (s11 + s22) / 2
    smax = (h + (h * h - (a * c) ** 2).clamp_min(0).sqrt()).sqrt()
    return torch.ceil(cut * smax).clamp(1, R).long()


def render(xy, L, rgb, base, H, W, R, cut=CUT, k=None):
    """xy (N,2) px, L (N,3) a,b,c, rgb (N,3), base (1,3,bh,bw) -> (H,W,3).
    cut > 0: blobs are grouped by their own patch radius (much less work);
    cut = 0: every blob uses the full (2R+1)^2 patch (old GSQ2 files)."""
    fast = KERNEL in KERNELS and xy.is_cuda and not cut and (k is None or KERNEL == "cuda")
    if fast:
        img = KERNELS[KERNEL](xy, L, rgb, H, W, R, k).view(H, W, 3)
        if base is not None:
            img = img + F.interpolate(base, size=(H, W), mode="bicubic",
                                      align_corners=False)[0].permute(1, 2, 0)
        return img
    out = torch.zeros(H * W, 3, device=xy.device)
    if cut:
        rad = patch_radius(L.detach(), R, cut)
        for r in torch.unique(rad).tolist():
            i = (rad == r).nonzero().squeeze(1)
            out = _splat(out, xy[i], L[i], rgb[i], H, W, r, None if k is None else k[i])
    else:
        out = _splat(out, xy, L, rgb, H, W, R, k)
    img = out.view(H, W, 3)
    if base is not None:
        img = img + F.interpolate(base, size=(H, W), mode="bicubic",
                                  align_corners=False)[0].permute(1, 2, 0)
    return img


def _splat(out, xy, L, rgb, H, W, R, k=None):
    dev = xy.device
    d = torch.arange(-R, R + 1, device=dev)
    dy, dx = torch.meshgrid(d, d, indexing="ij")
    dx, dy = dx.reshape(1, -1), dy.reshape(1, -1)
    chunk = max(1, (1 << 24) // (2 * R + 1) ** 2)
    for s in range(0, xy.shape[0], chunk):
        p, l, c = xy[s:s + chunk], L[s:s + chunk], rgb[s:s + chunk]
        kk = None if k is None else k[s:s + chunk, None]
        cx = torch.floor(p[:, 0:1]).long()
        cy = torch.floor(p[:, 1:2]).long()
        px, py = cx + dx, cy + dy
        ox = px.to(p.dtype) + 0.5 - p[:, 0:1]
        oy = py.to(p.dtype) + 0.5 - p[:, 1:2]
        u1 = ox / l[:, 0:1]
        u2 = (oy - l[:, 1:2] * u1) / l[:, 2:3]
        q = u1 * u1 + u2 * u2
        if kk is None:
            w = torch.exp(-0.5 * q)
        else:
            t = 1 - q / (2 * kk)
            w = torch.where(t > 0, torch.exp(kk * torch.log(t.clamp_min(1e-6))), torch.zeros_like(t))
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        w = w * ok
        idx = (py.clamp(0, H - 1) * W + px.clamp(0, W - 1)).reshape(-1)
        out = out.index_add(0, idx, (w.unsqueeze(-1) * c.unsqueeze(1)).reshape(-1, 3))
    return out


def base_shape(H, W):
    return max(2, math.ceil(H / 32)), max(2, math.ceil(W / 32))


class GaussImage(torch.nn.Module):
    def __init__(self, target, N, R=12, beta=False):
        super().__init__()
        H, W, _ = target.shape
        self.H, self.W, self.R = H, W, R
        self.smax = R / 3.5
        dev = target.device
        xy = self._sample(self._grad_map(target), N) + torch.rand(N, 2, device=dev)
        s0 = min(self.smax * 0.9, max(0.6, 0.6 * math.sqrt(H * W / N)))
        t = (s0 - 0.35) / (self.smax - 0.35)
        raw = math.log(t / (1 - t))
        self.xy = torch.nn.Parameter(xy)
        self.sraw = torch.nn.Parameter(torch.tensor([raw, 0.0, raw], device=dev).repeat(N, 1)
                                       + 0.05 * torch.randn(N, 3, device=dev))
        self.rgb = torch.nn.Parameter(torch.zeros(N, 3, device=dev))
        bh, bw = base_shape(H, W)
        b = F.adaptive_avg_pool2d(target.permute(2, 0, 1)[None], (bh, bw))
        self.base = torch.nn.Parameter(b.clone())
        self._s0raw = raw
        self.beta = beta
        self._k0raw = math.log((K0 - KMIN) / (KMAX - K0))
        if beta:
            self.kraw = torch.nn.Parameter(torch.full((N,), self._k0raw, device=dev))

    def K(self):
        return KMIN + (KMAX - KMIN) * torch.sigmoid(self.kraw) if getattr(self, "beta", False) else None

    @staticmethod
    def _grad_map(t):
        g = t.mean(-1)[None, None]
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=g.dtype, device=g.device)[None, None]
        gx = F.conv2d(g, kx, padding=1)
        gy = F.conv2d(g, kx.transpose(-1, -2), padding=1)
        return (gx * gx + gy * gy).sqrt()[0, 0]

    def _sample(self, weight, n):
        H, W = weight.shape
        p = 0.6 * weight.flatten() / weight.sum().clamp_min(1e-8) + 0.4 / (H * W)
        idx = torch.multinomial(p, n, replacement=n > H * W)
        return torch.stack([(idx % W).float(), (idx // W).float()], 1)

    def L(self):
        s = torch.sigmoid(self.sraw)
        a = 0.35 + (self.smax - 0.35) * s[:, 0]
        c = 0.35 + (self.smax - 0.35) * s[:, 2]
        b = self.smax * torch.tanh(self.sraw[:, 1])
        return torch.stack([a, b, c], 1)

    def forward(self):
        return render(self.xy, self.L(), self.rgb, self.base, self.H, self.W, self.R, k=self.K())

    @torch.no_grad()
    def relocate(self, opt, err, frac=0.05):
        """Move the weakest Gaussians to where the error is largest."""
        L = self.L()
        score = self.rgb.abs().sum(1) * L[:, 0] * L[:, 2]
        k = int(frac * score.numel())
        weak = torch.topk(score, k, largest=False).indices
        self.xy[weak] = self._sample(err, k) + torch.rand(k, 2, device=err.device)
        self.rgb[weak] = 0
        self.sraw[weak] = torch.tensor([self._s0raw, 0.0, self._s0raw], device=err.device) * 0.7
        ps = [self.xy, self.rgb, self.sraw]
        if self.beta:
            self.kraw[weak] = self._k0raw
            ps.append(self.kraw)
        for p in ps:
            st = opt.state.get(p)
            if st:
                st["exp_avg"][weak] = 0
                st["exp_avg_sq"][weak] = 0


def psnr(a, b):
    mse = F.mse_loss(a.clamp(0, 1), b).item()
    return 10 * math.log10(1 / max(mse, 1e-12))


def fit(target, N, iters=6000, R=20, log=None, lr=(0.05, 0.05, 0.02, 0.005), reloc_every=250, reloc_frac=0.03, reloc_until=0.7, beta=False, klr=0.05, kfreeze=0.3):
    torch.manual_seed(0)
    m = GaussImage(target, N, R, beta)
    groups = [
        {"params": [m.xy], "lr": lr[0]},
        {"params": [m.sraw], "lr": lr[1]},
        {"params": [m.rgb], "lr": lr[2]},
        {"params": [m.base], "lr": lr[3]},
    ]
    if beta:
        groups.append({"params": [m.kraw], "lr": klr})
    opt = torch.optim.Adam(fused=True, params=groups)
    lr0 = [g["lr"] for g in opt.param_groups]
    for it in range(iters):
        f = 0.5 * (1 + math.cos(math.pi * it / iters))
        for g, l0 in zip(opt.param_groups, lr0):
            g["lr"] = l0 * (0.05 + 0.95 * f)
        if beta and it < kfreeze * iters:   # let shapes settle before k moves
            opt.param_groups[-1]["lr"] = 0.0
        img = m()
        loss = F.mse_loss(img, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it and it % reloc_every == 0 and it < reloc_until * iters:
            m.relocate(opt, ((img.detach() - target) ** 2).sum(-1), reloc_frac)
        if log and (it % 500 == 0 or it == iters - 1):
            log(f"  it {it:5d}  psnr {psnr(img.detach(), target):.2f}")
    return m
