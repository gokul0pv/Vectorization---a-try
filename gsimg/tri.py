"""Triton GPU kernels for the blob renderer (forward + hand-written backward).

One GPU program per blob walks its (2R+1)^2 patch. Forward adds
w * rgb into the image with atomics. Backward reads the image gradient on
the same patch and reduces the blob's 8 parameter gradients in-register
(no atomics). Same maths as model._splat:
    u1 = ox / a,  u2 = (oy - b*u1) / c,  w = exp(-(u1^2 + u2^2) / 2)
Needs the triton-windows package; model.render
falls back to plain PyTorch when it is missing.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd(xy, L, rgb, out, H, W, R: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    P: tl.constexpr = 2 * R + 1
    x = tl.load(xy + 2 * i)
    y = tl.load(xy + 2 * i + 1)
    a = tl.load(L + 3 * i)
    b = tl.load(L + 3 * i + 1)
    c = tl.load(L + 3 * i + 2)
    cr = tl.load(rgb + 3 * i)
    cg = tl.load(rgb + 3 * i + 1)
    cb = tl.load(rgb + 3 * i + 2)
    k = tl.arange(0, BLOCK)
    px = tl.floor(x).to(tl.int32) + k % P - R
    py = tl.floor(y).to(tl.int32) + k // P - R
    ox = px.to(tl.float32) + 0.5 - x
    oy = py.to(tl.float32) + 0.5 - y
    u1 = ox / a
    u2 = (oy - b * u1) / c
    w = tl.exp(-0.5 * (u1 * u1 + u2 * u2))
    m = (k < P * P) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    p = (py * W + px) * 3
    tl.atomic_add(out + p, w * cr, mask=m)
    tl.atomic_add(out + p + 1, w * cg, mask=m)
    tl.atomic_add(out + p + 2, w * cb, mask=m)


@triton.jit
def _bwd(xy, L, rgb, gout, gxy, gL, grgb, H, W, R: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    P: tl.constexpr = 2 * R + 1
    x = tl.load(xy + 2 * i)
    y = tl.load(xy + 2 * i + 1)
    a = tl.load(L + 3 * i)
    b = tl.load(L + 3 * i + 1)
    c = tl.load(L + 3 * i + 2)
    cr = tl.load(rgb + 3 * i)
    cg = tl.load(rgb + 3 * i + 1)
    cb = tl.load(rgb + 3 * i + 2)
    k = tl.arange(0, BLOCK)
    px = tl.floor(x).to(tl.int32) + k % P - R
    py = tl.floor(y).to(tl.int32) + k // P - R
    ox = px.to(tl.float32) + 0.5 - x
    oy = py.to(tl.float32) + 0.5 - y
    u1 = ox / a
    u2 = (oy - b * u1) / c
    m = (k < P * P) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    w = tl.where(m, tl.exp(-0.5 * (u1 * u1 + u2 * u2)), 0.0)
    p = (py * W + px) * 3
    g0 = tl.load(gout + p, mask=m, other=0.0)
    g1 = tl.load(gout + p + 1, mask=m, other=0.0)
    g2 = tl.load(gout + p + 2, mask=m, other=0.0)
    tl.store(grgb + 3 * i, tl.sum(w * g0, 0))
    tl.store(grgb + 3 * i + 1, tl.sum(w * g1, 0))
    tl.store(grgb + 3 * i + 2, tl.sum(w * g2, 0))
    dq = -0.5 * w * (g0 * cr + g1 * cg + g2 * cb)      # dLoss/dq, q = u1^2 + u2^2
    du2 = dq * 2.0 * u2                                # partial wrt u2
    du1 = dq * 2.0 * u1 - du2 * b / c                  # total wrt u1 (u2 depends on u1)
    tl.store(gxy + 2 * i, tl.sum(-du1 / a, 0))
    tl.store(gxy + 2 * i + 1, tl.sum(-du2 / c, 0))
    tl.store(gL + 3 * i, tl.sum(-du1 * u1 / a, 0))
    tl.store(gL + 3 * i + 1, tl.sum(-du2 * u1 / c, 0))
    tl.store(gL + 3 * i + 2, tl.sum(-du2 * u2 / c, 0))


def _block(R):
    return triton.next_power_of_2((2 * R + 1) ** 2)


class Splat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy, L, rgb, H, W, R):
        xy, L, rgb = (t.contiguous().float() for t in (xy, L, rgb))
        out = torch.zeros(H * W, 3, device=xy.device)
        if xy.shape[0]:
            _fwd[(xy.shape[0],)](xy, L, rgb, out, H, W, R=R, BLOCK=_block(R), num_warps=8)
        ctx.save_for_backward(xy, L, rgb)
        ctx.dims = (H, W, R)
        return out

    @staticmethod
    def backward(ctx, gout):
        xy, L, rgb = ctx.saved_tensors
        H, W, R = ctx.dims
        gxy, gL, grgb = torch.empty_like(xy), torch.empty_like(L), torch.empty_like(rgb)
        if xy.shape[0]:
            _bwd[(xy.shape[0],)](xy, L, rgb, gout.contiguous(), gxy, gL, grgb, H, W,
                                 R=R, BLOCK=_block(R), num_warps=8)
        return gxy, gL, grgb, None, None, None


def splat(xy, L, rgb, H, W, R):
    return Splat.apply(xy, L, rgb, H, W, R)
