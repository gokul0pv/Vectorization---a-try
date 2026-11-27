"""PyTorch autograd wrapper for the CUDA C++ kernels in gsq_cuda/gsq_splat.dll.

Loaded with ctypes (no torch C++ extension, no ninja). Raises ImportError
when the DLL is missing so model.render can fall back to Triton / PyTorch.
k (per-blob Beta sharpness) is optional: None draws Gaussians.
"""
import ctypes
from pathlib import Path

import torch

_DLL = Path(__file__).resolve().parent.parent / "gsq_cuda" / "gsq_splat.dll"
if not _DLL.exists():
    raise ImportError(f"{_DLL} not built")
_lib = ctypes.CDLL(str(_DLL))
_P, _I = ctypes.c_void_p, ctypes.c_int
_lib.gsq_forward.argtypes = [_P, _P, _P, _P, _P, _I, _I, _I, _I, _P]
_lib.gsq_backward.argtypes = [_P, _P, _P, _P, _P, _P, _P, _P, _P, _I, _I, _I, _I, _P]


def _stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _check(code):
    if code:
        raise RuntimeError(f"CUDA kernel launch failed (cudaError {code})")


def _ptr(t):
    return None if t is None else t.data_ptr()


class Splat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy, L, rgb, k, H, W, R):
        xy, L, rgb = (t.detach().contiguous().float() for t in (xy, L, rgb))
        k = None if k is None else k.detach().contiguous().float()
        out = torch.zeros(H * W, 3, device=xy.device)
        _check(_lib.gsq_forward(xy.data_ptr(), L.data_ptr(), rgb.data_ptr(), _ptr(k), out.data_ptr(),
                                xy.shape[0], H, W, R, _stream()))
        ctx.save_for_backward(xy, L, rgb, k)
        ctx.dims = (H, W, R)
        return out

    @staticmethod
    def backward(ctx, gout):
        xy, L, rgb, k = ctx.saved_tensors
        H, W, R = ctx.dims
        gout = gout.contiguous()
        gxy, gL, grgb = torch.empty_like(xy), torch.empty_like(L), torch.empty_like(rgb)
        gk = None if k is None else torch.empty_like(k)
        _check(_lib.gsq_backward(xy.data_ptr(), L.data_ptr(), rgb.data_ptr(), _ptr(k), gout.data_ptr(),
                                 gxy.data_ptr(), gL.data_ptr(), grgb.data_ptr(), _ptr(gk),
                                 xy.shape[0], H, W, R, _stream()))
        return gxy, gL, grgb, gk, None, None, None


def splat(xy, L, rgb, H, W, R, k=None):
    return Splat.apply(xy, L, rgb, k, H, W, R)
