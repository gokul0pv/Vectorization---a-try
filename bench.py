"""Fit, compress, decode, compare with JPEG / WebP / AVIF at the same bytes.

usage: python bench.py IMG [IMG ...] [--mode sq|vq] [--n 4000,8000] [--lam 0.02,0.05]
       [--size 512] [--iters 2000] [--tag run1]
Baselines are fair: best of 4:2:0 and 4:4:4 colour (JPEG/AVIF), sharp YUV (WebP).
"""
import argparse
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gsimg import codec, sq
from gsimg.codec import render_decoded
from gsimg.model import fit, psnr

ROOT = Path(__file__).parent
OUT = ROOT / "out"
CACHE = OUT / "cache"
BASE_OPTS = {"JPEG": [dict(subsampling=2), dict(subsampling=0)],
             "AVIF": [dict(subsampling="4:2:0"), dict(subsampling="4:4:4")],
             "WEBP": [dict(method=6), dict(method=6, use_sharp_yuv=True)]}


def load(p, size):
    im = Image.open(p).convert("RGB")
    s = size / max(im.size)
    return im.resize((round(im.width * s) // 2 * 2, round(im.height * s) // 2 * 2), Image.LANCZOS)


def to_t(im):
    return torch.tensor(np.asarray(im), dtype=torch.float32, device="cuda") / 255


def to_im(t):
    return Image.fromarray((t.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy())


def psnr8(rec, tgt):
    """PSNR of the decoded picture after rounding to 8-bit pixels (as saved)."""
    r = (rec.clamp(0, 1) * 255 + 0.5).floor()
    t = (tgt * 255).round()
    mse = ((r - t) ** 2).mean().item()
    return 10 * math.log10(255 ** 2 / max(mse, 1e-12))


def curve(im, fmt):
    """[(bytes, psnr, save_kwargs)] over qualities and colour options."""
    ref = np.asarray(im, np.float64) / 255
    pts = []
    for o in BASE_OPTS[fmt]:
        for q in list(range(5, 85, 3)) + list(range(85, 101)):
            b = io.BytesIO()
            im.save(b, fmt, quality=q, **o)
            d = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), np.float64) / 255
            pts.append((b.tell(), 10 * np.log10(1 / max(((d - ref) ** 2).mean(), 1e-12)), dict(quality=q, **o)))
    return pts


def at_size(pts, nbytes):
    ok = [p for p in pts if p[0] <= nbytes]
    return max(ok, key=lambda p: p[1]) if ok else None


def bytes_for(pts, target_psnr):
    """Bytes the codec needs for target_psnr: log-linear interpolation on its
    best-quality-per-size envelope (None if it never gets there)."""
    env = []
    for b, p, *_ in sorted(pts, key=lambda x: x[0]):
        if not env or p > env[-1][1]:
            env.append((b, p))
    if target_psnr > env[-1][1]:
        return None
    if target_psnr <= env[0][1]:
        return env[0][0]
    for (b0, p0), (b1, p1) in zip(env, env[1:]):
        if p0 <= target_psnr <= p1:
            f = (target_psnr - p0) / (p1 - p0)
            return round(math.exp(math.log(b0) + f * (math.log(b1) - math.log(b0))))


def cached_fit(name, tgt, N, iters, beta=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}_{tgt.shape[1]}_{N}_{iters}_r20{'_beta' if beta else ''}.pt"
    if f.exists():
        return torch.load(f, weights_only=False)
    m = fit(tgt, N, iters, log=print, beta=beta)
    torch.save(m, f)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imgs", nargs="+")
    ap.add_argument("--mode", default="sq", choices=["sq", "vq"])
    ap.add_argument("--n", default="4000,8000")
    ap.add_argument("--lam", default="0.05")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--k", type=float, default=sq.K_POS)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--beta", action="store_true", help="Beta blobs (sharpness k per blob, GSQ4)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    res = OUT / f"results_{a.tag}.json"
    rows = []
    for path in a.imgs:
        name = Path(path).stem
        im = load(path, a.size)
        im.save(OUT / f"{name}_orig.png")
        tgt = to_t(im)
        png = (OUT / f"{name}_orig.png").stat().st_size
        curves = {f: curve(im, f) for f in ("JPEG", "WEBP", "AVIF")}
        lams = [float(x) for x in a.lam.split(",")] if a.mode == "sq" else [None]
        for N in map(int, a.n.split(",")):
            m = cached_fit(name, tgt, N, a.iters, a.beta)
            raw = psnr(m().detach(), tgt)
            for lam in lams:
                print(f"{name} {im.size} N={N} mode={a.mode} lam={lam}")
                t0 = time.time()
                if a.mode == "sq":
                    q = sq.train(m, tgt, lam, k=a.k, log=print)
                    buf, dec, ext = sq.encode(q), sq.decode, "gsq"
                else:
                    q = codec.quantize(m, tgt, 1, 1, 1500, log=print)
                    buf, dec, ext = codec.encode(q), codec.decode, "gsi"
                t_enc = time.time() - t0
                stem = f"{name}_{N}_{a.mode}{'' if lam is None else f'_{lam}'}{'_beta' if a.beta else ''}"
                (OUT / f"{stem}.{ext}").write_bytes(buf)
                t1 = time.time()
                d = dec(buf)
                with torch.no_grad():
                    rec = render_decoded(d)
                    torch.cuda.synchronize()
                    t_dec = time.time() - t1
                    # measured on the real 8-bit picture, exactly like JPEG/WebP/AVIF
                    p = psnr8(rec, tgt)
                    to_im(rec).save(OUT / f"{stem}_gs.png")
                row = dict(image=name, N=N, beta=a.beta, kept=int(d["xy"].shape[0]), mode=a.mode, lam=lam,
                           bytes=len(buf), bpp=round(8 * len(buf) / (im.width * im.height), 3),
                           psnr_float=round(raw, 2), psnr_file=round(p, 2), png_bytes=png,
                           enc_s=round(t_enc, 1), dec_s=round(t_dec, 2))
                for f, pts in curves.items():
                    r = at_size(pts, len(buf))
                    row[f.lower()] = round(r[1], 2) if r else None
                    row[f.lower() + "_bytes_same_psnr"] = bytes_for(pts, p)
                    if r and f == "AVIF":
                        b = io.BytesIO()
                        im.save(b, f, **r[2])
                        Image.open(io.BytesIO(b.getvalue())).convert("RGB").save(OUT / f"{stem}_avif.png")
                print("  ->", json.dumps(row))
                rows.append(row)
                res.write_text(json.dumps(rows, indent=1))
    print("\n image      N    kept   lam    bytes   GS    JPEG   WebP   AVIF  | AVIF bytes for same PSNR")
    for r in rows:
        print(f" {r['image'][:8]:8} {r['N']:6} {r['kept']:6} {r['lam']!s:5} {r['bytes']:7}  {r['psnr_file']:5}"
              f"  {r['jpeg']!s:5}  {r['webp']!s:5}  {r['avif']!s:5} | {r['avif_bytes_same_psnr']}")


if __name__ == "__main__":
    main()
