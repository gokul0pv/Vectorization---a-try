"""Explain-it picture: original | our file | AVIF same size | the blobs.
usage: python make_figure.py NAME N LAM  (needs out/NAME_N_sq_LAM.gsq from bench)"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from gsimg import sq
from gsimg.codec import render_decoded

name, N, lam = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = Path(__file__).parent / "out"
stem = f"{name}_{N}_sq_{lam}"
buf = (OUT / f"{stem}.gsq").read_bytes()
orig = Image.open(OUT / f"{name}_orig.png").convert("RGB")
ours = Image.open(OUT / f"{stem}_gs.png").convert("RGB")
avif = Image.open(OUT / f"{stem}_avif.png").convert("RGB")
W, H = orig.size
d = sq.decode(buf)
font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 15)
small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 13)


def psnr(a):
    x, y = np.asarray(a, np.float64) / 255, np.asarray(orig, np.float64) / 255
    return 10 * np.log10(1 / ((x - y) ** 2).mean())


# blob map: every Gaussian drawn as its 1-sigma ellipse, coloured by its colour
xy = d["xy"].cpu().numpy()
L = d["L"].cpu().numpy()
rgb = d["rgb"].cpu().numpy()
blob = Image.new("RGB", (W * 2, H * 2), (18, 18, 22))
bd = ImageDraw.Draw(blob)
t = np.linspace(0, 2 * np.pi, 20)
circ = np.stack([np.cos(t), np.sin(t)])
for (x, y), (a, b, c), col in zip(xy, L, rgb):
    pts = np.array([[a, 0], [b, c]]) @ circ          # 1-sigma ellipse
    pts = (pts + np.array([[x], [y]])) * 2
    cc = tuple(int(v) for v in np.clip(np.abs(col) / max(np.abs(col).max(), 1e-3) * 255, 40, 255))
    bd.line(list(map(tuple, pts.T)) + [tuple(pts[:, 0])], fill=cc, width=1)
blob = blob.resize((W, H), Image.LANCZOS)

# 2x render from the same file
x2 = render_decoded(d, 2.0)
x2 = Image.fromarray((x2.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy())

kb = len(buf) / 1024
png_kb = (OUT / f"{name}_orig.png").stat().st_size / 1024
tiles = [
    (orig, "Original", f"PNG {png_kb:.0f} KB (lossless)"),
    (ours, "Our blob file", f"{kb:.0f} KB  -  quality {psnr(ours):.1f} dB"),
    (avif, "AVIF, same size", f"{kb:.0f} KB  -  quality {psnr(avif):.1f} dB"),
    (blob, f"The {len(xy)} blobs inside our file", "each = position + shape + colour"),
]
# zoom crops (3x) of the most detailed area
g = np.asarray(orig.convert("L"), np.float32)
e = np.abs(np.diff(g, axis=0))[:, :-1] + np.abs(np.diff(g, axis=1))[:-1]
cw, ch = W // 4, H // 4
best = max(((x, y) for x in range(0, W - cw, 16) for y in range(0, H - ch, 16)),
           key=lambda p: e[p[1]:p[1] + ch, p[0]:p[0] + cw].sum())
box = (best[0], best[1], best[0] + cw, best[1] + ch)
zoom = [(im.crop(box).resize((W, H), Image.NEAREST), "Zoom: " + lab, "") for im, lab, _ in tiles[:3]]
zx2 = x2.crop(tuple(v * 2 for v in box)).resize((W, H), Image.LANCZOS)
zoom.append((zx2, "Zoom: our file drawn at 2x", "same file, sharper; no extra bytes"))

pad, top = 10, 46
fig = Image.new("RGB", (4 * W + 5 * pad, 2 * (H + top) + 3 * pad), "white")
fd = ImageDraw.Draw(fig)
for r, row in enumerate((tiles, zoom)):
    for c, (im, title, sub) in enumerate(row):
        x, y = pad + c * (W + pad), pad + r * (H + top + pad)
        fd.text((x, y), title, font=font, fill=(20, 20, 20))
        fd.text((x, y + 21), sub, font=small, fill=(90, 90, 90))
        fig.paste(im, (x, y + top))
    if r == 0:
        x0, y0 = pad + box[0], pad + top + box[1]
        fd.rectangle((x0, y0, x0 + cw, y0 + ch), outline=(255, 40, 40), width=2)
fig.save(OUT / f"figure_{name}.png")
print(OUT / f"figure_{name}.png")
