"""Explain-it picture: original | our file | AVIF same size | what's in the file,
then the same three zoomed in, plus our file drawn at 2x.
usage: python make_figure.py NAME N LAM [SUFFIX] [--zoom X,Y]
  needs out/NAME_N_sq_LAM[SUFFIX].gsq from bench (SUFFIX is _sym for bench.py --symbols);
  --zoom X,Y puts the top-left of the 4x zoom there (default: the busiest area)."""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from gsimg import sq
from gsimg.codec import render_decoded
from gsimg.rects import render_rects

args = [a for a in sys.argv[1:] if not a.startswith("--")]
zoom_at = None
if "--zoom" in sys.argv:
    zoom_at = tuple(int(v) for v in sys.argv[sys.argv.index("--zoom") + 1].split(","))
    args = [a for a in args if a != sys.argv[sys.argv.index("--zoom") + 1]]
name, N, lam = args[0], args[1], args[2]
suffix = args[3] if len(args) > 3 else ""
OUT = Path(__file__).parent / "out"
stem = f"{name}_{N}_sq_{lam}{suffix}"
buf = (OUT / f"{stem}.gsq").read_bytes()
orig = Image.open(OUT / f"{name}_orig.png").convert("RGB")
ours = Image.open(OUT / f"{stem}_gs.png").convert("RGB")
avif = Image.open(OUT / f"{stem}_avif.png").convert("RGB")
W, H = orig.size
d = sq.decode(buf)
font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 15)
small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 13)
BLUE, ORANGE, GREY = (42, 120, 214), (235, 104, 52), (120, 120, 116)


def psnr(a):
    x, y = np.asarray(a, np.float64) / 255, np.asarray(orig, np.float64) / 255
    return 10 * np.log10(1 / ((x - y) ** 2).mean())


# ---- what's in the file: boxes (grey), repeated shapes (blue), free blobs (orange)
S = 2                                                       # drawn at 2x, then reduced
xy, L = d["xy"].cpu().numpy(), d["L"].cpu().numpy()
nf = d.get("n_free", len(xy))
has_boxes = d.get("boxes") is not None
if has_boxes:   # faint picture of the boxes alone, so the layout reads
    bimg = render_rects(d["boxes"], d["brgb"], H, W) + torch.nn.functional.interpolate(
        d["base"], size=(H, W), mode="bicubic", align_corners=False)[0].permute(1, 2, 0)
    under = (bimg.clamp(0, 1) * 255).byte().cpu().numpy()
else:
    under = np.full((H, W, 3), 255, np.uint8)
under = (0.35 * under + 0.65 * 250).astype(np.uint8)        # washed out
blob = Image.fromarray(under).resize((W * S, H * S), Image.NEAREST)
bd = ImageDraw.Draw(blob)
if has_boxes:
    for x0, y0, x1, y1 in d["boxes"].cpu().numpy().tolist():
        bd.rectangle((x0 * S, y0 * S, x1 * S - 1, y1 * S - 1), outline=GREY, width=1)
t = np.linspace(0, 2 * np.pi, 24)
circ = np.stack([np.cos(t), np.sin(t)])
for i, ((x, y), (a, b, c)) in enumerate(zip(xy, L)):
    pts = (np.array([[a, 0], [b, c]]) @ circ + np.array([[x], [y]])) * S   # 1-sigma ellipse
    bd.line(list(map(tuple, pts.T)) + [tuple(pts[:, 0])], fill=ORANGE if i < nf else BLUE, width=1)
blob = blob.resize((W, H), Image.LANCZOS)

# 2x render from the same file
x2 = render_decoded(d, 2.0)
x2 = Image.fromarray((x2.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy())

kb = len(buf) / 1024
png_kb = (OUT / f"{name}_orig.png").stat().st_size / 1024
parts = [f"{nf:,} free blobs"]
if d.get("n_protos"):
    parts.append(f"{d['n_protos']} shapes ({d['n_proto_blobs']:,} blobs) drawn {d['n_copies']} times")
if has_boxes:
    parts.append(f"{len(d['boxes'])} boxes")
keys = [("free blobs", ORANGE)] + ([("repeated shapes", BLUE)] if d.get("n_protos") else []) +        ([("boxes", GREY)] if has_boxes else [])
ld = ImageDraw.Draw(blob)
lw, lh = 28 + max(ld.textlength(k, font=small) for k, _ in keys), 18 * len(keys)
ink = np.abs(np.asarray(orig, np.float32) - np.median(np.asarray(orig, np.float32), (0, 1))).sum(2)
corners = [(8, 8), (W - lw - 8, 8), (8, H - lh - 8), (W - lw - 8, H - lh - 8)]
lx, ly = min(corners, key=lambda c: ink[int(c[1]):int(c[1] + lh), int(c[0]):int(c[0] + lw)].sum())   # emptiest corner
ld.rectangle((lx - 4, ly - 4, lx + lw, ly + lh + 2), fill=(255, 255, 255), outline=(210, 210, 206))
for j, (k, col) in enumerate(keys):
    ld.line((lx, ly + 18 * j + 8, lx + 18, ly + 18 * j + 8), fill=col, width=3)
    ld.text((lx + 24, ly + 18 * j), k, font=small, fill=(40, 40, 40))
tiles = [
    (orig, "Original", f"PNG {png_kb:.0f} KB (lossless)"),
    (ours, "Our file", f"{kb:.1f} KB  -  quality {psnr(ours):.1f} dB"),
    (avif, "AVIF, same size", f"{kb:.1f} KB  -  quality {psnr(avif):.1f} dB"),
    (blob, "What's in our file", ", ".join(parts)),
]
# 4x zoom of the chosen (or busiest) area
cw, ch = W // 4, H // 4
if zoom_at is None:
    g = np.asarray(orig.convert("L"), np.float32)
    e = np.abs(np.diff(g, axis=0))[:, :-1] + np.abs(np.diff(g, axis=1))[:-1]
    zoom_at = max(((x, y) for x in range(0, W - cw, 16) for y in range(0, H - ch, 16)),
                  key=lambda p: e[p[1]:p[1] + ch, p[0]:p[0] + cw].sum())
box = (zoom_at[0], zoom_at[1], zoom_at[0] + cw, zoom_at[1] + ch)
zoom = [(im.crop(box).resize((W, H), Image.NEAREST), "Zoom: " + lab, "") for im, lab, _ in tiles[:3]]
zx2 = x2.crop(tuple(v * 2 for v in box)).resize((W, H), Image.LANCZOS)
zoom.append((zx2, "Zoom: our file drawn at 2x", "same file, no extra bytes; shows how the parts scale"))

pad, top = 10, 46
fig = Image.new("RGB", (4 * W + 5 * pad, 2 * (H + top) + 3 * pad), "white")
fd = ImageDraw.Draw(fig)
for r, row in enumerate((tiles, zoom)):
    for c, (im, title, sub) in enumerate(row):
        x, y = pad + c * (W + pad), pad + r * (H + top + pad)
        while fd.textlength(title, font=font) > W and " " in title:      # keep titles in their column
            title = title.rsplit(" ", 1)[0]
        fd.text((x, y), title, font=font, fill=(20, 20, 20))
        fd.text((x, y + 21), sub, font=small, fill=(90, 90, 90))
        fig.paste(im, (x, y + top))
    if r == 0:
        x0, y0 = pad + box[0], pad + top + box[1]
        fd.rectangle((x0, y0, x0 + cw, y0 + ch), outline=(255, 40, 40), width=2)
fig.save(OUT / f"figure_{name}.png")
print(OUT / f"figure_{name}.png")
