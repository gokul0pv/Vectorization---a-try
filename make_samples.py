"""Make the published sample images (all free to share):
  samples/spheres/original.png   - glossy spheres, ray-traced here (NumPy)
  samples/ribbons/original.png   - abstract glossy colour ribbons, generated here
  samples/earthrise/original.png - Apollo 8 "Earthrise" (NASA, public domain),
                                   from images-assets.nasa.gov as08-14-2383~large.jpg
usage: python make_samples.py [path/to/as08-14-2383~large.jpg]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent / "samples"
W, H, SS = 512, 320, 2   # output size, supersampling


def norm(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def spheres():
    w, h = W * SS, H * SS
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    cam = np.array([0.0, 1.1, -5.0])
    d = norm(np.stack([(xs - w / 2) / h * 1.2, -(ys - h / 2) / h * 1.2, np.ones_like(xs)], -1))
    S = [  # centre, radius, colour, reflectivity
        (np.array([-1.55, 0.75, 0.6]), 0.75, np.array([0.95, 0.30, 0.18]), 0.35),
        (np.array([0.15, 0.95, 1.4]), 0.95, np.array([0.20, 0.45, 0.95]), 0.45),
        (np.array([1.75, 0.55, 0.2]), 0.55, np.array([0.98, 0.78, 0.20]), 0.30),
        (np.array([0.75, 0.32, -0.9]), 0.32, np.array([0.30, 0.85, 0.55]), 0.25),
        (np.array([-0.55, 0.25, -1.3]), 0.25, np.array([0.85, 0.40, 0.85]), 0.25),
    ]
    light = norm(np.array([-0.6, 0.9, -0.5]))

    def sky(dirs):
        t = np.clip(dirs[..., 1:2] * 1.5 + 0.3, 0, 1)
        return (1 - t) * np.array([0.98, 0.86, 0.74]) + t * np.array([0.38, 0.55, 0.85])

    def hit(o, dirs):
        best = np.full(dirs.shape[:-1], np.inf)
        idx = np.full(dirs.shape[:-1], -1)
        for k, (c, r, _, _) in enumerate(S):
            oc = o - c
            b = (oc * dirs).sum(-1)
            q = (oc * oc).sum(-1) - r * r
            disc = b * b - q
            t = -b - np.sqrt(np.maximum(disc, 0))
            ok = (disc > 0) & (t > 1e-4) & (t < best)
            best = np.where(ok, t, best)
            idx = np.where(ok, k, idx)
        tp = np.where(dirs[..., 1] < -1e-6, -o[..., 1] / np.where(dirs[..., 1] < -1e-6, dirs[..., 1], -1), np.inf)
        ok = (tp > 1e-4) & (tp < best)
        best = np.where(ok, tp, best)
        idx = np.where(ok, len(S), idx)       # ground plane
        return best, idx

    def shade(o, dirs, depth):
        t, idx = hit(o, dirs)
        col = sky(dirs)
        p = o + dirs * np.where(np.isinf(t), 0, t)[..., None]
        n = np.zeros_like(p)
        base = np.zeros_like(p)
        refl = np.zeros(p.shape[:-1])
        for k, (c, r, cc, rf) in enumerate(S):
            m = idx == k
            n[m] = (p[m] - c) / r
            base[m] = cc
            refl[m] = rf
        g = idx == len(S)
        n[g] = [0, 1, 0]
        dist = np.hypot(p[..., 0], p[..., 2] - 0.5)
        base[g] = np.array([0.86, 0.84, 0.80]) * (0.75 + 0.25 * np.exp(-dist[g] / 6))[:, None]
        refl[g] = 0.12
        st, _ = hit(p + n * 1e-3, np.broadcast_to(light, p.shape))
        lit = np.isinf(st) * 1.0
        diff = np.clip((n * light).sum(-1), 0, 1) * (0.25 + 0.75 * lit)
        hv = norm(light - dirs)
        spec = np.clip((n * hv).sum(-1), 0, 1) ** 80 * lit
        c = base * (0.22 + 0.85 * diff)[..., None] + spec[..., None] * 0.9
        if depth > 0:
            rd = norm(dirs - 2 * (dirs * n).sum(-1, keepdims=True) * n)
            c = c * (1 - refl[..., None]) + shade(p + n * 1e-3, rd, depth - 1) * refl[..., None]
        return np.where((idx >= 0)[..., None], c, col)

    img = shade(np.broadcast_to(cam, d.shape), d, 2)
    img = np.clip(img, 0, 1) ** (1 / 1.1)
    return Image.fromarray((img * 255 + 0.5).astype(np.uint8)).resize((W, H), Image.LANCZOS)


def ribbons():
    w, h = W * SS, H * SS
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    x, y = xs / h, ys / h
    img = np.zeros((h, w, 3)) + np.array([0.07, 0.06, 0.10])
    img += (1 - y[..., None]) * np.array([0.05, 0.03, 0.10])
    hues = [(0.95, 0.25, 0.35), (0.98, 0.55, 0.15), (0.20, 0.75, 0.85), (0.55, 0.35, 0.95), (0.98, 0.80, 0.25)]
    for k, c in enumerate(hues):
        c = np.array(c)
        yc = 0.25 + 0.13 * k + 0.12 * np.sin(2.1 * x + 1.3 * k) + 0.05 * np.sin(5.3 * x + k)
        wd = 0.035 + 0.02 * np.sin(1.7 * x + k) ** 2
        u = (y - yc) / wd                       # across the ribbon, -1..1 inside
        a = np.clip(1.2 - np.abs(u), 0, 1) ** 1.5
        shade = 0.45 + 0.55 * np.cos(np.clip(u, -1.5, 1.5) * 1.1) ** 2
        gloss = np.exp(-((u + 0.35) / 0.18) ** 2) * (0.6 + 0.4 * np.sin(3 * x + k))
        col = c * shade[..., None] + gloss[..., None] * 0.8
        img = img * (1 - a[..., None]) + col * a[..., None]
    img = np.clip(img, 0, 1)
    return Image.fromarray((img * 255 + 0.5).astype(np.uint8)).resize((W, H), Image.LANCZOS)


def earthrise(src):
    im = Image.open(src).convert("RGB")
    s = W / im.width                            # crop a 512x320 band around Earth + horizon
    im = im.resize((W, round(im.height * s)), Image.LANCZOS)
    g = np.asarray(im.convert("L"), np.float32)
    rows = g.mean(1)
    top = int(np.clip(np.argmax(rows > 25) - 40, 0, im.height - H))
    return im.crop((0, top, W, top + H))


if __name__ == "__main__":
    out = {"spheres": spheres(), "ribbons": ribbons()}
    if len(sys.argv) > 1:
        out["earthrise"] = earthrise(sys.argv[1])
    for name, im in out.items():
        (ROOT / name).mkdir(parents=True, exist_ok=True)
        im.save(ROOT / name / "original.png")
        print(ROOT / name / "original.png", im.size)
