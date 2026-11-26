"""Re-measure our PSNR on the saved 8-bit decoded pictures (out/*_gs.png),
like JPEG/WebP/AVIF, and redo 'bytes the other codec needs for the same
quality'. Fixes results written before bench.py measured on 8-bit output.
usage: python rescore8.py out/results_TAG.json [...]"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from bench import bytes_for, curve

OUT = Path(__file__).parent / "out"
for f in sys.argv[1:]:
    rows = json.load(open(f))
    curves = {}
    for r in rows:
        name = r["image"]
        stem = f"{name}_{r['N']}_{r['mode']}" + ("" if r["lam"] is None else f"_{r['lam']}")
        orig = Image.open(OUT / f"{name}_orig.png").convert("RGB")
        dec = np.asarray(Image.open(OUT / f"{stem}_gs.png").convert("RGB"), np.float64)
        mse = ((dec - np.asarray(orig, np.float64)) ** 2).mean()
        old, r["psnr_file"] = r["psnr_file"], round(10 * np.log10(255 ** 2 / max(mse, 1e-12)), 2)
        if name not in curves:
            curves[name] = {c: curve(orig, c) for c in ("JPEG", "WEBP", "AVIF")}
        for c, pts in curves[name].items():
            r[c.lower() + "_bytes_same_psnr"] = bytes_for(pts, r["psnr_file"])
        r["psnr_measure"] = "8-bit"
        print(f"{stem:28} {old:6} -> {r['psnr_file']:6}")
    json.dump(rows, open(f, "w"), indent=1)
