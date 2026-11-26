"""Re-score baselines fairly: best of 4:2:0 and 4:4:4 at each size."""
import io, json
import numpy as np
from PIL import Image
r = json.load(open("out/results.json"))
cache = {}
def curve(name, fmt):
    k = (name, fmt)
    if k in cache: return cache[k]
    im = Image.open(f"out/{name}_orig.png").convert("RGB"); ref = np.asarray(im, np.float64) / 255
    opts = {"JPEG": [dict(subsampling=2), dict(subsampling=0)],
            "AVIF": [dict(subsampling="4:2:0"), dict(subsampling="4:4:4")],
            "WEBP": [dict(method=6), dict(method=6, use_sharp_yuv=True)]}[fmt]
    pts = []
    for o in opts:
        for q in list(range(5, 96, 3)) + [97, 99, 100]:
            b = io.BytesIO(); im.save(b, fmt, quality=q, **o)
            d = np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"), np.float64) / 255
            pts.append((b.tell(), 10 * np.log10(1 / max(((d - ref) ** 2).mean(), 1e-12))))
    cache[k] = pts; return pts
print(" image     N     bytes  GS(file) GS(float) JPEG   WebP   AVIF   (best PSNR dB within same bytes)")
for x in r:
    for f in ("JPEG", "WEBP", "AVIF"):
        ok = [p for p in curve(x["image"], f) if p[0] <= x["bytes"]]
        x[f.lower() + "_fair"] = round(max(p[1] for p in ok), 2) if ok else None
    print(f" {x['image']:8} {x['N']:6} {x['bytes']:7}  {x['psnr_file']:6}   {x['psnr_float']:6}   {x['jpeg_fair']!s:6} {x['webp_fair']!s:6} {x['avif_fair']!s:6}")
json.dump(r, open("out/results.json", "w"), indent=1)
