"""Charts for the README (docs/*.png) from results/*.json.

journey.png    quality gap vs AVIF per milestone (img27, 8,000 blobs)
rd_curves.png  file size vs quality: ours (v6, v5 faded) vs JPEG / WebP / AVIF / JPEG XL
screens.png    screenshots: our file size as a multiple of AVIF's, v5 -> layers
kodak.png      Kodak set (24 photos): our size vs each format at the same quality
The rate curves need the original test images in out/*_orig.png (the
Windows wallpapers are not in the repo; the synthetic doc/chart/ui are).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from PIL import Image

from bench import curve

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"Ours (blobs)": "#2a78d6", "AVIF": "#eb6834", "JPEG XL": "#8a5cd1", "WebP": "#1baf7a", "JPEG": "#eda100"}
MARK = {"Ours (blobs)": "o", "AVIF": "s", "JPEG XL": "P", "WebP": "^", "JPEG": "D"}
FMT = {"AVIF": "AVIF", "JPEG XL": "JXL", "WebP": "WEBP", "JPEG": "JPEG"}
plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def journey():
    steps = [
        ("1. Shared shape lists\n(k-means codebooks)", "v1_vq_codebooks", None),
        ("2. Learned quantisation\n+ arithmetic coder", "v2_learned_quant", None),
        ("3. Bigger blobs,\nlonger training", "v3_bigger_blobs", 1.0),
        ("4. Positions to 1/16 px", "v4_fine_positions", 1.0),
        ("5. Adaptive positions", "v5_adaptive_positions_photos", 1.0),
        ("6. Finer base grid, adaptive\nshape precision, coder (today)", "v6_photos", 0.8),
    ]
    labels, gaps = [], []
    for label, f, lam in steps:
        for r in json.load(open(ROOT / "results" / f"{f}.json")):
            if r["image"].replace("_orig", "") == "img27" and r["N"] == 8000 and (lam is None or r.get("lam") == lam):
                gaps.append(r["psnr_file"] - r.get("avif_fair", r.get("avif")))
                labels.append(label)
                break
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = range(len(labels))[::-1]
    ax.barh(list(y), gaps, color=SERIES["Ours (blobs)"], height=0.55)
    ax.axvline(0, color=INK2, linewidth=1.2)
    for yi, g in zip(y, gaps):
        ax.text(g + (0.08 if g >= 0 else -0.08), yi, f"{g:+.2f} dB", va="center",
                ha="left" if g >= 0 else "right", color=INK, fontsize=10)
    ax.set_yticks(list(y), labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(gaps) - 1.2, max(gaps) + 1.0)
    ax.set_xlabel("Quality difference vs AVIF at the same file size (dB)   ← AVIF better  |  ours better →")
    ax.set_title("How far the blob codec has come (3D render, 8,000 blobs)", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(DOCS / "journey.png", dpi=150)
    print("docs/journey.png", [round(g, 2) for g in gaps])


def rd_curves():
    photos = [("img27", "3D render"), ("img103", "Colourful stripes"),
              ("img102", "Smooth landscape"), ("img50", "Photo collage")]
    screens = [("doc", "Text document"), ("chart", "Chart"), ("ui", "App screen")]
    ours, old = {}, {}
    for f, dst in (("v6_photos", ours), ("v6_screens", ours),
                   ("v5_adaptive_positions_photos", old), ("v5_adaptive_positions_screens", old)):
        for r in json.load(open(ROOT / "results" / f"{f}.json")):
            dst.setdefault(r["image"].replace("_orig", ""), []).append((r["bytes"] / 1024, r["psnr_file"]))

    def pareto(rows):
        pts = []
        for b, p in sorted(rows):
            if not pts or p > pts[-1][1]:
                pts.append((b, p))
        return pts
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4))
    for row, group in enumerate((photos, screens)):
        for col in range(4):
            ax = axes[row][col]
            if col >= len(group):
                ax.axis("off")
                continue
            name, title = group[col]
            im = Image.open(ROOT / "out" / f"{name}_orig.png").convert("RGB")
            pts = pareto(ours[name])                   # best settings only
            hi = max(b for b, _ in pts) * 1.6
            lo = min(b for b, _ in pts) / 1.6
            for codec in ("JPEG", "WebP", "JPEG XL", "AVIF"):
                env = []
                for b, p, _ in sorted(curve(im, FMT[codec]), key=lambda x: x[0]):
                    if b / 1024 <= hi and p < 60 and (not env or p > env[-1][1]):
                        env.append((b / 1024, p))
                if env:
                    ax.plot(*zip(*env), color=SERIES[codec], linewidth=2, marker=MARK[codec],
                            markersize=4, label=codec)
            if name in old:
                ax.plot(*zip(*pareto(old[name])), color=SERIES["Ours (blobs)"], linewidth=1.5, alpha=0.35,
                        linestyle="--", marker="o", markersize=4, label="Ours before (v5)")
            ax.set_xlim(left=lo)
            ax.plot(*zip(*pts), color=SERIES["Ours (blobs)"], linewidth=2, marker="o",
                    markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, label="Ours (blobs)", zorder=5)
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_title(title, loc="left", fontsize=11, color=INK)
            ax.set_xlabel("File size (KB, log scale)")
            if col == 0:
                ax.set_ylabel("Quality (PSNR dB, higher = better)")
    h, l = axes[0][0].get_legend_handles_labels()
    order = [l.index(k) for k in list(SERIES) + ["Ours before (v5)"] if k in l]
    axes[0][3].legend([h[i] for i in order], [l[i] for i in order], frameon=False, loc="lower right")
    axes[1][3].axis("off")
    axes[1][3].text(0.0, 0.55, "Top row: rendered images and photos.\nBottom row: screenshots\n"
                    "(boxes + symbol layer).\n\nUp and left is better:\nmore quality for fewer bytes.\n\n"
                    "Dashed: our codec before (v5).", fontsize=11, color=INK2,
                    va="center", transform=axes[1][3].transAxes)
    fig.suptitle("File size vs quality: blob codec against JPEG, WebP, AVIF and JPEG XL (512 px images)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(DOCS / "rd_curves.png", dpi=130)
    print("docs/rd_curves.png")


def screens():
    """Our size as a multiple of AVIF's size at the same quality (best setting)."""
    names = [("doc", "Text document"), ("ui", "App screen"), ("chart", "Chart")]
    stages = [("v5: blobs only", "v5_adaptive_positions_screens", "#a9c7ee"),
              ("v6: + boxes + symbol layer", "v6_screens", SERIES["Ours (blobs)"])]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.36
    for k, (label, f, col) in enumerate(stages):
        rows = json.load(open(ROOT / "results" / f"{f}.json"))
        vals = []
        for n, _ in names:
            r = [x["bytes"] / x["avif_bytes_same_psnr"] for x in rows
                 if x["image"] == n and x.get("avif_bytes_same_psnr")]
            vals.append(min(r))
        xs = [i + (k - 0.5) * w for i in range(len(names))]
        ax.bar(xs, vals, width=w, color=col, label=label)
        for x, v in zip(xs, vals):
            ax.text(x, v * 1.04, f"{v:.2f}×", ha="center", va="bottom", fontsize=10, color=INK)
    ax.axhline(1, color=INK2, linewidth=1.2)
    ax.text(len(names) - 0.5, 1.03, "AVIF", ha="right", va="bottom", color=INK2, fontsize=9)
    ax.set_xticks(range(len(names)), [t for _, t in names])
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 4, 8])
    ax.set_ylim(0.5, 14)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Our file size ÷ AVIF size\n(same quality, lower is better)")
    ax.set_title("Screenshots: what boxes and the symbol layer did", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(DOCS / "screens.png", dpi=150)
    print("docs/screens.png")


def kodak():
    """Kodak 24 photos: our size ÷ each format's size at our quality."""
    import math
    rows = json.load(open(ROOT / "results" / "kodak.json"))
    fmts = [("AVIF", "avif"), ("JPEG XL", "jxl"), ("WebP", "webp"), ("JPEG", "jpeg")]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (label, k) in enumerate(fmts):
        r = [x["bytes"] / x[f"{k}_bytes_same_psnr"] for x in rows if x.get(f"{k}_bytes_same_psnr")]
        g = math.exp(sum(math.log(v) for v in r) / len(r))
        ax.scatter(r, [i] * len(r), color=SERIES[label], alpha=0.55, s=22)
        ax.plot([g, g], [i - 0.25, i + 0.25], color=INK, linewidth=2)
        ax.text(g, i - 0.28, f"average {g:.2f}×", ha="center", va="bottom", fontsize=9, color=INK)
    ax.axvline(1, color=INK2, linewidth=1.2)
    ax.set_yticks(range(len(fmts)), [f"vs {l}" for l, _ in fmts])
    ax.set_ylim(len(fmts) - 0.5, -0.75)
    ax.set_xscale("log")
    ax.set_xticks([0.75, 1, 1.5, 2, 3])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", visible=False)
    mean_p = sum(x["psnr_file"] for x in rows) / len(rows)
    ax.set_xlabel("Our file size ÷ theirs at the same quality   (left of 1× = ours smaller)")
    ax.set_title(f"Kodak photos (24 images, 768×512, 8,000 blobs, about {mean_p:.0f} dB): dots = images, "
                 "bar = average", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(DOCS / "kodak.png", dpi=150)
    print("docs/kodak.png")


if __name__ == "__main__":
    journey()
    rd_curves()
    screens()
    kodak()
