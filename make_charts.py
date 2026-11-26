"""Charts for the README (docs/*.png) from results/*.json.

journey.png    quality gap vs AVIF per milestone (img27, 8,000 blobs)
rd_curves.png  file size vs quality: ours vs JPEG / WebP / AVIF, per image
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
SERIES = {"Ours (blobs)": "#2a78d6", "AVIF": "#eb6834", "WebP": "#1baf7a", "JPEG": "#eda100"}
MARK = {"Ours (blobs)": "o", "AVIF": "s", "WebP": "^", "JPEG": "D"}
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
        ("5. Adaptive positions\n(today)", "v5_adaptive_positions_photos", 1.0),
    ]
    labels, gaps = [], []
    for label, f, lam in steps:
        for r in json.load(open(ROOT / "results" / f"{f}.json")):
            if r["image"] == "img27" and r["N"] == 8000 and (lam is None or r.get("lam") == lam):
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
    ours = {}
    for f in ("v5_adaptive_positions_photos", "v5_adaptive_positions_screens"):
        for r in json.load(open(ROOT / "results" / f"{f}.json")):
            ours.setdefault(r["image"], []).append((r["bytes"] / 1024, r["psnr_file"]))
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4))
    for row, group in enumerate((photos, screens)):
        for col in range(4):
            ax = axes[row][col]
            if col >= len(group):
                ax.axis("off")
                continue
            name, title = group[col]
            im = Image.open(ROOT / "out" / f"{name}_orig.png").convert("RGB")
            pts = []                                   # best (Pareto) settings only
            for b, p in sorted(ours[name]):
                if not pts or p > pts[-1][1]:
                    pts.append((b, p))
            hi = max(b for b, _ in pts) * 1.6
            for codec in ("JPEG", "WebP", "AVIF"):
                env = []
                for b, p, _ in sorted(curve(im, codec.upper()), key=lambda x: x[0]):
                    if b / 1024 <= hi and p < 60 and (not env or p > env[-1][1]):
                        env.append((b / 1024, p))
                if env:
                    ax.plot(*zip(*env), color=SERIES[codec], linewidth=2, marker=MARK[codec],
                            markersize=4, label=codec)
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
    order = [l.index(k) for k in SERIES if k in l]
    axes[0][3].legend([h[i] for i in order], [l[i] for i in order], frameon=False, loc="lower right")
    axes[1][3].axis("off")
    axes[1][3].text(0.0, 0.55, "Top row: rendered images and photos.\nBottom row: screenshots.\n\n"
                    "Up and left is better:\nmore quality for fewer bytes.", fontsize=11, color=INK2,
                    va="center", transform=axes[1][3].transAxes)
    fig.suptitle("File size vs quality: blob codec against JPEG, WebP and AVIF (512 px images)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(DOCS / "rd_curves.png", dpi=130)
    print("docs/rd_curves.png")


if __name__ == "__main__":
    journey()
    rd_curves()
