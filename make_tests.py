"""Make synthetic screenshot-type test images (no real screen content).
Writes testimgs/doc.png, chart.png, ui.png at 512x320."""
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "testimgs"
OUT.mkdir(exist_ok=True)
W, H = 512, 320
FONT = "C:/Windows/Fonts/segoeui.ttf"
f = lambda s: ImageFont.truetype(FONT, s)
rnd = random.Random(0)
WORDS = ("the image is stored as shapes so each repeated part costs only a few bytes and a "
         "computer can count search and compare them without decoding pixels first").split()


def doc():
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((24, 14), "Shape-based image format", font=f(20), fill=(20, 20, 20))
    y = 48
    for _ in range(14):
        line = " ".join(rnd.choice(WORDS) for _ in range(12))
        d.text((24, y), line, font=f(13), fill=(40, 40, 40))
        y += 18
    return im


def chart():
    im = Image.new("RGB", (W, H), (250, 250, 252))
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = 50, 30, 490, 280
    d.line((x0, y1, x1, y1), fill=(90, 90, 90), width=1)
    d.line((x0, y0, x0, y1), fill=(90, 90, 90), width=1)
    for i in range(6):
        yy = y1 - i * (y1 - y0) / 5
        d.line((x0, yy, x1, yy), fill=(225, 225, 230))
        d.text((14, yy - 8), f"{i * 20}", font=f(11), fill=(80, 80, 80))
    cols = [(31, 119, 180), (255, 127, 14), (44, 160, 44)]
    for s, c in enumerate(cols):
        pts = [(x0 + i * 22, y1 - 40 - s * 50 - 40 * math.sin(i / 3 + s) - rnd.random() * 10) for i in range(21)]
        d.line(pts, fill=c, width=2)
        for p in pts[::4]:
            d.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=c)
    for i in range(8):
        h = 20 + rnd.random() * 60
        d.rectangle((x0 + 10 + i * 54, y1 - h, x0 + 34 + i * 54, y1), fill=(180, 200, 230))
    d.text((x0 + 150, 6), "Throughput by week", font=f(14), fill=(30, 30, 30))
    return im


def ui():
    im = Image.new("RGB", (W, H), (243, 244, 246))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 36), fill=(30, 41, 59))
    d.text((14, 8), "Inspector", font=f(15), fill="white")
    d.rectangle((0, 36, 120, H), fill=(255, 255, 255))
    for i, name in enumerate(["Home", "Jobs", "Results", "Models", "Settings"]):
        if i == 1:
            d.rectangle((6, 46 + i * 34, 114, 74 + i * 34), fill=(219, 234, 254))
        d.ellipse((14, 54 + i * 34, 26, 66 + i * 34), fill=(100, 116, 139))
        d.text((34, 51 + i * 34), name, font=f(13), fill=(30, 41, 59))
    for r in range(2):
        for c in range(3):
            x, y = 134 + c * 124, 50 + r * 132
            d.rounded_rectangle((x, y, x + 114, y + 120), 8, fill="white", outline=(226, 232, 240))
            d.rectangle((x + 8, y + 8, x + 106, y + 70), fill=(rnd.randint(150, 230), rnd.randint(150, 230), rnd.randint(150, 230)))
            d.text((x + 8, y + 76), f"Job {r * 3 + c + 1}", font=f(13), fill=(15, 23, 42))
            ok = rnd.random() > 0.3
            d.rounded_rectangle((x + 8, y + 96, x + 52, y + 112), 6, fill=(220, 252, 231) if ok else (254, 226, 226))
            d.text((x + 13, y + 96), "PASS" if ok else "FAIL", font=f(11), fill=(22, 101, 52) if ok else (153, 27, 27))
    return im


for name, fn in (("doc", doc), ("chart", chart), ("ui", ui)):
    fn().save(OUT / f"{name}.png")
    print(OUT / f"{name}.png")
