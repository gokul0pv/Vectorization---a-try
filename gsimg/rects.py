"""Rectangle layer: flat panels, bars, swatches and 1 px lines (borders,
gridlines) as pixel-aligned boxes. Blobs are poor at long straight edges
and big flat areas; a box is 4 small integers + a colour.

A box covers pixels x0 <= x < x1, y0 <= y < y1 and ADDS its colour (like
blobs add theirs), so boxes can nest: a button on a panel adds the
difference. Drawn exactly with a difference image + two prefix sums.
"""
import cv2
import numpy as np
import torch


def _runs(row, tol):
    """Maximal runs of near-equal colour in one row: list of (x0, x1)."""
    d = np.abs(np.diff(row.astype(np.int32), axis=0)).max(1) > tol
    cut = np.r_[0, np.nonzero(d)[0] + 1, len(row)]
    return list(zip(cut[:-1], cut[1:]))


def _merge(img, mask, tol, min_len):
    """Runs (len >= min_len, inside mask) merged downwards into boxes with
    the same x range and colour. Returns [(x0, y0, x1, y1, colour)]."""
    H, W, _ = img.shape
    open_, out = {}, []
    for y in range(H):
        row = img[y]
        nxt = {}
        for x0, x1 in _runs(row, tol):
            if x1 - x0 < min_len or not mask[y, x0:x1].all():
                continue
            col = row[x0:x1].mean(0)
            key = (x0, x1)
            r = open_.pop(key, None)
            if r is not None and np.abs(r[4] - col).max() <= tol:
                r[3] = y + 1
                nxt[key] = r
            else:
                if r is not None:
                    out.append(r)
                nxt[key] = [x0, y, x1, y + 1, col]
        out += open_.values()
        open_ = nxt
    out += open_.values()
    return out


def _regions(img, min_area, fill=0.85):
    """One box per connected region of one exact colour: its bounding box,
    if the region plus the regions lying inside that box fill >= `fill` of
    it (nested boxes add their colour, so holes are filled by the boxes
    inside). Rounded corners and anti-aliased edges are left to blobs."""
    H, W, _ = img.shape
    key = (img[..., 0].astype(np.int64) << 16) | (img[..., 1].astype(np.int64) << 8) | img[..., 2]
    vals, cnt = np.unique(key, return_counts=True)
    regs = []
    for v in vals[cnt >= min_area]:
        n, lab, st, _ = cv2.connectedComponentsWithStats((key == v).astype(np.uint8), connectivity=4)
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if area >= min_area:
                regs.append((x, y, x + w, y + h, area))
    regs.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
    out = []
    for j, (x0, y0, x1, y1, area) in enumerate(regs):
        inner = sum(r[4] for r in regs[:j] if r[0] >= x0 and r[1] >= y0 and r[2] <= x1 and r[3] <= y1)
        if (area + inner) >= fill * (x1 - x0) * (y1 - y0):
            out.append([x0, y0, x1, y1, None])
    return out


def find_rects(a, tol=2, min_area=24, line_len=10, fg_thr=24):
    """a: H x W x 3 float array (0..255). Returns int array (n, 4) of boxes
    x0, y0, x1, y1 (flat regions first, then thin lines)."""
    a8 = a.astype(np.uint8)
    bg = cv2.medianBlur(a8, 15)
    fg = np.abs(a - bg.astype(np.float32)).max(2) > fg_thr
    # 1) flat regions: the picture with small detail replaced by its background
    clean = np.where(fg[..., None], bg, a8)
    flat = _regions(clean, min_area)
    # 2) thin lines in the detail: long runs, horizontally and vertically
    hl = [r for r in _merge(a8, fg, tol, line_len) if r[3] - r[1] <= 2]
    vl = _merge(np.ascontiguousarray(a8.transpose(1, 0, 2)), fg.T, tol, line_len)
    vl = [[r[1], r[0], r[3], r[2], r[4]] for r in vl if r[3] - r[1] <= 2]
    boxes = [r[:4] for r in flat + hl + vl]
    return np.array(boxes, np.int64).reshape(-1, 4)


def render_rects(boxes, rgb, H, W):
    """boxes (n, 4) int tensor x0 y0 x1 y1, rgb (n, 3) -> (H, W, 3), exact."""
    D = torch.zeros(H + 1, W + 1, 3, device=rgb.device, dtype=rgb.dtype)
    x0, y0, x1, y1 = boxes.unbind(1)
    ys = torch.cat([y0, y0, y1, y1])
    xs = torch.cat([x0, x1, x0, x1])
    v = torch.cat([rgb, -rgb, -rgb, rgb])
    D = D.index_put((ys, xs), v, accumulate=True)
    return D.cumsum(0).cumsum(1)[:H, :W]


def init_colours(boxes, target):
    """Colours that reproduce the target's mean inside each box, painting
    big boxes first and adding differences for boxes on top of them."""
    H, W, _ = target.shape
    order = np.argsort(-((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
    canvas = torch.zeros_like(target)
    rgb = torch.zeros(len(boxes), 3, device=target.device)
    for i in order:
        x0, y0, x1, y1 = boxes[i].tolist()
        d = (target[y0:y1, x0:x1] - canvas[y0:y1, x0:x1]).reshape(-1, 3).mean(0)
        rgb[i] = d
        canvas[y0:y1, x0:x1] += d
    return rgb
