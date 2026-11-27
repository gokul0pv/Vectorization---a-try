"""Experiment: a symbol layer. Shapes that repeat (letters, icons) are drawn by
ONE prototype group of blobs, stored once; every copy is just
(prototype id, position). Free blobs draw the rest. All trained together,
so each prototype learns from every copy of itself.

Compares at the same number of STORED blobs (free + prototype blobs):
    plain : N free blobs
    symbol: N - (prototype blobs) free blobs + prototypes + copies
usage: python exp_symbols.py IMG N [ITERS]"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bench import load, psnr8, to_t
from gsimg.model import fit
from gsimg.symbols import SymbolImage, find_symbols, fit_symbols


if __name__ == "__main__":
    path, N = sys.argv[1], int(sys.argv[2])
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
    im = load(path, 512)
    t = to_t(im)
    syms = find_symbols(np.asarray(im).astype(np.float32))
    copies = sum(len(s["origins"]) for s in syms)
    print(f"{Path(path).stem}: {len(syms)} prototypes, {copies} copies")
    t0 = time.time()
    m0 = fit(t, N, iters)
    with torch.no_grad():
        print(f"  plain  {N} free blobs: {psnr8(m0(), t):.2f} dB  ({time.time() - t0:.0f}s)", flush=True)
    probe = SymbolImage(t, 1, syms)
    n_free = N - probe.n_proto_blobs
    t0 = time.time()
    m1 = fit_symbols(t, n_free, syms, iters)
    with torch.no_grad():
        print(f"  symbol {n_free} free + {m1.n_proto_blobs} prototype blobs (= {N} stored) "
              f"+ {m1.n_copies} copies: {psnr8(m1(), t):.2f} dB  ({time.time() - t0:.0f}s)", flush=True)
