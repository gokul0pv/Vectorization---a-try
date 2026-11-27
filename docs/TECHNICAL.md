# Technical notes

A pixel image is turned into layers of simple parts, fitted on the GPU, and stored in a small file:

```
pixel = base_grid(p)                                   tiny bicubic colour grid
      + sum over boxes      colour_b * [p inside box b]   flat panels, bars, 1 px lines
      + sum over blobs      colour_i * w(p - position_i)   soft ellipses (2D Gaussians)
      + sum over copies c   sum over the blobs of prototype(c), moved to c's position
```

The blobs drawn by copies are the **symbol layer**: a shape that repeats, such as a letter or an icon, is stored once as a group of blobs (a prototype), and each copy is just a prototype number and a position. All layers are trained together, so a prototype learns from every copy of itself. A blob's weight is `w = exp(-q/2)`, where `q = dᵀ Σ⁻¹ d`. A file may instead store Beta blobs, `w = (1 - q/2k)₊^k` with a sharpness `k` per blob.

## Codec `.gsq` (`--mode sq`, the default)

1. **Find layers** (`bench.py --symbols`, `gsimg/rects.py`, `gsimg/symbols.py`), only for pictures that have them:
   - **Boxes:** flat regions are connected areas of one exact colour in the picture with small detail replaced by its local background (median, 15 px). Each region becomes one bounding box if the region plus the regions inside the box fill at least 85% of it. Boxes add their colour, so nested boxes fill the holes. Thin lines are long single-colour runs in the remaining detail. Boxes are pixel-aligned and drawn exactly with a difference image and two prefix sums.
   - **Symbols:** these are small shapes that differ from their local background, found as connected components. Shapes with the same box and a mean difference below 6 levels are one prototype, and only shapes that occur at least twice are used. A prototype gets about one blob per two ink pixels (3 to 40).
2. **Fit** (`gsimg/model.py`): Adam for 6000 steps, with blob radius limited to 20 px. Every 250 steps, the weakest 3% of blobs move to where the error is largest. One step (render, backward, fused Adam with tensor learning rates) is captured once as a **CUDA graph** and replayed; relocation runs eagerly between replays. `GSQ_GRAPHS=0` turns this off.
3. **Learned quantisation** (`gsimg/sq.py`): each blob's shape (log a, b, log c), colour (YCoCg) and, for Beta files, log k are snapped to per-channel grids whose steps are learned. Training minimises `bits + λ·255²·MSE` with a Laplace rate model per channel. Blobs drift to cheap, common values; blobs whose colour rounds to exactly 0 are dropped.
4. **Adaptive precision**: the decoder computes all of these from values it has already read, so they cost no bits.
   - **Shape:** strong blobs get a finer shape step and faint blobs a coarser one. The step is multiplied by `2^-g`, where `g = round(log2(amp / 0.05))` is clipped to −2…2 and `amp` is the blob's quantised colour amplitude. Colour is coded first, and each shape channel is coded per group `g` with its own model.
   - **Position:** the step is `2^-e` px, up to 1/128 px, with `e` from the blob's sharpness and colour amplitude.
5. **Entropy coding** (`gsimg/ac.py`): an adaptive arithmetic coder in pure Python. Each column with more than 64 values tries three codings and keeps the smallest: direct, bit-length, and bucketed (`value >> s` adaptively plus `s` raw bits).

`λ` trades size against quality, and the number of blobs `N` sets the quality ceiling. With layers, `N` counts free blobs plus prototype blobs, the ones actually stored.

### File format `GSQ4`

A header holds size, `R`, the blob count, `K_POS` and a flags byte, followed by these optional parts:
- **bit 0:** Beta blobs, adding a 7th channel log k.
- **bit 1:** adaptive shape precision, followed by float aref and byte G.
- **bit 2:** symbol layer, followed by P prototypes, C copies and the copy position step.
- **bit 3:** boxes, followed by the box count.

Then come the learned steps and one 7-byte header per coded column, then the arithmetic-coded body in this order:
1. base grid
2. free blobs (band counts, x deltas, rows in band, attribute columns, position fraction bits)
3. prototypes (blob counts, attribute columns, positions with adaptive fractions)
4. copies in reading order (prototype id, row delta, x delta; 1/64 px)
5. boxes in reading order (row delta, x0, width, height, r, g, b at 1/510)

`GSQ2` files (6 channels, base step 1/127.5) still decode, pixel-identically.

## Run

```
<gsq-env>/Scripts/python.exe bench.py IMG [IMG...] --n 4000,8000 --lam 0.2,0.8 --tag NAME
<gsq-env>/Scripts/python.exe bench.py screenshot.png --n 1000,2000 --lam 0.02,0.05 --symbols --tag NAME
<gsq-env>/Scripts/python.exe bench.py IMG --beta ...          # Beta blobs
```

`<gsq-env>` is a Python 3.12 venv with torch 2.11 (CUDA 12.8), triton-windows 3.7.1 and pillow-jxl-plugin 1.3.8.

Results:
- `out/results_NAME.json`
- `out/*_gs.png`: decoded image
- `out/*_avif.png`: AVIF at the same size

Baselines:
- JPEG and AVIF: the best of 4:2:0 and 4:4:4 colour.
- WebP: sharp YUV.
- JPEG XL: effort 7, plus its lossless size.

Fits are cached in `out/cache`. `get_kodak.sh` downloads the Kodak set (`kodak/`, not committed).

## Decoders

- **Rust (`gsq_rs/`):** `gsq_view.exe IN.gsq OUT.png [--scale 4] [--csv blobs.csv]`. Build: `cargo build --release --offline` with `%USERPROFILE%\.cargo\bin` and the MinGW `bin` on PATH.
- **C++ (`gsq_cpp/`):** `gsq.hpp` is a single-header library (C++17, no dependencies), used as `gsq::Blobs b = gsq::decode(bytes); auto rgb = gsq::render(b, scale, &w, &h);`. `gsq_view.cpp` is the same command-line tool as the Rust one. Build with GCC 16.2 (WinLibs, `%LOCALAPPDATA%\winlibs-gcc16\mingw64\bin`): `g++ -O3 -march=native -std=c++17 gsq_view.cpp -o gsq_view.exe -static`.

Both read `GSQ2` and every `GSQ4` option: Beta blobs, adaptive shape precision, the symbol layer, boxes and the bucketed columns. Both match the Python decoder within 1 brightness level on every file tested. At other scales, box corners are rounded to whole pixels.

## GPU kernel in CUDA C++: `gsq_cuda/`

- `splat.cu` holds the forward and hand-written backward pass for Gaussian and Beta blobs, with one 32×8 thread block per blob. Its gradients match PyTorch within float rounding.
- It's built as `gsq_splat.dll` with `gsq_cuda\build.bat` (portable MSVC 14.44 + CUDA 12.8, sm_120) and called through ctypes.
- `model.render` picks CUDA, then Triton (Gaussian only), then PyTorch. `GSQ_KERNEL=…` forces one.

## Speed (RTX 5060)

| Step | Before | Now |
|---|---|---|
| One fitting step, 8k blobs | 27 ms (PyTorch) → 4.97 ms (CUDA kernel, Python loop) | **3.07 ms** (CUDA graph) |
| Fit with boxes + symbols, 2000 steps | 27.6 s | **7.2 s** (CUDA graph) |
| Decode a file | about 0.35 s (Python) | **5 ms** (C++) / 6 ms (Rust) |

The graph timings above were measured while other jobs shared the GPU. Quantisation training (1600 steps) still runs eagerly, because it computes adaptive position precision on the CPU every step.

## What was tried

| Idea | Result |
|---|---|
| Beta kernel (sharpness per blob) | Text +1 to +6 dB at the same bytes; no change on other images. It needs k held fixed for the first 30% of fitting, or most blobs collapse to k < 1 and lose 7 dB. Kept as `--beta`. |
| Learned blob shapes (`exp_shape.py`) | A shared learned profile, one learned 16×16 stamp, and 8 stamps with one picked per blob: none beat the Gaussian. Blending all 8 stamps per blob costs 8 numbers per blob, and twice as many Gaussians is better for that budget. Dropped. |
| Finer base grid (1/510) | +0.1 to +2.7 dB at the same bytes. Kept. |
| Adaptive shape precision | Up to 19.5% smaller and up to +0.9 dB. Kept. |
| Mixed quantisation (rate from noise, picture from rounding) | Only 1–3% smaller. Off (`sq.MIXED`). |
| Rate model check (`exp_rate.py`) | The Laplace model is within 2–5% of the real coder. The quality lost in quantisation is the rate/quality trade, not a mispriced rate. |
| Symbol layer | Text 48–53 KB → about 9 KB at 44 dB. Kept. Fixed 1/16 px prototype and 1/8 px copy positions capped text at 41 dB; adaptive positions lifted that. |
| Boxes | UI screen and chart files 2–3× smaller. Kept. |
| Bucketed columns | 0–2.4% smaller. Kept. |
| Colour context (Co, Cg coded given Y) | +1.3% to −2% of the file after model-learning cost. Dropped. |
| Per-blob patch radius (`model.CUT`) | Most blobs need the full radius, and grouping was 3× slower. Off. |
| Codebook index hopping, two-stage codebooks, neighbour prediction | No gain. Dropped. |
