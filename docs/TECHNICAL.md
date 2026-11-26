# Technical notes

A pixel image is turned into a few thousand soft elliptical blobs (2D Gaussians), fitted on the GPU, and stored in a small file. The file renders at any resolution.

```
pixel = base_grid(p) + sum_i colour_i * exp(-0.5 * d^T Sigma_i^-1 d)
```

## Codec `.gsq` (default, `--mode sq`)
1. **Fit** (`gsimg/model.py`): 6000 Adam steps. Blob radius is limited to 20 px. Every 250 steps, the weakest 3% of blobs are moved to where the error is largest.
2. **Learned quantisation** (`gsimg/sq.py`): each blob's shape (log a, b, log c) and colour (YCoCg) are snapped to a per-channel grid whose step is learned. Training minimises `bits + λ·255²·MSE`, with a Laplace rate model per channel. Blobs drift to cheap, common values, and invisible ones (colour exactly 0) are dropped.
3. **Positions**: kept to 1/16 px (1/4 px cost 4–9 dB) and coded in 8 px bands.
4. **Entropy coding** (`gsimg/ac.py`): an adaptive arithmetic coder, written in pure Python.

`λ` trades size against quality. The number of blobs `N` sets the quality ceiling.

## Older codec `.gsi` (`--mode vq`)
`gsimg/codec.py` holds k-means shape and colour codebooks (256 entries each, 1 byte per index). It's about 4 dB worse; kept for comparison.

## Run
Encode and benchmark with the fast GPU kernels (Triton):
```
<gsq-env>/Scripts/python.exe bench.py IMG [IMG...] --n 4000,8000 --lam 0.3,1.0 --tag NAME
```
`<gsq-env>` is a Python 3.12 venv with torch 2.11 (CUDA 12.8) and triton-windows 3.7.1. Without Triton, the code falls back to plain PyTorch, which is about 7× slower per step.

Results:
- `out/results_NAME.json`
- `out/*_gs.png`: decoded image
- `out/*_avif.png`: AVIF at the same size

Baselines are fair: the best of 4:2:0 and 4:4:4 colour for JPEG and AVIF, and sharp-YUV for WebP. Fits are cached in `out/cache`.

## Decoder in Rust: `gsq_rs/`
```
gsq_rs/target/release/gsq_view.exe IN.gsq OUT.png [--scale 4] [--csv blobs.csv]
```
- No Python and no GPU needed.
- Decodes in 3–7 ms and renders in about 50 ms on all CPU cores, matching Python within 1 brightness level.
- `--scale` draws at any size; `--csv` lists every blob (x, y, shape, colour).
- Build: `PATH=%USERPROFILE%\.cargo\bin;%LOCALAPPDATA%\mingw-binutils\bin` then `cargo build --release --offline`.

## Decoder in C++: `gsq_cpp/`
- `gsq.hpp` is a single-header library (C++17, no dependencies) that any C++ program can include:
  `gsq::Blobs b = gsq::decode(bytes); auto rgb = gsq::render(b, scale, &w, &h);`
- `gsq_view.cpp` is the same CLI as the Rust one (`--scale`, `--csv`). Its PNG writer is built in and uncompressed, so output files are about 4× bigger than the Rust ones but written faster.
- Compiler: GCC 16.2 (WinLibs MinGW-w64 UCRT) at `%LOCALAPPDATA%\winlibs-gcc16\mingw64\bin`; no install or admin rights needed.
- Build:
  ```
  set PATH=%LOCALAPPDATA%\winlibs-gcc16\mingw64\bin;%PATH%
  g++ -O3 -march=native -std=c++17 gsq_view.cpp -o gsq_view.exe -static
  ```
- Matches Python within 1 level on all 28 test files.

Rendering 8k blobs:
| Output | C++ | Rust |
|---|---|---|
| 512×320 | 28 ms | 60 ms |
| 2048×1280 | 450 ms | 800 ms |

The decode step itself is about 5 ms in both.

## GPU kernel in CUDA C++: `gsq_cuda/`
- `splat.cu` holds the forward and hand-written backward pass. One thread block (32×8 threads) per blob walks the blob's patch, with no integer division per pixel.
- It's built as `gsq_splat.dll` and called from Python through ctypes (`gsimg/cuda_splat.py`), so it needs no torch C++ extension and no ninja.
- Build: `gsq_cuda\build.bat`. It uses portable MSVC 14.44 (`%LOCALAPPDATA%\msvc-portable\setup_x64.bat`) and CUDA 12.8 nvcc (`%LOCALAPPDATA%\cuda-12.8`), targeting sm_120 (RTX 5060).
- `model.render` picks the CUDA kernel first, then Triton, then PyTorch. `GSQ_KERNEL=cuda|triton|torch` forces one.

Step time (8k blobs, render + backward; difference vs PyTorch is about 1e-3 relative, i.e. float rounding):

| Kernel | ms per step |
|---|---|
| PyTorch | 27.1 |
| Triton | 2.9 |
| **CUDA C++** | **2.7** |

A full encode takes 28 s with either GPU kernel. The rest is Python overhead per step, which CUDA graphs would be the next lever for.

## Speed (8k blobs, 512×320)
| Step | Before | Now |
|---|---|---|
| Fit + quantise + encode | about 225 s | **about 28 s** (Triton kernel, `gsimg/tri.py`) |
| Decode | about 0.35 s (Python) | **5 ms** (C++) / 6 ms (Rust) |

## What was tried and dropped
- Per-blob patch radius (`model.CUT`): most blobs are long and need the full radius-20 patch, so grouping by size was 3× slower. It is off; `GSQ3` would mark files that use it.
- Codebook index hopping (straight-through VQ): unstable, no gain.
- Two-stage residual codebooks: cost more bytes than they gained.
- Predicting blob values from neighbours: no entropy gain, because blob values are spatially independent.
