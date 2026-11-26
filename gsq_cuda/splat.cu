// CUDA C++ blob renderer: forward + hand-written backward (same maths as
// gsimg/tri.py and gsimg/model._splat). Built as a DLL, called via ctypes.
//   u1 = ox / a,  u2 = (oy - b*u1) / c,  w = exp(-(u1^2 + u2^2) / 2)
// One thread block per blob. Threads form a 32 x 8 grid that walks the
// (2R+1)^2 patch by columns and rows: no integer division per pixel.
#include <cuda_runtime.h>

#define API extern "C" __declspec(dllexport)
constexpr int TX = 32, TY = 8, THREADS = TX * TY;

__global__ void __launch_bounds__(THREADS)
fwd(const float* __restrict__ xy, const float* __restrict__ L,
    const float* __restrict__ rgb, float* __restrict__ out, int H, int W, int R) {
    const int i = blockIdx.x, P = 2 * R + 1;
    const float x = xy[2 * i], y = xy[2 * i + 1];
    const float ia = 1.f / L[3 * i], b = L[3 * i + 1], ic = 1.f / L[3 * i + 2];
    const float cr = rgb[3 * i], cg = rgb[3 * i + 1], cb = rgb[3 * i + 2];
    const int x0 = max((int)floorf(x) - R, 0), x1 = min((int)floorf(x) - R + P, W);
    const int y0 = max((int)floorf(y) - R, 0), y1 = min((int)floorf(y) - R + P, H);
    for (int py = y0 + threadIdx.y; py < y1; py += TY) {
        const float oy = py + 0.5f - y;
        for (int px = x0 + threadIdx.x; px < x1; px += TX) {
            const float u1 = (px + 0.5f - x) * ia;
            const float u2 = (oy - b * u1) * ic;
            const float w = __expf(-0.5f * (u1 * u1 + u2 * u2));
            float* p = out + 3 * (py * W + px);
            atomicAdd(p, w * cr);
            atomicAdd(p + 1, w * cg);
            atomicAdd(p + 2, w * cb);
        }
    }
}

template <int K>
__device__ __forceinline__ void block_sum(float (&v)[K], float (*sh)[THREADS / 32]) {
    const int t = threadIdx.y * TX + threadIdx.x, lane = t & 31, warp = t >> 5;
#pragma unroll
    for (int j = 0; j < K; ++j) {
        float s = v[j];
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
        if (lane == 0) sh[j][warp] = s;
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int j = 0; j < K; ++j) {
            float s = lane < THREADS / 32 ? sh[j][lane] : 0.f;
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
            v[j] = s;   // valid in thread 0
        }
    }
}

__global__ void __launch_bounds__(THREADS)
bwd(const float* __restrict__ xy, const float* __restrict__ L,
    const float* __restrict__ rgb, const float* __restrict__ gout,
    float* gxy, float* gL, float* grgb, int H, int W, int R) {
    __shared__ float sh[8][THREADS / 32];
    const int i = blockIdx.x, P = 2 * R + 1;
    const float x = xy[2 * i], y = xy[2 * i + 1];
    const float ia = 1.f / L[3 * i], b = L[3 * i + 1], ic = 1.f / L[3 * i + 2];
    const float cr = rgb[3 * i], cg = rgb[3 * i + 1], cb = rgb[3 * i + 2];
    const int x0 = max((int)floorf(x) - R, 0), x1 = min((int)floorf(x) - R + P, W);
    const int y0 = max((int)floorf(y) - R, 0), y1 = min((int)floorf(y) - R + P, H);
    float s[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // r g b | x y | a b c (pre-scaled)
    for (int py = y0 + threadIdx.y; py < y1; py += TY) {
        const float oy = py + 0.5f - y;
        for (int px = x0 + threadIdx.x; px < x1; px += TX) {
            const float u1 = (px + 0.5f - x) * ia;
            const float u2 = (oy - b * u1) * ic;
            const float w = __expf(-0.5f * (u1 * u1 + u2 * u2));
            const float* g = gout + 3 * (py * W + px);
            const float g0 = g[0], g1 = g[1], g2 = g[2];
            s[0] += w * g0; s[1] += w * g1; s[2] += w * g2;
            const float dq = -0.5f * w * (g0 * cr + g1 * cg + g2 * cb);
            const float du2 = dq * 2.f * u2;
            const float du1 = dq * 2.f * u1 - du2 * b * ic;
            s[3] += du1;           // * -1/a   (applied once, below)
            s[4] += du2;           // * -1/c
            s[5] += du1 * u1;      // * -1/a
            s[6] += du2 * u1;      // * -1/c
            s[7] += du2 * u2;      // * -1/c
        }
    }
    block_sum<8>(s, sh);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        grgb[3 * i] = s[0]; grgb[3 * i + 1] = s[1]; grgb[3 * i + 2] = s[2];
        gxy[2 * i] = -s[3] * ia; gxy[2 * i + 1] = -s[4] * ic;
        gL[3 * i] = -s[5] * ia; gL[3 * i + 1] = -s[6] * ic; gL[3 * i + 2] = -s[7] * ic;
    }
}

// Pointers are CUDA device pointers (torch tensor.data_ptr()); stream is the
// caller's cudaStream_t (torch.cuda.current_stream().cuda_stream).
API int gsq_forward(const float* xy, const float* L, const float* rgb, float* out,
                    int N, int H, int W, int R, void* stream) {
    if (N > 0) fwd<<<N, dim3(TX, TY), 0, (cudaStream_t)stream>>>(xy, L, rgb, out, H, W, R);
    return (int)cudaGetLastError();
}

API int gsq_backward(const float* xy, const float* L, const float* rgb, const float* gout,
                     float* gxy, float* gL, float* grgb, int N, int H, int W, int R, void* stream) {
    if (N > 0) bwd<<<N, dim3(TX, TY), 0, (cudaStream_t)stream>>>(xy, L, rgb, gout, gxy, gL, grgb, H, W, R);
    return (int)cudaGetLastError();
}
