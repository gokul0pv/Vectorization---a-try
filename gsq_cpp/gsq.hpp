// gsq.hpp - single-header decoder + renderer for .gsq blob images.
//   GSQ2: 6 channels, base step 1/127.5
//   GSQ4: flags byte (bit 0: Beta blobs, 7th channel log k; bit 1: adaptive shape
//         precision, then float aref + uint8 G; bit 2: symbol layer, then uint16 P +
//         uint32 C + uint8 copy_e), base step 1/510
// Exact port of gsimg/sq.py decode + gsimg/model.py render (C++17, no deps).
//
//   gsq::Blobs b = gsq::decode(bytes);          // throws std::runtime_error
//   std::vector<float> rgb = gsq::render(b, 2.0f, &w, &h);   // any scale
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace gsq {

// ---- adaptive arithmetic decoder (gsimg/ac.py) ------------------------------
namespace detail {
constexpr uint64_t HALF = 1ull << 31, QUARTER = 1ull << 30, MASK = (1ull << 32) - 1, INC = 24;

struct Model {
    size_t n, top;
    std::vector<uint64_t> t;
    uint64_t total;
    explicit Model(size_t n_) : n(n_), t(n_ + 1), total(n_) {
        for (size_t i = 1; i <= n; ++i) t[i] = i & (~i + 1);
        size_t bits = 0;
        while ((n >> bits) != 0) ++bits;
        top = size_t(1) << bits;
    }
    void add(size_t i) {
        total += INC;
        for (++i; i <= n; i += i & (~i + 1)) t[i] += INC;
    }
    uint64_t prefix(size_t i) const {
        uint64_t s = 0;
        for (; i > 0; i -= i & (~i + 1)) s += t[i];
        return s;
    }
    size_t find(uint64_t v) const {
        size_t pos = 0;
        for (size_t step = top; step; step >>= 1) {
            size_t nxt = pos + step;
            if (nxt <= n && t[nxt] <= v) { pos = nxt; v -= t[nxt]; }
        }
        return pos;
    }
};

struct Decoder {
    const uint8_t* d;
    size_t len, pos = 0;
    uint64_t low = 0, high = MASK, code = 0;
    Decoder(const uint8_t* d_, size_t len_) : d(d_), len(len_) {
        for (int i = 0; i < 32; ++i) code = (code << 1) | bit();
    }
    uint64_t bit() {
        size_t p = pos++;
        return (p >> 3) < len ? (d[p >> 3] >> (7 - (p & 7))) & 1 : 0;
    }
    void narrow(uint64_t lo, uint64_t hi, uint64_t total) {
        uint64_t rng = high - low + 1;
        high = low + hi * rng / total - 1;
        low = low + lo * rng / total;
        while (((low ^ high) & HALF) == 0) {
            code = ((code << 1) & MASK) | bit();
            low = (low << 1) & MASK;
            high = ((high << 1) & MASK) | 1;
        }
        while (low & ~high & QUARTER) {
            code = (code & HALF) | ((code << 1) & (MASK >> 1)) | bit();
            low = (low << 1) ^ HALF;
            high = ((high ^ HALF) << 1) | HALF | 1;
        }
    }
    uint64_t value(uint64_t total) const { return ((code - low + 1) * total - 1) / (high - low + 1); }
    size_t sym(Model& m) {
        size_t s = m.find(value(m.total));
        narrow(m.prefix(s), m.prefix(s + 1), m.total);
        m.add(s);
        return s;
    }
    int64_t raw(unsigned k) {
        int64_t v = 0;
        for (unsigned j = 0; j < k; ++j) {
            uint64_t b = value(2) >= 1 ? 1 : 0;
            narrow(b, b + 1, 2);
            v = (v << 1) | int64_t(b);
        }
        return v;
    }
    struct Head { uint8_t mode; int32_t lo; int16_t a; };
    std::vector<int64_t> ints(size_t n, Head h) {
        Model m(size_t(h.a));
        std::vector<int64_t> out(n);
        for (auto& v : out) {
            int64_t x;
            if (h.mode == 0) x = int64_t(sym(m));
            else {
                unsigned k = unsigned(sym(m));
                x = k == 0 ? 0 : k == 1 ? 1 : (int64_t(1) << (k - 1)) | raw(k - 1);
            }
            v = x + h.lo;
        }
        return out;
    }
};

template <class T> T rd(const uint8_t* b) { T v; std::memcpy(&v, b, sizeof v); return v; }

inline unsigned pos_exp(const int64_t* s, const float* steps, double k, double shape_mult = 1.0) {
    double v[6];
    for (int j = 0; j < 6; ++j) v[j] = double(s[j]) * double(steps[j]) * (j < 3 ? shape_mult : 1.0);
    double a = std::exp(v[0]), b = v[1], c = std::exp(v[2]);
    double y = v[3], co = v[4], cg = v[5];
    double amp = std::max({std::fabs(y + co - cg), std::fabs(y + cg), std::fabs(y - co - cg), 1e-3});
    double s11 = a * a, s22 = b * b + c * c, h = (s11 + s22) / 2;
    double smin = std::sqrt(std::max(h - std::sqrt(std::max(h * h - (a * c) * (a * c), 0.0)), 1e-12));
    double r = std::clamp(k * smin / amp, std::pow(2.0, -7.0), 1.0);
    return unsigned(std::clamp(std::nearbyint(-std::log2(r)), 0.0, 7.0));   // ties to even
}
// Shape precision group from the quantised colour (same as sq.shape_g, f64).
inline int shape_g(const int64_t* csym, const float* csteps, double aref, int G) {  // csym: Y, Co, Cg
    double y = double(csym[0]) * double(csteps[0]), co = double(csym[1]) * double(csteps[1]),
           cg = double(csym[2]) * double(csteps[2]);
    double amp = std::max({std::fabs(y + co - cg), std::fabs(y + cg), std::fabs(y - co - cg), 1e-4});
    double g = std::nearbyint(std::log2(amp / aref));   // ties to even
    return int(std::clamp(g, double(-G), double(G)));
}
}  // namespace detail

struct Blobs {
    int h = 0, w = 0, r = 0, bh = 0, bw = 0;
    std::vector<float> xy, l, rgb;   // N*2, N*3 (a,b,c), N*3
    std::vector<float> base;         // 3*bh*bw
    std::vector<float> k;            // Beta sharpness per blob; empty = Gaussians
    size_t size() const { return xy.size() / 2; }
};

constexpr float KMIN = 0.25f, KMAX = 12.f;   // gsimg/model.py

inline Blobs decode(const std::vector<uint8_t>& buf) {
    using namespace detail;
    if (buf.size() < 18) throw std::runtime_error("not a .gsq file");
    const bool v4 = std::memcmp(buf.data(), "GSQ4", 4) == 0;
    if (!v4 && std::memcmp(buf.data(), "GSQ2", 4) != 0) throw std::runtime_error("not a GSQ2/GSQ4 file");
    const uint8_t* p = buf.data();
    Blobs B;
    B.h = rd<uint16_t>(p + 4); B.w = rd<uint16_t>(p + 6); B.r = p[8];
    size_t n = rd<uint32_t>(p + 9);
    double k = rd<float>(p + 13);
    size_t o = 17;
    const uint8_t flags = v4 ? p[o++] : 0;
    const bool beta = flags & 1, adaptive = flags & 2, symb = flags & 4;
    double aref = 0; int G = 0;
    if (adaptive) {
        if (buf.size() < o + 5) throw std::runtime_error("truncated header");
        aref = rd<float>(p + o); G = p[o + 4]; o += 5;
    }
    size_t P = 0, C = 0;
    int copy_e = 0;
    if (symb) {
        if (buf.size() < o + 7) throw std::runtime_error("truncated header");
        P = rd<uint16_t>(p + o); C = rd<uint32_t>(p + o + 2); copy_e = p[o + 6]; o += 7;
    }
    const int nch = beta ? 7 : 6, na = nch + (adaptive ? 6 * G : 0);
    const int nheads = 4 + na + (symb ? 1 + na + 2 + 3 : 0);
    const size_t hdr = o + 4 * nch + 7 * size_t(nheads);
    if (buf.size() < hdr) throw std::runtime_error("truncated header");
    float steps[7];
    for (int j = 0; j < nch; ++j, o += 4) steps[j] = rd<float>(p + o);
    std::vector<Decoder::Head> heads(static_cast<size_t>(nheads));
    for (int i = 0; i < nheads; ++i, o += 7) heads[size_t(i)] = {p[o], rd<int32_t>(p + o + 1), rd<int16_t>(p + o + 5)};
    Decoder d(p + hdr, buf.size() - hdr);
    B.bh = std::max(2, (B.h + 31) / 32);
    B.bw = std::max(2, (B.w + 31) / 32);
    const int BAND = 8;
    size_t nb = size_t((B.h + 2 * B.r) / BAND + 1);
    auto base_q = d.ints(size_t(3 * B.bh * B.bw), heads[0]);
    auto counts = d.ints(nb, heads[1]);
    auto dx = d.ints(n, heads[2]);
    auto yin = d.ints(n, heads[3]);
    size_t hi = 4;

    // attribute columns (sq._read_attrs): sym is m x nch, mult = shape step multiplier
    auto read_attrs = [&](size_t m, std::vector<int64_t>& sym, std::vector<double>& mult) {
        sym.assign(m * nch, 0);
        mult.assign(m, 1.0);
        auto col_all = [&](int j) {
            auto col = d.ints(m, heads[hi++]);
            for (size_t i = 0; i < m; ++i) sym[i * nch + j] = col[i];
        };
        if (!adaptive) {
            for (int j = 0; j < nch; ++j) col_all(j);
            return;
        }
        for (int j = 3; j < 6; ++j) col_all(j);
        std::vector<int> g(m);
        for (size_t i = 0; i < m; ++i) {
            g[i] = shape_g(&sym[i * nch + 3], steps + 3, aref, G);
            mult[i] = std::ldexp(1.0, -g[i]);
        }
        for (int j = 0; j < 3; ++j)
            for (int v = -G; v <= G; ++v) {
                size_t cnt = 0;
                for (size_t i = 0; i < m; ++i) cnt += g[i] == v;
                auto col = d.ints(cnt, heads[hi++]);
                size_t c = 0;
                for (size_t i = 0; i < m; ++i) if (g[i] == v) sym[i * nch + j] = col[c++];
            }
        for (int j = 6; j < nch; ++j) col_all(j);
    };
    auto push_blob = [&](const int64_t* s, double m, float x, float y) {
        float v[7];
        for (int j = 0; j < nch; ++j) v[j] = float(double(s[j]) * (j < 3 ? m : 1.0)) * steps[j];
        B.xy.push_back(x); B.xy.push_back(y);
        B.l.push_back(std::exp(v[0])); B.l.push_back(v[1]); B.l.push_back(std::exp(v[2]));
        B.rgb.push_back(v[3] + v[4] - v[5]); B.rgb.push_back(v[3] + v[5]); B.rgb.push_back(v[3] - v[4] - v[5]);
        if (beta) B.k.push_back(std::clamp(std::exp(v[6]), KMIN, KMAX));
    };

    std::vector<int64_t> sym;
    std::vector<double> mult;
    read_attrs(n, sym, mult);
    std::vector<unsigned> e(n);
    for (size_t i = 0; i < n; ++i) e[i] = pos_exp(&sym[i * nch], steps, k, mult[i]);
    std::vector<int64_t> fx(n), fy(n);
    for (size_t i = 0; i < n; ++i) { fx[i] = d.raw(e[i]); fy[i] = d.raw(e[i]); }

    size_t i = 0;
    for (size_t band = 0; band < nb; ++band) {
        int64_t cx = 0;
        for (int64_t c = 0; c < counts[band]; ++c, ++i) {
            if (i >= n) throw std::runtime_error("bad band counts");
            cx += dx[i];
            int64_t cy = int64_t(band) * BAND + yin[i];
            double s = double(int64_t(1) << e[i]);
            push_blob(&sym[i * nch], mult[i], float(double((cx << e[i]) + fx[i]) / s - B.r),
                      float(double((cy << e[i]) + fy[i]) / s - B.r));
        }
    }
    if (i != n) throw std::runtime_error("bad band counts");

    if (symb) {   // symbol layer: prototypes (stored once) drawn at every copy
        const double COPY_Q = std::ldexp(1.0, copy_e);
        auto pcount = d.ints(P, heads[hi++]);
        size_t npb = 0;
        std::vector<size_t> first(P);
        for (size_t q = 0; q < P; ++q) { first[q] = npb; npb += size_t(pcount[q]); }
        std::vector<int64_t> psym;
        std::vector<double> pmult;
        read_attrs(npb, psym, pmult);
        std::vector<unsigned> ep(npb);
        for (size_t b = 0; b < npb; ++b) ep[b] = pos_exp(&psym[b * nch], steps, k, pmult[b]);
        auto px = d.ints(npb, heads[hi++]);
        auto py = d.ints(npb, heads[hi++]);
        auto cproto = d.ints(C, heads[hi++]);
        auto dy = d.ints(C, heads[hi++]);
        auto cxq = d.ints(C, heads[hi++]);
        std::vector<double> pxf(npb), pyf(npb);
        for (size_t b = 0; b < npb; ++b) {
            int64_t fx = d.raw(ep[b]), fy = d.raw(ep[b]);
            double s = std::ldexp(1.0, int(ep[b]));
            pxf[b] = double(px[b] * (int64_t(1) << ep[b]) + fx) / s;
            pyf[b] = double(py[b] * (int64_t(1) << ep[b]) + fy) / s;
        }
        int64_t oy = 0, ox = 0;
        for (size_t c = 0; c < C; ++c) {
            oy += dy[c];
            ox = (c > 0 && dy[c] == 0) ? ox + cxq[c] : cxq[c];
            size_t q = size_t(cproto[c]);
            if (q >= P) throw std::runtime_error("bad prototype id");
            for (size_t b = first[q]; b < first[q] + size_t(pcount[q]); ++b)
                push_blob(&psym[b * nch], pmult[b], float(pxf[b] + double(ox) / COPY_Q),
                          float(pyf[b] + double(oy) / COPY_Q));
        }
    }
    B.base.resize(base_q.size());
    const float bs = v4 ? 510.f : 127.5f;
    for (size_t j = 0; j < base_q.size(); ++j) B.base[j] = float(base_q[j]) / bs - 0.5f;
    return B;
}

// ---- renderer ----------------------------------------------------------------
namespace detail {
inline void cubic_w(float t, float w[4]) {
    const float A = -0.75f;
    auto c1 = [&](float x) { return ((A + 2) * x - (A + 3)) * x * x + 1; };
    auto c2 = [&](float x) { return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A; };
    w[0] = c2(t + 1); w[1] = c1(t); w[2] = c1(1 - t); w[3] = c2(2 - t);
}

// PyTorch bicubic: A = -0.75, align_corners = False, edge clamp. Adds into out.
inline void bicubic_add(const std::vector<float>& src, int sh, int sw, int oh, int ow, std::vector<float>& out) {
    struct Tap { int i[4]; float w[4]; };
    auto taps = [](int o, float sc, int n) {
        Tap t;
        float real = sc * (float(o) + 0.5f) - 0.5f, f = std::floor(real);
        for (int k = 0; k < 4; ++k) t.i[k] = std::clamp(int(f) - 1 + k, 0, n - 1);
        cubic_w(real - f, t.w);
        return t;
    };
    float sy = float(sh) / float(oh), sx = float(sw) / float(ow);
    std::vector<Tap> xt(static_cast<size_t>(ow));
    for (int x = 0; x < ow; ++x) xt[size_t(x)] = taps(x, sx, sw);
    for (int y = 0; y < oh; ++y) {
        Tap ty = taps(y, sy, sh);
        for (int x = 0; x < ow; ++x) {
            const Tap& tx = xt[size_t(x)];
            for (int ch = 0; ch < 3; ++ch) {
                const float* plane = &src[size_t(ch) * sh * sw];
                float acc = 0;
                for (int a = 0; a < 4; ++a) {
                    const float* row = plane + size_t(ty.i[a]) * sw;
                    float r = 0;
                    for (int b = 0; b < 4; ++b) r += tx.w[b] * row[tx.i[b]];
                    acc += ty.w[a] * r;
                }
                out[(size_t(y) * ow + x) * 3 + ch] += acc;
            }
        }
    }
}
}  // namespace detail

// Sum of Gaussians on a (2R+1)^2 patch each, + bicubic base. All cores.
inline std::vector<float> render(const Blobs& B, float scale, int* out_w, int* out_h) {
    int oh = int(std::nearbyint(double(B.h) * scale)), ow = int(std::nearbyint(double(B.w) * scale));
    int rr = int(std::ceil(float(B.r) * scale));
    size_t n = B.size(), px_count = size_t(oh) * ow * 3;
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::vector<float>> acc(nt, std::vector<float>(px_count, 0.f));
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; ++t) {
        th.emplace_back([&, t] {
            std::vector<float>& img = acc[t];
            for (size_t i = t; i < n; i += nt) {
                float x = B.xy[2 * i] * scale, y = B.xy[2 * i + 1] * scale;
                float a = B.l[3 * i] * scale, b = B.l[3 * i + 1] * scale, c = B.l[3 * i + 2] * scale;
                const float* col = &B.rgb[3 * i];
                const bool bk = !B.k.empty();
                const float kk = bk ? B.k[i] : 0.f, ik = bk ? 0.5f / kk : 0.f;
                int cx = int(std::floor(x)), cy = int(std::floor(y));
                for (int py = std::max(cy - rr, 0); py <= std::min(cy + rr, oh - 1); ++py) {
                    float oy = float(py) + 0.5f - y;
                    float* row = &img[size_t(py) * ow * 3];
                    for (int px = std::max(cx - rr, 0); px <= std::min(cx + rr, ow - 1); ++px) {
                        float ox = float(px) + 0.5f - x;
                        float u1 = ox / a, u2 = (oy - b * u1) / c;
                        float q = u1 * u1 + u2 * u2, w;
                        if (!bk) w = std::exp(-0.5f * q);
                        else {   // (1 - q/2k)_+^k
                            float t1 = 1.f - q * ik;
                            if (t1 <= 0.f) continue;
                            w = std::exp(kk * std::log(std::max(t1, 1e-6f)));
                        }
                        float* p = row + size_t(px) * 3;
                        p[0] += w * col[0]; p[1] += w * col[1]; p[2] += w * col[2];
                    }
                }
            }
        });
    }
    for (auto& t : th) t.join();
    std::vector<float> out = std::move(acc[0]);
    for (unsigned t = 1; t < nt; ++t)
        for (size_t j = 0; j < px_count; ++j) out[j] += acc[t][j];
    detail::bicubic_add(B.base, B.bh, B.bw, oh, ow, out);
    *out_w = ow; *out_h = oh;
    return out;
}

}  // namespace gsq
