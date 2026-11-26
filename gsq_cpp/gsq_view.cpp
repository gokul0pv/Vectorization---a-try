// gsq_view (C++): open a .gsq blob image, render at any scale, save PNG.
//   gsq_view IN.gsq OUT.png [--scale S] [--csv BLOBS.csv]
// Build: g++ -O3 -march=native -std=c++17 gsq_view.cpp -o gsq_view.exe -static
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>

#include "gsq.hpp"

// Minimal PNG writer: zlib stored (uncompressed) blocks, no dependencies.
static uint32_t crc32(const uint8_t* d, size_t n, uint32_t c = 0) {
    static uint32_t tab[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t r = i;
            for (int k = 0; k < 8; ++k) r = r & 1 ? 0xEDB88320u ^ (r >> 1) : r >> 1;
            tab[i] = r;
        }
        init = true;
    }
    c = ~c;
    for (size_t i = 0; i < n; ++i) c = tab[(c ^ d[i]) & 0xFF] ^ (c >> 8);
    return ~c;
}

static void put32(std::vector<uint8_t>& v, uint32_t x) {
    for (int s = 24; s >= 0; s -= 8) v.push_back(uint8_t(x >> s));
}

static void chunk(std::ofstream& f, const char* type, const std::vector<uint8_t>& data) {
    std::vector<uint8_t> c;
    put32(c, uint32_t(data.size()));
    c.insert(c.end(), type, type + 4);
    c.insert(c.end(), data.begin(), data.end());
    std::vector<uint8_t> crc;
    put32(crc, crc32(c.data() + 4, c.size() - 4));
    f.write(reinterpret_cast<const char*>(c.data()), std::streamsize(c.size()));
    f.write(reinterpret_cast<const char*>(crc.data()), 4);
}

static void write_png(const std::string& path, const std::vector<uint8_t>& rgb, int w, int h) {
    std::vector<uint8_t> raw;
    raw.reserve(size_t(h) * (size_t(w) * 3 + 1));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgb.begin() + size_t(y) * w * 3, rgb.begin() + size_t(y + 1) * w * 3);
    }
    std::vector<uint8_t> z = {0x78, 0x01};
    uint32_t a = 1, b = 0;
    for (uint8_t v : raw) { a = (a + v) % 65521; b = (b + a) % 65521; }
    for (size_t o = 0; o < raw.size() || o == 0; o += 65535) {
        size_t len = std::min<size_t>(65535, raw.size() - o);
        z.push_back(o + len >= raw.size() ? 1 : 0);
        z.push_back(uint8_t(len)); z.push_back(uint8_t(len >> 8));
        z.push_back(uint8_t(~len)); z.push_back(uint8_t(~len >> 8));
        z.insert(z.end(), raw.begin() + o, raw.begin() + o + len);
        if (len == 0) break;
    }
    put32(z, (b << 16) | a);
    std::vector<uint8_t> ihdr;
    put32(ihdr, uint32_t(w)); put32(ihdr, uint32_t(h));
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});
    std::ofstream f(path, std::ios::binary);
    const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    f.write(reinterpret_cast<const char*>(sig), 8);
    chunk(f, "IHDR", ihdr);
    chunk(f, "IDAT", z);
    chunk(f, "IEND", {});
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: gsq_view IN.gsq OUT.png [--scale S] [--csv BLOBS.csv]\n";
        return 2;
    }
    float scale = 1.f;
    std::string csv;
    for (int i = 3; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--scale") scale = std::stof(argv[i + 1]);
        else if (a == "--csv") csv = argv[i + 1];
    }
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    auto t0 = clk::now();
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(in)), {});
    gsq::Blobs B;
    try { B = gsq::decode(buf); } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
    auto t1 = clk::now();
    int w, h;
    auto img = gsq::render(B, scale, &w, &h);
    auto t2 = clk::now();
    std::vector<uint8_t> px(img.size());
    for (size_t i = 0; i < img.size(); ++i) px[i] = uint8_t(std::clamp(img[i], 0.f, 1.f) * 255.f + 0.5f);
    write_png(argv[2], px, w, h);
    if (!csv.empty()) {
        std::ofstream c(csv);
        c << "x,y,shape_a,shape_b,shape_c,red,green,blue\n";
        for (size_t i = 0; i < B.size(); ++i)
            c << B.xy[2 * i] << ',' << B.xy[2 * i + 1] << ',' << B.l[3 * i] << ',' << B.l[3 * i + 1] << ','
              << B.l[3 * i + 2] << ',' << B.rgb[3 * i] << ',' << B.rgb[3 * i + 1] << ',' << B.rgb[3 * i + 2] << '\n';
    }
    auto t3 = clk::now();
    std::printf("%zu blobs, %zu bytes -> %dx%d | decode %.1f ms, render %.1f ms, png %.1f ms\n",
                B.size(), buf.size(), w, h, ms(t0, t1), ms(t1, t2), ms(t2, t3));
    return 0;
}
