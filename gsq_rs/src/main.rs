//! gsq_view: open a .gsq blob image, render it at any scale, save PNG.
//!
//!   gsq_view IN.gsq OUT.png [--scale S] [--csv BLOBS.csv]
//!
//! Exact port of gsimg/sq.py decode + gsimg/model.py render (CPU, all cores).

mod ac;

use rayon::prelude::*;
use std::time::Instant;

const MAGIC: &[u8; 4] = b"GSQ3"; // GSQ3: per-blob patch radius min(R, ceil(CUT*sigma_max))
const CUT: f32 = 4.0;
const BAND: i64 = 8;
const EMAX: f64 = 7.0;

pub struct Blobs {
    pub h: usize,
    pub w: usize,
    pub r: usize,
    pub xy: Vec<[f32; 2]>,
    pub l: Vec<[f32; 3]>,   // a, b, c of the Cholesky factor
    pub rgb: Vec<[f32; 3]>,
    pub base: Vec<f32>,     // 3 x bh x bw
    pub bh: usize,
    pub bw: usize,
    pub cut: f32,         // 0 = full (2R+1)^2 patch (GSQ2)
}

fn rd<const N: usize>(b: &[u8], o: usize) -> [u8; N] {
    b[o..o + N].try_into().unwrap()
}

/// Per-blob position precision exponent (same formula as sq.pos_exp, f64).
fn pos_exp(sym: &[i64; 6], steps: &[f32; 6], k: f64) -> u32 {
    let v: Vec<f64> = (0..6).map(|j| sym[j] as f64 * steps[j] as f64).collect();
    let (a, b, c) = (v[0].exp(), v[1], v[2].exp());
    let (y, co, cg) = (v[3], v[4], v[5]);
    let amp = [y + co - cg, y + cg, y - co - cg].iter().fold(0f64, |m, x| m.max(x.abs())).max(1e-3);
    let (s11, s22) = (a * a, b * b + c * c);
    let h = (s11 + s22) / 2.0;
    let smin = (h - (h * h - (a * c) * (a * c)).max(0.0).sqrt()).max(1e-12).sqrt();
    let r = (k * smin / amp).clamp(2f64.powf(-EMAX), 1.0);
    (-r.log2()).round_ties_even().clamp(0.0, EMAX) as u32
}

pub fn decode(buf: &[u8]) -> Result<Blobs, String> {
    if buf.len() < 111 || (&buf[..4] != MAGIC && &buf[..4] != b"GSQ2") {
        return Err("not a GSQ2/GSQ3 file".into());
    }
    let cut = if &buf[..4] == MAGIC { CUT } else { 0.0 };
    let h = u16::from_le_bytes(rd(buf, 4)) as usize;
    let w = u16::from_le_bytes(rd(buf, 6)) as usize;
    let r = buf[8] as usize;
    let n = u32::from_le_bytes(rd(buf, 9)) as usize;
    let k = f32::from_le_bytes(rd(buf, 13)) as f64;
    let mut steps = [0f32; 6];
    for (j, s) in steps.iter_mut().enumerate() {
        *s = f32::from_le_bytes(rd(buf, 17 + 4 * j));
    }
    let mut heads = Vec::new();
    for i in 0..10 {
        let o = 41 + 7 * i;
        heads.push((buf[o], i32::from_le_bytes(rd(buf, o + 1)), i16::from_le_bytes(rd(buf, o + 5))));
    }
    let mut d = ac::Decoder::new(&buf[111..]);
    let bh = 2.max(h.div_ceil(32));
    let bw = 2.max(w.div_ceil(32));
    let nb = (h + 2 * r) / BAND as usize + 1;
    let base_q = d.ints(3 * bh * bw, heads[0]);
    let counts = d.ints(nb, heads[1]);
    let dx = d.ints(n, heads[2]);
    let yin = d.ints(n, heads[3]);
    let cols: Vec<Vec<i64>> = (0..6).map(|j| d.ints(n, heads[4 + j])).collect();
    let sym: Vec<[i64; 6]> = (0..n).map(|i| std::array::from_fn(|j| cols[j][i])).collect();
    let e: Vec<u32> = sym.iter().map(|s| pos_exp(s, &steps, k)).collect();
    let frac: Vec<(i64, i64)> = e.iter().map(|&ei| (d.raw(ei), d.raw(ei))).collect();

    let mut xy = Vec::with_capacity(n);
    let mut i = 0usize;
    for (band, &c) in counts.iter().enumerate() {
        let mut cx = 0i64;
        for _ in 0..c {
            cx += dx[i];
            let cy = band as i64 * BAND + yin[i];
            let s = (1i64 << e[i]) as f64;
            let ux = ((cx << e[i]) + frac[i].0) as f64;
            let uy = ((cy << e[i]) + frac[i].1) as f64;
            xy.push([(ux / s - r as f64) as f32, (uy / s - r as f64) as f32]);
            i += 1;
        }
    }
    if i != n {
        return Err(format!("band counts sum to {i}, expected {n}"));
    }
    let mut l = Vec::with_capacity(n);
    let mut rgb = Vec::with_capacity(n);
    for s in &sym {
        let v: [f32; 6] = std::array::from_fn(|j| s[j] as f32 * steps[j]);
        l.push([v[0].exp(), v[1], v[2].exp()]);
        let (y, co, cg) = (v[3], v[4], v[5]);
        rgb.push([y + co - cg, y + cg, y - co - cg]);
    }
    let base = base_q.iter().map(|&q| q as f32 / 127.5 - 0.5).collect();
    Ok(Blobs { h, w, r, xy, l, rgb, base, bh, bw, cut })
}

/// PyTorch bicubic (A = -0.75, align_corners = False, edge clamp).
fn cubic_w(t: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let c1 = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let c2 = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    [c2(t + 1.0), c1(t), c1(1.0 - t), c2(2.0 - t)]
}

fn bicubic(src: &[f32], sh: usize, sw: usize, oh: usize, ow: usize, out: &mut [f32]) {
    let (sy, sx) = (sh as f32 / oh as f32, sw as f32 / ow as f32);
    let taps = |o: usize, sc: f32, n: usize| {
        let real = sc * (o as f32 + 0.5) - 0.5;
        let f = real.floor();
        let idx: [usize; 4] = std::array::from_fn(|k| (f as i64 - 1 + k as i64).clamp(0, n as i64 - 1) as usize);
        (idx, cubic_w(real - f))
    };
    let xt: Vec<_> = (0..ow).map(|x| taps(x, sx, sw)).collect();
    for y in 0..oh {
        let (iy, wy) = taps(y, sy, sh);
        for (x, (ix, wx)) in xt.iter().enumerate() {
            for ch in 0..3 {
                let plane = &src[ch * sh * sw..];
                let mut acc = 0f32;
                for a in 0..4 {
                    let row = &plane[iy[a] * sw..];
                    let mut r = 0f32;
                    for b in 0..4 {
                        r += wx[b] * row[ix[b]];
                    }
                    acc += wy[a] * r;
                }
                out[(y * ow + x) * 3 + ch] += acc;
            }
        }
    }
}

/// Sum of Gaussians on a (2R+1)^2 patch each (as gsimg.model.render), + base.
pub fn render(bl: &Blobs, scale: f32) -> (usize, usize, Vec<f32>) {
    let oh = (bl.h as f64 * scale as f64).round_ties_even() as usize;
    let ow = (bl.w as f64 * scale as f64).round_ties_even() as usize;
    let rr = (bl.r as f32 * scale).ceil() as i64;
    // One image buffer per thread (blobs strided across threads), then summed.
    let nt = rayon::current_num_threads();
    let n = bl.xy.len();
    let mut bufs: Vec<Vec<f32>> = (0..nt)
        .into_par_iter()
        .map(|t| {
                let mut acc = vec![0f32; oh * ow * 3];
                for i in (t..n).step_by(nt) {
                    let (x, y) = (bl.xy[i][0] * scale, bl.xy[i][1] * scale);
                    let [a, b, c] = bl.l[i].map(|v| v * scale);
                    let col = bl.rgb[i];
                    let (cx, cy) = (x.floor() as i64, y.floor() as i64);
                    let rr = if bl.cut > 0.0 {
                        let (s11, s22) = (a * a, b * b + c * c);
                        let h = (s11 + s22) / 2.0;
                        let smax = (h + (h * h - (a * c) * (a * c)).max(0.0).sqrt()).sqrt();
                        ((bl.cut * smax).ceil() as i64).clamp(1, rr)
                    } else {
                        rr
                    };
                    for py in (cy - rr).max(0)..=(cy + rr).min(oh as i64 - 1) {
                        let oy = py as f32 + 0.5 - y;
                        let row = py as usize * ow;
                        for px in (cx - rr).max(0)..=(cx + rr).min(ow as i64 - 1) {
                            let ox = px as f32 + 0.5 - x;
                            let u1 = ox / a;
                            let u2 = (oy - b * u1) / c;
                            let wgt = (-0.5 * (u1 * u1 + u2 * u2)).exp();
                            let p = (row + px as usize) * 3;
                            acc[p] += wgt * col[0];
                            acc[p + 1] += wgt * col[1];
                            acc[p + 2] += wgt * col[2];
                        }
                    }
                }
                acc
        })
        .collect();
    let mut img = bufs.swap_remove(0);
    for b in &bufs {
        img.par_chunks_mut(1 << 16).zip(b.par_chunks(1 << 16)).for_each(|(x, y)| {
            x.iter_mut().zip(y).for_each(|(p, q)| *p += q);
        });
    }
    bicubic(&bl.base, bl.bh, bl.bw, oh, ow, &mut img);
    (oh, ow, img)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: gsq_view IN.gsq OUT.png [--scale S] [--csv BLOBS.csv]");
        std::process::exit(2);
    }
    let mut scale = 1f32;
    let mut csv = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--scale" => { scale = args[i + 1].parse().expect("scale"); i += 2; }
            "--csv" => { csv = Some(args[i + 1].clone()); i += 2; }
            a => panic!("unknown arg {a}"),
        }
    }
    let t0 = Instant::now();
    let buf = std::fs::read(&args[1]).expect("read input");
    let bl = decode(&buf).unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(1) });
    let t1 = Instant::now();
    let (oh, ow, img) = render(&bl, scale);
    let t2 = Instant::now();
    let px: Vec<u8> = img.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
    let f = std::io::BufWriter::new(std::fs::File::create(&args[2]).expect("create output"));
    let mut enc = png::Encoder::new(f, ow as u32, oh as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().and_then(|mut w| w.write_image_data(&px)).expect("write png");
    if let Some(p) = csv {
        let mut s = String::from("x,y,shape_a,shape_b,shape_c,red,green,blue\n");
        for i in 0..bl.xy.len() {
            let (q, l, c) = (bl.xy[i], bl.l[i], bl.rgb[i]);
            s += &format!("{},{},{},{},{},{},{},{}\n", q[0], q[1], l[0], l[1], l[2], c[0], c[1], c[2]);
        }
        std::fs::write(p, s).expect("write csv");
    }
    let t3 = Instant::now();
    println!(
        "{} blobs, {} bytes -> {}x{} | decode {:.1} ms, render {:.1} ms, png {:.1} ms",
        bl.xy.len(), buf.len(), ow, oh,
        (t1 - t0).as_secs_f64() * 1e3, (t2 - t1).as_secs_f64() * 1e3, (t3 - t2).as_secs_f64() * 1e3
    );
}
