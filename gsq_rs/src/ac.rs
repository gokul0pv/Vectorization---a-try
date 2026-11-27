//! Adaptive arithmetic decoder: exact port of gsimg/ac.py.

const HALF: u64 = 1 << 31;
const QUARTER: u64 = 1 << 30;
const MASK: u64 = (1 << 32) - 1;
const INC: u64 = 24;

/// Adaptive frequency table over 0..n (Fenwick tree, every symbol starts at 1).
pub struct Model {
    n: usize,
    t: Vec<u64>,
    total: u64,
    top: usize,
}

impl Model {
    pub fn new(n: usize) -> Self {
        let mut t = vec![0u64; n + 1];
        for (i, v) in t.iter_mut().enumerate().skip(1) {
            *v = (i & i.wrapping_neg()) as u64;
        }
        let bits = usize::BITS - n.leading_zeros();
        Model { n, t, total: n as u64, top: 1usize << bits }
    }

    fn add(&mut self, i: usize) {
        self.total += INC;
        let mut i = i + 1;
        while i <= self.n {
            self.t[i] += INC;
            i += i & i.wrapping_neg();
        }
    }

    fn prefix(&self, mut i: usize) -> u64 {
        let mut s = 0;
        while i > 0 {
            s += self.t[i];
            i -= i & i.wrapping_neg();
        }
        s
    }

    fn find(&self, mut v: u64) -> usize {
        let (mut pos, mut step) = (0usize, self.top);
        while step > 0 {
            let nxt = pos + step;
            if nxt <= self.n && self.t[nxt] <= v {
                pos = nxt;
                v -= self.t[nxt];
            }
            step >>= 1;
        }
        pos
    }
}

pub struct Decoder<'a> {
    data: &'a [u8],
    pos: usize,
    low: u64,
    high: u64,
    code: u64,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut d = Decoder { data, pos: 0, low: 0, high: MASK, code: 0 };
        for _ in 0..32 {
            d.code = (d.code << 1) | d.bit();
        }
        d
    }

    #[inline]
    fn bit(&mut self) -> u64 {
        let p = self.pos;
        self.pos += 1;
        match self.data.get(p >> 3) {
            Some(b) => ((b >> (7 - (p & 7))) & 1) as u64,
            None => 0,
        }
    }

    #[inline]
    fn narrow(&mut self, lo: u64, hi: u64, total: u64) {
        let rng = self.high - self.low + 1;
        self.high = self.low + hi * rng / total - 1;
        self.low += lo * rng / total;
        while (self.low ^ self.high) & HALF == 0 {
            self.code = ((self.code << 1) & MASK) | self.bit();
            self.low = (self.low << 1) & MASK;
            self.high = ((self.high << 1) & MASK) | 1;
        }
        while self.low & !self.high & QUARTER != 0 {
            self.code = (self.code & HALF) | ((self.code << 1) & (MASK >> 1)) | self.bit();
            self.low = (self.low << 1) ^ HALF;
            self.high = ((self.high ^ HALF) << 1) | HALF | 1;
        }
    }

    #[inline]
    fn value(&self, total: u64) -> u64 {
        let rng = self.high - self.low + 1;
        ((self.code - self.low + 1) * total - 1) / rng
    }

    pub fn sym(&mut self, m: &mut Model) -> usize {
        let s = m.find(self.value(m.total));
        let lo = m.prefix(s);
        let hi = m.prefix(s + 1);
        self.narrow(lo, hi, m.total);
        m.add(s);
        s
    }

    pub fn raw(&mut self, k: u32) -> i64 {
        let mut v = 0i64;
        for _ in 0..k {
            let b = if self.value(2) >= 1 { 1 } else { 0 };
            self.narrow(b, b + 1, 2);
            v = (v << 1) | b as i64;
        }
        v
    }

    /// Integer stream: mode 0 = direct (v - lo), mode 1 = bit-length bucket + raw bits.
    pub fn ints(&mut self, n: usize, head: (u8, i32, i16)) -> Vec<i64> {
        let (mode, lo, a) = head;
        let mut m = Model::new(a as usize);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let v = if mode == 0 {
                self.sym(&mut m) as i64
            } else if mode >= 2 {
                // bucket + (mode - 2) raw low bits
                let sh = (mode - 2) as u32;
                let b = self.sym(&mut m) as i64;
                (b << sh) | if sh > 0 { self.raw(sh) } else { 0 }
            } else {
                let k = self.sym(&mut m) as u32;
                match k {
                    0 => 0,
                    1 => 1,
                    _ => (1i64 << (k - 1)) | self.raw(k - 1),
                }
            };
            out.push(v + lo as i64);
        }
        out
    }
}
