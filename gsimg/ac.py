"""Adaptive arithmetic coder (32-bit integer, carry-less with underflow
handling) with Fenwick-tree frequency models. Pure Python; ~1 s per 100k
symbols."""
import numpy as np

FULL = 1 << 32
HALF = 1 << 31
QUARTER = 1 << 30
MASK = FULL - 1
INC = 24  # adaptive count increment


class Model:
    """Adaptive frequency table over 0..n-1 (every symbol starts at 1)."""

    def __init__(self, n):
        self.n = n
        self.t = [0] * (n + 1)
        for i in range(1, n + 1):
            self.t[i] = i & -i          # Fenwick tree of all ones
        self.total = n
        self.top = 1 << n.bit_length()

    def add(self, i, d=INC):
        self.total += d
        i += 1
        t, n = self.t, self.n
        while i <= n:
            t[i] += d
            i += i & -i

    def prefix(self, i):
        s, t = 0, self.t
        while i > 0:
            s += t[i]
            i -= i & -i
        return s

    def find(self, v):
        pos, step, t, n = 0, self.top, self.t, self.n
        while step:
            nxt = pos + step
            if nxt <= n and t[nxt] <= v:
                pos = nxt
                v -= t[nxt]
            step >>= 1
        return pos


class Encoder:
    def __init__(self):
        self.low, self.high, self.under, self.bits = 0, MASK, 0, []

    def code(self, lo, hi, total):
        rng = self.high - self.low + 1
        self.high = self.low + hi * rng // total - 1
        self.low = self.low + lo * rng // total
        while ((self.low ^ self.high) & HALF) == 0:
            b = self.low >> 31
            self.bits.append(b)
            if self.under:
                self.bits.extend([b ^ 1] * self.under)
                self.under = 0
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) & MASK) | 1
        while self.low & ~self.high & QUARTER:
            self.under += 1
            self.low = (self.low << 1) ^ HALF
            self.high = ((self.high ^ HALF) << 1) | HALF | 1

    def sym(self, m, s):
        lo = m.prefix(s)
        self.code(lo, lo + m.prefix(s + 1) - lo, m.total)
        m.add(s)

    def raw(self, v, k):
        for j in range(k - 1, -1, -1):
            b = (v >> j) & 1
            self.code(b, b + 1, 2)

    def finish(self):
        self.bits.append(1)
        return np.packbits(np.array(self.bits, np.uint8)).tobytes()


class Decoder:
    def __init__(self, data):
        self.bits = np.unpackbits(np.frombuffer(data, np.uint8)).tolist()
        self.pos, self.low, self.high, self.code_ = 0, 0, MASK, 0
        for _ in range(32):
            self.code_ = (self.code_ << 1) | self._rb()

    def _rb(self):
        p = self.pos
        self.pos += 1
        return self.bits[p] if p < len(self.bits) else 0

    def _narrow(self, lo, hi, total):
        rng = self.high - self.low + 1
        self.high = self.low + hi * rng // total - 1
        self.low = self.low + lo * rng // total
        while ((self.low ^ self.high) & HALF) == 0:
            self.code_ = ((self.code_ << 1) & MASK) | self._rb()
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) & MASK) | 1
        while self.low & ~self.high & QUARTER:
            self.code_ = (self.code_ & HALF) | ((self.code_ << 1) & (MASK >> 1)) | self._rb()
            self.low = (self.low << 1) ^ HALF
            self.high = ((self.high ^ HALF) << 1) | HALF | 1

    def _value(self, total):
        rng = self.high - self.low + 1
        return ((self.code_ - self.low + 1) * total - 1) // rng

    def sym(self, m):
        s = m.find(self._value(m.total))
        lo = m.prefix(s)
        self._narrow(lo, m.prefix(s + 1), m.total)
        m.add(s)
        return s

    def raw(self, k):
        v = 0
        for _ in range(k):
            b = 1 if self._value(2) >= 1 else 0
            self._narrow(b, b + 1, 2)
            v = (v << 1) | b
        return v


# ---- integer streams -------------------------------------------------------
# direct: symbol = v - min over an adaptive table (small alphabets)
# bucket: zigzag(v - min) -> bit length (adaptive) + raw low bits (big ones)
DIRECT_MAX = 512


BUCKET_MAX = 4096


def _put_mode(enc, u, mode, a):
    """u = vals - lo (>= 0). mode 0: direct (a symbols); 1: bit length
    (a = 64) + raw bits; 2 + s: bucket u >> s (a buckets) + s raw low bits."""
    m = Model(a)
    if mode == 0:
        for v in u:
            enc.sym(m, v)
    elif mode == 1:
        for v in u:
            k = v.bit_length()
            enc.sym(m, k)
            if k > 1:
                enc.raw(v & ((1 << (k - 1)) - 1), k - 1)
    else:
        sh = mode - 2
        for v in u:
            enc.sym(m, v >> sh)
            enc.raw(v & ((1 << sh) - 1), sh)


def put_ints(enc, vals):
    """Codes an int array; returns the (mode, min, size) header it needs.
    Large alphabets try several codings and keep the smallest (the adaptive
    model learns a 400-symbol alphabet slowly; buckets of 8 learn faster)."""
    vals = np.asarray(vals, np.int64)
    if vals.size == 0:
        return (0, 0, 1)
    lo = int(vals.min())
    a = int(vals.max()) - lo + 1
    u = (vals - lo).tolist()
    if a <= 64:
        _put_mode(enc, u, 0, a)
        return (0, lo, a)
    cands = [(1, 64)]
    if a <= BUCKET_MAX:
        cands.append((0, a))
    for sh in range(1, 6):
        nb = ((a - 1) >> sh) + 1
        if nb <= BUCKET_MAX and nb >= 8:
            cands.append((2 + sh, nb))
    best = None
    for mode, size in cands:
        t = Encoder()
        _put_mode(t, u, mode, size)
        n = len(t.finish())
        if best is None or n < best[0]:
            best = (n, mode, size)
    _, mode, size = best
    _put_mode(enc, u, mode, size)
    return (mode, lo, size)


def get_ints(dec, n, head):
    mode, lo, a = head
    out = np.empty(n, np.int64)
    m = Model(a)
    if mode == 0:
        for i in range(n):
            out[i] = dec.sym(m) + lo
        return out
    if mode >= 2:
        sh = mode - 2
        for i in range(n):
            b = dec.sym(m)
            out[i] = ((b << sh) | dec.raw(sh)) + lo
        return out
    for i in range(n):
        k = dec.sym(m)
        v = 0 if k == 0 else (1 << (k - 1)) | (dec.raw(k - 1) if k > 1 else 0)
        out[i] = v + lo
    return out
