//! Exact / quantized brute-force kNN over a packed vector column, per
//! segment, with a roaring filter bitmap and a bounded top-k heap per
//! segment merged across segments. Measures what xerj's exact kNN path
//! would cost if it read a packed column instead of stored documents.
//!
//!   cargo run --release -- --rows a.f16.bin,b.f16.bin --queries a.q.f16.bin,b.q.f16.bin
//!
//! Input: claude-mem-search-eval release files (rows of 384 f16, renormalised
//! to unit length exactly as research/claude-mem/lib.ts `loadVectors` does).
//! Output: one TSV row per (method, rescore depth, k, filter, threads).
use rayon::prelude::*;
use roaring::RoaringBitmap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Instant;

const D: usize = 384;
const WORDS: usize = D / 64;

// ── data ────────────────────────────────────────────────────────────────

fn load_f16(paths: &[String], limit: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for p in paths {
        let bytes = std::fs::read(p).expect("read vectors");
        for row in bytes.chunks_exact(2 * D) {
            if out.len() / D >= limit {
                break;
            }
            let mut v: Vec<f32> = row
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= n);
            out.extend_from_slice(&v);
        }
    }
    out
}

struct Store {
    n: usize,
    f32s: Vec<f32>,
    f16s: Vec<u16>,
    i8s: Vec<i8>,
    i8_scale: f32,
    bits: Vec<u64>,
}

impl Store {
    fn new(f32s: Vec<f32>) -> Self {
        let n = f32s.len() / D;
        let f16s = f32s.iter().map(|x| half::f16::from_f32(*x).to_bits()).collect();
        let max = f32s.iter().fold(0f32, |m, x| m.max(x.abs()));
        let i8_scale = 127.0 / max;
        let i8s = f32s.iter().map(|x| (x * i8_scale).round() as i8).collect();
        let mut bits = vec![0u64; n * WORDS];
        for (r, v) in f32s.chunks_exact(D).enumerate() {
            for (d, x) in v.iter().enumerate() {
                if *x > 0.0 {
                    bits[r * WORDS + d / 64] |= 1 << (d % 64);
                }
            }
        }
        Store { n, f32s, f16s, i8s, i8_scale, bits }
    }
    fn f32_row(&self, r: u32) -> &[f32] {
        &self.f32s[r as usize * D..(r as usize + 1) * D]
    }
}

// ── kernels ─────────────────────────────────────────────────────────────

/// xerj's arithmetic today: `vector_column::dot_f64`, strictly sequential f64.
fn dot_f64_seq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum::<f64>() as f32
}

/// Sequential f32: what a plain loop compiles to without reassociation.
fn dot_f32_seq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
mod simd {
    use super::{D, WORDS};
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    /// f32 · f32, four 8-lane FMA accumulators.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_f32(q: &[f32], v: &[f32]) -> f32 {
        let (q, v) = (q.as_ptr(), v.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i < D {
            for (j, a) in acc.iter_mut().enumerate() {
                let o = i + 8 * j;
                *a = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(o)), _mm256_loadu_ps(v.add(o)), *a);
            }
            i += 32;
        }
        hsum(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])))
    }

    /// f32 query · f16 row (F16C widen, then FMA).
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_f16(q: &[f32], v: &[u16]) -> f32 {
        let (q, v) = (q.as_ptr(), v.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i < D {
            for (j, a) in acc.iter_mut().enumerate() {
                let o = i + 8 * j;
                let h = _mm256_cvtph_ps(_mm_loadu_si128(v.add(o) as *const __m128i));
                *a = _mm256_fmadd_ps(_mm256_loadu_ps(q.add(o)), h, *a);
            }
            i += 32;
        }
        hsum(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])))
    }

    /// i16 query · i8 row: sign-extend 16 i8 to i16, `vpmaddwd`, i32 lanes.
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_i8(q: &[i16], v: &[i8]) -> i32 {
        let (q, v) = (q.as_ptr(), v.as_ptr());
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut i = 0;
        while i < D {
            let x0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(v.add(i) as *const __m128i));
            let x1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(v.add(i + 16) as *const __m128i));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(x0, _mm256_loadu_si256(q.add(i) as *const __m256i)));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(x1, _mm256_loadu_si256(q.add(i + 16) as *const __m256i)));
            i += 32;
        }
        let a = _mm256_add_epi32(a0, a1);
        let s = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
        _mm_cvtsi128_si32(s)
    }

    #[target_feature(enable = "popcnt")]
    pub unsafe fn hamming(q: &[u64], v: &[u64]) -> u32 {
        let mut s = 0;
        for w in 0..WORDS {
            s += (q[w] ^ v[w]).count_ones();
        }
        s
    }
}

// ── top-k ───────────────────────────────────────────────────────────────

/// Ordered so that the heap's max is the WORST entry: lower score is worse,
/// and on an exact tie the higher row is worse. A total order, so the result
/// is deterministic regardless of segment or thread interleaving.
#[derive(Clone, Copy, Debug)]
struct Entry {
    score: f32,
    row: u32,
}
impl Ord for Entry {
    fn cmp(&self, o: &Self) -> Ordering {
        o.score.total_cmp(&self.score).then(self.row.cmp(&o.row))
    }
}
impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl PartialEq for Entry {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Entry {}

fn top_k(seg_lo: u32, filter: &RoaringBitmap, k: usize, score: impl Fn(u32) -> f32) -> Vec<Entry> {
    let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(k + 1);
    for local in filter.iter() {
        let row = seg_lo + local;
        let e = Entry { score: score(row), row };
        if heap.len() < k {
            heap.push(e);
        } else if e < *heap.peek().unwrap() {
            *heap.peek_mut().unwrap() = e;
        }
    }
    heap.into_vec()
}

fn merge(mut parts: Vec<Entry>, k: usize) -> Vec<Entry> {
    parts.sort_unstable();
    parts.truncate(k);
    parts
}

// ── methods ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
enum Method {
    F64Seq,
    F32Seq,
    F32Avx2,
    F16Avx2,
    /// int8 scan to the best `R`, then exact f32 rescore of those `R`.
    I8Rescore(usize),
    /// 1-bit sign codes (Hamming) to the best `R`, then exact f32 rescore.
    BinRescore(usize),
}
impl Method {
    fn name(self) -> (&'static str, usize) {
        match self {
            Method::F64Seq => ("f64-seq (xerj today)", 0),
            Method::F32Seq => ("f32-seq", 0),
            Method::F32Avx2 => ("f32-avx2", 0),
            Method::F16Avx2 => ("f16-avx2", 0),
            Method::I8Rescore(r) => ("i8-avx2+f32-rescore", r),
            Method::BinRescore(r) => ("bin-popcnt+f32-rescore", r),
        }
    }
}

struct Segments {
    bounds: Vec<(u32, u32)>,
}

fn search(
    store: &Store,
    segs: &Segments,
    filters: &[RoaringBitmap],
    q: &[f32],
    k: usize,
    m: Method,
    parallel: bool,
) -> Vec<Entry> {
    let qi16: Vec<i16> = q.iter().map(|x| (x * store.i8_scale).round() as i16).collect();
    let mut qbits = [0u64; WORDS];
    for (d, x) in q.iter().enumerate() {
        if *x > 0.0 {
            qbits[d / 64] |= 1 << (d % 64);
        }
    }
    let depth = match m {
        Method::I8Rescore(r) | Method::BinRescore(r) => r.max(k),
        _ => k,
    };
    let per_seg = |s: usize| -> Vec<Entry> {
        let (lo, _) = segs.bounds[s];
        let f = &filters[s];
        unsafe {
            match m {
                Method::F64Seq => top_k(lo, f, depth, |r| dot_f64_seq(q, store.f32_row(r))),
                Method::F32Seq => top_k(lo, f, depth, |r| dot_f32_seq(q, store.f32_row(r))),
                Method::F32Avx2 => top_k(lo, f, depth, |r| simd::dot_f32(q, store.f32_row(r))),
                Method::F16Avx2 => top_k(lo, f, depth, |r| {
                    simd::dot_f16(q, &store.f16s[r as usize * D..(r as usize + 1) * D])
                }),
                Method::I8Rescore(_) => top_k(lo, f, depth, |r| {
                    simd::dot_i8(&qi16, &store.i8s[r as usize * D..(r as usize + 1) * D]) as f32
                }),
                Method::BinRescore(_) => top_k(lo, f, depth, |r| {
                    -(simd::hamming(&qbits, &store.bits[r as usize * WORDS..(r as usize + 1) * WORDS]) as f32)
                }),
            }
        }
    };
    let parts: Vec<Entry> = if parallel {
        (0..segs.bounds.len()).into_par_iter().flat_map_iter(per_seg).collect()
    } else {
        (0..segs.bounds.len()).flat_map(per_seg).collect()
    };
    let cand = merge(parts, depth);
    match m {
        Method::I8Rescore(_) | Method::BinRescore(_) => merge(
            cand.into_iter()
                .map(|e| Entry { score: unsafe { simd::dot_f32(q, store.f32_row(e.row)) }, row: e.row })
                .collect(),
            k,
        ),
        _ => cand,
    }
}

// ── driver ──────────────────────────────────────────────────────────────

fn arg(name: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| default.to_string())
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() as f64 - 1.0) * p).round() as usize]
}

fn vm_hwm_mib() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines()
        .find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

fn main() {
    let rows: Vec<String> = arg("--rows", "").split(',').map(String::from).collect();
    let qpaths: Vec<String> = arg("--queries", "").split(',').map(String::from).collect();
    let n: usize = arg("--n", "300306").parse().unwrap();
    let nq: usize = arg("--nq", "200").parse().unwrap();
    let nseg: usize = arg("--segs", "16").parse().unwrap();
    let threads: usize = arg("--threads", "16").parse().unwrap();
    let only: String = arg("--only", "");
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();

    let t = Instant::now();
    let store = Store::new(load_f16(&rows, n));
    assert_eq!(store.n, n, "fewer rows than --n");
    // Queries: spread evenly over every query file given.
    let mut queries: Vec<Vec<f32>> = Vec::new();
    for p in &qpaths {
        let all = load_f16(std::slice::from_ref(p), usize::MAX);
        let m = all.len() / D;
        let want = nq / qpaths.len();
        for i in 0..want {
            let r = i * m / want;
            queries.push(all[r * D..(r + 1) * D].to_vec());
        }
    }
    eprintln!(
        "loaded {} rows, {} queries in {:.1}s; f32 {:.0} MiB, f16 {:.0} MiB, i8 {:.0} MiB, bits {:.1} MiB",
        store.n,
        queries.len(),
        t.elapsed().as_secs_f64(),
        (store.f32s.len() * 4) as f64 / 1048576.0,
        (store.f16s.len() * 2) as f64 / 1048576.0,
        store.i8s.len() as f64 / 1048576.0,
        (store.bits.len() * 8) as f64 / 1048576.0
    );

    let segs = Segments {
        bounds: (0..nseg)
            .map(|s| ((s * n / nseg) as u32, ((s + 1) * n / nseg) as u32))
            .collect(),
    };
    // Filters as per-segment bitmaps of segment-local positions. 100% is a
    // full range; 10% / 1% are uniform random (xorshift, fixed seed), the
    // worst case for locality.
    let mut rng = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let selectivities = [1.0f64, 0.10, 0.01];
    let filters: Vec<Vec<RoaringBitmap>> = selectivities
        .iter()
        .map(|&sel| {
            segs.bounds
                .iter()
                .map(|&(lo, hi)| {
                    let mut b = RoaringBitmap::new();
                    if sel >= 1.0 {
                        b.insert_range(0..hi - lo);
                    } else {
                        for i in 0..hi - lo {
                            if (next() % 1_000_000) as f64 / 1e6 < sel {
                                b.insert(i);
                            }
                        }
                    }
                    b
                })
                .collect()
        })
        .collect();

    println!("method\trescore\tk\tfilter\tthreads\tp50_ms\tp95_ms\tmean_ms\trecall\trecall_min\torder_eq_f64");
    for k in [10usize, 400] {
        for (fi, &sel) in selectivities.iter().enumerate() {
            let f = &filters[fi];
            let truth: Vec<Vec<Entry>> = queries
                .iter()
                .map(|q| search(&store, &segs, f, q, k, Method::F64Seq, true))
                .collect();
            let mut methods = vec![Method::F64Seq, Method::F32Seq, Method::F32Avx2, Method::F16Avx2];
            for mult in [1usize, 2, 4, 10] {
                methods.push(Method::I8Rescore(k * mult));
            }
            for mult in [4usize, 10, 40] {
                methods.push(Method::BinRescore(k * mult));
            }
            for m in methods {
                if !only.is_empty() && !m.name().0.starts_with(only.as_str()) {
                    continue;
                }
                for parallel in [false, true] {
                    for q in queries.iter().take(3) {
                        std::hint::black_box(search(&store, &segs, f, q, k, m, parallel));
                    }
                    let mut ms = Vec::with_capacity(queries.len());
                    let mut recall = Vec::with_capacity(queries.len());
                    let mut set_eq = 0usize;
                    for (qi, q) in queries.iter().enumerate() {
                        let t = Instant::now();
                        let got = search(&store, &segs, f, q, k, m, parallel);
                        ms.push(t.elapsed().as_secs_f64() * 1e3);
                        let want: std::collections::HashSet<u32> = truth[qi].iter().map(|e| e.row).collect();
                        let hit = got.iter().filter(|e| want.contains(&e.row)).count();
                        recall.push(hit as f64 / want.len().max(1) as f64);
                        if got.iter().map(|e| e.row).eq(truth[qi].iter().map(|e| e.row)) {
                            set_eq += 1;
                        }
                    }
                    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let (name, r) = m.name();
                    println!(
                        "{name}\t{r}\t{k}\t{:.0}%\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.4}\t{:.2}\t{}/{}",
                        sel * 100.0,
                        if parallel { threads } else { 1 },
                        pct(&ms, 0.5),
                        pct(&ms, 0.95),
                        ms.iter().sum::<f64>() / ms.len() as f64,
                        recall.iter().sum::<f64>() / recall.len() as f64,
                        recall.iter().cloned().fold(1.0, f64::min),
                        set_eq,
                        queries.len()
                    );
                }
            }
        }
    }
    eprintln!("peak RSS {:.0} MiB", vm_hwm_mib());
}
