//! Parity and timing of the ColBERT encoder against PyLate.
//!
//! The fixture is claude-mem-search-eval's `fixtures/colbert-parity.jsonl`
//! (20 queries, 50 documents, PyLate 1.6.0 ids, masks, vectors and the 20×50
//! MaxSim matrix); it is not vendored here.
//!
//!   colbert check <colbert-parity.jsonl>
//!   colbert bench <colbert-parity.jsonl> <rows.jsonl>
//!
//! The backend is candle unless `COLBERT_ONNX` names an ONNX file in the hub
//! repo (`onnx/model.onnx`, `onnx/model_int8.onnx`, …; needs the
//! `onnx-experimental` feature). Threads: `RAYON_NUM_THREADS` for candle,
//! and the same value for ONNX Runtime's intra-op pool;
//! `COLBERT_ONNX_SPIN=0` stops its idle threads spinning.
//!
//! `check` fails unless ids, masks and kept positions match exactly, every
//! fixture token vector has cosine ≥ 0.9999, and each query ranks the 50
//! documents in the fixture's order. `bench` times single-query encodes over
//! the fixture's queries, document encodes over the `text` of each line of
//! `rows.jsonl` (claude-mem-search-eval `src/export-rows.ts`), and MaxSim
//! over 50 random candidates of 54 and of 1,000 tokens.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::time::Instant;
use xerj_cmem_embed::colbert::{
    from_f16, maxsim, to_f16, Colbert, ColbertEncoder, DIM, QUERY_TOKENS,
};
use xerj_cmem_embed::Batching;

const MIN_COSINE: f64 = 0.9999;

struct Text {
    id: String,
    text: String,
    ids: Vec<u32>,
    mask: Vec<u32>,
    kept: Vec<usize>,
    positions: Vec<usize>,
    vectors: Vec<Vec<f32>>,
}

struct Fixture {
    queries: Vec<Text>,
    docs: Vec<Text>,
    scores: Vec<Vec<f64>>,
}

fn read_fixture(path: &str) -> Result<Fixture> {
    let body = std::fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    let (mut queries, mut docs, mut scores) = (Vec::new(), Vec::new(), Vec::new());
    for line in body.lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line)?;
        let nums = |k: &str| -> Vec<u64> {
            v[k].as_array()
                .map(|a| a.iter().filter_map(Value::as_u64).collect())
                .unwrap_or_default()
        };
        let floats = |row: &Value| -> Vec<f64> {
            row.as_array()
                .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect())
                .unwrap_or_default()
        };
        match v["kind"].as_str() {
            Some("scores") => {
                scores = v["maxsim"]
                    .as_array()
                    .context("maxsim")?
                    .iter()
                    .map(floats)
                    .collect()
            }
            Some(kind) => {
                let t = Text {
                    id: v["id"].as_str().unwrap_or_default().into(),
                    text: v["text"].as_str().context("text")?.into(),
                    ids: nums("input_ids").into_iter().map(|x| x as u32).collect(),
                    mask: nums("attention_mask")
                        .into_iter()
                        .map(|x| x as u32)
                        .collect(),
                    kept: nums("kept").into_iter().map(|x| x as usize).collect(),
                    positions: nums("positions").into_iter().map(|x| x as usize).collect(),
                    vectors: v["vectors"]
                        .as_array()
                        .context("vectors")?
                        .iter()
                        .map(|r| floats(r).into_iter().map(|x| x as f32).collect())
                        .collect(),
                };
                if kind == "query" {
                    &mut queries
                } else {
                    &mut docs
                }
                .push(t);
            }
            None => bail!("line without kind"),
        }
    }
    if scores.len() != queries.len() {
        bail!("{} score rows for {} queries", scores.len(), queries.len());
    }
    Ok(Fixture {
        queries,
        docs,
        scores,
    })
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// Document indices best first; ties keep index order.
fn ranking(scores: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    order
}

fn check(model: &dyn ColbertEncoder, fx: &Fixture) -> Result<()> {
    let mut failures = 0;
    let mut worst = (1.0f64, String::new());
    let mut cosines = Vec::new();
    let mut compare = |t: &Text, ids: &[u32], mask: &[u32], kept: &[usize], out: &[f32]| {
        if ids != t.ids || mask != t.mask || kept != t.kept {
            println!("INPUT MISMATCH {}: ids {:?}", t.id, ids);
            failures += 1;
            return;
        }
        if out.len() != kept.len() * DIM {
            println!("ROWS {}: {} for {} kept", t.id, out.len() / DIM, kept.len());
            failures += 1;
            return;
        }
        for (&p, want) in t.positions.iter().zip(&t.vectors) {
            let c = cosine(&out[p * DIM..(p + 1) * DIM], want);
            cosines.push(c);
            if c < worst.0 {
                worst = (c, format!("{} row {p}", t.id));
            }
            if c < MIN_COSINE {
                println!("LOW {c:.7} {} row {p}", t.id);
                failures += 1;
            }
        }
    };
    let mut qv = Vec::new();
    for t in &fx.queries {
        let input = model.tokens().query_input(&t.text)?;
        let out = model.encode_query(&t.text)?;
        let all: Vec<usize> = (0..QUERY_TOKENS).collect();
        compare(t, &input.ids, &input.mask, &all, &out);
        qv.push(out);
    }
    let texts: Vec<&str> = fx.docs.iter().map(|t| t.text.as_str()).collect();
    let dv = model.encode_documents(&texts, Batching::default())?;
    for (t, out) in fx.docs.iter().zip(&dv) {
        let input = model.tokens().document_input(&t.text)?;
        compare(
            t,
            &input.ids,
            &input.mask,
            &model.tokens().kept(&input),
            out,
        );
    }
    cosines.sort_by(f64::total_cmp);
    println!(
        "inputs: {} queries, {} docs; token vectors compared={} cosine min={:.9} ({}, 1-min={:.2e}) median={:.9}",
        fx.queries.len(),
        fx.docs.len(),
        cosines.len(),
        worst.0,
        worst.1,
        1.0 - worst.0,
        quantile(&cosines, 0.5)
    );

    let mut max_err = 0.0f64;
    let mut rank_mismatch = 0;
    for (qi, q) in qv.iter().enumerate() {
        let ours: Vec<f64> = dv.iter().map(|d| maxsim(q, d) as f64).collect();
        for (a, b) in ours.iter().zip(&fx.scores[qi]) {
            max_err = max_err.max((a - b).abs());
        }
        let (r_ours, r_ref) = (ranking(&ours), ranking(&fx.scores[qi]));
        if r_ours != r_ref {
            let at = r_ours
                .iter()
                .zip(&r_ref)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            println!(
                "RANK {}: differs from place {at}: ours d{} {:.5}, fixture d{} {:.5}",
                fx.queries[qi].id,
                r_ours[at],
                ours[r_ours[at]],
                r_ref[at],
                fx.scores[qi][r_ref[at]]
            );
            rank_mismatch += 1;
        }
    }
    println!(
        "maxsim {}×{}: max abs error {max_err:.2e} (fixture rounded to 1e-5); rankings identical {}/{}",
        qv.len(),
        dv.len(),
        qv.len() - rank_mismatch,
        qv.len()
    );
    if failures + rank_mismatch > 0 {
        bail!("{failures} input/vector failures, {rank_mismatch} rankings differ");
    }
    Ok(())
}

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|s| s.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
        .unwrap_or_default()
}

/// Deterministic unit vectors, `rows × DIM`.
fn random_rows(rows: usize, seed: &mut u64) -> Vec<f32> {
    let mut v: Vec<f32> = (0..rows * DIM)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            (*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect();
    for r in v.chunks_exact_mut(DIM) {
        let n = r.iter().map(|x| x * x).sum::<f32>().sqrt();
        r.iter_mut().for_each(|x| *x /= n);
    }
    v
}

fn bench(model: &dyn ColbertEncoder, fx: &Fixture, rows_path: &str) -> Result<()> {
    println!(
        "threads={} load={}",
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into()),
        loadavg()
    );
    let queries: Vec<&str> = fx.queries.iter().map(|t| t.text.as_str()).collect();
    for q in &queries {
        model.encode_query(q)?;
    }
    let mut lat = Vec::new();
    for _ in 0..10 {
        for q in &queries {
            let t = Instant::now();
            model.encode_query(q)?;
            lat.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    lat.sort_by(f64::total_cmp);
    println!(
        "query encode ms ({} encodes): p50={:.2} p95={:.2} max={:.2} load={}",
        lat.len(),
        quantile(&lat, 0.5),
        quantile(&lat, 0.95),
        lat[lat.len() - 1],
        loadavg()
    );

    let body = std::fs::read_to_string(rows_path).with_context(|| format!("read {rows_path}"))?;
    let texts: Vec<String> = body
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Value = serde_json::from_str(l)?;
            Ok(v["text"].as_str().unwrap_or_default().to_string())
        })
        .collect::<Result<_>>()?;
    let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
    let tokens_in: usize = texts
        .iter()
        .map(|t| model.tokens().document_input(t).map(|i| i.ids.len()))
        .sum::<xerj_cmem_embed::Result<usize>>()?;
    model.encode_documents(&texts[..8], Batching::default())?;
    for max_rows in [64, 8] {
        let b = Batching {
            max_rows,
            ..Batching::default()
        };
        let t = Instant::now();
        let out = model.encode_documents(&texts, b)?;
        let s = t.elapsed().as_secs_f64();
        let kept: usize = out.iter().map(|v| v.len() / DIM).sum();
        println!(
            "documents={} tokens_in={tokens_in} kept={kept} max_rows={max_rows}: {:.0} tokens/s, {:.1} docs/s ({s:.1}s) load={}",
            texts.len(),
            tokens_in as f64 / s,
            texts.len() as f64 / s,
            loadavg()
        );
    }

    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let query = random_rows(QUERY_TOKENS, &mut seed);
    for tokens in [54, 1000] {
        let docs: Vec<Vec<f32>> = (0..50).map(|_| random_rows(tokens, &mut seed)).collect();
        let stored: Vec<_> = docs.iter().map(|d| to_f16(d)).collect();
        let mut f32_ms = Vec::new();
        let mut f16_ms = Vec::new();
        let mut sink = 0.0f32;
        for _ in 0..200 {
            let t = Instant::now();
            sink += docs.iter().map(|d| maxsim(&query, d)).sum::<f32>();
            f32_ms.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            sink += stored
                .iter()
                .map(|d| maxsim(&query, &from_f16(d)))
                .sum::<f32>();
            f16_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        f32_ms.sort_by(f64::total_cmp);
        f16_ms.sort_by(f64::total_cmp);
        println!(
            "maxsim 50 × {tokens} tokens, 1 thread ms: f32 p50={:.3} p95={:.3}; from fp16 p50={:.3} p95={:.3} ({:.3} µs/token) [{sink:.0}] load={}",
            quantile(&f32_ms, 0.5),
            quantile(&f32_ms, 0.95),
            quantile(&f16_ms, 0.5),
            quantile(&f16_ms, 0.95),
            quantile(&f32_ms, 0.5) * 1e3 / (50 * tokens) as f64,
            loadavg()
        );
    }
    Ok(())
}

fn load() -> Result<Box<dyn ColbertEncoder>> {
    let Ok(onnx_file) = std::env::var("COLBERT_ONNX") else {
        println!("backend=candle");
        return Ok(Box::new(Colbert::load(None)?));
    };
    #[cfg(feature = "onnx-experimental")]
    {
        let threads = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|t| t.parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
        let spinning = std::env::var("COLBERT_ONNX_SPIN").map_or(true, |v| v != "0");
        println!(
            "backend=onnxruntime file={onnx_file} intra_threads={threads} spinning={spinning}"
        );
        Ok(Box::new(xerj_cmem_embed::colbert::onnx::OrtColbert::load(
            None, &onnx_file, threads, spinning,
        )?))
    }
    #[cfg(not(feature = "onnx-experimental"))]
    bail!("COLBERT_ONNX={onnx_file}: built without the onnx-experimental feature")
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (Some(cmd), Some(path)) = (args.get(1), args.get(2)) else {
        bail!("usage: colbert check <colbert-parity.jsonl> | bench <colbert-parity.jsonl> <rows.jsonl>");
    };
    let fx = read_fixture(path)?;
    let t = Instant::now();
    let model = load()?;
    eprintln!("model loaded in {:.2}s", t.elapsed().as_secs_f64());
    match (cmd.as_str(), args.get(3)) {
        ("check", _) => check(model.as_ref(), &fx),
        ("bench", Some(rows)) => bench(model.as_ref(), &fx, rows),
        _ => bail!("unknown command {cmd}"),
    }
}
