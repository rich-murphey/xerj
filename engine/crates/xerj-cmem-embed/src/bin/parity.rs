//! Parity and throughput against claude-mem's stored vectors.
//!
//! Input is JSON Lines, one stored `observation_embeddings` row per line:
//! `{observation_id, part, kind, metadata, content, embedding: [f32; 384], why?}`,
//! as exported with psql (`embedding::real[]`, `observations.metadata`).
//!
//!   parity check <rows.jsonl>   re-embed each row's text, cosine vs stored
//!   parity bench <rows.jsonl>   passages/s by batch size, single-query latency
//!
//! `MINILM_ONNX` names another ONNX file (a path in the hub repo or an
//! absolute path; default `onnx/model.onnx`); `ORT_INTRA_THREADS` sizes ONNX
//! Runtime's intra-op pool (default: ONNX Runtime's choice).

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::time::Instant;
use xerj_cmem_embed::{observation_parts, Batching, Embedder, MAX_TOKENS, ONNX_FILE};

struct Row {
    id: String,
    part: String,
    why: String,
    text: String,
    stored: Vec<f32>,
}

fn read_rows(path: &str) -> Result<Vec<Row>> {
    let body = std::fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    let mut rows = Vec::new();
    let mut missing = 0;
    for line in body.lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line)?;
        let s = |k: &str| v[k].as_str().unwrap_or_default().to_string();
        let part = s("part");
        let parts = observation_parts(&s("kind"), &v["metadata"], &s("content"));
        let Some(text) = parts.into_iter().find(|p| p.part == part).map(|p| p.text) else {
            eprintln!("no part {part} for observation {}", s("observation_id"));
            missing += 1;
            continue;
        };
        let stored = v["embedding"]
            .as_array()
            .context("embedding")?
            .iter()
            .map(|x| x.as_f64().unwrap_or(f64::NAN) as f32)
            .collect();
        rows.push(Row {
            id: s("observation_id"),
            part,
            why: s("why"),
            text,
            stored,
        });
    }
    if missing > 0 {
        bail!("{missing} rows: part not reconstructed from the observation");
    }
    Ok(rows)
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

fn check(emb: &Embedder, rows: &[Row]) -> Result<()> {
    let texts: Vec<&str> = rows.iter().map(|r| r.text.as_str()).collect();
    let vectors = emb.embed(&texts, Batching::default())?;
    let mut capped = 0;
    let mut sep_ok = 0;
    let mut cos = Vec::with_capacity(rows.len());
    for (r, v) in rows.iter().zip(&vectors) {
        let ids = emb.token_ids(&r.text)?;
        if ids.len() == MAX_TOKENS {
            capped += 1;
            sep_ok += usize::from(ids.last() == Some(&102));
        }
        let c = cosine(v, &r.stored);
        if c < 0.9999 {
            println!(
                "LOW {c:.7} {} {} ({}, {} tokens)",
                r.id,
                r.part,
                r.why,
                ids.len()
            );
        }
        cos.push(c);
    }
    let mut sorted = cos.clone();
    sorted.sort_by(f64::total_cmp);
    println!(
        "rows={} at_cap({MAX_TOKENS})={capped} ending_[SEP]={sep_ok}",
        rows.len()
    );
    println!(
        "cosine min={:.9} p1={:.9} median={:.9} max={:.9} below_0.9999={} (1-min={:.2e})",
        sorted[0],
        quantile(&sorted, 0.01),
        quantile(&sorted, 0.5),
        sorted[sorted.len() - 1],
        sorted.iter().filter(|&&c| c < 0.9999).count(),
        1.0 - sorted[0]
    );
    let capped_cos: Vec<f64> = rows
        .iter()
        .zip(&cos)
        .filter(|(r, _)| {
            emb.token_ids(&r.text)
                .map(|i| i.len() == MAX_TOKENS)
                .unwrap_or(false)
        })
        .map(|(_, c)| *c)
        .collect();
    if !capped_cos.is_empty() {
        let min = capped_cos.iter().copied().fold(f64::INFINITY, f64::min);
        println!("at-cap rows: min cosine {min:.9} (1-min={:.2e})", 1.0 - min);
    }
    Ok(())
}

fn bench(emb: &Embedder, rows: &[Row]) -> Result<()> {
    // The random rows are the store's own mix of part lengths.
    let texts: Vec<&str> = rows
        .iter()
        .filter(|r| r.why != "long")
        .map(|r| r.text.as_str())
        .collect();
    let tokens: usize = texts
        .iter()
        .map(|t| emb.token_ids(t).map(|i| i.len()).unwrap_or(0))
        .sum();
    println!(
        "passages={} mean_tokens={:.1} threads={}",
        texts.len(),
        tokens as f64 / texts.len() as f64,
        std::env::var("ORT_INTRA_THREADS").unwrap_or_else(|_| "default".into())
    );
    emb.embed(&texts[..8], Batching::default())?;
    for max_rows in [64, 16, 4, 1] {
        let b = Batching {
            max_rows,
            padded_token_budget: Batching::default()
                .padded_token_budget
                .max(max_rows * MAX_TOKENS / 4),
        };
        let t = Instant::now();
        emb.embed(&texts, b)?;
        let s = t.elapsed().as_secs_f64();
        println!(
            "max_rows={max_rows:>3} budget={:>5}: {:.1} passages/s ({s:.2}s)",
            b.padded_token_budget,
            texts.len() as f64 / s
        );
    }
    let query = ["how did we fix the pgvector HNSW index build on dv"];
    emb.embed(&query, Batching::default())?;
    let mut lat: Vec<f64> = (0..200)
        .map(|_| {
            let t = Instant::now();
            emb.embed(&query, Batching::default())
                .map(|_| t.elapsed().as_secs_f64() * 1e3)
        })
        .collect::<std::result::Result<_, _>>()?;
    lat.sort_by(f64::total_cmp);
    println!(
        "query latency ({} tokens) ms: p50={:.2} p99={:.2} max={:.2}",
        emb.token_ids(query[0])?.len(),
        quantile(&lat, 0.5),
        quantile(&lat, 0.99),
        lat[lat.len() - 1]
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (Some(cmd), Some(path)) = (args.get(1), args.get(2)) else {
        bail!("usage: parity check|bench <rows.jsonl>");
    };
    let rows = read_rows(path)?;
    let t = Instant::now();
    let onnx_file = std::env::var("MINILM_ONNX").unwrap_or_else(|_| ONNX_FILE.into());
    let threads = std::env::var("ORT_INTRA_THREADS")
        .ok()
        .and_then(|t| t.parse().ok())
        .unwrap_or(0);
    let emb = Embedder::load_with(None, &onnx_file, threads)?;
    eprintln!("model loaded in {:.2}s", t.elapsed().as_secs_f64());
    match cmd.as_str() {
        "check" => check(&emb, &rows),
        "bench" => bench(&emb, &rows),
        _ => bail!("unknown command {cmd}"),
    }
}
