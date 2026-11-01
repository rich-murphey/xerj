//! all-MiniLM-L6-v2 sentence embedder that reproduces the vectors claude-mem
//! stores in `observation_embeddings` under the model tag [`MODEL_TAG`].
//!
//! claude-mem (rich-murphey/claude-mem, `src/services/vector/LocalEmbedder.ts`)
//! embeds with transformers.js and Xenova/all-MiniLM-L6-v2 fp32: the tokenizer
//! cut to [`MAX_TOKENS`] word pieces, the closing `[SEP]` that transformers.js
//! drops on truncation put back, attention-masked mean pooling, L2 norm. That
//! is what Chroma's default embedding function does, and what the HF
//! `tokenizers` crate does natively: its truncation reserves room for the
//! special tokens, so a long text encodes as `[CLS]` + 254 word pieces +
//! `[SEP]` with no repair step.
//!
//! The network is the very file claude-mem runs: `onnx/model.onnx` of
//! [`MODEL_ID`] at [`REVISION`] (sha256 [`ONNX_SHA256`]), with that
//! revision's `tokenizer.json`, on ONNX Runtime (`ort`). Pooling and the norm
//! are done here.

pub mod colbert;

use hf_hub::{Repo, RepoType};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tokenizers::{Encoding, Tokenizer, TruncationParams};

/// The `model` column claude-mem writes for these vectors.
pub const MODEL_TAG: &str = "Xenova/all-MiniLM-L6-v2/fp32/mean/l2/256";

/// Hub repository of the ONNX export transformers.js (and so claude-mem) runs.
pub const MODEL_ID: &str = "Xenova/all-MiniLM-L6-v2";

/// The hub revision whose `onnx/model.onnx` is byte-identical to the file in
/// the live claude-mem containers (`@huggingface/transformers` 4.3.1 cache).
pub const REVISION: &str = "751bff37182d3f1213fa05d7196b954e230abad9";

/// The fp32 export, inside the hub repo or a local model directory.
pub const ONNX_FILE: &str = "onnx/model.onnx";

/// sha256 of [`ONNX_FILE`] at [`REVISION`].
pub const ONNX_SHA256: &str = "759c3cd2b7fe7e93933ad23c4c9181b7396442a2ed746ec7c1d46192c469c46e";

/// Width of the model's output and of every vector.
pub const DIMS: usize = 384;

/// Tokens per input, `[CLS]` and `[SEP]` included: the length the model was
/// trained at and Chroma truncates to, not the tokenizer's 512.
pub const MAX_TOKENS: usize = 256;

/// UTF-16 code units of a text handed to the tokenizer, as claude-mem's
/// `text.slice(0, MAX_TEXT_CHARS)`.
pub const MAX_TEXT_UTF16: usize = 32 * 1024;

/// Facts embedded per observation (claude-mem `MAX_FACT_PARTS`).
pub const MAX_FACT_PARTS: usize = 64;

const SUMMARY_FIELDS: [&str; 6] = [
    "request",
    "investigated",
    "learned",
    "completed",
    "next_steps",
    "notes",
];

const PAD_TOKEN_ID: u32 = 0;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("model files: {0}")]
    Hub(#[from] hf_hub::api::sync::ApiError),
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error("safetensors: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("weights: {0}")]
    Weights(String),
    #[error("onnx runtime: {0}")]
    Ort(#[from] ort::Error),
    #[error("onnx output: {0}")]
    OrtOutput(String),
    #[error("onnx runtime environment: {0:#}")]
    OrtEnvironment(anyhow::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Rows per forward pass and the `rows × padded length` ceiling; rows are
/// grouped by token length so short texts are not padded to a long one.
/// Grouping does not change any vector: padded positions are masked out.
#[derive(Debug, Clone, Copy)]
pub struct Batching {
    pub max_rows: usize,
    pub padded_token_budget: usize,
}

impl Default for Batching {
    /// xerj-ai's measured knee (`neural.rs` MAX_BATCH_ROWS, PADDED_TOKEN_BUDGET).
    fn default() -> Self {
        Self {
            max_rows: 64,
            padded_token_budget: 4_096,
        }
    }
}

/// One text embedded for an observation row, named as claude-mem's `part` column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Part {
    pub part: String,
    pub text: String,
}

/// JavaScript `String.prototype.trim()` whitespace: Unicode White_Space plus BOM.
fn js_trim(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || c == '\u{feff}')
}

/// The texts claude-mem embeds for one observation row, in its order: a port of
/// `observationEmbeddingParts` (`src/storage/postgres/observation-embeddings.ts`).
/// Each text is embedded as-is: no title, prefix or field name is prepended.
pub fn observation_parts(kind: &str, metadata: &Value, content: &str) -> Vec<Part> {
    let mut parts = Vec::new();
    let add = |parts: &mut Vec<Part>, part: String, value: Option<&Value>| {
        if let Some(Value::String(s)) = value {
            if !js_trim(s).is_empty() {
                parts.push(Part {
                    part,
                    text: s.clone(),
                });
            }
        }
    };
    if kind == "summary" {
        for field in SUMMARY_FIELDS {
            add(&mut parts, field.to_string(), metadata.get(field));
        }
    } else {
        add(&mut parts, "narrative".into(), metadata.get("narrative"));
        add(
            &mut parts,
            "text".into(),
            metadata.get("legacy").and_then(|l| l.get("text")),
        );
        // Blank facts are dropped before numbering, as ChromaSync numbers them.
        let facts = metadata
            .get("facts")
            .and_then(Value::as_array)
            .map(|a| a.as_slice())
            .unwrap_or_default()
            .iter()
            .filter(|f| f.as_str().is_some_and(|s| !js_trim(s).is_empty()))
            .take(MAX_FACT_PARTS);
        for (i, fact) in facts.enumerate() {
            add(&mut parts, format!("fact_{i}"), Some(fact));
        }
        if parts.is_empty() {
            let title = match metadata.get("title") {
                Some(Value::String(t)) if js_trim(t) != "Untitled" => js_trim(t),
                _ => "",
            };
            let subtitle = match metadata.get("subtitle") {
                Some(Value::String(s)) => js_trim(s),
                _ => "",
            };
            let heading = [title, subtitle]
                .into_iter()
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            add(
                &mut parts,
                "title".into(),
                Some(&Value::String(heading.clone())),
            );
            if js_trim(content) != heading {
                add(
                    &mut parts,
                    "content".into(),
                    Some(&Value::String(content.into())),
                );
            }
        }
    }
    if parts.is_empty() {
        add(
            &mut parts,
            "content".into(),
            Some(&Value::String(content.into())),
        );
    }
    parts
}

/// `text.slice(0, MAX_TEXT_UTF16)`, cut at a char boundary.
fn clip_utf16(text: &str) -> &str {
    let mut units = 0;
    for (i, c) in text.char_indices() {
        units += c.len_utf16();
        if units > MAX_TEXT_UTF16 {
            return &text[..i];
        }
    }
    text
}

/// A file of a hub model: `local_dir/name` when a directory is given, else
/// from the HF hub cache at the pinned revision (downloading on a miss).
/// Blocking.
pub(crate) fn model_file(local_dir: Option<&Path>, repo: &Repo, name: &str) -> Result<PathBuf> {
    Ok(match local_dir {
        _ if Path::new(name).is_absolute() => name.into(),
        Some(dir) => dir.join(name),
        None => hf_hub::api::sync::Api::new()?
            .repo(repo.clone())
            .get(name)?,
    })
}

/// An ONNX Runtime session on the process's one environment
/// (`xerj_ai::onnx::environment`), all graph optimisations on.
/// `intra_threads` 0 leaves the pool size to ONNX Runtime.
pub(crate) fn session(onnx: &Path, intra_threads: usize, spinning: bool) -> Result<Session> {
    let env = xerj_ai::onnx::environment().map_err(Error::OrtEnvironment)?;
    Ok(Session::builder(env)?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(ort::Error::from)?
        .with_intra_threads(intra_threads)
        .map_err(ort::Error::from)?
        .with_intra_op_spinning(spinning)
        .map_err(ort::Error::from)?
        .commit_from_file(onnx)?)
}

/// Ids and masks as the `int64` tensors a BERT export takes, plus all-zero
/// `token_type_ids`, for one `[rows, seq_len]` batch.
pub(crate) fn bert_inputs(
    ids: Vec<u32>,
    mask: Vec<u32>,
    rows: usize,
    seq_len: usize,
) -> Result<[Tensor<i64>; 3]> {
    let shape = [rows, seq_len];
    let as_i64 = |v: Vec<u32>| v.into_iter().map(i64::from).collect::<Vec<_>>();
    let types = vec![0i64; ids.len()];
    Ok([
        Tensor::from_array((shape, as_i64(ids)))?,
        Tensor::from_array((shape, as_i64(mask)))?,
        Tensor::from_array((shape, types))?,
    ])
}

/// The safe `ort` API runs a session through `&mut`, so the session sits
/// behind a mutex; throughput comes from batching, not concurrent runs.
pub struct Embedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl Embedder {
    /// Load `tokenizer.json` and [`ONNX_FILE`] from `local_dir` (laid out as
    /// the hub repo), or from the HF hub cache at [`REVISION`] (downloading on
    /// a miss) when `local_dir` is `None`. ONNX Runtime sizes its own intra-op
    /// pool. Blocking.
    pub fn load(local_dir: Option<&Path>) -> Result<Self> {
        Self::load_with(local_dir, ONNX_FILE, 0)
    }

    /// As [`Embedder::load`], with another ONNX file (a path in the repo or
    /// directory, or an absolute path) and `intra_threads` intra-op threads
    /// (0: ONNX Runtime's choice). Idle threads do not spin, as in xerj-ai's
    /// MiniLM backend. Blocking.
    pub fn load_with(
        local_dir: Option<&Path>,
        onnx_file: &str,
        intra_threads: usize,
    ) -> Result<Self> {
        let repo = Repo::with_revision(MODEL_ID.into(), RepoType::Model, REVISION.into());
        let onnx = model_file(local_dir, &repo, onnx_file)?;
        let mut tokenizer = Tokenizer::from_file(model_file(local_dir, &repo, "tokenizer.json")?)?;
        // The hub tokenizer.json truncates at 128 (and the
        // sentence-transformers one also pads to 128); padding is done per
        // batch here, and the cap is MAX_TOKENS.
        tokenizer.with_padding(None);
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: MAX_TOKENS,
            ..Default::default()
        }))?;
        Ok(Self {
            session: Mutex::new(session(&onnx, intra_threads, false)?),
            tokenizer,
        })
    }

    pub fn dims(&self) -> usize {
        DIMS
    }

    /// Token ids the model sees for `text`, special tokens included.
    pub fn token_ids(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(clip_utf16(text), true)?
            .get_ids()
            .to_vec())
    }

    /// Unit-length vectors, in input order. Blocking and CPU-bound.
    pub fn embed(&self, texts: &[&str], batching: Batching) -> Result<Vec<Vec<f32>>> {
        let clipped: Vec<&str> = texts.iter().map(|t| clip_utf16(t)).collect();
        let encodings = self.tokenizer.encode_batch(clipped, true)?;
        let lengths: Vec<usize> = encodings.iter().map(|e| e.get_ids().len()).collect();
        let mut out = vec![Vec::new(); texts.len()];
        for rows in group_by_padded_cost(&lengths, batching) {
            let seq_len = rows.iter().map(|&i| lengths[i]).max().unwrap_or(0);
            let vectors = self.forward(&encodings, &rows, seq_len)?;
            for (i, v) in rows.into_iter().zip(vectors) {
                out[i] = v;
            }
        }
        Ok(out)
    }

    fn forward(
        &self,
        encodings: &[Encoding],
        rows: &[usize],
        seq_len: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let batch = rows.len();
        let mut ids = Vec::with_capacity(batch * seq_len);
        let mut mask = Vec::with_capacity(batch * seq_len);
        for &row in rows {
            let e = &encodings[row];
            ids.extend_from_slice(e.get_ids());
            ids.resize(ids.len() + seq_len - e.get_ids().len(), PAD_TOKEN_ID);
            mask.extend_from_slice(e.get_attention_mask());
            mask.resize(mask.len() + seq_len - e.get_attention_mask().len(), 0);
        }
        let [input_ids, attention_mask, token_type_ids] =
            bert_inputs(ids, mask.clone(), batch, seq_len)?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| Error::OrtOutput("session mutex poisoned".into()))?;
        let outputs = session.run(ort::inputs! {
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
            "token_type_ids" => token_type_ids,
        })?;
        let (shape, hidden) = outputs
            .get("last_hidden_state")
            .ok_or_else(|| Error::OrtOutput("no last_hidden_state".into()))?
            .try_extract_tensor::<f32>()?;
        if shape[..] != [batch as i64, seq_len as i64, DIMS as i64] {
            return Err(Error::OrtOutput(format!(
                "last_hidden_state shape {shape:?} for {batch} × {seq_len} input"
            )));
        }
        // Attention-masked mean, then L2 norm, as transformers.js.
        Ok(hidden
            .chunks_exact(seq_len * DIMS)
            .zip(mask.chunks_exact(seq_len))
            .map(|(tokens, mask)| {
                let mut sum = vec![0f32; DIMS];
                let mut count = 0f32;
                for (t, &m) in tokens.chunks_exact(DIMS).zip(mask) {
                    if m != 0 {
                        sum.iter_mut().zip(t).for_each(|(s, x)| *s += x);
                        count += 1.0;
                    }
                }
                sum.iter_mut().for_each(|s| *s /= count);
                let norm = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
                sum.iter_mut().for_each(|s| *s /= norm);
                sum
            })
            .collect())
    }
}

/// xerj-ai `microbatch::group_by_padded_cost`: sort by length, cut a batch
/// when it would exceed the row cap or the padded-token budget.
pub(crate) fn group_by_padded_cost(lengths: &[usize], b: Batching) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    order.sort_by_key(|&i| lengths[i]);
    let mut batches = Vec::new();
    let mut batch: Vec<usize> = Vec::new();
    let mut longest = 0;
    for i in order {
        let next = longest.max(lengths[i]);
        if !batch.is_empty()
            && (batch.len() >= b.max_rows || next * (batch.len() + 1) > b.padded_token_budget)
        {
            batches.push(std::mem::take(&mut batch));
            longest = 0;
        }
        longest = longest.max(lengths[i]);
        batch.push(i);
    }
    if !batch.is_empty() {
        batches.push(batch);
    }
    batches
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parts_number_facts_after_dropping_blanks() {
        let m = json!({"narrative": "n", "facts": ["a", "  ", "b"], "title": "t"});
        let parts = observation_parts("discovery", &m, "body");
        let names: Vec<_> = parts.iter().map(|p| p.part.as_str()).collect();
        assert_eq!(names, ["narrative", "fact_0", "fact_1"]);
        assert_eq!(parts[2].text, "b");
    }

    #[test]
    fn summary_parts_follow_field_order() {
        let m = json!({"notes": "z", "request": "r", "learned": ""});
        let names: Vec<_> = observation_parts("summary", &m, "")
            .into_iter()
            .map(|p| p.part)
            .collect();
        assert_eq!(names, ["request", "notes"]);
    }

    #[test]
    fn title_and_content_when_nothing_else() {
        let m = json!({"title": "T", "subtitle": "S"});
        let parts = observation_parts("discovery", &m, "body");
        assert_eq!(
            parts[0],
            Part {
                part: "title".into(),
                text: "T\nS".into()
            }
        );
        assert_eq!(parts[1].part, "content");
        let untitled = observation_parts("discovery", &json!({"title": "Untitled"}), "x");
        assert_eq!(
            untitled,
            [Part {
                part: "content".into(),
                text: "x".into()
            }]
        );
    }

    #[test]
    fn clip_counts_utf16_units() {
        let s = "\u{1F600}".repeat(MAX_TEXT_UTF16 / 2 + 1);
        assert_eq!(clip_utf16(&s).encode_utf16().count(), MAX_TEXT_UTF16);
    }
}
