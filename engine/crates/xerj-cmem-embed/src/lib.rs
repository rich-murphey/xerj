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
//! The model is the same network as xerj-ai's `neural` backend (candle BERT,
//! `sentence-transformers/all-MiniLM-L6-v2` safetensors); only the token cap
//! differs (xerj-ai: 512).

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokenizers::{Encoding, Tokenizer, TruncationParams};

/// The `model` column claude-mem writes for these vectors.
pub const MODEL_TAG: &str = "Xenova/all-MiniLM-L6-v2/fp32/mean/l2/256";

/// Hub repository holding the safetensors weights candle reads.
pub const DEFAULT_MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";

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
    #[error("model config: {0}")]
    Config(#[from] serde_json::Error),
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
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

pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dims: usize,
}

impl Embedder {
    /// Load from a directory holding `config.json`, `tokenizer.json` and
    /// `model.safetensors`, or from the HF hub cache (downloading on a miss)
    /// when `local_dir` is `None`. Blocking.
    pub fn load(local_dir: Option<&Path>) -> Result<Self> {
        let (config_path, tokenizer_path, weights_path) = match local_dir {
            Some(dir) => (
                dir.join("config.json"),
                dir.join("tokenizer.json"),
                dir.join("model.safetensors"),
            ),
            None => {
                let repo = hf_hub::api::sync::Api::new()?.model(DEFAULT_MODEL_ID.into());
                (
                    repo.get("config.json")?,
                    repo.get("tokenizer.json")?,
                    repo.get("model.safetensors")?,
                )
            }
        };
        let config_json = std::fs::read_to_string(&config_path).map_err(|source| Error::Io {
            path: config_path.clone(),
            source,
        })?;
        let config: Config = serde_json::from_str(&config_json)?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)?;
        // The hub tokenizer.json pads to a fixed 128 and truncates at 128;
        // padding is done per batch here, and the cap is MAX_TOKENS.
        tokenizer.with_padding(None);
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: MAX_TOKENS,
            ..Default::default()
        }))?;

        let device = Device::Cpu;
        // SAFETY: the weights file is not modified while mapped.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&weights_path),
                DTYPE,
                &device,
            )?
        };
        let model = BertModel::load(vb, &config)?;
        Ok(Self {
            model,
            tokenizer,
            device,
            dims: config.hidden_size,
        })
    }

    pub fn dims(&self) -> usize {
        self.dims
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
        let input_ids = Tensor::from_vec(ids, (batch, seq_len), &self.device)?;
        let attention_mask = Tensor::from_vec(mask, (batch, seq_len), &self.device)?;
        let token_type_ids = input_ids.zeros_like()?;
        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
        let mask_f = attention_mask.to_dtype(DType::F32)?.unsqueeze(2)?;
        let mean = hidden
            .broadcast_mul(&mask_f)?
            .sum(1)?
            .broadcast_div(&mask_f.sum(1)?)?;
        let norm = mean.sqr()?.sum_keepdim(1)?.sqrt()?;
        Ok(mean.broadcast_div(&norm)?.to_vec2::<f32>()?)
    }
}

/// xerj-ai `microbatch::group_by_padded_cost`: sort by length, cut a batch
/// when it would exceed the row cap or the padded-token budget.
fn group_by_padded_cost(lengths: &[usize], b: Batching) -> Vec<Vec<usize>> {
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
