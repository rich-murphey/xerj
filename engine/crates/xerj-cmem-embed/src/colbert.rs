//! answerdotai/answerai-colbert-small-v1 late-interaction encoder and MaxSim
//! scorer, reproducing the token vectors PyLate 1.6.0 writes in
//! claude-mem-search-eval (`src/colbert_encode.py`; the spec is
//! `results/colbert-rerank.md` § Porting to Rust, checked against
//! `fixtures/colbert-parity.jsonl` by the `colbert` binary).
//!
//! BERT (12 layers, 384 wide) then `linear` 384 → [`DIM`] without bias, then
//! L2 norm per token. A query is `[CLS] [Q] text [SEP]` padded with `[MASK]`
//! to exactly [`QUERY_TOKENS`] ids, the padding not attended to but output:
//! every one of the 32 rows enters MaxSim. A document is `[CLS] [D] text
//! [SEP]`, at most [`DOC_MAX_TOKENS`] ids, with punctuation tokens attended
//! to and then dropped from the output.
//!
//! BERT runs on ONNX Runtime: either the hub's own export at [`REVISION`]
//! ([`FP32_FILE`], or a quantised sibling), which stops at
//! `last_hidden_state` [batch, seq, 384] so the head is applied here from
//! `model.safetensors`, or a re-export with the head folded in
//! (`research/claude-mem/colbert_export.py`, output `token_vectors`
//! [batch, seq, 96], already unit length), used as-is.
//!
//! The safe `ort` API runs a session through `&mut`, so the session sits
//! behind a mutex; throughput comes from batching, not concurrent runs.

use crate::{bert_inputs, group_by_padded_cost, model_file, session, Batching, Error, Result};
use half::f16;
use hf_hub::{Repo, RepoType};
use ort::session::Session;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::{Tokenizer, TruncationParams};

pub const MODEL_ID: &str = "answerdotai/answerai-colbert-small-v1";

/// The revision the eval's vectors and fixture were made with.
pub const REVISION: &str = "934fa8bb4ce2284f4c2baa232d81aca4d076fa5e";

/// The hub's fp32 export, the one of its files that passes parity.
pub const FP32_FILE: &str = "onnx/model.onnx";

/// Width of BERT's hidden state.
const HIDDEN: usize = 384;

/// Values per token vector.
pub const DIM: usize = 96;

/// Ids per query, marker and `[MASK]` padding included.
pub const QUERY_TOKENS: usize = 32;

/// Ids per document at most, marker included: the eval's cut, not the 300
/// the model was trained at.
pub const DOC_MAX_TOKENS: usize = 512;

const MASK: u32 = 103;
/// `[unused0]`, inserted after `[CLS]`.
const QUERY_MARKER: u32 = 1;
/// `[unused1]`, inserted after `[CLS]`.
const DOC_MARKER: u32 = 2;
/// PyLate's document skiplist: Python's `string.punctuation`.
const PUNCTUATION: &str = r##"!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"##;

/// The ids and attention mask BERT sees for one text, marker included.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Input {
    pub ids: Vec<u32>,
    pub mask: Vec<u32>,
}

/// Tokenisation and output selection.
pub struct Tokens {
    query_tokenizer: Tokenizer,
    doc_tokenizer: Tokenizer,
    skiplist: HashSet<u32>,
}

impl Tokens {
    fn load(tokenizer_json: &Path) -> Result<Self> {
        // Text is cut one short of the target with [CLS] and [SEP] kept, so
        // a truncated input still ends in [SEP] once the marker is in.
        let tokenizer = |max_length: usize| -> Result<Tokenizer> {
            let mut t = Tokenizer::from_file(tokenizer_json)?;
            t.with_padding(None);
            t.with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))?;
            Ok(t)
        };
        let query_tokenizer = tokenizer(QUERY_TOKENS - 1)?;
        let doc_tokenizer = tokenizer(DOC_MAX_TOKENS - 1)?;
        let skiplist = PUNCTUATION
            .chars()
            .filter_map(|c| doc_tokenizer.token_to_id(&c.to_string()))
            .collect();
        Ok(Self {
            query_tokenizer,
            doc_tokenizer,
            skiplist,
        })
    }

    pub fn query_input(&self, text: &str) -> Result<Input> {
        let mut ids = marked(&self.query_tokenizer, text, QUERY_MARKER)?;
        let mut mask = vec![1; ids.len()];
        ids.resize(QUERY_TOKENS, MASK);
        mask.resize(QUERY_TOKENS, 0);
        Ok(Input { ids, mask })
    }

    pub fn document_input(&self, text: &str) -> Result<Input> {
        let ids = marked(&self.doc_tokenizer, text, DOC_MARKER)?;
        let mask = vec![1; ids.len()];
        Ok(Input { ids, mask })
    }

    /// Positions of a document input whose vectors are output: all but
    /// the skiplist's.
    pub fn kept(&self, input: &Input) -> Vec<usize> {
        (0..input.ids.len())
            .filter(|&i| !self.skiplist.contains(&input.ids[i]))
            .collect()
    }
}

/// `linear` 384 → [`DIM`] without bias, then L2 norm per token.
struct Head {
    /// `linear.weight`, row-major `[DIM, 384]`.
    weight: Vec<f32>,
}

impl Head {
    fn load(safetensors_file: &Path) -> Result<Self> {
        let bytes = std::fs::read(safetensors_file).map_err(|source| Error::Io {
            path: safetensors_file.into(),
            source,
        })?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let w = tensors.tensor("linear.weight")?;
        if w.dtype() != Dtype::F32 || w.shape() != [DIM, HIDDEN] {
            return Err(Error::Weights(format!(
                "linear.weight is {:?} {:?}, not F32 [{DIM}, {HIDDEN}]",
                w.dtype(),
                w.shape()
            )));
        }
        let weight = w
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Ok(Self { weight })
    }

    /// Unit vectors at every position of each input's own length, from
    /// BERT's `[batch, seq_len, 384]` last hidden state.
    fn apply(&self, hidden: &[f32], seq_len: usize, inputs: &[&Input]) -> Vec<Vec<f32>> {
        inputs
            .iter()
            .zip(hidden.chunks_exact(seq_len * HIDDEN))
            .map(|(input, rows)| {
                let mut out = Vec::with_capacity(input.ids.len() * DIM);
                for h in rows.chunks_exact(HIDDEN).take(input.ids.len()) {
                    let start = out.len();
                    out.extend(self.weight.chunks_exact(HIDDEN).map(|w| dot(w, h)));
                    let v = &mut out[start..];
                    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    v.iter_mut().for_each(|x| *x /= norm);
                }
                out
            })
            .collect()
    }
}

/// Ids and mask of a batch, row-major `[inputs, seq_len]`, padded with
/// [MASK] under attention mask 0.
fn pad(inputs: &[&Input], seq_len: usize) -> (Vec<u32>, Vec<u32>) {
    let mut ids = Vec::with_capacity(inputs.len() * seq_len);
    let mut mask = Vec::with_capacity(inputs.len() * seq_len);
    for input in inputs {
        ids.extend_from_slice(&input.ids);
        ids.resize(ids.len() + seq_len - input.ids.len(), MASK);
        mask.extend_from_slice(&input.mask);
        mask.resize(mask.len() + seq_len - input.mask.len(), 0);
    }
    (ids, mask)
}

pub struct Colbert {
    session: Mutex<Session>,
    /// `None` when the graph has the head folded in.
    head: Option<Head>,
    tokens: Tokens,
}

impl Colbert {
    /// `onnx_file` is an absolute path, a path inside the hub repo at
    /// [`REVISION`] (e.g. [`FP32_FILE`]) or, with `local_dir`, inside that
    /// directory, beside `tokenizer.json` (and `model.safetensors` when the
    /// graph has no head). `intra_threads` sizes ONNX Runtime's intra-op
    /// pool; `spinning` lets its idle threads spin. Blocking.
    pub fn load(
        local_dir: Option<&Path>,
        onnx_file: &str,
        intra_threads: usize,
        spinning: bool,
    ) -> Result<Self> {
        let repo = repo();
        let session = session(
            &model_file(local_dir, &repo, onnx_file)?,
            intra_threads.max(1),
            spinning,
        )?;
        let has_head = session
            .outputs()
            .iter()
            .any(|o| o.name() == "token_vectors");
        let head = match has_head {
            true => None,
            false => Some(Head::load(&model_file(
                local_dir,
                &repo,
                "model.safetensors",
            )?)?),
        };
        Ok(Self {
            session: Mutex::new(session),
            head,
            tokens: Tokens::load(&model_file(local_dir, &repo, "tokenizer.json")?)?,
        })
    }

    pub fn tokens(&self) -> &Tokens {
        &self.tokens
    }

    /// `QUERY_TOKENS × DIM` values, row-major. Blocking and CPU-bound.
    pub fn encode_query(&self, text: &str) -> Result<Vec<f32>> {
        let input = self.tokens.query_input(text)?;
        let mut rows = self.forward(&[&input], QUERY_TOKENS)?;
        Ok(rows.pop().unwrap_or_default())
    }

    /// Per text, in input order, `kept × DIM` values, row-major. Blocking
    /// and CPU-bound.
    pub fn encode_documents(&self, texts: &[&str], batching: Batching) -> Result<Vec<Vec<f32>>> {
        let tokens = &self.tokens;
        let inputs = texts
            .iter()
            .map(|t| tokens.document_input(t))
            .collect::<Result<Vec<_>>>()?;
        let lengths: Vec<usize> = inputs.iter().map(|i| i.ids.len()).collect();
        let mut out = vec![Vec::new(); texts.len()];
        for rows in group_by_padded_cost(&lengths, batching) {
            let seq_len = rows.iter().map(|&i| lengths[i]).max().unwrap_or(0);
            let batch: Vec<&Input> = rows.iter().map(|&i| &inputs[i]).collect();
            let vectors = self.forward(&batch, seq_len)?;
            for (i, all) in rows.into_iter().zip(vectors) {
                out[i] = tokens
                    .kept(&inputs[i])
                    .into_iter()
                    .flat_map(|p| &all[p * DIM..(p + 1) * DIM])
                    .copied()
                    .collect();
            }
        }
        Ok(out)
    }

    /// Unit vectors at every position of each input's own length.
    fn forward(&self, inputs: &[&Input], seq_len: usize) -> Result<Vec<Vec<f32>>> {
        let (ids, mask) = pad(inputs, seq_len);
        let [input_ids, attention_mask, token_type_ids] =
            bert_inputs(ids, mask, inputs.len(), seq_len)?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| Error::OrtOutput("session mutex poisoned".into()))?;
        let outputs = session.run(ort::inputs! {
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
            "token_type_ids" => token_type_ids,
        })?;
        let (name, width) = match self.head {
            Some(_) => ("last_hidden_state", HIDDEN),
            None => ("token_vectors", DIM),
        };
        let (shape, values) = outputs
            .get(name)
            .ok_or_else(|| Error::OrtOutput(format!("no {name}")))?
            .try_extract_tensor::<f32>()?;
        if shape[..] != [inputs.len() as i64, seq_len as i64, width as i64] {
            return Err(Error::OrtOutput(format!(
                "{name} shape {shape:?} for {} × {seq_len} input",
                inputs.len()
            )));
        }
        Ok(match &self.head {
            Some(head) => head.apply(values, seq_len, inputs),
            None => inputs
                .iter()
                .zip(values.chunks_exact(seq_len * DIM))
                .map(|(input, rows)| rows[..input.ids.len() * DIM].to_vec())
                .collect(),
        })
    }
}

fn repo() -> Repo {
    Repo::with_revision(MODEL_ID.into(), RepoType::Model, REVISION.into())
}

/// `[CLS] marker text… [SEP]`, the text cut so the whole is one longer than
/// the tokenizer's truncation length.
fn marked(tokenizer: &Tokenizer, text: &str, marker: u32) -> Result<Vec<u32>> {
    let mut ids = tokenizer.encode(text, true)?.get_ids().to_vec();
    ids.insert(1, marker);
    Ok(ids)
}

/// Σ over query rows of the best dot product with any document row. Both
/// are row-major `DIM`-wide unit vectors; a document with no rows scores −∞.
pub fn maxsim(query: &[f32], doc: &[f32]) -> f32 {
    let mut best = vec![f32::NEG_INFINITY; query.len() / DIM];
    for d in doc.chunks_exact(DIM) {
        for (b, q) in best.iter_mut().zip(query.chunks_exact(DIM)) {
            *b = b.max(dot(q, d));
        }
    }
    best.iter().sum()
}

/// Eight independent partial sums so the loop vectorises.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    for (x, y) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum()
}

/// Storage form: the eval's release v5 holds fp16 rows.
pub fn to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

pub fn from_f16(v: &[f16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maxsim_sums_best_match_per_query_row() {
        let unit = |i: usize| {
            let mut v = vec![0.0; DIM];
            v[i] = 1.0;
            v
        };
        let query = [unit(0), unit(1)].concat();
        let doc = [unit(1), unit(2)].concat();
        assert_eq!(maxsim(&query, &doc), 1.0);
        let half_0 = vec![0.5; DIM];
        assert_eq!(maxsim(&query, &[doc, half_0].concat()), 1.5);
    }

    #[test]
    fn f16_round_trip() {
        let v = [0.1f32, -0.25, 1.0];
        let back = from_f16(&to_f16(&v));
        assert!(v.iter().zip(&back).all(|(a, b)| (a - b).abs() < 1e-3));
    }
}
