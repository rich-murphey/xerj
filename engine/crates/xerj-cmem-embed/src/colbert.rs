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

use crate::{group_by_padded_cost, load_bert, Batching, ModelFiles, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::bert::BertModel;
use half::f16;
use hf_hub::{Repo, RepoType};
use std::collections::HashSet;
use std::path::Path;
use tokenizers::{Tokenizer, TruncationParams};

#[cfg(feature = "onnx-experimental")]
pub mod onnx;

pub const MODEL_ID: &str = "answerdotai/answerai-colbert-small-v1";

/// The revision the eval's vectors and fixture were made with.
pub const REVISION: &str = "934fa8bb4ce2284f4c2baa232d81aca4d076fa5e";

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

/// Tokenisation and output selection, shared by every backend.
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
    /// `linear.weight` transposed, `[384, DIM]`.
    linear: Tensor,
}

impl Head {
    fn new(weight: Tensor) -> Result<Self> {
        Ok(Self {
            linear: weight.t()?.contiguous()?,
        })
    }

    /// Unit vectors at every position of each input's own length, from
    /// BERT's `[batch, seq_len, 384]` last hidden state.
    fn apply(&self, hidden: &Tensor, inputs: &[&Input]) -> Result<Vec<Vec<f32>>> {
        let projected = hidden.broadcast_matmul(&self.linear)?;
        let norm = projected.sqr()?.sum_keepdim(2)?.sqrt()?;
        let unit = projected.broadcast_div(&norm)?;
        Ok(unit
            .flatten_from(1)?
            .to_vec2::<f32>()?
            .into_iter()
            .zip(inputs)
            .map(|(mut v, input)| {
                v.truncate(input.ids.len() * DIM);
                v
            })
            .collect())
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

/// A backend: BERT's forward pass and the head, over shared tokenisation.
pub trait ColbertEncoder {
    fn tokens(&self) -> &Tokens;

    /// Unit vectors at every position of each input's own length.
    fn forward(&self, inputs: &[&Input], seq_len: usize) -> Result<Vec<Vec<f32>>>;

    /// `QUERY_TOKENS × DIM` values, row-major. Blocking and CPU-bound.
    fn encode_query(&self, text: &str) -> Result<Vec<f32>> {
        let input = self.tokens().query_input(text)?;
        let mut rows = self.forward(&[&input], QUERY_TOKENS)?;
        Ok(rows.pop().unwrap_or_default())
    }

    /// Per text, in input order, `kept × DIM` values, row-major. Blocking
    /// and CPU-bound.
    fn encode_documents(&self, texts: &[&str], batching: Batching) -> Result<Vec<Vec<f32>>> {
        let tokens = self.tokens();
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
}

/// The candle backend, the default and the one musl builds have.
pub struct Colbert {
    bert: BertModel,
    head: Head,
    tokens: Tokens,
    device: Device,
}

impl Colbert {
    /// Load from a directory holding `config.json`, `tokenizer.json` and
    /// `model.safetensors`, or from the HF hub cache at [`REVISION`]
    /// (downloading on a miss) when `local_dir` is `None`. Blocking.
    pub fn load(local_dir: Option<&Path>) -> Result<Self> {
        let files = ModelFiles::resolve(local_dir, repo())?;
        let device = Device::Cpu;
        let (bert, config, vb) = load_bert(&files, &device)?;
        let head = Head::new(vb.get((DIM, config.hidden_size), "linear.weight")?)?;
        Ok(Self {
            bert,
            head,
            tokens: Tokens::load(&files.tokenizer)?,
            device,
        })
    }
}

impl ColbertEncoder for Colbert {
    fn tokens(&self) -> &Tokens {
        &self.tokens
    }

    fn forward(&self, inputs: &[&Input], seq_len: usize) -> Result<Vec<Vec<f32>>> {
        let (ids, mask) = pad(inputs, seq_len);
        let shape = (inputs.len(), seq_len);
        let input_ids = Tensor::from_vec(ids, shape, &self.device)?;
        let attention_mask = Tensor::from_vec(mask, shape, &self.device)?;
        let token_type_ids = input_ids.zeros_like()?;
        let hidden = self
            .bert
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
        self.head.apply(&hidden, inputs)
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
