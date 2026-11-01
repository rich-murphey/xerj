//! EXPERIMENTAL ONNX Runtime backend (`onnx-experimental`, GNU/Linux), as
//! xerj-ai's: the hub's own export of the BERT body at [`super::REVISION`]
//! (`onnx/model.onnx`, fp32, or a quantised sibling), with tokenisation and
//! output selection shared with the candle backend. The export stops at
//! `last_hidden_state` [batch, seq, 384]; the 384 → 96 head is applied here
//! from `model.safetensors`.
//!
//! The safe `ort` API runs a session through `&mut`, so the session sits
//! behind a mutex; throughput comes from batching, not concurrent runs.

use super::{pad, repo, ColbertEncoder, Head, Input, Tokens};
use crate::{Error, ModelFiles, Result};
use candle_core::{safetensors::MmapedSafetensors, Device, Tensor};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor as OrtTensor;
use std::path::Path;
use std::sync::Mutex;

/// The fp32 export, the one that passes parity.
pub const FP32_FILE: &str = "onnx/model.onnx";

pub struct OrtColbert {
    session: Mutex<Session>,
    head: Head,
    tokens: Tokens,
}

impl OrtColbert {
    /// `onnx_file` is a path inside the hub repo (e.g. [`FP32_FILE`]) or, with
    /// `local_dir`, inside that directory, beside `tokenizer.json` and
    /// `model.safetensors`. `intra_threads` sizes ONNX Runtime's intra-op
    /// pool; `spinning` lets its idle threads spin. Blocking.
    pub fn load(
        local_dir: Option<&Path>,
        onnx_file: &str,
        intra_threads: usize,
        spinning: bool,
    ) -> Result<Self> {
        let files = ModelFiles::resolve(local_dir, repo())?;
        let onnx_path = match local_dir {
            Some(dir) => dir.join(onnx_file),
            None => hf_hub::api::sync::Api::new()?.repo(repo()).get(onnx_file)?,
        };
        // SAFETY: the weights file is not modified while mapped.
        let weights = unsafe { MmapedSafetensors::new(&files.weights)? };
        let head = Head::new(weights.load("linear.weight", &Device::Cpu)?)?;
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort::Error::from)?
            .with_intra_threads(intra_threads.max(1))
            .map_err(ort::Error::from)?
            .with_intra_op_spinning(spinning)
            .map_err(ort::Error::from)?
            .commit_from_file(&onnx_path)?;
        Ok(Self {
            session: Mutex::new(session),
            head,
            tokens: Tokens::load(&files.tokenizer)?,
        })
    }
}

impl ColbertEncoder for OrtColbert {
    fn tokens(&self) -> &Tokens {
        &self.tokens
    }

    fn forward(&self, inputs: &[&Input], seq_len: usize) -> Result<Vec<Vec<f32>>> {
        let (ids, mask) = pad(inputs, seq_len);
        let shape = [inputs.len(), seq_len];
        let as_i64 = |v: Vec<u32>| v.into_iter().map(i64::from).collect::<Vec<_>>();
        let token_types = vec![0i64; ids.len()];
        let input_ids = OrtTensor::from_array((shape, as_i64(ids)))?;
        let attention_mask = OrtTensor::from_array((shape, as_i64(mask)))?;
        let token_type_ids = OrtTensor::from_array((shape, token_types))?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| Error::OrtOutput("session mutex poisoned".into()))?;
        let outputs = session.run(ort::inputs! {
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
            "token_type_ids" => token_type_ids,
        })?;
        let output = outputs
            .get("last_hidden_state")
            .ok_or_else(|| Error::OrtOutput("no last_hidden_state".into()))?;
        let (dims, hidden) = output.try_extract_tensor::<f32>()?;
        let dims: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
        let [batch, len, width] = dims[..] else {
            return Err(Error::OrtOutput(format!("output shape {dims:?}")));
        };
        if (batch, len) != (inputs.len(), seq_len) {
            return Err(Error::OrtOutput(format!(
                "output shape {dims:?} for {} × {seq_len} input",
                inputs.len()
            )));
        }
        let hidden = Tensor::from_slice(hidden, (batch, len, width), &Device::Cpu)?;
        self.head.apply(&hidden, inputs)
    }
}
