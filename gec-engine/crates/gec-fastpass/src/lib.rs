use anyhow::Result;
use gec_core::{Edit, EditCategory};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

pub trait EncoderSession {
    fn forward(&self, input_ids: &[i64], attention_mask: &[i64]) -> Result<Array2<f32>>;
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TagVocab {
    pub tags: Vec<String>,
}

pub struct FastPass<E: EncoderSession> {
    encoder: E,
    vocab: TagVocab,
}

impl<E: EncoderSession> FastPass<E> {
    pub fn with_encoder(encoder: E, vocab: TagVocab) -> Self {
        Self { encoder, vocab }
    }

    pub fn edits_for_tokens(&self, tokens: &[&str]) -> Result<Vec<Edit>> {
        let n = tokens.len() as i64;
        let ids: Vec<i64> = (0..n).collect();
        let mask: Vec<i64> = vec![1; n as usize];
        let logits = self.encoder.forward(&ids, &mask)?;
        let mut edits = Vec::new();
        let mut byte_offset = 0usize;
        for (idx, tok) in tokens.iter().enumerate() {
            let row = logits.row(idx);
            let (tag_idx, &conf) = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));
            let tag = self
                .vocab
                .tags
                .get(tag_idx)
                .map(String::as_str)
                .unwrap_or("$KEEP");
            let (kind, value) = decode_tag_string(tag);
            let tok_len = tok.len();
            let span = byte_offset..byte_offset + tok_len;
            match kind.as_str() {
                "KEEP" => {}
                "DELETE" => edits.push(Edit {
                    span: span.clone(),
                    original: (*tok).into(),
                    replacement: String::new(),
                    category: EditCategory::Grammar,
                    confidence: conf,
                }),
                "REPLACE" => edits.push(Edit {
                    span: span.clone(),
                    original: (*tok).into(),
                    replacement: value,
                    category: EditCategory::Grammar,
                    confidence: conf,
                }),
                "APPEND" => edits.push(Edit {
                    span: span.end..span.end,
                    original: String::new(),
                    replacement: format!(" {}", value),
                    category: EditCategory::Grammar,
                    confidence: conf,
                }),
                _ => {}
            }
            byte_offset += tok_len + 1;
        }
        Ok(edits)
    }
}

pub fn decode_tag_string(s: &str) -> (String, String) {
    let body = s.strip_prefix('$').unwrap_or(s);
    if body == "KEEP" {
        return ("KEEP".into(), String::new());
    }
    if body == "DELETE" {
        return ("DELETE".into(), String::new());
    }
    match body.split_once('_') {
        Some((head, value)) => (head.to_string(), value.to_string()),
        None => (body.to_string(), String::new()),
    }
}

#[cfg(feature = "ort-runtime")]
pub mod ort_runtime {
    use std::path::Path;
    use std::sync::Mutex;

    use anyhow::Context;
    use ndarray::{Array, Array2};
    use ort::session::Session;
    use ort::value::Value;

    use super::{EncoderSession, Result};

    /// ort-backed ONNX encoder. The session is wrapped in a Mutex so the
    /// `EncoderSession` trait (which takes `&self`) can mutate the session
    /// during `run`. ort 2.0 `Session::run` requires `&mut self`.
    pub struct OrtEncoder {
        session: Mutex<Session>,
    }

    impl OrtEncoder {
        pub fn from_file(path: &Path) -> Result<Self> {
            let session = Session::builder()
                .context("ort Session::builder")?
                .commit_from_file(path)
                .with_context(|| format!("ort commit_from_file({})", path.display()))?;
            Ok(Self {
                session: Mutex::new(session),
            })
        }
    }

    impl EncoderSession for OrtEncoder {
        fn forward(&self, input_ids: &[i64], attention_mask: &[i64]) -> Result<Array2<f32>> {
            let seq_len = input_ids.len();
            anyhow::ensure!(
                seq_len == attention_mask.len(),
                "input_ids and attention_mask must agree in length"
            );

            let ids = Array::from_shape_vec((1, seq_len), input_ids.to_vec())
                .context("input_ids shape")?;
            let mask = Array::from_shape_vec((1, seq_len), attention_mask.to_vec())
                .context("attention_mask shape")?;

            let mut session = self
                .session
                .lock()
                .map_err(|_| anyhow::anyhow!("ort session mutex poisoned"))?;

            let outputs = session
                .run(ort::inputs![
                    "input_ids" => Value::from_array(ids)?,
                    "attention_mask" => Value::from_array(mask)?,
                ])
                .context("ort session.run")?;

            // Expected output name is "logits". ort rc.10 returns &Value here.
            let logits_value = outputs
                .get("logits")
                .ok_or_else(|| anyhow::anyhow!("ort session has no `logits` output"))?;

            let (shape, data) = logits_value
                .try_extract_tensor::<f32>()
                .context("extract logits as f32")?;

            // ort 2.0 Shape is iterable into i64. Accept [1, seq, num_tags]
            // or [seq, num_tags] by collecting + matching.
            let dims: Vec<i64> = shape.iter().copied().collect();
            let (seq, num_tags) = match dims.as_slice() {
                [1, s, t] => (*s as usize, *t as usize),
                [s, t] => (*s as usize, *t as usize),
                other => anyhow::bail!("unexpected logits shape: {:?}", other),
            };
            anyhow::ensure!(
                seq == seq_len,
                "ort returned seq_len={} but expected {}",
                seq,
                seq_len
            );

            let arr = Array2::from_shape_vec((seq, num_tags), data.to_vec())
                .context("Array2 from logits buffer")?;
            Ok(arr)
        }
    }
}
