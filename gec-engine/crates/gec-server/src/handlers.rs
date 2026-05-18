use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use axum::{
    extract::{Json as JsonExtract, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use gec_aligner::{apply_edits, merge_edits};
use gec_core::{CorrectionRequest, CorrectionResponse, ResponseStats, RewriteRequest};
use gec_deeppass::llama_runtime::LlamaCppLlm;
use gec_deeppass::DeepPass;
use gec_fastpass::ort_runtime::OrtEncoder;
use gec_fastpass::{FastPass, TagVocab};
use gec_tokenizer::Tokenizer;

/// Server state. Each pass is optional so an operator can start the server
/// without one of the models and still hit /healthz; handlers degrade
/// gracefully when their backing model is absent.
#[derive(Clone)]
pub struct AppState {
    fast: Option<Arc<FastPassBundle>>,
    deep: Option<Arc<DeepPassBundle>>,
}

pub struct FastPassBundle {
    pub tokenizer: Tokenizer,
    pub pass: FastPass<OrtEncoder>,
}

pub struct DeepPassBundle {
    pub pass: DeepPass<LlamaCppLlm>,
}

impl AppState {
    pub fn empty() -> Self {
        Self {
            fast: None,
            deep: None,
        }
    }

    pub fn with_fastpass(mut self, bundle: FastPassBundle) -> Self {
        self.fast = Some(Arc::new(bundle));
        self
    }

    pub fn with_deeppass(mut self, bundle: DeepPassBundle) -> Self {
        self.deep = Some(Arc::new(bundle));
        self
    }

    pub fn load(
        encoder_onnx: Option<&Path>,
        encoder_tags: Option<&Path>,
        encoder_tokenizer: Option<&Path>,
        llm_gguf: Option<&Path>,
    ) -> Result<Self> {
        let mut state = Self::empty();

        if let (Some(onnx), Some(tags), Some(tokjson)) =
            (encoder_onnx, encoder_tags, encoder_tokenizer)
        {
            let encoder = OrtEncoder::from_file(onnx).context("load ONNX encoder")?;
            let vocab: TagVocab = serde_json::from_str(
                &std::fs::read_to_string(tags)
                    .with_context(|| format!("read {}", tags.display()))?,
            )
            .context("parse tags.json")?;
            let tokenizer = Tokenizer::from_file(tokjson).context("load tokenizer.json")?;
            state = state.with_fastpass(FastPassBundle {
                tokenizer,
                pass: FastPass::with_encoder(encoder, vocab),
            });
        }

        if let Some(gguf) = llm_gguf {
            let llm = LlamaCppLlm::from_file(gguf).context("load LLM GGUF")?;
            state = state.with_deeppass(DeepPassBundle {
                pass: DeepPass::with_llm(llm),
            });
        }

        Ok(state)
    }
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/correct", post(correct))
        .route("/rewrite", post(rewrite))
        .with_state(state)
}

async fn healthz() -> &'static str {
    "ok"
}

async fn correct(
    State(state): State<AppState>,
    JsonExtract(req): JsonExtract<CorrectionRequest>,
) -> Result<Json<CorrectionResponse>, (StatusCode, String)> {
    let started = Instant::now();
    let request_id = request_id();

    if req.text.is_empty() {
        return Ok(Json(CorrectionResponse {
            edits: vec![],
            corrected_text: String::new(),
            degraded: false,
            partial: false,
            request_id,
            stats: ResponseStats::default(),
        }));
    }

    let Some(fast) = state.fast.as_ref() else {
        return Ok(Json(CorrectionResponse {
            edits: vec![],
            corrected_text: req.text,
            degraded: true,
            partial: false,
            request_id,
            stats: ResponseStats::default(),
        }));
    };

    let words: Vec<&str> = req.text.split_whitespace().collect();
    if words.is_empty() {
        return Ok(Json(CorrectionResponse {
            edits: vec![],
            corrected_text: req.text,
            degraded: false,
            partial: false,
            request_id,
            stats: ResponseStats::default(),
        }));
    }

    let edits_result = tokio::task::block_in_place(|| fast.pass.edits_for_tokens(&words));
    let edits = match edits_result {
        Ok(e) => merge_edits(e),
        Err(err) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("fast-pass error: {err}"),
            ));
        }
    };

    let corrected_text = apply_edits(&req.text, &edits);
    let fast_ms = started.elapsed().as_millis() as u64;

    Ok(Json(CorrectionResponse {
        edits,
        corrected_text,
        degraded: false,
        partial: false,
        request_id,
        stats: ResponseStats {
            fast_ms,
            deep_ms: 0,
            n_chunks: 1,
            tokens_generated: 0,
        },
    }))
}

async fn rewrite(
    State(state): State<AppState>,
    JsonExtract(req): JsonExtract<RewriteRequest>,
) -> Result<Json<CorrectionResponse>, (StatusCode, String)> {
    let started = Instant::now();
    let request_id = request_id();

    let Some(deep) = state.deep.as_ref() else {
        return Ok(Json(CorrectionResponse {
            edits: vec![],
            corrected_text: req.text,
            degraded: true,
            partial: false,
            request_id,
            stats: ResponseStats::default(),
        }));
    };

    let result = tokio::task::block_in_place(|| deep.pass.rewrite(&req.text, req.tone.clone()));
    match result {
        Ok((rewritten, edits)) => {
            let deep_ms = started.elapsed().as_millis() as u64;
            Ok(Json(CorrectionResponse {
                edits,
                corrected_text: rewritten,
                degraded: false,
                partial: false,
                request_id,
                stats: ResponseStats {
                    fast_ms: 0,
                    deep_ms,
                    n_chunks: 1,
                    tokens_generated: 0,
                },
            }))
        }
        Err(err) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("deep-pass error: {err}"),
        )),
    }
}

fn request_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{:x}-{:x}", dur.as_secs(), dur.subsec_nanos())
}
