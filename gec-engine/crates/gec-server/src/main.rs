use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use gec_server::{router, AppState};

#[derive(Parser, Debug)]
#[command(name = "gec-server")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,
    #[arg(long)]
    encoder_onnx: Option<PathBuf>,
    #[arg(long)]
    encoder_tags: Option<PathBuf>,
    #[arg(long)]
    encoder_tokenizer: Option<PathBuf>,
    #[arg(long)]
    llm_gguf: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let state = AppState::load(
        args.encoder_onnx.as_deref(),
        args.encoder_tags.as_deref(),
        args.encoder_tokenizer.as_deref(),
        args.llm_gguf.as_deref(),
    )?;
    let app = router(state);
    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    tracing::info!("listening on {}", args.bind);
    axum::serve(listener, app).await?;
    Ok(())
}
