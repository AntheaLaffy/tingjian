use anyhow::{Context, Result};
use burn_wgpu::WgpuDevice;
use clap::{Parser, Subcommand, ValueEnum};
use qwen3_asr_core::{audio, model::QwenAsr, tokenizer::Language};
use std::{path::PathBuf, sync::atomic::AtomicBool, time::Instant};

#[derive(Parser)]
#[command(name = "asr", version, about = "Local Qwen3-ASR with CubeCL/wgpu")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Check the real GPU and validate compute/readback.
    GpuCheck,
    /// Transcribe a local WAV with the native CubeCL/wgpu graph.
    Transcribe {
        audio: PathBuf,
        #[arg(long, default_value = "models/Qwen3-ASR-0.6B")]
        model: PathBuf,
        #[arg(long, value_enum, default_value = "auto")]
        language: LanguageArg,
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        #[arg(long, default_value_t = 20.0)]
        segment_seconds: f64,
        #[arg(long)]
        json: bool,
        /// Export first-segment encoder activations as little-endian f32.
        #[arg(long)]
        dump_encoder: Option<PathBuf>,
        /// Use mixed FP16 projections with FP32 normalization and attention.
        #[arg(long)]
        fp16: bool,
        /// Repeat the fixed workload without reloading the model, for warm-run measurement.
        #[arg(long, default_value_t = 1)]
        repeat: usize,
    },
    /// Export log-mel features without loading model weights.
    Mel { audio: PathBuf, output: PathBuf },
}

#[derive(Clone, Copy, ValueEnum)]
enum LanguageArg {
    Auto,
    Chinese,
    English,
}
impl From<LanguageArg> for Language {
    fn from(value: LanguageArg) -> Self {
        match value {
            LanguageArg::Auto => Self::Auto,
            LanguageArg::Chinese => Self::Chinese,
            LanguageArg::English => Self::English,
        }
    }
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::GpuCheck => {
            let report = qwen3_asr_core::gpu::smoke(&WgpuDevice::DefaultDevice)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Mel {
            audio: path,
            output,
        } => {
            let mel = audio::log_mel(&audio::read_wav(&path)?)?;
            std::fs::write(
                output,
                mel.data
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            println!("{} x {} f32 (frequency-major)", audio::MEL_BINS, mel.frames);
        }
        Command::Transcribe {
            audio: path,
            model,
            language,
            max_tokens,
            segment_seconds,
            json,
            dump_encoder,
            fp16,
            repeat,
        } => {
            anyhow::ensure!(
                (1.0..=30.0).contains(&segment_seconds),
                "Segment duration must be between 1 and 30 seconds"
            );
            let device = WgpuDevice::DefaultDevice;
            anyhow::ensure!(
                (1..=100).contains(&repeat),
                "Repeat count must be between 1 and 100"
            );
            let gpu = qwen3_asr_core::gpu::initialize(&device)?;
            eprintln!("GPU: {} ({})", gpu.name, gpu.backend);
            let load_start = Instant::now();
            let mut model =
                QwenAsr::<qwen3_asr_core::gpu::GpuBackend>::load(&model, device, |message| {
                    eprintln!("{message}")
                })?;
            if fp16 {
                anyhow::ensure!(
                    gpu.supports_fp16,
                    "This adapter does not support FP16 shaders"
                );
                model.use_fp16(|message| eprintln!("{message}"));
            }
            let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
            let samples = audio::read_wav(&path)?;
            anyhow::ensure!(
                samples.len() >= 160,
                "WAV recording must contain at least 10 ms"
            );
            let cancel = AtomicBool::new(false);
            let segment_size = (segment_seconds * 16_000.0) as usize;
            let mut results = Vec::new();
            for iteration in 0..repeat {
                for (index, segment) in samples.chunks(segment_size).enumerate() {
                    if segment.len() < 160 {
                        break;
                    }
                    if iteration == 0
                        && index == 0
                        && let Some(output) = &dump_encoder
                    {
                        let values = model.encode_audio(segment, &cancel)?;
                        std::fs::write(
                            output,
                            values
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<_>>(),
                        )
                        .context("Writing encoder output")?;
                    }
                    let result = model.transcribe(
                        segment,
                        language.into(),
                        max_tokens,
                        &cancel,
                        |_| {},
                        |message| eprintln!("{message}"),
                    )?;
                    if !json {
                        println!("{}", result.text);
                    }
                    eprintln!(
                        "Segment {}: {:.2}s audio, {:.2}s compute, RTF {:.3}, {} tokens, finished={}",
                        index + 1,
                        result.audio_seconds,
                        result.elapsed_ms / 1000.0,
                        result.real_time_factor,
                        result.generated_tokens.len(),
                        result.finished
                    );
                    results.push(result);
                }
            }
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &serde_json::json!({"gpu": gpu, "load_ms": load_ms, "precision": if fp16 { "mixed-fp16" } else { "fp32" }, "repeat": repeat, "segments": results})
                    )?
                );
            }
        }
    }
    Ok(())
}
