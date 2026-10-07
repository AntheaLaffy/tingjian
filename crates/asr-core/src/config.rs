use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct AudioConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub downsample_hidden_size: usize,
    pub num_mel_bins: usize,
    pub n_window: usize,
    pub n_window_infer: usize,
    pub output_dim: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ThinkerConfig {
    pub audio_config: AudioConfig,
    pub text_config: TextConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub thinker_config: ThinkerConfig,
}

impl ModelConfig {
    pub fn load(directory: &Path) -> Result<Self> {
        let config: Self =
            serde_json::from_reader(std::fs::File::open(directory.join("config.json"))?)
                .context("Invalid Qwen3-ASR config.json")?;
        let a = &config.thinker_config.audio_config;
        let t = &config.thinker_config.text_config;
        ensure!(
            a.num_mel_bins == 128 && a.output_dim == t.hidden_size,
            "Unsupported audio configuration"
        );
        ensure!(
            t.num_attention_heads.is_multiple_of(t.num_key_value_heads),
            "Invalid grouped attention configuration"
        );
        ensure!(t.head_dim.is_multiple_of(2), "Invalid rotary dimension");
        // This first backend targets the specified model, so incompatible checkpoints fail before allocating GPU memory.
        ensure!(
            a.d_model == 896
                && a.encoder_layers == 18
                && t.hidden_size == 1024
                && t.num_hidden_layers == 28,
            "This backend requires Qwen3-ASR-0.6B weights"
        );
        Ok(config)
    }
}
