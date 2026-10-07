use crate::{
    audio::{MelFeatures, log_mel},
    config::{AudioConfig, ModelConfig, TextConfig},
    tokenizer::{ASR_TEXT, AsrTokenizer, Language, PREFIX, STOP_TOKENS},
    weights::WeightReader,
};
use anyhow::{Result, bail, ensure};
use burn::tensor::{
    DType, FloatDType, Tensor, TensorData,
    activation::{gelu, silu, softmax},
    module::conv2d,
    ops::ConvOptions,
};
use half::bf16;
use serde::Serialize;
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

#[derive(Clone)]
struct Linear<B: crate::kernels::MatmulBackend> {
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
}

impl<B: crate::kernels::MatmulBackend> Linear<B> {
    fn load(
        reader: &mut WeightReader,
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
        device: &B::Device,
    ) -> Result<Self> {
        let values = reader.read(&format!("{name}.weight"), &[output, input])?;
        let mut transposed = vec![0.0; values.len()];
        for o in 0..output {
            for i in 0..input {
                transposed[i * output + o] = values[o * input + i];
            }
        }
        let weight = tensor(transposed, [input, output], device);
        let bias = if bias {
            Some(tensor(
                reader.read(&format!("{name}.bias"), &[output])?,
                [output],
                device,
            ))
        } else {
            None
        };
        Ok(Self { weight, bias })
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let result = stable_matmul(x, self.weight.clone());
        match &self.bias {
            Some(bias) => result + bias.clone().unsqueeze::<2>(),
            None => result,
        }
    }
}

fn tensor<B: crate::kernels::MatmulBackend, const D: usize>(
    values: Vec<f32>,
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D> {
    Tensor::from_data(TensorData::new(values, shape), device)
}

#[derive(Clone)]
struct LayerNorm<B: crate::kernels::MatmulBackend> {
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
}
impl<B: crate::kernels::MatmulBackend> LayerNorm<B> {
    fn load(
        reader: &mut WeightReader,
        name: &str,
        dimension: usize,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: tensor(
                reader.read(&format!("{name}.weight"), &[dimension])?,
                [dimension],
                device,
            ),
            bias: tensor(
                reader.read(&format!("{name}.bias"), &[dimension])?,
                [dimension],
                device,
            ),
        })
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let centered = x.clone() - x.mean_dim(1);
        let inverse_std = (centered.clone().powf_scalar(2.0).mean_dim(1) + 1e-5)
            .sqrt()
            .recip();
        centered * inverse_std * self.weight.clone().unsqueeze::<2>()
            + self.bias.clone().unsqueeze::<2>()
    }
}

fn rms_norm<B: crate::kernels::MatmulBackend, const D: usize>(
    x: Tensor<B, D>,
    weight: Tensor<B, 1>,
    epsilon: f64,
) -> Tensor<B, D> {
    let inverse_rms = (x.clone().powf_scalar(2.0).mean_dim(D - 1) + epsilon)
        .sqrt()
        .recip();
    x * inverse_rms * weight.unsqueeze::<D>()
}

#[derive(Clone)]
struct EncoderLayer<B: crate::kernels::MatmulBackend> {
    attention_norm: LayerNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    ffn_norm: LayerNorm<B>,
    up: Linear<B>,
    down: Linear<B>,
}
impl<B: crate::kernels::MatmulBackend> EncoderLayer<B> {
    fn load(
        reader: &mut WeightReader,
        index: usize,
        config: &AudioConfig,
        device: &B::Device,
    ) -> Result<Self> {
        let p = format!("thinker.audio_tower.layers.{index}");
        let d = config.d_model;
        Ok(Self {
            attention_norm: LayerNorm::load(
                reader,
                &format!("{p}.self_attn_layer_norm"),
                d,
                device,
            )?,
            q: Linear::load(reader, &format!("{p}.self_attn.q_proj"), d, d, true, device)?,
            k: Linear::load(reader, &format!("{p}.self_attn.k_proj"), d, d, true, device)?,
            v: Linear::load(reader, &format!("{p}.self_attn.v_proj"), d, d, true, device)?,
            out: Linear::load(
                reader,
                &format!("{p}.self_attn.out_proj"),
                d,
                d,
                true,
                device,
            )?,
            ffn_norm: LayerNorm::load(reader, &format!("{p}.final_layer_norm"), d, device)?,
            up: Linear::load(
                reader,
                &format!("{p}.fc1"),
                d,
                config.encoder_ffn_dim,
                true,
                device,
            )?,
            down: Linear::load(
                reader,
                &format!("{p}.fc2"),
                config.encoder_ffn_dim,
                d,
                true,
                device,
            )?,
        })
    }
    fn forward(&self, x: Tensor<B, 2>, heads: usize) -> Tensor<B, 2> {
        let [length, dimension] = x.dims();
        let normalized = self.attention_norm.forward(x.clone());
        let q = self
            .q
            .forward(normalized.clone())
            .reshape([length, heads, dimension / heads])
            .swap_dims(0, 1);
        let k = self
            .k
            .forward(normalized.clone())
            .reshape([length, heads, dimension / heads])
            .swap_dims(0, 1);
        let v = self
            .v
            .forward(normalized)
            .reshape([length, heads, dimension / heads])
            .swap_dims(0, 1);
        let scores = stable_matmul(q, k.swap_dims(1, 2)) / ((dimension / heads) as f64).sqrt();
        let attention = stable_matmul(softmax(scores, 2), v)
            .swap_dims(0, 1)
            .reshape([length, dimension]);
        let x = x + self.out.forward(attention);
        let hidden = self
            .down
            .forward(gelu(self.up.forward(self.ffn_norm.forward(x.clone()))));
        x + hidden
    }
}

struct Encoder<B: crate::kernels::MatmulBackend> {
    config: AudioConfig,
    conv_weights: Vec<Tensor<B, 4>>,
    conv_biases: Vec<Tensor<B, 1>>,
    conv_out: Linear<B>,
    layers: Vec<EncoderLayer<B>>,
    norm: LayerNorm<B>,
    proj1: Linear<B>,
    proj2: Linear<B>,
}
impl<B: crate::kernels::MatmulBackend> Encoder<B> {
    fn load(
        reader: &mut WeightReader,
        config: &AudioConfig,
        device: &B::Device,
        progress: &mut impl FnMut(&str),
        cancel: &AtomicBool,
    ) -> Result<Self> {
        let p = "thinker.audio_tower";
        let channels = config.downsample_hidden_size;
        let mut conv_weights = Vec::new();
        let mut conv_biases = Vec::new();
        for i in 1..=3 {
            check_cancel(cancel)?;
            let inputs = if i == 1 { 1 } else { channels };
            conv_weights.push(tensor(
                reader.read(&format!("{p}.conv2d{i}.weight"), &[channels, inputs, 3, 3])?,
                [channels, inputs, 3, 3],
                device,
            ));
            conv_biases.push(tensor(
                reader.read(&format!("{p}.conv2d{i}.bias"), &[channels])?,
                [channels],
                device,
            ));
        }
        let conv_out = Linear::load(
            reader,
            &format!("{p}.conv_out"),
            channels * config.num_mel_bins.div_ceil(8),
            config.d_model,
            false,
            device,
        )?;
        let mut layers = Vec::new();
        for i in 0..config.encoder_layers {
            check_cancel(cancel)?;
            progress(&format!(
                "Loading encoder {}/{}",
                i + 1,
                config.encoder_layers
            ));
            layers.push(EncoderLayer::load(reader, i, config, device)?);
        }
        Ok(Self {
            config: config.clone(),
            conv_weights,
            conv_biases,
            conv_out,
            layers,
            norm: LayerNorm::load(reader, &format!("{p}.ln_post"), config.d_model, device)?,
            proj1: Linear::load(
                reader,
                &format!("{p}.proj1"),
                config.d_model,
                config.d_model,
                true,
                device,
            )?,
            proj2: Linear::load(
                reader,
                &format!("{p}.proj2"),
                config.d_model,
                config.output_dim,
                true,
                device,
            )?,
        })
    }

    fn forward(
        &self,
        mel: &MelFeatures,
        device: &B::Device,
        cancel: &AtomicBool,
        progress: &mut impl FnMut(&str),
    ) -> Result<Tensor<B, 2>> {
        let chunk_size = self.config.n_window * 2;
        let padded_width = mel.frames.min(chunk_size);
        let tokens_per_chunk = padded_width.div_ceil(8);
        let mut chunks = Vec::new();
        let d = self.config.d_model;
        let mut positions = vec![0.0; tokens_per_chunk * d];
        for t in 0..tokens_per_chunk {
            for i in 0..d / 2 {
                let phase = t as f64 * (-10000_f64.ln() * i as f64 / (d / 2 - 1) as f64).exp();
                positions[t * d + i] = phase.sin() as f32;
                positions[t * d + d / 2 + i] = phase.cos() as f32;
            }
        }
        let positions = tensor(positions, [tokens_per_chunk, d], device);
        for start in (0..mel.frames).step_by(chunk_size) {
            check_cancel(cancel)?;
            let valid = (mel.frames - start).min(chunk_size);
            let mut values = vec![0.0; self.config.num_mel_bins * padded_width];
            for m in 0..self.config.num_mel_bins {
                values[m * padded_width..m * padded_width + valid].copy_from_slice(
                    &mel.data[m * mel.frames + start..m * mel.frames + start + valid],
                );
            }
            let mut x = tensor(
                values,
                [1, 1, self.config.num_mel_bins, padded_width],
                device,
            );
            for i in 0..3 {
                if self.conv_weights[i].dtype() == DType::F16 {
                    x = x.cast(FloatDType::F16);
                }
                x = gelu(
                    conv2d(
                        x,
                        self.conv_weights[i].clone(),
                        Some(self.conv_biases[i].clone()),
                        ConvOptions::new([2, 2], [1, 1], [1, 1], 1),
                    )
                    .cast(FloatDType::F32),
                );
            }
            let [_, c, f, t] = x.dims();
            let x = self
                .conv_out
                .forward(x.permute([0, 3, 1, 2]).reshape([t, c * f]))
                + positions.clone();
            chunks.push(x.slice([0..valid.div_ceil(8), 0..d]));
        }
        let mut x = Tensor::cat(chunks, 0);
        let length = x.dims()[0];
        let window = tokens_per_chunk * (self.config.n_window_infer / chunk_size);
        for (i, layer) in self.layers.iter().enumerate() {
            check_cancel(cancel)?;
            progress(&format!("Encoding audio {}/{}", i + 1, self.layers.len()));
            let windows = (0..length)
                .step_by(window)
                .map(|start| {
                    layer.forward(
                        x.clone().slice([start..(start + window).min(length), 0..d]),
                        self.config.encoder_attention_heads,
                    )
                })
                .collect();
            x = Tensor::cat(windows, 0);
        }
        Ok(self
            .proj2
            .forward(gelu(self.proj1.forward(self.norm.forward(x)))))
    }
}

#[derive(Clone)]
struct DecoderLayer<B: crate::kernels::MatmulBackend> {
    attention_norm: Tensor<B, 1>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    q_norm: Tensor<B, 1>,
    k_norm: Tensor<B, 1>,
    ffn_norm: Tensor<B, 1>,
    gate: Linear<B>,
    up: Linear<B>,
    down: Linear<B>,
}
impl<B: crate::kernels::MatmulBackend> DecoderLayer<B> {
    fn load(
        reader: &mut WeightReader,
        index: usize,
        config: &TextConfig,
        device: &B::Device,
    ) -> Result<Self> {
        let p = format!("thinker.model.layers.{index}");
        let d = config.hidden_size;
        let norm = |reader: &mut WeightReader, name: &str, size: usize| -> Result<Tensor<B, 1>> {
            Ok(tensor(
                reader.read(&format!("{p}.{name}.weight"), &[size])?,
                [size],
                device,
            ))
        };
        Ok(Self {
            attention_norm: norm(reader, "input_layernorm", d)?,
            q: Linear::load(
                reader,
                &format!("{p}.self_attn.q_proj"),
                d,
                config.num_attention_heads * config.head_dim,
                false,
                device,
            )?,
            k: Linear::load(
                reader,
                &format!("{p}.self_attn.k_proj"),
                d,
                config.num_key_value_heads * config.head_dim,
                false,
                device,
            )?,
            v: Linear::load(
                reader,
                &format!("{p}.self_attn.v_proj"),
                d,
                config.num_key_value_heads * config.head_dim,
                false,
                device,
            )?,
            out: Linear::load(
                reader,
                &format!("{p}.self_attn.o_proj"),
                config.num_attention_heads * config.head_dim,
                d,
                false,
                device,
            )?,
            q_norm: norm(reader, "self_attn.q_norm", config.head_dim)?,
            k_norm: norm(reader, "self_attn.k_norm", config.head_dim)?,
            ffn_norm: norm(reader, "post_attention_layernorm", d)?,
            gate: Linear::load(
                reader,
                &format!("{p}.mlp.gate_proj"),
                d,
                config.intermediate_size,
                false,
                device,
            )?,
            up: Linear::load(
                reader,
                &format!("{p}.mlp.up_proj"),
                d,
                config.intermediate_size,
                false,
                device,
            )?,
            down: Linear::load(
                reader,
                &format!("{p}.mlp.down_proj"),
                config.intermediate_size,
                d,
                false,
                device,
            )?,
        })
    }
    fn forward(
        &self,
        x: Tensor<B, 2>,
        cache: &mut Option<KvCache<B>>,
        config: &TextConfig,
        cos: Tensor<B, 3>,
        sin: Tensor<B, 3>,
        mask: Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        let length = x.dims()[0];
        let h = config.num_attention_heads;
        let kv = config.num_key_value_heads;
        let d = config.head_dim;
        let normalized = rms_norm(x.clone(), self.attention_norm.clone(), config.rms_norm_eps);
        let q = rms_norm(
            self.q.forward(normalized.clone()).reshape([length, h, d]),
            self.q_norm.clone(),
            config.rms_norm_eps,
        )
        .swap_dims(0, 1);
        let k = rms_norm(
            self.k.forward(normalized.clone()).reshape([length, kv, d]),
            self.k_norm.clone(),
            config.rms_norm_eps,
        )
        .swap_dims(0, 1);
        let v = self
            .v
            .forward(normalized)
            .reshape([length, kv, d])
            .swap_dims(0, 1);
        let q = rotary(q, cos.clone(), sin.clone());
        let k = rotary(k, cos, sin);
        let (k, v) = match cache.take() {
            Some(previous) => (
                Tensor::cat(vec![previous.k, k], 1),
                Tensor::cat(vec![previous.v, v], 1),
            ),
            None => (k, v),
        };
        *cache = Some(KvCache {
            k: k.clone(),
            v: v.clone(),
        });
        let k = repeat_kv(k, h / kv);
        let v = repeat_kv(v, h / kv);
        let scores = stable_matmul(q, k.swap_dims(1, 2)) / (d as f64).sqrt() + mask;
        let attention = stable_matmul(softmax(scores, 2), v)
            .swap_dims(0, 1)
            .reshape([length, h * d]);
        let x = x + self.out.forward(attention);
        let normalized = rms_norm(x.clone(), self.ffn_norm.clone(), config.rms_norm_eps);
        let ffn = self
            .down
            .forward(silu(self.gate.forward(normalized.clone())) * self.up.forward(normalized));
        x + ffn
    }
}

fn repeat_kv<B: crate::kernels::MatmulBackend>(x: Tensor<B, 3>, repeats: usize) -> Tensor<B, 3> {
    let [heads, length, dimension] = x.dims();
    x.reshape([heads, 1, length, dimension])
        .expand([heads, repeats, length, dimension])
        .reshape([heads * repeats, length, dimension])
}

fn rotary<B: crate::kernels::MatmulBackend>(
    x: Tensor<B, 3>,
    cos: Tensor<B, 3>,
    sin: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let [heads, length, dimension] = x.dims();
    let first = x.clone().slice([0..heads, 0..length, 0..dimension / 2]);
    let second = x
        .clone()
        .slice([0..heads, 0..length, dimension / 2..dimension]);
    x * cos + Tensor::cat(vec![-second, first], 2) * sin
}

struct KvCache<B: crate::kernels::MatmulBackend> {
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
}

pub struct QwenAsr<B: crate::kernels::MatmulBackend> {
    config: ModelConfig,
    device: B::Device,
    encoder: Encoder<B>,
    layers: Vec<DecoderLayer<B>>,
    norm: Tensor<B, 1>,
    embedding: Vec<bf16>,
    head: Vec<Tensor<B, 2>>,
    tokenizer: AsrTokenizer,
}

#[derive(Clone, Debug, Serialize)]
pub struct Transcription {
    pub text: String,
    pub generated_tokens: Vec<u32>,
    pub audio_seconds: f64,
    pub mel_ms: f64,
    pub encoder_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub elapsed_ms: f64,
    pub real_time_factor: f64,
    pub finished: bool,
}

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        bail!("Operation cancelled");
    }
    Ok(())
}
pub fn is_cancelled(error: &anyhow::Error) -> bool {
    error.to_string() == "Operation cancelled"
}

impl<B: crate::kernels::MatmulBackend> QwenAsr<B> {
    /// Keep normalization and attention in FP32, while storing dense/conv weights and computing their projections in FP16.
    pub fn use_fp16(&mut self, mut progress: impl FnMut(&str)) {
        let convert = |linear: &mut Linear<B>| {
            linear.weight = linear.weight.clone().cast(FloatDType::F16);
            let _ = linear
                .weight
                .clone()
                .slice([0..1, 0..1])
                .cast(FloatDType::F32)
                .into_data();
        };
        for (index, layer) in self.encoder.layers.iter_mut().enumerate() {
            progress(&format!(
                "Preparing FP16 encoder {}/{}",
                index + 1,
                self.encoder.config.encoder_layers
            ));
            for linear in [
                &mut layer.q,
                &mut layer.k,
                &mut layer.v,
                &mut layer.out,
                &mut layer.up,
                &mut layer.down,
            ] {
                convert(linear);
            }
        }
        for linear in [
            &mut self.encoder.conv_out,
            &mut self.encoder.proj1,
            &mut self.encoder.proj2,
        ] {
            convert(linear);
        }
        for weight in &mut self.encoder.conv_weights {
            *weight = weight.clone().cast(FloatDType::F16);
            let _ = weight
                .clone()
                .slice([0..1, 0..1, 0..1, 0..1])
                .cast(FloatDType::F32)
                .into_data();
        }
        for bias in &mut self.encoder.conv_biases {
            *bias = bias.clone().cast(FloatDType::F16);
        }
        let layer_count = self.layers.len();
        for (index, layer) in self.layers.iter_mut().enumerate() {
            progress(&format!(
                "Preparing FP16 decoder {}/{}",
                index + 1,
                layer_count
            ));
            for linear in [
                &mut layer.q,
                &mut layer.k,
                &mut layer.v,
                &mut layer.out,
                &mut layer.gate,
                &mut layer.up,
                &mut layer.down,
            ] {
                convert(linear);
            }
        }
        for weight in &mut self.head {
            *weight = weight.clone().cast(FloatDType::F16);
            let _ = weight
                .clone()
                .slice([0..1, 0..1])
                .cast(FloatDType::F32)
                .into_data();
        }
    }

    pub fn load(directory: &Path, device: B::Device, progress: impl FnMut(&str)) -> Result<Self> {
        Self::load_cancellable(directory, device, &AtomicBool::new(false), progress)
    }

    pub fn load_cancellable(
        directory: &Path,
        device: B::Device,
        cancel: &AtomicBool,
        mut progress: impl FnMut(&str),
    ) -> Result<Self> {
        let config = ModelConfig::load(directory)?;
        let tokenizer = AsrTokenizer::load(directory)?;
        let mut reader = WeightReader::open(&directory.join("model.safetensors"))?;
        let a = &config.thinker_config.audio_config;
        let t = &config.thinker_config.text_config;
        check_cancel(cancel)?;
        let encoder = Encoder::load(&mut reader, a, &device, &mut progress, cancel)?;
        let mut layers = Vec::new();
        for i in 0..t.num_hidden_layers {
            check_cancel(cancel)?;
            progress(&format!(
                "Loading decoder {}/{}",
                i + 1,
                t.num_hidden_layers
            ));
            layers.push(DecoderLayer::load(&mut reader, i, t, &device)?);
        }
        let norm = tensor(
            reader.read("thinker.model.norm.weight", &[t.hidden_size])?,
            [t.hidden_size],
            &device,
        );
        progress("Loading tied output embeddings");
        let (metadata, bytes) = reader.raw("thinker.model.embed_tokens.weight")?;
        ensure!(
            metadata.dtype == "BF16" && metadata.shape == [t.vocab_size, t.hidden_size],
            "Invalid embedding table"
        );
        let embedding: Vec<bf16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| bf16::from_bits(u16::from_le_bytes(*v)))
            .collect();
        drop(bytes);
        let mut head = Vec::new();
        // Split the vocabulary projection to support adapters with a 128 MiB storage binding limit.
        for start in (0..t.vocab_size).step_by(4096) {
            check_cancel(cancel)?;
            let count = (t.vocab_size - start).min(4096);
            let mut values = vec![0.0; t.hidden_size * count];
            for i in 0..t.hidden_size {
                for o in 0..count {
                    values[i * count + o] = embedding[(start + o) * t.hidden_size + i].to_f32();
                }
            }
            head.push(tensor(values, [t.hidden_size, count], &device));
        }
        // Force completion of GPU uploads before reporting the model ready.
        let _ = norm.clone().into_data();
        progress("Model ready");
        Ok(Self {
            config,
            device,
            encoder,
            layers,
            norm,
            embedding,
            head,
            tokenizer,
        })
    }

    fn embed(&self, tokens: &[u32]) -> Result<Tensor<B, 2>> {
        let config = &self.config.thinker_config.text_config;
        let mut values = Vec::with_capacity(tokens.len() * config.hidden_size);
        for &token in tokens {
            ensure!(
                (token as usize) < config.vocab_size,
                "Invalid token ID {token}"
            );
            let start = token as usize * config.hidden_size;
            values.extend(
                self.embedding[start..start + config.hidden_size]
                    .iter()
                    .map(|v| v.to_f32()),
            );
        }
        Ok(tensor(
            values,
            [tokens.len(), config.hidden_size],
            &self.device,
        ))
    }

    fn decoder(
        &self,
        mut x: Tensor<B, 2>,
        cache: &mut [Option<KvCache<B>>],
        start: usize,
        cancel: &AtomicBool,
    ) -> Result<Tensor<B, 2>> {
        let config = &self.config.thinker_config.text_config;
        let length = x.dims()[0];
        let dimension = config.head_dim;
        let mut cosine = vec![0.0; length * dimension];
        let mut sine = vec![0.0; length * dimension];
        for pos in 0..length {
            for i in 0..dimension / 2 {
                let phase = (start + pos) as f64
                    * config.rope_theta.powf(-2.0 * i as f64 / dimension as f64);
                for offset in [i, i + dimension / 2] {
                    cosine[pos * dimension + offset] = phase.cos() as f32;
                    sine[pos * dimension + offset] = phase.sin() as f32;
                }
            }
        }
        let cos = tensor(cosine, [1, length, dimension], &self.device);
        let sin = tensor(sine, [1, length, dimension], &self.device);
        let mask = tensor(
            (0..length)
                .flat_map(|row| {
                    (0..start + length)
                        .map(move |column| if column <= start + row { 0.0 } else { -1e30 })
                })
                .collect(),
            [1, length, start + length],
            &self.device,
        );
        for (index, (layer, cache)) in self.layers.iter().zip(cache.iter_mut()).enumerate() {
            check_cancel(cancel)?;
            x = layer.forward(x, cache, config, cos.clone(), sin.clone(), mask.clone());
            trace_tensor(&x, start, index, "hidden")?;
            if let Some(cache) = cache {
                trace_tensor(&cache.k, start, index, "key")?;
                trace_tensor(&cache.v, start, index, "value")?;
            }
        }
        let last = x.slice([length - 1..length, 0..config.hidden_size]);
        Ok(rms_norm(last, self.norm.clone(), config.rms_norm_eps))
    }

    fn next_token(&self, hidden: Tensor<B, 2>) -> Result<u32> {
        let logits = Tensor::cat(
            self.head
                .iter()
                .map(|weight| stable_matmul(hidden.clone(), weight.clone()))
                .collect(),
            1,
        );
        let token = logits
            .argmax(1)
            .into_data()
            .to_vec::<i32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        Ok(token[0] as u32)
    }

    pub fn transcribe(
        &self,
        samples: &[f32],
        language: Language,
        max_tokens: usize,
        cancel: &AtomicBool,
        mut on_text: impl FnMut(&str),
        mut progress: impl FnMut(&str),
    ) -> Result<Transcription> {
        ensure!(
            max_tokens > 0 && max_tokens <= 2048,
            "Token limit must be between 1 and 2048"
        );
        ensure!(
            samples.len() <= 30 * 16_000,
            "Transcribe accepts at most 30 seconds; split longer recordings into segments"
        );
        check_cancel(cancel)?;
        let total_start = Instant::now();
        let start = Instant::now();
        let mel = log_mel(samples)?;
        let mel_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let audio = self
            .encoder
            .forward(&mel, &self.device, cancel, &mut progress)?;
        // A scalar readback closes the asynchronous GPU timing interval while retaining audio activations on-device.
        let _ = audio.clone().slice([0..1, 0..1]).into_data();
        let encoder_ms = start.elapsed().as_secs_f64() * 1000.0;
        check_cancel(cancel)?;
        let start = Instant::now();
        let suffix = self.tokenizer.suffix(language)?;
        let embeddings = Tensor::cat(vec![self.embed(&PREFIX)?, audio, self.embed(&suffix)?], 0);
        let prefill_length = embeddings.dims()[0];
        let mut cache: Vec<Option<KvCache<B>>> = (0..self.layers.len()).map(|_| None).collect();
        progress("Decoding transcript");
        let hidden = self.decoder(embeddings, &mut cache, 0, cancel)?;
        let mut token = self.next_token(hidden)?;
        let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let mut tokens = Vec::new();
        let mut text_tokens = Vec::new();
        let mut past_header = language != Language::Auto;
        let mut finished = false;
        for offset in 0..max_tokens {
            check_cancel(cancel)?;
            if STOP_TOKENS.contains(&token) {
                finished = true;
                break;
            }
            tokens.push(token);
            if token == ASR_TEXT {
                past_header = true;
            } else if past_header {
                text_tokens.push(token);
                on_text(&self.tokenizer.decode(&text_tokens)?);
            }
            let hidden = self.decoder(
                self.embed(&[token])?,
                &mut cache,
                prefill_length + offset,
                cancel,
            )?;
            token = self.next_token(hidden)?;
        }
        finished |= STOP_TOKENS.contains(&token);
        let decode_ms = start.elapsed().as_secs_f64() * 1000.0;
        let elapsed_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        let audio_seconds = samples.len() as f64 / 16_000.0;
        let text = self.tokenizer.decode(&text_tokens)?.trim().to_owned();
        Ok(Transcription {
            text,
            generated_tokens: tokens,
            audio_seconds,
            mel_ms,
            encoder_ms,
            prefill_ms,
            decode_ms,
            elapsed_ms,
            real_time_factor: elapsed_ms / (audio_seconds * 1000.0),
            finished,
        })
    }

    pub fn encode_audio(&self, samples: &[f32], cancel: &AtomicBool) -> Result<Vec<f32>> {
        let mel = log_mel(samples)?;
        self.encoder
            .forward(&mel, &self.device, cancel, &mut |_| {})?
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))
    }
}

/// The single-row CubeCL 0.10 vector kernels produce wrong results on the verified Intel RPL-P adapter.
/// A duplicate row selects the verified matrix path. Slice away the duplicate before attention/cache updates.
pub fn stable_matmul<B: crate::kernels::MatmulBackend, const D: usize>(
    mut lhs: Tensor<B, D>,
    rhs: Tensor<B, D>,
) -> Tensor<B, D> {
    let half = rhs.dtype() == DType::F16;
    if half {
        lhs = lhs.cast(FloatDType::F16);
    }
    let output = if lhs.dims()[D - 2] == 1
        && std::env::var_os("ASR_GEMV_MODE").as_deref() != Some(std::ffi::OsStr::new("padded"))
    {
        B::narrow_matmul(lhs, rhs)
    } else if lhs.dims()[D - 2] != 1 {
        lhs.matmul(rhs)
    } else {
        let output = Tensor::cat(vec![lhs.clone(), lhs], D - 2).matmul(rhs);
        let dimensions = output.dims();
        let mut ranges: [std::ops::Range<usize>; D] = std::array::from_fn(|i| 0..dimensions[i]);
        ranges[D - 2] = 0..1;
        output.slice(ranges)
    };
    if half {
        output.cast(FloatDType::F32)
    } else {
        output
    }
}

fn trace_tensor<B: crate::kernels::MatmulBackend, const D: usize>(
    x: &Tensor<B, D>,
    position: usize,
    layer: usize,
    name: &str,
) -> Result<()> {
    if let Some(directory) = std::env::var_os("ASR_TRACE_DIR")
        && position < 80
    {
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory)?;
        let values = x
            .clone()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        std::fs::write(
            directory.join(format!("{position}-layer-{layer}-{name}.f32")),
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    type Cpu = NdArray<f32>;
    #[test]
    fn grouped_attention_repeats_each_head_in_order() {
        let x = tensor::<Cpu, 3>(vec![1., 2., 3., 4.], [2, 1, 2], &Default::default());
        let data = repeat_kv(x, 2).into_data().to_vec::<f32>().unwrap();
        assert_eq!(data, [1., 2., 1., 2., 3., 4., 3., 4.]);
    }
    #[test]
    fn rotary_uses_split_halves() {
        let device = Default::default();
        let x = tensor::<Cpu, 3>(vec![1., 2., 3., 4.], [1, 1, 4], &device);
        let cos = tensor(vec![0.; 4], [1, 1, 4], &device);
        let sin = tensor(vec![1.; 4], [1, 1, 4], &device);
        assert_eq!(
            rotary(x, cos, sin).into_data().to_vec::<f32>().unwrap(),
            [-3., -4., 1., 2.]
        );
    }
}
