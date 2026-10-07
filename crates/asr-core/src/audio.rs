use anyhow::{Context, Result, ensure};
use rustfft::{FftPlanner, num_complex::Complex32};
use std::path::Path;

pub const SAMPLE_RATE: u32 = 16_000;
pub const MEL_BINS: usize = 128;
const FFT_SIZE: usize = 400;
const HOP: usize = 160;

#[derive(Clone, Debug)]
pub struct MelFeatures {
    /// Frequency-major [128, frames], matching WhisperFeatureExtractor.
    pub data: Vec<f32>,
    pub frames: usize,
}

pub fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("Cannot read WAV {}", path.display()))?;
    let spec = reader.spec();
    ensure!(
        spec.channels > 0 && spec.sample_rate > 0,
        "Invalid WAV format"
    );
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2_f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .samples::<i32>()
                .map(|v| v.map(|s| s as f32 / scale))
                .collect::<std::result::Result<_, _>>()?
        }
    };
    ensure!(
        samples.iter().all(|s| s.is_finite()),
        "Audio contains non-finite samples"
    );
    let channels = usize::from(spec.channels);
    let mono: Vec<f32> = samples
        .chunks_exact(channels)
        .map(|v| v.iter().sum::<f32>() / channels as f32)
        .collect();
    Ok(resample(&mono, spec.sample_rate, SAMPLE_RATE))
}

pub fn write_wav(path: &Path, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..=24 {
        term *= (x / (2.0 * k as f64)).powi(2);
        sum += term;
    }
    sum
}

/// Windowed sinc with an anti-aliasing cutoff; microphone processing calls this on bounded blocks.
pub fn resample(input: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate {
        return input.to_vec();
    }
    if input.is_empty() || source_rate == 0 || target_rate == 0 {
        return Vec::new();
    }
    let ratio = f64::from(target_rate) / f64::from(source_rate);
    let cutoff = ratio.min(1.0);
    let output_len = (input.len() as f64 * ratio).floor() as usize;
    let beta = 6.0;
    let normalization = bessel_i0(beta);
    (0..output_len)
        .map(|i| {
            let position = i as f64 / ratio;
            let center = position.floor() as isize;
            let mut value = 0.0;
            let mut total = 0.0;
            for j in center - 16..=center + 16 {
                if j < 0 || j >= input.len() as isize {
                    continue;
                }
                let distance = position - j as f64;
                if distance.abs() >= 16.0 {
                    continue;
                }
                let phase = std::f64::consts::PI * distance * cutoff;
                let sinc = if phase.abs() < 1e-12 {
                    1.0
                } else {
                    phase.sin() / phase
                };
                let window =
                    bessel_i0(beta * (1.0 - (distance / 16.0).powi(2)).sqrt()) / normalization;
                let weight = cutoff * sinc * window;
                value += f64::from(input[j as usize]) * weight;
                total += weight;
            }
            if total.abs() > 1e-12 {
                (value / total) as f32
            } else {
                0.0
            }
        })
        .collect()
}

fn hz_to_mel(hz: f64) -> f64 {
    if hz < 1000.0 {
        hz * 3.0 / 200.0
    } else {
        15.0 + (hz / 1000.0).ln() * 27.0 / 6.4_f64.ln()
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    if mel < 15.0 {
        mel * 200.0 / 3.0
    } else {
        1000.0 * ((mel - 15.0) * 6.4_f64.ln() / 27.0).exp()
    }
}

fn reflected_index(index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let period = 2 * (length as isize - 1);
    let index = index.rem_euclid(period);
    if index < length as isize {
        index as usize
    } else {
        (period - index) as usize
    }
}

pub fn log_mel(samples: &[f32]) -> Result<MelFeatures> {
    ensure!(samples.len() >= HOP, "Audio must contain at least 10 ms");
    ensure!(
        samples.iter().all(|s| s.is_finite()),
        "Audio contains non-finite samples"
    );
    let frames = samples.len() / HOP;
    let bins = FFT_SIZE / 2 + 1;
    let edges: Vec<f64> = (0..MEL_BINS + 2)
        .map(|i| mel_to_hz(hz_to_mel(8000.0) * i as f64 / (MEL_BINS + 1) as f64))
        .collect();
    let mut filters = vec![0_f32; MEL_BINS * bins];
    for m in 0..MEL_BINS {
        for k in 0..bins {
            let hz = k as f64 * f64::from(SAMPLE_RATE) / FFT_SIZE as f64;
            let rising = (hz - edges[m]) / (edges[m + 1] - edges[m]);
            let falling = (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]);
            filters[m * bins + k] =
                (rising.min(falling).max(0.0) * 2.0 / (edges[m + 2] - edges[m])) as f32;
        }
    }
    let window: Vec<f32> = (0..FFT_SIZE)
        .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / FFT_SIZE as f64).cos()) as f32)
        .collect();
    let fft = FftPlanner::new().plan_fft_forward(FFT_SIZE);
    let mut buffer = vec![Complex32::default(); FFT_SIZE];
    let mut scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
    let mut power = vec![0.0; bins];
    let mut data = vec![0.0; MEL_BINS * frames];
    for t in 0..frames {
        for i in 0..FFT_SIZE {
            let index = (t * HOP + i) as isize - (FFT_SIZE / 2) as isize;
            buffer[i] = Complex32::new(
                samples[reflected_index(index, samples.len())] * window[i],
                0.0,
            );
        }
        fft.process_with_scratch(&mut buffer, &mut scratch);
        for k in 0..bins {
            power[k] = buffer[k].norm_sqr();
        }
        for m in 0..MEL_BINS {
            let energy: f32 = power
                .iter()
                .zip(&filters[m * bins..(m + 1) * bins])
                .map(|(a, b)| a * b)
                .sum();
            data[m * frames + t] = energy.max(1e-10).log10();
        }
    }
    let floor = data.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 8.0;
    for value in &mut data {
        *value = (value.max(floor) + 4.0) / 4.0;
    }
    Ok(MelFeatures { data, frames })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn whisper_silence_and_frame_count() {
        let mel = log_mel(&vec![0.0; 16_001]).unwrap();
        assert_eq!(mel.frames, 100);
        assert!(mel.data.iter().all(|v| (*v + 1.5).abs() < 1e-6));
    }
    #[test]
    fn resampling_preserves_audible_sine() {
        let input: Vec<f32> = (0..48_000)
            .map(|i| (std::f64::consts::TAU * 440.0 * i as f64 / 48_000.0).sin() as f32)
            .collect();
        let output = resample(&input, 48_000, 16_000);
        assert_eq!(output.len(), 16_000);
        let error = output[32..output.len() - 32]
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                (v - (std::f64::consts::TAU * 440.0 * (i + 32) as f64 / 16_000.0).sin() as f32)
                    .abs()
            })
            .fold(0.0, f32::max);
        assert!(error < 0.003, "{error}");
    }
}
