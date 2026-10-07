use anyhow::{Context, Result, bail, ensure};
use half::{bf16, f16};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

#[derive(Clone, Debug, Deserialize)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],
}

/// Reads individual safetensors ranges instead of retaining a second full model in RAM.
pub struct WeightReader {
    file: File,
    tensors: BTreeMap<String, TensorMeta>,
    data_start: u64,
}

impl WeightReader {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file =
            File::open(path).with_context(|| format!("Cannot open {}", path.display()))?;
        let file_len = file.metadata()?.len();
        let mut length = [0; 8];
        file.read_exact(&mut length)?;
        let header_len = u64::from_le_bytes(length);
        ensure!(
            header_len <= 16 * 1024 * 1024 && header_len + 8 <= file_len,
            "Invalid safetensors header length"
        );
        let mut header = vec![0; header_len as usize];
        file.read_exact(&mut header)?;
        let mut values: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header)?;
        values.remove("__metadata__");
        let tensors: BTreeMap<String, TensorMeta> = values
            .into_iter()
            .map(|(name, value)| Ok((name, serde_json::from_value(value)?)))
            .collect::<Result<_>>()?;
        let data_start = header_len + 8;
        let mut ranges = Vec::new();
        for (name, meta) in &tensors {
            let bytes = match meta.dtype.as_str() {
                "F32" => 4,
                "BF16" | "F16" => 2,
                dtype => bail!("Unsupported {dtype} tensor {name}"),
            };
            let elements = meta
                .shape
                .iter()
                .try_fold(1usize, |v, &n| v.checked_mul(n))
                .context("Tensor size overflow")?;
            let [start, end] = meta.data_offsets;
            ensure!(
                end >= start
                    && end <= file_len - data_start
                    && end - start == (elements * bytes) as u64,
                "Invalid safetensors range for {name}"
            );
            ranges.push((start, end));
        }
        ranges.sort_unstable();
        let mut next = 0;
        for (start, end) in ranges {
            ensure!(start == next, "Safetensors ranges overlap or contain gaps");
            next = end;
        }
        ensure!(next == file_len - data_start, "Incomplete safetensors file");
        Ok(Self {
            file,
            tensors,
            data_start,
        })
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self
            .tensors
            .get(name)
            .with_context(|| format!("Missing tensor {name}"))?
            .shape
            .clone())
    }

    pub fn raw(&mut self, name: &str) -> Result<(TensorMeta, Vec<u8>)> {
        let meta = self
            .tensors
            .get(name)
            .with_context(|| format!("Missing tensor {name}"))?
            .clone();
        let [start, end] = meta.data_offsets;
        self.file.seek(SeekFrom::Start(self.data_start + start))?;
        let mut bytes = vec![0; (end - start) as usize];
        self.file.read_exact(&mut bytes)?;
        Ok((meta, bytes))
    }

    pub fn read(&mut self, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
        let (meta, bytes) = self.raw(name)?;
        ensure!(
            meta.shape == expected,
            "Wrong shape for {name}: {:?}, expected {expected:?}",
            meta.shape
        );
        Ok(match meta.dtype.as_str() {
            "F32" => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect(),
            "BF16" => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| bf16::from_bits(u16::from_le_bytes(*v)).to_f32())
                .collect(),
            "F16" => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| f16::from_bits(u16::from_le_bytes(*v)).to_f32())
                .collect(),
            _ => unreachable!(),
        })
    }
}
