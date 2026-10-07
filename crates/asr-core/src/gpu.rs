use anyhow::{Result, bail};
use burn::tensor::{Tensor, TensorData};
use burn_wgpu::{Wgpu, WgpuDevice, graphics::Vulkan, init_setup};
use serde::Serialize;

pub type GpuBackend = Wgpu<f32, i32>;

#[derive(Clone, Debug, Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub driver: String,
    pub max_storage_buffer_bytes: u64,
    pub supports_fp16: bool,
}

pub fn initialize(device: &WgpuDevice) -> Result<GpuInfo> {
    let setup = init_setup::<Vulkan>(device, Default::default());
    let info = setup.adapter.get_info();
    if format!("{:?}", info.device_type) == "Cpu" {
        bail!("A hardware GPU is required for the GPU inference backend");
    }
    Ok(GpuInfo {
        name: info.name,
        backend: format!("{:?}", info.backend),
        device_type: format!("{:?}", info.device_type),
        driver: format!("{} {}", info.driver, info.driver_info),
        max_storage_buffer_bytes: setup.adapter.limits().max_storage_buffer_binding_size,
        supports_fp16: format!("{:?}", setup.adapter.features()).contains("SHADER_F16"),
    })
}

/// Execute and read back real GPU operations before loading any model weights.
pub fn smoke(device: &WgpuDevice) -> Result<serde_json::Value> {
    let info = initialize(device)?;
    let a = Tensor::<GpuBackend, 2>::from_data(
        TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]),
        device,
    );
    let b = Tensor::<GpuBackend, 2>::from_data(
        TensorData::new(vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2]),
        device,
    );
    let data = a.matmul(b).into_data();
    let actual = data.to_vec::<f32>().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let expected = [58.0, 64.0, 139.0, 154.0];
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    if error > 1e-4 {
        bail!("GPU matmul readback failed: {actual:?}, max error {error}");
    }
    let mut single_row_error = 0.0_f64;
    for output in [896, 1024, 2048, 3072, 4096] {
        let input = 1024;
        let values: Vec<f32> = (0..input)
            .map(|i| ((i * 7 % 29) as f32 - 14.0) / 17.0)
            .collect();
        let weights: Vec<f32> = (0..input * output)
            .map(|i| ((i * 11 % 31) as f32 - 15.0) / 23.0)
            .collect();
        let reference: Vec<f64> = (0..output)
            .map(|o| {
                (0..input)
                    .map(|i| f64::from(values[i]) * f64::from(weights[i * output + o]))
                    .sum()
            })
            .collect();
        let x = Tensor::<GpuBackend, 2>::from_data(TensorData::new(values, [1, input]), device);
        let w =
            Tensor::<GpuBackend, 2>::from_data(TensorData::new(weights, [input, output]), device);
        let actual = crate::model::stable_matmul(x, w)
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        single_row_error = actual
            .iter()
            .zip(reference)
            .map(|(a, b)| (f64::from(*a) - b).abs())
            .fold(single_row_error, f64::max);
    }
    if single_row_error > 1e-4 {
        bail!("Single-row projection regression: {single_row_error}");
    }
    Ok(
        serde_json::json!({"gpu": info, "matmul": actual, "max_abs_error": error, "single_row_projection_max_abs_error": single_row_error}),
    )
}
