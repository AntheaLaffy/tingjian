use burn::tensor::{Tensor, TensorData};
use burn_wgpu::WgpuDevice;
use qwen3_asr_core::gpu::GpuBackend;

fn main() {
    let device = WgpuDevice::DefaultDevice;
    qwen3_asr_core::gpu::initialize(&device).unwrap();
    for output in [896, 1024, 2048, 3072, 4096] {
        let input = 1024;
        let a: Vec<f32> = (0..input).map(|i| ((i * 7 % 29) as f32 - 14.0) / 17.0).collect();
        let b: Vec<f32> = (0..input * output).map(|i| ((i * 11 % 31) as f32 - 15.0) / 23.0).collect();
        let expected: Vec<f64> = (0..output).map(|o| (0..input).map(|i| a[i] as f64 * b[i * output + o] as f64).sum()).collect();
        let mut padded = a.clone(); padded.extend(vec![0.0; input]);
        let x = Tensor::<GpuBackend, 2>::from_data(TensorData::new(padded, [2, input]), &device);
        let w = Tensor::<GpuBackend, 2>::from_data(TensorData::new(b, [input, output]), &device);
        let actual = x.matmul(w).into_data().to_vec::<f32>().unwrap();
        let error = actual.iter().zip(expected).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max);
        println!("linear 1x{input}x{output}: max_abs_error={error}");
        assert!(error < 0.0001);
    }
    let (heads, length, dimension) = (16, 74, 128);
    let q: Vec<f32> = (0..heads * dimension).map(|i| ((i * 7 % 29) as f32 - 14.0) / 17.0).collect();
    let k: Vec<f32> = (0..heads * length * dimension).map(|i| ((i * 11 % 31) as f32 - 15.0) / 23.0).collect();
    let expected: Vec<f64> = (0..heads).flat_map(|h| {
        let q = &q; let k = &k;
        (0..length).map(move |t| (0..dimension).map(|d| q[h * dimension + d] as f64 * k[(h * length + t) * dimension + d] as f64).sum())
    }).collect();
    let query = Tensor::<GpuBackend, 3>::from_data(TensorData::new(q, [heads, 1, dimension]), &device);
    let key = Tensor::<GpuBackend, 3>::from_data(TensorData::new(k, [heads, length, dimension]), &device).swap_dims(1, 2);
    let query = Tensor::cat(vec![query.clone(), query], 1);
    let actual = query.matmul(key).slice([0..heads,0..1,0..length]).into_data().to_vec::<f32>().unwrap();
    let error = actual.iter().zip(expected).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max);
    println!("batched cached attention: max_abs_error={error}");
    assert!(error < 0.0001);
}
