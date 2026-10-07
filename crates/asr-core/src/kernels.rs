use crate::gpu::GpuBackend;
use burn::tensor::{DType, Tensor as BurnTensor, backend::Backend};
use burn_wgpu::{CubeBackend, WgpuRuntime};
use cubecl::prelude::*;

type BaseBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

/// Backend extension for one-row projections. CPU helpers remain useful as independent small-shape references.
pub trait MatmulBackend: Backend {
    fn narrow_matmul<const D: usize>(
        lhs: BurnTensor<Self, D>,
        rhs: BurnTensor<Self, D>,
    ) -> BurnTensor<Self, D>;
}

impl MatmulBackend for burn::backend::NdArray<f32> {
    fn narrow_matmul<const D: usize>(
        lhs: BurnTensor<Self, D>,
        rhs: BurnTensor<Self, D>,
    ) -> BurnTensor<Self, D> {
        lhs.matmul(rhs)
    }
}

impl MatmulBackend for GpuBackend {
    fn narrow_matmul<const D: usize>(
        lhs: BurnTensor<Self, D>,
        rhs: BurnTensor<Self, D>,
    ) -> BurnTensor<Self, D> {
        gemv(lhs, rhs)
    }
}

// Each lane computes one output column. Adjacent lanes read adjacent columns in the dense decoder matrices.
// Accumulation stays in FP32 for both FP32 and FP16 storage; this kernel does not depend on subgroup shuffle behavior.
#[cube(launch)]
fn gemv_kernel<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    output: &mut Tensor<f32>,
    inner: usize,
    columns: usize,
    #[comptime] batched: bool,
) {
    let lane = usize::cast_from(UNIT_POS);
    let output_lane = lane % 32usize;
    let part_lane = lane / 32usize;
    let index = usize::cast_from(CUBE_POS_X) * 32usize + output_lane;
    let column = index % columns;
    let batch = index / columns;
    let lhs_last = if batched { 2usize } else { 1usize };
    let rhs_inner = if batched { 1usize } else { 0usize };
    let rhs_last = if batched { 2usize } else { 1usize };
    let mut lhs_base = 0usize;
    let mut rhs_base = 0usize;
    if batched {
        lhs_base = batch * lhs.stride(0);
        rhs_base = batch * rhs.stride(0);
    }
    let valid = if batched {
        index < lhs.shape(0) * columns
    } else {
        index < columns
    };
    let mut sum = 0.0f32;
    let mut k = part_lane;
    while k < inner && valid {
        let a = f32::cast_from(lhs[lhs_base + k * lhs.stride(lhs_last)]);
        let b = f32::cast_from(
            rhs[rhs_base + k * rhs.stride(rhs_inner) + column * rhs.stride(rhs_last)],
        );
        sum += a * b;
        k += 8usize;
    }
    let mut partial = SharedMemory::<f32>::new(256usize);
    partial[lane] = sum;
    sync_cube();
    if part_lane == 0usize && valid {
        let mut total = 0.0f32;
        for part in 0..8usize {
            total += partial[output_lane + part * 32usize];
        }
        output[index] = total;
    }
}

pub fn gemv<const D: usize>(
    lhs: BurnTensor<GpuBackend, D>,
    rhs: BurnTensor<GpuBackend, D>,
) -> BurnTensor<GpuBackend, D> {
    assert!(
        D == 2 || D == 3,
        "Native GEMV supports matrix and batched matrix inputs"
    );
    assert_eq!(lhs.dims()[D - 2], 1);
    let inner = lhs.dims()[D - 1];
    let columns = rhs.dims()[D - 1];
    let mut shape = lhs.dims();
    shape[D - 1] = columns;
    let device = lhs.device();
    let output = BurnTensor::<GpuBackend, D>::empty(shape, (&device, DType::F32));
    let left = lhs.into_primitive().tensor();
    let right = rhs.into_primitive().tensor();
    let out = output.clone().into_primitive().tensor();
    let left = left
        .client
        .clone()
        .resolve_tensor_float::<BaseBackend>(left);
    let right = right
        .client
        .clone()
        .resolve_tensor_float::<BaseBackend>(right);
    let out = out.client.clone().resolve_tensor_float::<BaseBackend>(out);
    let count = shape.iter().product::<usize>().div_ceil(32) as u32;
    match left.dtype {
        DType::F32 => gemv_kernel::launch::<f32, WgpuRuntime>(
            &out.client,
            CubeCount::Static(count, 1, 1),
            CubeDim::new_1d(256),
            left.into_tensor_arg(),
            right.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            inner,
            columns,
            D == 3,
        ),
        DType::F16 => gemv_kernel::launch::<half::f16, WgpuRuntime>(
            &out.client,
            CubeCount::Static(count, 1, 1),
            CubeDim::new_1d(256),
            left.into_tensor_arg(),
            right.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            inner,
            columns,
            D == 3,
        ),
        dtype => panic!("Unsupported GEMV input type {dtype:?}"),
    };
    output
}
