use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("./gpu_kernel")
        .copy_to("./kernels/gpu_kernel.ptx")
        .build()
        .unwrap();
}
