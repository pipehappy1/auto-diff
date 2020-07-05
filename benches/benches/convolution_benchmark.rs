use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};
use std::iter;

use tensor_rs::tensor::gen_tensor::*;
use tensor_rs::tensor::PaddingMode;
use tensor_rs::tensor::index_slicing::IndexSlicing;
use tensor_rs::tensor::convolution::{Convolution, gemm_conv};

extern crate ndarray;
extern crate ndarray_linalg;
extern crate openblas_src; // or another backend of your choice

//use ndarray;

fn varing_input_size_benchmark(c: &mut Criterion) {
    let ss = [10, 15, 20, 25, 30, 35, 50, 70, 100];

    let mut group = c.benchmark_group("varing_input_size");
    for size in &ss {
        let data = GenTensor::<f32>::fill(1., &vec![*size, *size]).reshape(&[1, 1, *size, *size]);
        let filter = GenTensor::<f32>::arange(9).reshape(&vec![1, 1, 3, 3]);
        let stride = vec![1, 1];
        let padding = vec![1, 1];
        let dilation = vec![1, 1];
        let padding_mode = PaddingMode::Zeros;
        group.bench_with_input(BenchmarkId::new("naive", size*size), size, |b, &size| {
            b.iter(|| {
                let result = data.conv_gen(&filter, &stride, &padding, &dilation, padding_mode);
            });
        });

        group.bench_with_input(BenchmarkId::new("dot_product", size*size), size, |b, &size| {
            b.iter(|| {
                let result = gemm_conv(&data, &filter, &stride, &padding, &dilation, padding_mode);
            });
        });
    }
    group.finish();

}

criterion_group!(benches, varing_input_size_benchmark);
criterion_main!(benches);
