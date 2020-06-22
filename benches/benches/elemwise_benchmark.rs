use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};
use std::iter;

use tensor_rs::tensor::gen_tensor::*;

extern crate ndarray;
extern crate ndarray_linalg;
extern crate openblas_src; // or another backend of your choice

//use ndarray;

fn elemwise_benchmark(c: &mut Criterion) {
    let ss = vec![10, 20, 30, 50, 70, 128];

    let mut group = c.benchmark_group("elemwise");
    for size in ss.iter() {
        let m1 = GenTensor::<f64>::fill(1., &vec![*size, *size]);
        let m2 = GenTensor::<f64>::fill(2., &vec![*size, *size]);
        group.bench_with_input(BenchmarkId::new("naive", size*size), size, |b, &size| {
            b.iter(|| {
                let tmp = m1.sub(&m2);
                let tmp2 = tmp.mul(&tmp);
            });
        });
        group.bench_with_input(BenchmarkId::new("local", size*size), size, |b, &size| {
            b.iter(|| {
                let m_result = GenTensor::<f64>::squared_error(&m1, &m2);
            });
        });
        let md1 = &ndarray::Array2::<f64>::zeros(((*size) as usize, (*size) as usize));
        let md2 = &ndarray::Array2::<f64>::ones(((*size) as usize, (*size) as usize));
        group.bench_with_input(BenchmarkId::new("ndarray", size*size), size, |b, &size| {
            b.iter(|| {
                let tmp = md1 - md2;
                let tmp2 = &tmp*&tmp;
            });
        });
    }
    group.finish();

}

criterion_group!(benches, elemwise_benchmark);
criterion_main!(benches);
