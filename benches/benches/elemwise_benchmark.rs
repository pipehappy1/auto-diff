use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};
use std::iter;

use auto_diff::Var;
use tensor_rs::tensor_impl::gen_tensor::GenTensor;
use tensor_rs::tensor::Tensor;

extern crate ndarray;
extern crate ndarray_linalg;
extern crate openblas_src; // or another backend of your choice

//use ndarray;

fn elemwise_benchmark(c: &mut Criterion) {
    let ss = vec![10, 20, 30, 50,];

    let mut group = c.benchmark_group("elemwise");
    for size in ss.iter() {
	
        let m1 = Var::fill_f64(&vec![*size, *size], 1.);
        let m2 = Var::fill_f64(&vec![*size, *size], 2.);
        group.bench_with_input(BenchmarkId::new("var", size*size), size, |b, &size| {
            b.iter(|| {
                let tmp = m1.sub(&m2).unwrap();
                let tmp2 = tmp.mul(&tmp).unwrap();
            });
        });
	
        let m1 = GenTensor::<f64>::fill(1., &vec![*size, *size]);
        let m2 = GenTensor::<f64>::fill(2., &vec![*size, *size]);
        group.bench_with_input(BenchmarkId::new("gentensor", size*size), size, |b, &size| {
            b.iter(|| {
                let m_result = GenTensor::<f64>::squared_error(&m1, &m2);
            });
        });

	let m1 = Tensor::fill_f64(&vec![*size, *size], 1.);
        let m2 = Tensor::fill_f64(&vec![*size, *size], 2.);
        group.bench_with_input(BenchmarkId::new("tensor", size*size), size, |b, &size| {
            b.iter(|| {
                let tmp = m1.sub(&m2);
		let tmp2 = tmp.mul(&tmp);
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
