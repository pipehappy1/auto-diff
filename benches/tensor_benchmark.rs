use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};
use std::iter;

use auto_diff::tensor::*;

fn single_add_benchmark(c: &mut Criterion) {
    let m1 = GenTensor::<f64>::new_val(1., &vec![10,10]);
    c.bench_function("single add", |b| b.iter(|| {
        let m3 = m1.add(&m1);
    }));
}

fn tensor_dim_benchmark(c: &mut Criterion) {
    let ss = vec![10, 20, 30, 50, 70, 128];

    let mut group = c.benchmark_group("tensor_dim");
    for size in ss.iter() {
        let m1 = GenTensor::<f64>::new_val(1., &vec![*size, *size]);
        group.bench_with_input(BenchmarkId::new("add", size*size), size, |b, &size| {
            b.iter(|| {
                let m_result = m1.add(&m1);
            });
        });
    }
    group.finish();

}

criterion_group!(benches, single_add_benchmark, tensor_dim_benchmark);
criterion_main!(benches);
