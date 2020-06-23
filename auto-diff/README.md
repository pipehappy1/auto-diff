# A simple machine learning toolset

[![crates.io version](https://img.shields.io/crates/v/auto-diff.svg)](https://crates.io/crates/auto-diff)
[![License](https://img.shields.io/crates/l/auto-diff.svg)](https://github.com/pipehappy1/auto-diff/blob/master/LICENSE.txt)

## Introduction

This is an auto-difference based learning library.

## Features

- A type less tensor.
- Variable over tensor with support for back propagation.
- Support for common operators, including convolution.

## Example

```rust,no_run
use tensor_rs::tensor::Tensor;
use auto_diff::rand::RNG;
use auto_diff::var::{Module};
use auto_diff::optim::{SGD, Optimizer};

fn main() {

    fn func(input: &Tensor) -> Tensor {
        input.matmul(&Tensor::from_vec_f32(&vec![2., 3.], &vec![2, 1])).add(&Tensor::from_vec_f32(&vec![1.], &vec![1]))
    }

    let N = 100;
    let mut rng = RNG::new();
    rng.set_seed(123);
    let data = rng.normal(&vec![N, 2], 0., 2.);
    let label = func(&data);


    let mut m = Module::new();
    
    let op1 = m.linear(Some(2), Some(1), true);
    let weights = op1.get_values().unwrap();
    rng.normal_(&weights[0], 0., 1.);
    rng.normal_(&weights[1], 0., 1.);
    op1.set_values(&weights);

    let op2 = op1.clone();
    let block = m.func(
        move |x| {
            op2.call(x)
        }
    );
    
    let loss_func = m.mse_loss();
    
    let mut opt = SGD::new(3.);

    for i in 0..200 {
        let input = m.var_value(data.clone());
        
        let y = block.call(&[&input]);
        
        let loss = loss_func.call(&[&y, &m.var_value(label.clone())]);
        println!("index: {}, loss: {}", i, loss.get().get_scale_f32());
        
        loss.backward(-1.);
        opt.step2(&block);

    }

    let weights = op1.get_values().expect("");
    println!("{:?}, {:?}", weights[0], weights[1]);
}
```


## Dependence

install gfortran is openblas-src = "0.9" is used.


