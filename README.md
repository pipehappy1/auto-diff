# A simple machine learning toolset

[![crates.io version](https://img.shields.io/crates/v/auto-diff.svg)](https://crates.io/crates/auto-diff)
[![License](https://img.shields.io/crates/l/auto-diff.svg)](https://github.com/pipehappy1/auto-diff/blob/master/LICENSE.txt)

## Introduction

## Install

## Example

    use auto_diff::tensor::Tensor;
    use auto_diff::rand::RNG;
    use auto_diff::op::{Linear, Op};
    use auto_diff::var::{Module, mseloss};
    use auto_diff::optim::{SGD, Optimizer};
    
    fn main() {
    
        fn func(input: &Tensor) -> Tensor {
            input.matmul(&Tensor::from_vec_f32(&vec![2., 3.], &vec![2, 1]))
        }
    
        let N = 10;
        let mut m = Module::new();
        let mut rng = RNG::new();
        rng.set_seed(123);
        let x = rng.normal(&vec![N, 2], 0., 2.);
    
        let y = func(&x);
        let op = Linear::new(Some(2), Some(1), true);
        rng.normal_(op.weight(), 0., 1.);
        rng.normal_(op.bias(), 0., 1.);
    
        let linear = Op::new(Box::new(op));
    
        let input = m.var();
        let output = input.to(&linear);
        let label = m.var();
    
        let loss = mseloss(&output, &label);
    
        input.set(x);
        label.set(y);
    
        let mut opt = SGD::new(0.2);
    
        for i in 0..100 {
            m.forward();
            m.backward(-1.);
    
            println!("{}", loss.get().get_scale_f32());
    
            opt.step(&m);
        }
    
    }

## Dependence

install gfortran is openblas-src = "0.9" is used.


