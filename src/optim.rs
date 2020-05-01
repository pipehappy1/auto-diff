//!
//! Gradient based optimization.
//!

use super::tensor::Tensor;
use super::var::Module;

pub trait Optimizer {
    fn step(&mut self, m: &Module);
}

pub struct SGD {
    lr: Tensor,
}
impl SGD {
    pub fn new(lr: f32) -> SGD {
        SGD {
            lr: Tensor::from_vec_f32(&vec![lr], &vec![1]),
        }
    }
}
impl Optimizer for SGD {
    fn step(&mut self, m: &Module) {
        
    }
}
