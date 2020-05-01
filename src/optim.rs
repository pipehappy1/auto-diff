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
        m._visit_op(|x| {
            //println!("{}", x.get_name());
            let weights = x.get_values();
            let grads = x.get_grads();
            let mut new_weight = Vec::new();
            for (i, j) in weights.iter().zip(grads.iter()) {
                new_weight.push(i.add(&j.mul(&self.lr)));
            }
            x.set_values(&new_weight);
        });
    }
}
