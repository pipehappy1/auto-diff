//!
//! Gradient based optimization.
//!

use super::tensor::Tensor;
use super::var::Module;
use crate::rand;

pub struct MiniBatch {
    rng: rand::RNG,
    size: usize,
}
impl MiniBatch {
    pub fn new(rng: rand::RNG, size: usize) -> MiniBatch {
        MiniBatch {
            rng: rng,
            size: size,
        }
    }

    pub fn next(&mut self, data: &Tensor, label: &Tensor) -> (Tensor, Tensor) {
        let sample_size = data.size()[0];
        let sample_size2 = label.size()[0];

        if sample_size != sample_size2 {
            panic!("minibatch needs data and label has the same N {}, {}",
                   sample_size, sample_size2);
        }
        
        let index = self.rng.gen_range_usize(0, sample_size, Some(self.size));
        (Tensor::new(), Tensor::new())
    }
}

pub trait Optimizer {
    fn step(&mut self, m: &Module);
}

// actually it's GD
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
            let weights = x.get_values();
            let grads = x.get_grads();
            // println!("name: {}, {}, {}", x.get_name(), weights.len(), grads.len());

            let mut new_weight = Vec::new();
            for (i, j) in weights.iter().zip(grads.iter()) {
                // println!("{:?}, {:?}, {:?}", i.size(), j.size(), self.lr.size());
                
                new_weight.push(i.add(&j.mul(&self.lr)));
            }
            x.set_values(&new_weight);
        });
    }
}
