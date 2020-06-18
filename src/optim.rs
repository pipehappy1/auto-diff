//!
//! Gradient based optimization.
//!
use std::cell::RefCell;
use super::tensor::Tensor;
use super::var::{Func, Module};
use crate::rand;

pub struct MiniBatch {
    rng: RefCell<rand::RNG>,
    size: usize,
}
impl MiniBatch {
    pub fn new(rng: rand::RNG, size: usize) -> MiniBatch {
        MiniBatch {
            rng: RefCell::new(rng),
            size: size,
        }
    }

    pub fn next(&self, data: &Tensor, label: &Tensor) -> (Tensor, Tensor) {
        let sample_size = data.size()[0];
        let sample_size2 = label.size()[0];

        if sample_size != sample_size2 {
            panic!("minibatch needs data and label has the same N {}, {}",
                   sample_size, sample_size2);
        }
        
        let index = self.rng.borrow_mut().gen_range_usize(0, sample_size, Some(self.size));
        //println!("minibatch index: {:?}", index);
        let index_t = Tensor::from_vec_usize(&index, &[index.len()]);

        let mdata = data.index_select(0, &index_t);
        let mlabel = label.index_select(0, &index_t);
        (mdata, mlabel)
    }
}

pub trait Optimizer {
    fn step(&mut self, m: &Module);
    fn step2(&mut self, m: &Func);
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

    fn step2(&mut self, m: &Func) {
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


#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;
    use crate::rand::RNG;
    use super::*;

    #[test]
    fn mini_batch() {
        let data = Tensor::ones(&[10, 3]);
        let label = Tensor::zeros(&[10]);
        
        let rng = RNG::new();
        let minibatch = MiniBatch::new(rng, 4);
        let (mdata, mlabel) = minibatch.next(&data, &label);

        assert_eq!(mdata.size(), [4, 3]);
        assert_eq!(mlabel.size(), [4]);
        println!("{:?}, {:?}", mdata, mlabel);
    }
}
