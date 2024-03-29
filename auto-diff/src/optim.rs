//!
//! Gradient based optimization.
//!
use super::compute_graph::Net;
use crate::err::AutoDiffError;
use crate::var::Var;
use rand::prelude::StdRng;
use std::cell::RefCell;
use std::rc::Rc;
use tensor_rs::tensor::Tensor;

/// Create random batch view from a large batch.
pub struct MiniBatch {
    rng: StdRng,
    size: usize,
}
impl MiniBatch {
    pub fn new(rng: StdRng, size: usize) -> MiniBatch {
        MiniBatch { rng, size }
    }

    pub fn next(&mut self, data: &Var, label: &Var) -> Result<(Var, Var), AutoDiffError> {
        let sample_size = data.size()[0];
        let sample_size2 = label.size()[0];

        if sample_size != sample_size2 {
            return Err(AutoDiffError::new(&format!(
                "minibatch needs data and label has the same N {}, {}",
                sample_size, sample_size2
            )));
        }
        let index_t = Var::rand_usize(&mut self.rng, &[self.size], 0, sample_size);

        let mdata = data.index_select(0, index_t.clone())?;
        let mlabel = label.index_select(0, index_t)?;
        mdata.reset_net();
        mlabel.reset_net();
        Ok((mdata, mlabel))
    }
}

pub trait Optimizer {
    fn step(&mut self, net: Rc<RefCell<Net>>);
}

// actually it's GD
pub struct SGD {
    lr: Tensor,
}
impl SGD {
    #[cfg(feature = "use-f64")]
    pub fn new(lr: f64) -> SGD {
        Self::new_f64(lr)
    }
    #[cfg(feature = "use-f32")]
    pub fn new(lr: f32) -> SGD {
        Self::new_f32(lr)
    }

    pub fn new_f64(lr: f64) -> SGD {
        SGD {
            lr: Tensor::from_vec_f64(&[lr], &[1]),
        }
    }
    pub fn new_f32(lr: f32) -> SGD {
        SGD {
            lr: Tensor::from_vec_f32(&[lr], &[1]),
        }
    }
}
impl Optimizer for SGD {
    fn step(&mut self, net: Rc<RefCell<Net>>) {
        net.borrow_mut().visit_op(
            |x| {
                let weights = x.get_values();
                let grads = x.get_grads();
                //println!("name: {}, {}, {}", x.get_name(), weights.len(), grads.len());

                let mut new_weight = Vec::new();
                for (i, j) in weights.iter().zip(grads.iter()) {
                    //println!("{:?}, {:?}, {:?}", i.size(), j.size(), self.lr.size());
                    new_weight.push(i.sub(&j.mul(&self.lr)));
                }
                x.set_values(&new_weight);
            },
            None,
            None,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::var::Var;
    use rand::prelude::*;

    #[test]
    fn mini_batch() {
        let data = Var::ones(&[10, 3]);
        let label = Var::zeros(&[10]);

        let rng = StdRng::seed_from_u64(671);
        let mut minibatch = MiniBatch::new(rng, 4);
        let (mdata, mlabel) = minibatch.next(&data, &label).unwrap();

        assert_eq!(mdata.size(), [4, 3]);
        assert_eq!(mlabel.size(), [4]);
        println!("{:?}, {:?}", mdata, mlabel);
    }
}
