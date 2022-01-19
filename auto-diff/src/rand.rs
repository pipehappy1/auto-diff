use rand::prelude::*;
use rand_distr::{Normal, Uniform, Distribution};

use tensor_rs::tensor::*;


pub struct RNG {
    rng: StdRng,
}

impl RNG {
    pub fn new() -> RNG {
        RNG {
            rng: StdRng::seed_from_u64(671),
        }
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }

    pub fn gen_range_usize(&mut self, left: usize, right: usize,
                           vec_size: Option<usize>) -> Vec::<usize> {

        let size = if let Some(val) = vec_size {val} else {1};
        let mut ret = Vec::new();

        let mut index = 0;
        loop {
            ret.push(self.rng.gen_range(left, right));
            index += 1;
            if index >= size {
                break;
            }
        }
        ret
    }
    
    pub fn bernoulli() {}
    pub fn cauchy() {}
    pub fn exponential() {}
    pub fn geometric() {}
    pub fn log_normal() {}
    
    pub fn normal(&mut self, dim: &[usize], mean: f32, std: f32) -> Tensor {
        let elem = dim.iter().product();
        
        let mut dta = Vec::<f32>::with_capacity(elem);
        let normal = Normal::new(mean, std).expect("");
        for _i in 0..elem {
            dta.push(normal.sample(&mut self.rng));
        }
        Tensor::from_vec_f32(&dta, dim)
    }
    
    //pub fn random() {}

    // TODO: will do generics
    //pub fn uniform<F>(dim: &[usize], from: F, to: F) -> Tensor
    //where F: num_traits::Float {
    //    Tensor::new()
    //}
    pub fn uniform(&mut self, dim: &[usize], from: f32, to: f32) -> Tensor {
        let elem: usize = dim.iter().product();

        let mut dta = Vec::<f32>::with_capacity(elem);
        let normal = Uniform::new(from, to);
        for _i in 0..elem {
            dta.push(normal.sample(&mut self.rng));
        }
        Tensor::from_vec_f32(&dta, dim)
    }

    //// in place operation
    pub fn normal_(&mut self, o: &Tensor, mean: f32, std: f32) {
        let t = self.normal(&o.size(), mean, std);
        o.swap(t);
    }

    pub fn uniform_(&mut self, o: &Tensor, from: f32, to: f32) {
        let t = self.uniform(&o.size(), from, to);
        o.swap(t);
    }
}
