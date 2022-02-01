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
}
