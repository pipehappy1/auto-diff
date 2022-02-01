use super::GenTensor;
use crate::tensor_trait::rand::Random;

use rand::prelude::*;
use rand_distr::{Normal, Uniform, Distribution, StandardNormal};

impl<T> Random for GenTensor<T> where T: num_traits::Float + rand_distr::uniform::SampleUniform, StandardNormal: Distribution<T> {
    type TensorType = GenTensor<T>;
    type ElementType = T;

    fn bernoulli() -> Self::TensorType {
        unimplemented!();
    }
    fn cauchy() -> Self::TensorType {
        unimplemented!();
    }
    fn exponential() -> Self::TensorType {
        unimplemented!();
    }
    fn geometric() -> Self::TensorType {
        unimplemented!();
    }
    fn log_normal() -> Self::TensorType {
        unimplemented!();
    }
    fn normal(rng: &mut StdRng,
              dim: &[usize],
              mean: Self::ElementType,
              std: Self::ElementType) -> Self::TensorType {
        let elem = dim.iter().product();
        
        let mut dta = Vec::<Self::ElementType>::with_capacity(elem);
        let normal = Normal::<Self::ElementType>::new(mean, std).expect("");
        for _i in 0..elem {
            dta.push(normal.sample(rng));
        }
        GenTensor::new_raw(&dta, dim)
    }
    fn uniform(rng: &mut StdRng,
               dim: &[usize],
               from: Self::ElementType,
               to: Self::ElementType) -> Self::TensorType {
        let elem: usize = dim.iter().product();

        let mut dta = Vec::<Self::ElementType>::with_capacity(elem);
        let normal = Uniform::<Self::ElementType>::new(from, to);
        for _i in 0..elem {
            dta.push(normal.sample(rng));
        }
        GenTensor::new_raw(&dta, dim)
    }
}
