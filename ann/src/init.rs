use ::rand::prelude::StdRng;

use auto_diff::{Var, AutoDiffError};
use tensor_rs::tensor::Tensor;

pub fn normal(data: &Tensor, mean: Option<Var>, std: Option<Var>, rng: &mut StdRng) -> Result<(), AutoDiffError>{
    let size = data.size();
    let mean = if let Some(v) = mean {f64::try_from(v)?} else {0.};
    let std = if let Some(v) = std {f64::try_from(v)?} else {1.};
    data.swap(&Var::normal(rng, &size, mean, std).val());
    Ok(())
}
