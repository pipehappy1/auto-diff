use rand::prelude::StdRng;

pub trait Random {
    type TensorType;
    type ElementType;

    /// Generate a random int close on left, open on right.
    fn rand_usize(rng: &mut StdRng,
                dim: &[usize],
                left: usize, right: usize) -> Self::TensorType;
    fn bernoulli() -> Self::TensorType;
    fn cauchy() -> Self::TensorType;
    fn exponential() -> Self::TensorType;
    fn geometric() -> Self::TensorType;
    fn log_normal() -> Self::TensorType;
    fn normal(rng: &mut StdRng,
              dim: &[usize],
              mean: Self::ElementType,
              std: Self::ElementType) -> Self::TensorType;
    fn uniform(rng: &mut StdRng,
               dim: &[usize],
               from: Self::ElementType,
               to: Self::ElementType) -> Self::TensorType;
}
