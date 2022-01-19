#[cfg(feature = "use-cuda")]
use crate::tensor::cuda_tensor::CudaTensor;
use crate::tensor_trait::reduction::ReduceTensor;









//////////////
// cuda tensor
//////////////
#[cfg(feature = "use-cuda")]
impl ReduceTensor for CudaTensor {

    fn argmax(&self, dim: Option<&[usize]>, keep_dim: bool) -> Self {
        todo!();
    }
    fn argmin(&self, dim: Option<&[usize]>, keep_dim: bool) -> Self {
        todo!();
    }
    fn dist() {
        todo!();
    }
    fn logsumexp(&self, dim: Option<&[usize]>, keep_dim: bool) -> Self {
        todo!();
    }
    fn mean(&self, dim: Option<&[usize]>, keepdim: bool) -> Self {
        todo!();
    }
    fn median() {
        todo!();
    }
    fn mode() {
        todo!();
    }
    fn prod(&self, dim: Option<&[usize]>, keepdim: bool) -> Self {
        todo!();
    }
    fn std(&self, dim: Option<&[usize]>, keepdim: bool) -> Self {
        todo!();
    }
    fn std_mean() {
        todo!();
    }
    //fn sum(&self, dim: usize, keepdim: bool) -> Self::TensorType;
    fn sum(&self, dim: Option<&[usize]>, keepdim: bool) -> Self {
        todo!();
    }
    fn unique() {
        todo!();
    }
    fn unique_consecutive() {
        todo!();
    }
    fn var(&self, dim: Option<&[usize]>, keepdim: bool) -> Self {
        todo!();
    }
    fn var_mean() {
        todo!();
    }

    fn max(&self, dim: Option<&[usize]>, keepdim: bool) -> Self {
        todo!();
    }
    fn min(&self, dim: Option<&[usize]>, keepdim: bool) -> Self {
        todo!();
    }
}
