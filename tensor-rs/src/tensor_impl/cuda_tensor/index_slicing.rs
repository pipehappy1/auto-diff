#[cfg(feature = "use-cuda")]
use super::cuda_tensor::CudaTensor;
use crate::tensor_trait::index_slicing::IndexSlicing;




/****************/
// Cuda tensor ops
/****************/
#[cfg(feature = "use-cuda")]
impl IndexSlicing for CudaTensor {
    fn cat(&self, tensors: &[&Self], dim: usize) -> Self {
        todo!();
    }
    fn chunk(&self, chunks: usize, dim: usize) -> Vec<Self> {
        todo!();
    }
    fn gather(&self, dim: usize, index: &Self) -> Self {
        todo!();
    }
    fn index_select(&self, dim: usize, index: &Self) -> Self
    {
        todo!();
    }
    // fn masked_select();
    //pub fn narrow() {}
    //pub fn nonzero() {}
    fn reshape(&self, new_shape: &[usize]) -> Self {
        todo!();
    }
    fn split(&self, sections: &[usize], dim: usize) -> Vec<Self> {
        todo!();
    }
    fn squeeze(&self, dim: Option<usize>) -> Self {
        todo!();
    }
    fn stack(tensors: &[&Self], dim: usize) -> Self {
        todo!();
    }
    //pub fn t() {}
    fn take(&self, index: &[usize]) -> Self {
        todo!();
    }
    //pub fn transpose() {}
    //pub fn unbind() {}
    fn permute(&self, dims: &[usize]) -> Self {
        todo!();
    }
    fn unsqueeze(&self, dim: usize) -> Self {
        todo!();
    }
    //pub fn condition() {} // this is pytorch where
    fn conditional_select(&self, x: &Self, y: &Self) -> Self {
        todo!();
    }
    fn repeat(&self, sizes: &[usize]) -> Self {
        todo!();
    }

}
