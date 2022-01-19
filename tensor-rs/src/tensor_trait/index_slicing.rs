use std::marker::Sized;

pub trait IndexSlicing where Self: Sized {

    fn cat(&self, tensors: &[&Self], dim: usize) -> Self;
    fn chunk(&self, chunks: usize, dim: usize) -> Vec<Self>;
    fn gather(&self, dim: usize, index: &Self) -> Self;
    fn index_select(&self, dim: usize, index: &Self) -> Self;
    // fn masked_select();
    //pub fn narrow() {}
    //pub fn nonzero() {}
    fn reshape(&self, new_shape: &[usize]) -> Self;
    fn split(&self, sections: &[usize], dim: usize) -> Vec<Self>;
    fn squeeze(&self, dim: Option<usize>) -> Self;
    fn stack(tensors: &[&Self], dim: usize) -> Self;
    fn t(&self) -> Self;
    fn take(&self, index: &[usize]) -> Self;
    //pub fn transpose() {}
    //pub fn unbind() {}
    fn permute(&self, dims: &[usize]) -> Self;
    fn unsqueeze(&self, dim: usize) -> Self;
    //pub fn condition() {} // this is pytorch where
    fn conditional_select(&self, x: &Self, y: &Self) -> Self;

    fn repeat(&self, sizes: &[usize]) -> Self;
}
