use std::marker::Sized;

pub trait IndexSlicing where Self: Sized {

    /// Concatenates the given sequence of seq tensors
    /// in the given dimension.
    /// The input tensor should all have the same size except
    /// on the given dimension.
    /// The output tensor will have all the same size as the input
    /// except the given dimension, which will be the sum of
    /// the inputs on the given dimension.
    /// Apply cat on [tensor(5, 3, 2), tensor(5, 7, 2), ]
    /// will get a tensor(5, 10, 2).
    fn cat(&self, tensors: &[Self], dim: usize) -> Self;
    
    /// Splits a tensor into a specific number of chunks.
    fn chunk(&self, chunks: usize, dim: usize) -> Vec<Self>;
    
    /// Pick elements on the given dimension by the index,
    /// and gather them in the output.
    /// A restriction is that self.size() and index.size()
    /// should be the same on other dimensions.
    fn gather(&self, dim: usize, index: &Self) -> Self;
    /// The opposite of gather.
    /// Self will be replaced with value along dim by index.
    fn spread(&self, dim: usize, index: &Self, value: &Self) -> Self;

    /// Select on dim and collect those subtensor by index.
    fn index_select(&self, dim: usize, index: &Self) -> Self;

    /// Inverse of index_select, remove those subtensor by index along dim.
    fn index_exclude(&self, dim: usize, index: &Self) -> Self;
    // fn masked_select();
    //pub fn narrow() {}
    //pub fn nonzero() {}

    /// Just change the index boundary.
    fn reshape(&self, new_shape: &[usize]) -> Self;

    /// Inverse of cat(), split tensor along dim dimension,
    /// the length of each section on dim is specified by sections.
    fn split(&self, sections: &[usize], dim: usize) -> Vec<Self>;

    /// Remove dimension with length of 1.
    fn squeeze(&self, dim: Option<usize>) -> Self;

    /// Stack tensor with the same size along a new dimension
    /// specified by dim.
    /// The difference from cat is that cat don't create new dimension.
    fn stack(&self, tensors: &[Self], dim: usize) -> Self;

    /// Transpose
    fn t(&self) -> Self;

    /// Returns a new tensor with the elements of input at the given indices. 
    /// The input tensor is treated as if it were viewed as a 1-D tensor.
    /// The result takes the same shape as the indices.
    fn take(&self, index: &[usize]) -> Self;
    //pub fn transpose() {}
    //pub fn unbind() {}

    /// 
    fn permute(&self, dims: &[usize]) -> Self;

    /// Add size 1 dimension at dim.
    fn unsqueeze(&self, dim: usize) -> Self;
    //pub fn condition() {} // this is pytorch where

    /// Self is the bool condition, at each position of self,
    /// select from x if self at the position is positive or zero,
    /// Otherwise , use value from y if self at the position is negative.
    /// The restriction is that, self, x, and y all have the same size.
    fn conditional_select(&self, x: &Self, y: &Self) -> Self;
    /// Repeat the tensor along all dimensions,
    /// the number of repeat is specified in sizes.
    /// Thus the restriction is that self.size().len() is
    /// equal to sizes.len().
    fn repeat(&self, sizes: &[usize]) -> Self;
}
