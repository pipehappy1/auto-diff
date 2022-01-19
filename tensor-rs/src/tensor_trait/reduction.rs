pub trait ReduceTensor where Self: std::marker::Sized {

    fn argmax(&self, dim: Option<&[usize]>, keep_dim: bool) -> Self;
    fn argmin(&self, dim: Option<&[usize]>, keep_dim: bool) -> Self;
    fn dist();
    fn logsumexp(&self, dim: Option<&[usize]>, keep_dim: bool) -> Self;
    fn mean(&self, dim: Option<&[usize]>, keepdim: bool) -> Self;
    fn median();
    fn mode();
    fn prod(&self, dim: Option<&[usize]>, keepdim: bool) -> Self;
    fn std(&self, dim: Option<&[usize]>, keepdim: bool) -> Self;
    fn std_mean();
    //fn sum(&self, dim: usize, keepdim: bool) -> Self::TensorType;
    fn sum(&self, dim: Option<&[usize]>, keepdim: bool) -> Self;
    fn unique();
    fn unique_consecutive();
    fn var(&self, dim: Option<&[usize]>, keepdim: bool) -> Self;
    fn var_mean();

    fn max(&self, dim: Option<&[usize]>, keepdim: bool) -> Self;
    fn min(&self, dim: Option<&[usize]>, keepdim: bool) -> Self;
}
