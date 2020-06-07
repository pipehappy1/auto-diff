use crate::tensor::gen_tensor::GenTensor;

pub trait ReduceTensor {
    type TensorType;

    fn argmax();
    fn argmin();
    fn dist();
    fn logsumexp(&self, dim: Option<&[usize]>, keep_dim: bool) -> Self::TensorType;
    fn mean(&self, dim: Option<&[usize]>, keepdim: bool) -> Self::TensorType;
    fn median();
    fn mode();
    fn norm();
    fn prod(&self, dim: Option<&[usize]>, keepdim: bool) -> Self::TensorType;
    fn std(&self, dim: Option<&[usize]>, keepdim: bool) -> Self::TensorType;
    fn std_mean();
    //fn sum(&self, dim: usize, keepdim: bool) -> Self::TensorType;
    fn sum(&self, dim: Option<&[usize]>, keepdim: bool) -> Self::TensorType;
    fn unique();
    fn unique_consecutive();
    fn var(&self, dim: Option<&[usize]>, keepdim: bool) -> Self::TensorType;
    fn var_mean();

    fn max(&self, dim: Option<&[usize]>, keepdim: bool) -> Self::TensorType;
    fn min(&self, dim: Option<&[usize]>, keepdim: bool) -> Self::TensorType;
}

impl<T> ReduceTensor for GenTensor<T> where T: num_traits::Float {
    type TensorType = GenTensor<T>;

    fn argmax() {unimplemented!();}
    fn argmin() {unimplemented!();}
    fn dist() {unimplemented!();}
    fn logsumexp(&self, dim: Option<&[usize]>, keep_dim: bool) -> Self::TensorType {
        self._iter_patch(dim, keep_dim,
                         |x| {
                             let mut max = x[0];
                             for i in x {
                                 if max < *i {
                                     max = *i;
                                 }
                             }

                             let mut sum = T::zero();
                             for i in x {
                                 sum = sum + (*i - max).exp();
                             }
                             max + sum.ln()
                         }
        )
    }
    /// Returns the mean value of the tensor along dim row.
    fn mean(&self, dim: Option<&[usize]>, keep_dim: bool) -> GenTensor<T> {
        self._iter_patch(dim, keep_dim,
                         |x| {
                             let n = x.len();
                             let mut sum = T::zero();
                             for i in x {
                                 sum = sum + *i;
                             }
                             sum / T::from(n).expect("")
                         }
        )
    }
    fn median(){unimplemented!();}
    fn mode() {unimplemented!();}
    fn norm() {unimplemented!();}
    fn prod(&self, dim: Option<&[usize]>, keep_dim: bool) -> GenTensor<T> {
        self._iter_patch(dim, keep_dim,
                         |x| {
                             let mut p = T::one();
                             for i in x {
                                 p = p * (*i);
                             }
                             p
                         }
        )
    }
    fn std(&self, dim: Option<&[usize]>, keep_dim: bool) -> GenTensor<T> {
        self._iter_patch(dim, keep_dim,
                         |x| {
                             let n = x.len();
                             let mut sum = T::zero();
                             let mut sum2 = T::zero();
                             for i in x {
                                 sum = sum + *i;
                                 sum2 = sum2 + *i*(*i);
                             }
                             let sum2 = sum2 / T::from(n).expect("");
                             let sum = sum / T::from(n).expect("");
                             (sum2 - sum*sum).sqrt()
                         }
        )
    }
    fn std_mean() {unimplemented!();}
    //fn sum(&self, dim: usize, keepdim: bool) -> Self::TensorType {}
    /// Returns the sum of all elements.
    /// ```
    /// # use auto_diff::tensor::gen_tensor::*;
    /// # use crate::auto_diff::tensor::reduction::ReduceTensor;
    /// let m1 = GenTensor::<f64>::new_raw(&vec![1.,2.,3.,4.,], &vec![2,2]);
    /// assert_eq!(m1.sum(None, false).get_scale(), 10.);
    /// ```
    fn sum(&self, dim: Option<&[usize]>, keep_dim: bool) -> GenTensor<T> {
        self._iter_patch(dim, keep_dim,
                         |x| {
                             let mut sum = T::zero();
                             for i in x {
                                 sum = sum + *i;
                             }
                             sum
                         }
        )
    }
    fn unique(){unimplemented!();}
    fn unique_consecutive() {unimplemented!();}
    fn var(&self, dim: Option<&[usize]>, keep_dim: bool) -> GenTensor<T> {
        self._iter_patch(dim, keep_dim,
                         |x| {
                             let n = x.len();
                             let mut sum = T::zero();
                             let mut sum2 = T::zero();
                             for i in x {
                                 sum = sum + *i;
                                 sum2 = sum2 + *i*(*i);
                             }
                             let sum2 = sum2 / T::from(n).expect("");
                             let sum = sum / T::from(n).expect("");
                             sum2 - sum*sum
                         }
        )
    }

    fn var_mean() {unimplemented!();}

    fn max(&self, dim: Option<&[usize]>, keep_dim: bool) -> Self::TensorType {
        self._iter_patch(dim, keep_dim,
                         |x| {
                             let mut max = x[0];
                             for i in x {
                                 if max < *i {
                                     max = *i;
                                 }
                             }
                             max
                         }
        )
    }
    fn min(&self, dim: Option<&[usize]>, keep_dim: bool) -> Self::TensorType {
        self._iter_patch(dim, keep_dim,
                         |x| {
                             let mut min = x[0];
                             for i in x {
                                 if min < *i {
                                     min = *i;
                                 }
                             }
                             min
                         }
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::gen_tensor::GenTensor;
    use super::*;

    #[test]
    fn logsumexp() {
        let a = GenTensor::<f32>::new_raw(&vec![1., 2., 3., 4., 5., 6., ], &vec![3, 2]);
        let b = a.logsumexp(Some(&[1]), false);
        assert_eq!(b, GenTensor::<f32>::new_raw(&vec![2.3132617, 4.3132615, 6.3132615], &vec![3]));
    }

    #[test]
    fn mean() {
        let a = GenTensor::<f32>::fill(1., &vec![3, 4, 3]);
        let b = a.mean(Some(&[1]), false);
        assert_eq!(*b.size(), vec![3, 3]);
        assert_eq!(b.numel(), 9);
        //println!("{}", b);
        let c = a.mean(Some(&[1]), true);
        assert_eq!(*c.size(), vec![3, 1, 3]);
        assert_eq!(c.numel(), 9);
        //println!("{}", c);
    }

    #[test]
    fn var() {
        let a = GenTensor::<f32>::new_raw(&vec![1., 2., 3., 4., 5., 6., ], &vec![3, 2]);
        let b = a.var(Some(&[0]), false);
        assert_eq!(*b.size(), vec![2]);
        assert_eq!(b.numel(), 2);
        assert_eq!(b, GenTensor::<f32>::new_raw(&vec![2.666667, 2.666666], &vec![2]));
        //println!("{}", b);
        let c = a.var(Some(&[1]), true);
        assert_eq!(*c.size(), vec![3, 1]);
        assert_eq!(c.numel(), 3);
        assert_eq!(c, GenTensor::<f32>::new_raw(&vec![0.25, 0.25, 0.25], &vec![3, 1]));
        //println!("{}", c);
    }

    #[test]
    fn std() {
        let a = GenTensor::<f32>::new_raw(&vec![1., 2., 3., 4., 5., 6., ], &vec![3, 2]);
        let b = a.std(Some(&[0]), false);
        assert_eq!(*b.size(), vec![2]);
        assert_eq!(b.numel(), 2);
        assert_eq!(b, GenTensor::<f32>::new_raw(&vec![1.6329932, 1.632993], &vec![2]));
        //println!("{}", b);
        let c = a.std(Some(&[1]), true);
        assert_eq!(*c.size(), vec![3, 1]);
        assert_eq!(c.numel(), 3);
        assert_eq!(c, GenTensor::<f32>::new_raw(&vec![0.5, 0.5, 0.5], &vec![3, 1]));
        //println!("{}", c);
    }
}
