use crate::tensor::gen_tensor::GenTensor;

pub trait ReduceTensor {
    type TensorType;

    fn argmax();
    fn argmin();
    fn dist();
    fn logsumexp();
    fn mean(&self, dim: usize, keepdim: bool) -> Self::TensorType;
    fn median();
    fn mode();
    fn norm();
    fn prod();
    fn std(&self, dim: usize, keepdim: bool) -> Self::TensorType;
    fn std_mean();
    //fn sum(&self, dim: usize, keepdim: bool) -> Self::TensorType;
    fn sum(&self) -> Self::TensorType;
    fn unique();
    fn unique_consecutive();
    fn var(&self, dim: usize, keepdim: bool) -> Self::TensorType;
    fn var_mean();

}

impl<T> ReduceTensor for GenTensor<T> where T: num_traits::Float {
    type TensorType = GenTensor<T>;

    fn argmax() {unimplemented!();}
    fn argmin() {unimplemented!();}
    fn dist() {unimplemented!();}
    fn logsumexp(){unimplemented!();}
    /// Returns the mean value of the tensor along dim row.
    fn mean(&self, dim: usize, keepdim: bool) -> GenTensor<T> {
        self._dim_statistic(dim, keepdim,
                            |over, k, j, inner_size, step| {
                                let mut sum = T::zero();
                                for i in 0..over {
                                    let index = k*inner_size*over + j +i*step;
                                    //println!("mean: {}", index);
                                    sum = sum + self.get_data()[index];
                                }
                                sum = sum / T::from(over).expect("N");
                                sum
                            })
    }
    fn median(){unimplemented!();}
    fn mode() {unimplemented!();}
    fn norm() {unimplemented!();}
    fn prod() {unimplemented!();}
    fn std(&self, dim: usize, keepdim: bool) -> GenTensor<T> {
        self._dim_statistic(dim, keepdim,
                            |over, k, j, inner_size, step| {
                                let mut sum = T::zero();
                                let mut sum2 = T::zero();
                                for i in 0..over {
                                    let index = k*inner_size*over + j +i*step;
                                    //println!("mean: {}", index);
                                    sum = sum + self.get_data()[index];
                                    sum2 = sum2 + self.get_data()[index]*self.get_data()[index];
                                }
                                sum = sum / T::from(over).expect("N");
                                sum2 = sum2 / T::from(over).expect("N");
                                (sum2 - sum*sum).sqrt()
                            })
    }
    fn std_mean() {unimplemented!();}
    //fn sum(&self, dim: usize, keepdim: bool) -> Self::TensorType {}
    /// Returns the sum of all elements.
    /// ```
    /// # use auto_diff::tensor::gen_tensor::*;
    /// # use crate::auto_diff::tensor::reduction::ReduceTensor;
    /// let m1 = GenTensor::<f64>::new_raw(&vec![1.,2.,3.,4.,], &vec![2,2]);
    /// assert_eq!(m1.sum().get_scale(), 10.);
    /// ```
    fn sum(&self) -> GenTensor<T> {
        let mut sum = T::zero();
        for i in self.get_data() {
            sum = sum + *i;
        }
        GenTensor::new_raw(&vec![sum], &vec![1])
    }
    fn unique(){unimplemented!();}
    fn unique_consecutive() {unimplemented!();}
    fn var(&self, dim: usize, keepdim: bool) -> GenTensor<T> {
        self._dim_statistic(dim, keepdim,
                            |over, k, j, inner_size, step| {
                                let mut sum = T::zero();
                                let mut sum2 = T::zero();
                                for i in 0..over {
                                    let index = k*inner_size*over + j +i*step;
                                    //println!("mean: {}", index);
                                    sum = sum + self.get_data()[index];
                                    sum2 = sum2 + self.get_data()[index]*self.get_data()[index];
                                }
                                sum = sum / T::from(over).expect("N");
                                sum2 = sum2 / T::from(over).expect("N");
                                sum2 - sum*sum
                            })
    }

    fn var_mean() {unimplemented!();}


}

#[cfg(test)]
mod tests {
    use crate::tensor::gen_tensor::GenTensor;
    use super::*;

    #[test]
    fn mean() {
        let a = GenTensor::<f32>::fill(1., &vec![3, 4, 3]);
        let b = a.mean(1, false);
        assert_eq!(*b.size(), vec![3, 3]);
        assert_eq!(b.numel(), 9);
        //println!("{}", b);
        let c = a.mean(1, true);
        assert_eq!(*c.size(), vec![3, 1, 3]);
        assert_eq!(c.numel(), 9);
        //println!("{}", c);
    }

    #[test]
    fn var() {
        let a = GenTensor::<f32>::new_raw(&vec![1., 2., 3., 4., 5., 6., ], &vec![3, 2]);
        let b = a.var(0, false);
        assert_eq!(*b.size(), vec![2]);
        assert_eq!(b.numel(), 2);
        assert_eq!(b, GenTensor::<f32>::new_raw(&vec![2.666667, 2.666666], &vec![2]));
        //println!("{}", b);
        let c = a.var(1, true);
        assert_eq!(*c.size(), vec![3, 1]);
        assert_eq!(c.numel(), 3);
        assert_eq!(c, GenTensor::<f32>::new_raw(&vec![0.25, 0.25, 0.25], &vec![3, 1]));
        //println!("{}", c);
    }

    #[test]
    fn std() {
        let a = GenTensor::<f32>::new_raw(&vec![1., 2., 3., 4., 5., 6., ], &vec![3, 2]);
        let b = a.std(0, false);
        assert_eq!(*b.size(), vec![2]);
        assert_eq!(b.numel(), 2);
        assert_eq!(b, GenTensor::<f32>::new_raw(&vec![1.6329932, 1.632993], &vec![2]));
        //println!("{}", b);
        let c = a.std(1, true);
        assert_eq!(*c.size(), vec![3, 1]);
        assert_eq!(c.numel(), 3);
        assert_eq!(c, GenTensor::<f32>::new_raw(&vec![0.5, 0.5, 0.5], &vec![3, 1]));
        //println!("{}", c);
    }
}
