use crate::tensor::gen_tensor::GenTensor;
#[cfg(feature = "use-cuda")]
use crate::tensor::cuda_tensor::CudaTensor;
use crate::tensor_trait::reduction::ReduceTensor;


impl<T> GenTensor<T> where T: num_traits::Float {
    fn _argmax_min(&self, dim: Option<&[usize]>, keep_dim: bool, max: bool) -> Self {
        if keep_dim {
            panic!("argmax cannot keep dim");
        }
        let dim2aggregate;
        if let Some(dim_val) = dim {
            dim2aggregate = (0..self.size().len()).filter(|x| dim_val.contains(x)).collect();
        } else {
            dim2aggregate = self.size().to_vec();
        }
        let dim = dim2aggregate;
        
        // build return tensor dimension.
        let mut aggregated = false;
        let ret_dim: Vec<usize> = (0..self.size().len()).map(|x|
                                                             if dim.contains(&x) {
                                                                 if !aggregated {
                                                                     aggregated = true;
                                                                     dim.len()
                                                                 } else {
                                                                     1
                                                                 }
                                                             } else {
                                                                 self.size()[x]
                                                             }
        ).collect();
        let mut ret = Self::zeros(&ret_dim);
        //println!("{:?}, {:?}, {:?}", ret.size(), self.size(), dim);

        let kept_dim: Vec<usize> = (0..self.size().len()).filter(|x| !dim.contains(x)).collect();
        let mut index = vec![0; kept_dim.len()]; // index for the loop.

        loop {
            let mut patch_index: Vec::<(usize, usize)> = Vec::new();
            let mut output_index: Vec<usize> = Vec::new();
            let mut kept_dim_step = 0;
            let mut aggregated = false;
            for i in 0..self.size().len() {
                if dim.contains(&i) {
                    patch_index.push((0, self.size()[i]));
                    if !aggregated {
                        output_index.push(0);
                        aggregated = true;
                    }
                } else {
                    patch_index.push((index[kept_dim_step], index[kept_dim_step]+1));
                    output_index.push(index[kept_dim_step]);
                    kept_dim_step += 1;
                }
            }
            //println!("index: {:?}, patch_index: {:?}, output_index: {:?}", index, patch_index, output_index);

            //let value = closure(self.get_patch(&patch_index, None).get_data());
            let the_patch = self.get_patch(&patch_index, None);
            let mut max_value = the_patch.get_data()[0];
            let mut max_index = 0;
            for (elem_index, i) in the_patch.get_data().iter().enumerate() {
                if max {
                    if max_value < *i {
                        max_value = *i;
                        max_index = elem_index;
                    }
                } else if max_value > *i {
                    max_value = *i;
                    max_index = elem_index;
                }
            }
            let dimpos_elem = the_patch.index2dimpos(max_index);
            let mut dimpos_elem2 = Vec::new();
            for (dim_index, v) in dimpos_elem.iter().enumerate() {
                if dim.contains(&dim_index) {
                    dimpos_elem2.push(*v);
                } 
            }
            let dimpos_elem = dimpos_elem2;
            //println!("dispos_elem: {:?}", dimpos_elem);
            for (set_index, i) in dimpos_elem.iter().enumerate() {
                let mut dest_index = output_index.to_vec();
                dest_index[dim[0]] = set_index;
                //println!("dest_index: {:?}", dest_index);
                ret.set(&dest_index, T::from(*i).unwrap());
            }
            
            for i in 0..index.len() {
                index[kept_dim.len() -i -1] += 1;
                if index[kept_dim.len() -i -1] >= self.size()[kept_dim[kept_dim.len() -i -1]] {
                    index[kept_dim.len() -i -1] = 0;
                } else {
                    break
                }
            }

            if index == vec![0; kept_dim.len()] {
                break
            }
        }
        
        ret
    }
}

impl<T> ReduceTensor for GenTensor<T> where T: num_traits::Float {

    fn argmax(&self, dim: Option<&[usize]>, keep_dim: bool) -> Self {
        self._argmax_min(dim, keep_dim, true)
    }
    fn argmin(&self, dim: Option<&[usize]>, keep_dim: bool) -> Self {
        self._argmax_min(dim, keep_dim, false)
    }
    fn dist() {unimplemented!();}
    fn logsumexp(&self, dim: Option<&[usize]>, keep_dim: bool) -> Self {
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
    /// # use crate::tensor_rs::tensor::gen_tensor::*;
    /// # use crate::tensor_rs::tensor_trait::reduction::ReduceTensor;
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

    fn max(&self, dim: Option<&[usize]>, keep_dim: bool) -> Self {
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
    fn min(&self, dim: Option<&[usize]>, keep_dim: bool) -> Self {
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
    fn argmax() {
        let a = GenTensor::<f32>::new_raw(&vec![1., 2., 3., 4., 5., 6., ], &vec![3, 2]);
        let b = a.argmax(Some(&[0]), false);
        println!("{:?}", b);
        assert_eq!(b, GenTensor::<f32>::new_raw(&[2., 2.,], &[1, 2]));

        let b = a.argmax(Some(&[1]), false);
        println!("{:?}", b);
        assert_eq!(b, GenTensor::<f32>::new_raw(&[1., 1., 1.,], &[3, 1]));
    }

    #[test]
    fn argmin() {
        let a = GenTensor::<f32>::new_raw(&vec![1., 2., 3., 4., 5., 6., ], &vec![3, 2]);
        let b = a.argmin(Some(&[0]), false);
        println!("{:?}", b);
        assert_eq!(b, GenTensor::<f32>::new_raw(&[0., 0.,], &[1, 2]));

        let b = a.argmin(Some(&[1]), false);
        println!("{:?}", b);
        assert_eq!(b, GenTensor::<f32>::new_raw(&[0., 0., 0.,], &[3, 1]));
    }

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
