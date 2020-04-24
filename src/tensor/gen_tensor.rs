use std::fmt;

/// Naive tensor implementation, single thread, no check.
pub struct GenTensor<T> {
    d: Vec<T>,
    dim: Vec<usize>,
}
impl<T> GenTensor<T> where T: num_traits::Float {
    pub fn new() -> GenTensor<T> {
        GenTensor { d: Vec::<T>::new(), dim: Vec::new() }
    }

    /// Create a tensor with given Vec.
    pub fn new_raw(data: &[T], shape: &[usize]) -> GenTensor<T> {
        let new_data = data.to_vec();
        let new_dim = shape.to_vec();
        GenTensor {
            d: new_data,
            dim: new_dim,
        }
    }

    /// Create a tensor filled with the same value d
    ///
    /// ```
    /// # use auto_diff::tensor::gen_tensor::*;
    /// let m1 = GenTensor::<f64>::fill(1., &vec![3,5,2]);
    /// ```
    pub fn fill(d: T, shape: &[usize]) -> GenTensor<T> {
        let mut dsize = 1;
        for i in shape {
            dsize *= *i;
        }
        GenTensor {
            d: vec![d; dsize],
            dim: shape.to_vec(),
        }
    }
    
    /// Right dimension changes fastest.
    /// Right dimension has the stride 1.
    ///
    /// ```
    /// # use auto_diff::tensor::gen_tensor::*;
    /// let m1 = GenTensor::<f64>::new_raw(&vec![0.; 3*5*2], &vec![3,5,2]);
    /// assert_eq!(m1.stride(), vec![10,2,1]);
    /// ```
    pub fn stride(&self) -> Vec<usize> {
        let mut ret = vec![0; self.dim.len()];
        let dsize = ret.len();
        for i in 0..dsize {
            if i == 0 {
                ret[dsize-1] = 1;
            } else {
                ret[dsize-i-1] = ret[dsize-i]*self.dim[dsize-i];
            }
        }
        ret
    }
    /// Return value at the index of the tensor.
    ///
    /// ```
    /// # use auto_diff::tensor::gen_tensor::*;
    /// let m1 = GenTensor::<f64>::new_raw(&vec![1.,2.,3.,4.,5.,6.], &vec![2,3]);
    /// assert_eq!(m1.get(&vec![1,1]), 5.);
    /// ```
    pub fn get(&self, o: &[usize]) -> T {
        let stride = self.stride();
        let dsize = o.len();
        let mut index = 0;
        for i in 0..dsize {
            index += stride[i]*o[i];
        }
        self.d[index]
    }
    pub fn get_mut(&mut self, o: &[usize]) -> &mut T {
        let stride = self.stride();
        let dsize = o.len();
        let mut index = 0;
        for i in 0..dsize {
            index += stride[i]*o[i];
        }
        &mut self.d[index]
    }

    /// dump the underlying vec
    pub fn get_raw(&self) -> Vec<T> {
        self.d.to_vec()
    }
    /// dump the single value in the tensor
    /// if it is the single value in the tensor.
    pub fn get_scale(&self) -> T {
        if self.dim.len() <= 1 && self.d.len() == 1 {
            self.d[0]
        } else {
            panic!("Only one element tensor can get_scale()");
        }
    }
    
    // get NCHW elements
    /// get NCHW elements, always return the size of left most dimension.
    pub fn get_N(&self) -> GenTensor<T> {
        GenTensor {
            d: vec![T::from(self.dim[0]).expect("N")],
            dim: vec![],
        }
    }
    /// get NCHW elements, always return the size of second left most dimension.
    pub fn get_C(&self) -> GenTensor<T> {
        GenTensor {
            d: vec![T::from(self.dim[1]).expect("N")],
            dim: vec![],
        }
    }
    /// get NCDHW elements, will require the self.dim has 5 dimensions.
    pub fn get_D(&self) -> GenTensor<T> {
        if self.dim.len() == 5 {
            GenTensor {
                d: vec![T::from(self.dim[2]).expect("N")],
                dim: vec![],
            }            
        } else {
            panic!("Bad shape for get_D");
        }

    }
    /// get NCDHW elements, will require the self.dim has 5 dimensions or 4 dimensions.
    pub fn get_H(&self) -> GenTensor<T> {
        if self.dim.len() == 5 {
            GenTensor {
                d: vec![T::from(self.dim[3]).expect("N")],
                dim: vec![],
            }
        } else if self.dim.len() == 4 {
            GenTensor {
                d: vec![T::from(self.dim[2]).expect("N")],
                dim: vec![],
            }
        } else {
            panic!("Bad shape for get_D");
        }
    }
    /// get NCDHW elements, will require the self.dim has 5 dimensions or 4 dimensions.
    pub fn get_W(&self) -> GenTensor<T> {
        if self.dim.len() == 5 {
            GenTensor {
                d: vec![T::from(self.dim[4]).expect("N")],
                dim: vec![],
            }
        } else if self.dim.len() == 4 {
            GenTensor {
                d: vec![T::from(self.dim[3]).expect("N")],
                dim: vec![],
            }
        } else {
            panic!("Bad shape for get_D");
        }
    }

    /// Returns the size of the self tensor.
    pub fn size(&self) -> Vec<usize> {
        self.dim.to_vec()
    }

    /// Returns the total number of elements in the input tensor
    pub fn numel(&self) -> usize {
        self.d.len()
    }

    /// Returns the sum of all elements.
    /// ```
    /// # use auto_diff::tensor::gen_tensor::*;
    /// let m1 = GenTensor::<f64>::new_raw(&vec![1.,2.,3.,4.,], &vec![2,2]);
    /// assert_eq!(m1.sum().get_scale(), 10.);
    /// ```
    pub fn sum(&self) -> GenTensor<T> {
        let mut sum = T::zero();
        for i in &self.d {
            sum = sum + *i;
        }
        GenTensor {
            d: vec![sum],
            dim: vec![],
        }
    }

    pub fn unsqueeze(&self, dim: &[usize]) {
        
    }
    
    /// element-wise add with right-hand broadcast.
    ///
    /// ```
    /// # use auto_diff::tensor::gen_tensor::*;
    /// let m1 = GenTensor::<f64>::new_raw(&vec![1.,2.,3.,4.,], &vec![2,2]);
    /// let m2 = GenTensor::<f64>::new_raw(&vec![1.,2.,3.,4.,], &vec![2,2]);
    /// let m3 = m1.add(&m2);
    /// assert_eq!(m3.get(&vec![0,0]), 2.);
    /// assert_eq!(m3.get(&vec![1,1]), 8.);
    /// ```
    pub fn add(&self, o: &GenTensor<T>) -> GenTensor<T> {
        let mut ret = GenTensor {
            d: Vec::with_capacity(self.d.len()),
            dim: self.dim.clone(),
        };
        if self.d.len() == o.d.len() {
            for (v1, v2) in self.d.iter().zip(o.d.iter()) {
                ret.d.push(*v1 + *v2);
            }
        } else {
            if self.d.len() < o.d.len() {
                panic!("right-hand broadcast only.");
            }
            if self.dim.len() <= o.dim.len() {
                panic!("unmatched dimension.");
            }
            for i in 0..o.dim.len() {
                if o.dim[o.dim.len()-i-1] != self.dim[self.dim.len()-i-1] {
                    panic!("unmatched size.");
                }
            }

            // do repeat add
            let mut index = 0;
            for i in 0..self.d.len() {
                ret.d.push(self.d[i] + o.d[index]);
                index += 1;
                if index >= o.d.len() {
                    index = 0;
                }
            }
        }
        ret
    }
    pub fn sub(&self, o: &GenTensor<T>) -> GenTensor<T> {
        let mut ret = GenTensor {
            d: Vec::with_capacity(self.d.len()),
            dim: self.dim.clone(),
        };
        for (v1, v2) in self.d.iter().zip(o.d.iter()) {
            ret.d.push(*v1 - *v2);
        }
        ret
    }
    pub fn mul(&self, o: &GenTensor<T>) -> GenTensor<T> {
        let mut ret = GenTensor {
            d: Vec::with_capacity(self.d.len()),
            dim: self.dim.clone(),
        };
        for (v1, v2) in self.d.iter().zip(o.d.iter()) {
            ret.d.push(*v1 * *v2);
        }
        ret
    }
    pub fn div(&self, o: &GenTensor<T>) -> GenTensor<T> {
        let mut ret = GenTensor {
            d: Vec::with_capacity(self.d.len()),
            dim: self.dim.clone(),
        };
        for (v1, v2) in self.d.iter().zip(o.d.iter()) {
            ret.d.push(*v1 / *v2);
        }
        ret
    }
    

    /// matrix multiplication
    ///
    /// ```
    /// # use auto_diff::tensor::gen_tensor::*;
    /// let m1 = GenTensor::<f64>::new_raw(&vec![1.,2.,3.,4.,5.,6.], &vec![3,2]);
    /// let m2 = GenTensor::<f64>::new_raw(&vec![2.,3.,4.,5.,6.,7.], &vec![2,3]);
    /// let mut result = m1.mm(&m2);
    /// assert!(result == GenTensor::<f64>::new_raw(&vec![12.,15.,18.,26.,33.,40.,40.,51.,62.,], &vec![3,3]), "");
    /// ```
    pub fn mm(&self, o: &GenTensor<T>) -> GenTensor<T>{
        let ls = self.dim[0];
        let rs = o.dim[1];
        let mut ret = GenTensor {
            d: Vec::with_capacity(ls*rs),
            dim: vec![ls, rs],
        };
        let lstride = self.stride();
        let rstride = o.stride();
        for i in 0..ls {
            for j in 0..rs {
                let mut tsum = T::zero();
                for k in 0..self.dim[1] {
                    tsum = tsum
                        + self.d[i*lstride[0] + k] * o.d[k*rstride[0] + j];
                }
                ret.d.push(tsum);
            }
        }
        ret
    }

    /// matrix multiplication of two tensor
    pub fn matmul(&self, o: &GenTensor<T>) -> GenTensor<T> {
        let inner = o.dim[0];
        let mut cap = 1;
        let mut odim = Vec::new();
        let mut lloop = 1;
        let mut rloop = 1;
        for i in 0..self.dim.len()-1 {
            cap *= self.dim[i];
            odim.push(self.dim[i]);
            lloop *= self.dim[i];
        }
        for i in 1..o.dim.len() {
            cap *= o.dim[i];
            odim.push(o.dim[i]);
            rloop *= o.dim[i];
        }

        let mut ret = GenTensor {
            d: Vec::with_capacity(cap),
            dim: odim,
        };
        
        let lstride = self.stride();
        let rstride = o.stride();
        for i in 0..lloop {
            for j in 0..rloop {
                let mut tsum = T::zero();
                for k in 0..inner {
                    tsum = tsum
                        + self.d[i*lstride[0] + k] * o.d[k*rstride[0] + j];
                }
                ret.d.push(tsum);
            }
        }
        ret
    }

    pub fn squared_error(t1: &Self, t2: &Self) -> GenTensor<T> {
        let mut ret = GenTensor {
            d: Vec::with_capacity(t1.d.len()),
            dim: t1.dim.to_vec(),
        };
        for (v1, v2) in t1.d.iter().zip(t2.d.iter()) {
            ret.d.push((*v1 - *v2)*(*v1 - *v2));
        }
        ret
    }

    /// Concatenates sequence of tensors along a new dimension.
    ///
    /// All tensors need to be of the same size.
    /// ```
    /// # use auto_diff::tensor::gen_tensor::*;
    /// let m1 = GenTensor::<f64>::new_raw(&vec![1.,2.,3.,4.,5.,6.], &vec![3,2]);
    /// let m2 = GenTensor::<f64>::new_raw(&vec![2.,3.,4.,5.,6.,7.], &vec![3,2]);
    /// let result = GenTensor::<f64>::stack(&vec![&m1, &m2], 1);
    /// let raw = result.get_raw();
    /// for i in raw {
    ///     println!("{}", i);
    /// }
    /// assert_eq!(result.size(), vec![3,2,2]);
    /// ```
    pub fn stack(tensors: &[&Self], dim: usize) -> GenTensor<T> {
        let cap = tensors.len()*tensors[0].d.len();
        let mut odim = Vec::new();
        for i in 0..tensors[0].dim.len() {
            if i == dim {
                odim.push(tensors.len());
            }
            odim.push(tensors[0].dim[i]);
        }
        if odim.len() == tensors[0].dim.len() {
            odim.push(tensors.len());
        }
        
        let mut ret = GenTensor {
            d: Vec::with_capacity(cap),
            dim: odim,
        };
        
        let mut outter_loop = 1;
        let mut inner_loop = 1;
        for i in 0..tensors[0].dim.len() {
            if i < dim {
                outter_loop *= tensors[0].dim[i];
            } else {
                inner_loop *= tensors[0].dim[i];
            }
        }
        for i in 0..outter_loop {
            for j in 0..tensors.len() {
                for k in 0..inner_loop {
                    ret.d.push(tensors[j].d[k + i*inner_loop]);
                }
            }
        }
        ret
    }

    /// Permute the dimensions of this tensor.
    ///
    /// ```
    /// # use auto_diff::tensor::gen_tensor::*;
    /// let mut m1 = GenTensor::<f64>::fill(1., &vec![2, 3, 5]);
    /// m1.permute(&vec![2, 0, 1]);
    /// ```
    pub fn permute(&mut self, dims: &[usize]) {
        let dim_len = self.dim.len();
        let mut target_dim = vec![0; dim_len];
        for i in 0..dim_len {
            target_dim[i] = self.dim[dims[i]];
        }

        let mut new_d = self.d.to_vec();
        let mut index = vec![0; dim_len];
        let mut old_index = vec![0; dim_len];
        let old_stride = self.stride();
        self.dim = target_dim.to_vec();
        let new_stride = self.stride();
        for i in 0..self.d.len() {
            for j in 0..dim_len {
                old_index[dims[j]] = index[j];
            }

            let mut item_index = 0;
            let mut new_item_index = 0;
            for j in 0..dim_len {
                item_index += old_stride[j]*old_index[j];
                new_item_index += new_stride[j]*index[j];
            }
            new_d[new_item_index] = self.d[item_index];
            
            index[dim_len-1] += 1;
            let mut next_dim = dim_len-1;
            while index[next_dim] >= target_dim[next_dim] {
                if next_dim <= 0 {
                    break
                } else {
                    index[next_dim] = 0;
                    index[next_dim-1] += 1;
                    next_dim -= 1;                    
                }

            }

        }
        self.d = new_d;
    }

    /// Computes element-wise equality
    /// use eq_t instead, as eq is reserved for == overloading.
    ///
    /// ```
    /// # use auto_diff::tensor::gen_tensor::*;
    /// let m1 = GenTensor::<f64>::new_raw(&vec![1.,2.,3.,4.,5.,6.], &vec![3,2]);
    /// let m2 = GenTensor::<f64>::new_raw(&vec![1.,2.,3.,4.,5.,6.], &vec![3,2]);
    /// assert_eq!(m1.eq_t(&m2).get(&vec![0,0]), 1.);
    /// assert_eq!(m1.eq_t(&m2).get(&vec![2,1]), 1.);
    /// ```
    pub fn eq_t(&self, o: &GenTensor<T>) -> GenTensor<T> {
        let mut cmp = Vec::<T>::with_capacity(self.d.len());
        for (v1, v2) in self.d.iter().zip(o.d.iter()) {
            if (*v1-*v2).abs() < T::min_positive_value().sqrt() {
                cmp.push(T::one(),);
            } else {
                cmp.push(T::zero());
            }
        }
        GenTensor {
            d: cmp,
            dim: self.dim.to_vec(),
        }
    }

    /// true if two tensors have the same size and elements, false otherwise.
    ///
    /// ```
    /// # use auto_diff::tensor::gen_tensor::*;
    /// let m1 = GenTensor::<f64>::fill(1., &vec![3,5,2]);
    /// let m2 = GenTensor::<f64>::fill(1., &vec![3,5,2]);
    /// assert_eq!(m1.equal(&m2), Ok(()))
    /// ```
    pub fn equal(&self, o: &GenTensor<T>) -> Result<(), ()> {
        let mut same = Ok(());
        for (v1, v2) in self.d.iter().zip(o.d.iter()) {
            if (*v1-*v2).abs() > T::min_positive_value().sqrt() {
                same = Err(());
                break;
            }
        }
        same
    }

}

/// ```
/// # use auto_diff::tensor::gen_tensor::*;
/// let m1 = GenTensor::<f64>::fill(1., &vec![3,5,2]);
/// let m2 = GenTensor::<f64>::fill(1., &vec![3,5,2]);
/// assert_eq!(m1==m2, true)
/// ```
impl<T> PartialEq for GenTensor<T> where T: num_traits::Float {
    fn eq(&self, other: &Self) -> bool {
        if self.equal(other) == Ok(()) {
            true
        } else {
            false
        }
    }
}
impl<T> Eq for GenTensor<T> where T: num_traits::Float {}

impl fmt::Display for GenTensor<f32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dim.len() == 2 {
            write!(f, "[");
            for i in 0..self.dim[0] {
                write!(f, "[");
                for j in 0..self.dim[1] {
                    write!(f, "{}, ", self.get(&vec![i, j]));
                }
                write!(f, "]\n");
            }
            write!(f, "]\n")
        } else {
            write!(f, "{:?}\n", self.dim);
            write!(f, "{:?}", self.d)            
        }
    }
}
impl fmt::Display for GenTensor<f64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.dim);
        write!(f, "{:?}", self.d)
    }
}

impl<T> Clone for GenTensor<T> where T: num_traits::Float {
    fn clone(&self) -> Self {
        GenTensor {
            d: self.d.to_vec(),
            dim: self.dim.to_vec(),
        }
    }
}
