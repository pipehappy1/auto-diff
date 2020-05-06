use std::fmt;

/// Naive tensor implementation, single thread
pub struct GenTensor<T> {
    d: Vec<T>,
    dim: Vec<usize>,
}
impl<T> GenTensor<T> where T: num_traits::Float {

    /// Creation Ops
    
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

    // 
    // as_tensor
    // as_strided
    // from_ndarray
    // zeros
    pub fn zeros_like(&self) -> GenTensor<T> {
        let mut new_data = Vec::with_capacity(self.d.len());
        for i in 0..self.d.len() {
            new_data.push(T::zero());
        }
        let new_dim = self.dim.to_vec();
        GenTensor {
            d: new_data,
            dim: new_dim,
        }
    }

    // ones
    pub fn ones_like(&self) -> GenTensor<T> {
        let mut new_data = Vec::with_capacity(self.d.len());
        for i in 0..self.d.len() {
            new_data.push(T::one());
        }
        let new_dim = self.dim.to_vec();
        GenTensor {
            d: new_data,
            dim: new_dim,
        }
    }
    // arange
    // range
    // linspace
    // logspace
    // eye
    pub fn empty(shape: &[usize]) -> GenTensor<T> {
        let mut elem = 1;
        for i in shape {
            elem *= i;
        }
        
        let mut new_data = Vec::with_capacity(elem);
        unsafe{ new_data.set_len(elem); }
        let new_dim = shape.to_vec();
        GenTensor {
            d: new_data,
            dim: new_dim,
        }
    }
    // empty_like
    // empty_stided
    // full
    // full_like
    // quantize_per_tensor
    // quantize_per_channel
    // 

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
    pub fn from_record(&mut self, row: usize, record: &[f32]) -> Result<(), ()> {
        for (i, index) in record.iter().zip(0..self.dim[self.dim.len()-1]) {
            self.d[row*self.dim[self.dim.len()-1] + index] = T::from(*i).expect("");
        }
        Ok(())
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
            dim: vec![1],
        }
    }
    /// get NCHW elements, always return the size of second left most dimension.
    pub fn get_C(&self) -> GenTensor<T> {
        GenTensor {
            d: vec![T::from(self.dim[1]).expect("N")],
            dim: vec![1],
        }
    }
    /// get NCDHW elements, will require the self.dim has 5 dimensions.
    pub fn get_D(&self) -> GenTensor<T> {
        if self.dim.len() == 5 {
            GenTensor {
                d: vec![T::from(self.dim[2]).expect("N")],
                dim: vec![1],
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
                dim: vec![1],
            }
        } else if self.dim.len() == 4 {
            GenTensor {
                d: vec![T::from(self.dim[2]).expect("N")],
                dim: vec![1],
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
                dim: vec![1],
            }
        } else if self.dim.len() == 4 {
            GenTensor {
                d: vec![T::from(self.dim[3]).expect("N")],
                dim: vec![1],
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

    /// Returns the total number of elements in the input tensor
    pub fn numel_tensor(&self) -> GenTensor<T> {
        GenTensor {
            d: vec![T::from(self.d.len()).expect(""),],
            dim: vec![1],
        }
    }


    // Indexing, Slicing, Joining, Mutating Ops
    
    pub fn cat(&self, tensors: &[&GenTensor<T>], dim: usize) -> GenTensor<T> {
        GenTensor::new()
    }
    //pub fn chunk() {}
    //pub fn gather() {}
    //pub fn index_select() {}
    //pub fn masked_select() {}
    //pub fn narrow() {}
    //pub fn nonzero() {}
    //pub fn reshape() {}
    //pub fn split() {}
    //pub fn squeeze() {}
    //pub fn stack() {}
    //pub fn t() {}
    //pub fn take() {}
    //pub fn transpose() {}
    //pub fn unbind() {}
    //
    //pub fn permute(&self, dim: &[usize]) -> Tensor {
    //    Tensor {
    //        v: Rc::new(RefCell::new(self.v.borrow().permute(dim))),
    //    }
    //}
    //
    ///// Returns a new tensor with a dimension of size one inserted at the specified position.
    ///// 
    ///// The returned tensor shares the same underlying data with this tensor.
    /////
    ///// 
    //pub fn unsqueeze(&mut self, dim: &[usize]) -> &Tensor {
    //    self.v.borrow_mut().unsqueeze(dim);
    //    self
    //}
    //
    //pub fn condition() {} // this is pytorch where

    

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
            dim: vec![1],
        }
    }

    pub fn _dim_statistic<F>(&self, dim: usize, keepdim: bool, closure: F) -> GenTensor<T>
    where F: Fn(usize, usize, usize, usize, usize) -> T {
        if self.dim.len() <= dim {
            panic!("Tensor has dimension {:?}, mean() get dim of {}", self.dim, dim);
        }
        
        let mut ret_dim;
        if keepdim {
            ret_dim = self.dim.to_vec();
            ret_dim[dim] = 1;
        } else {
            ret_dim = Vec::new();
            for (i, index) in self.dim.iter().zip(0..self.dim.len()) {
                if index != dim {
                    ret_dim.push(*i);
                }
            }
        }
        
        let mut cap = 1;
        for i in &ret_dim {
            cap *= i;
        }

        let mut outer_size = 1;
        let mut inner_size = 1;
        for i in 0..self.dim.len() {
            if i < dim {
                outer_size *= self.dim[i];
            }
            if i > dim {
                inner_size *= self.dim[i];
            }
        }
        
        let mut data = Vec::with_capacity(cap);
        let over = self.dim[dim];
        let stride = self.stride();
        let step = stride[dim];

        for k in 0..outer_size {
            for j in 0..inner_size {
                let val = closure(over, k, j, inner_size, step);
                data.push(val);
            }
        }
        
        GenTensor {
            d: data,
            dim: ret_dim,
        }
    }

    pub fn var(&self, dim: usize, keepdim: bool) -> GenTensor<T> {
        self._dim_statistic(dim, keepdim,
                            |over, k, j, inner_size, step| {
                                let mut sum = T::zero();
                                let mut sum2 = T::zero();
                                for i in 0..over {
                                    let index = k*inner_size*over + j +i*step;
                                    //println!("mean: {}", index);
                                    sum = sum + self.d[index];
                                    sum2 = sum2 + self.d[index]*self.d[index];
                                }
                                sum = sum / T::from(over).expect("N");
                                sum2 = sum2 / T::from(over).expect("N");
                                sum2 - sum*sum
                            })
    }

    pub fn std(&self, dim: usize, keepdim: bool) -> GenTensor<T> {
        self._dim_statistic(dim, keepdim,
                            |over, k, j, inner_size, step| {
                                let mut sum = T::zero();
                                let mut sum2 = T::zero();
                                for i in 0..over {
                                    let index = k*inner_size*over + j +i*step;
                                    //println!("mean: {}", index);
                                    sum = sum + self.d[index];
                                    sum2 = sum2 + self.d[index]*self.d[index];
                                }
                                sum = sum / T::from(over).expect("N");
                                sum2 = sum2 / T::from(over).expect("N");
                                (sum2 - sum*sum).sqrt()
                            })
    }

    /// Returns the mean value of the tensor along dim row.
    pub fn mean(&self, dim: usize, keepdim: bool) -> GenTensor<T> {
        self._dim_statistic(dim, keepdim,
                            |over, k, j, inner_size, step| {
                                let mut sum = T::zero();
                                for i in 0..over {
                                    let index = k*inner_size*over + j +i*step;
                                    //println!("mean: {}", index);
                                    sum = sum + self.d[index];
                                }
                                sum = sum / T::from(over).expect("N");
                                sum
                            })
    }

    pub fn unsqueeze(&self, dim: &[usize]) {
        
    }


    // Pointwise Ops
    
    fn _pointwise<F>(&self, closure: F) -> GenTensor<T>
    where F: Fn(&T) -> T {
        let mut ret = GenTensor {
            d: Vec::with_capacity(self.d.len()),
            dim: self.dim.clone(),
        };

        for i in &self.d {
            ret.d.push(closure(i));
        }
        ret
    }
    // abs
    pub fn abs(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.abs()
        })
    }
    // acos
    pub fn acos(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.acos()
        })
    }
    // add, there is one.
    // addcdiv
    // addcmul
    // angle
    // asin
    pub fn asin(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.asin()
        })
    }
    // atan
    pub fn atan(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.atan()
        })
    }
    // atan2
    // bitwise_not
    // bitwise_and
    // bitwise_or
    // bitwise_xor
    // ceil
    pub fn ceil(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.ceil()
        })
    }
    // clamp
    pub fn clamp(&self, min: T, max: T) -> GenTensor<T> {
        let mut ret = GenTensor {
            d: Vec::with_capacity(self.d.len()),
            dim: self.dim.clone(),
        };

        for i in &self.d {
            let value;
            if *i < min {
                value = min;
            } else if *i <= max {
                value = *i;
            } else {
                value = max;
            }
            ret.d.push(value);
        }
        ret
    }
    // conj
    // cos
    pub fn cos(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.cos()
        })
    }
    // cosh
    pub fn cosh(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.cosh()
        })
    }
    // div, there is one.
    // digamma
    //pub fn digamma(&self) -> GenTensor<T> {
    //    self._pointwise(|x| {
    //        x.digamma()
    //    })
    //}
    // erf
    // erfc
    // erfinv
    // exp
    pub fn exp(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.exp()
        })
    }
    // expm1
    pub fn expm1(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.exp_m1()
        })
    }
    // floor
    pub fn floor(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.floor()
        })
    }
    // floor_divide
    // fmod
    // frac
    pub fn frac(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.fract()
        })
    }
    // imag
    // lerp, this is on Tensor.
    // lgamma
    // log
    pub fn log(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.ln()
        })
    }
    // log10
    pub fn log10(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.log10()
        })
    }
    // log1p
    pub fn log1p(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.ln_1p()
        })
    }

    /// Better log(1 + exp(x))
    /// see https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    pub fn log1pexp(&self) -> GenTensor<T> {
        let mut ret = GenTensor {
            d: Vec::with_capacity(self.d.len()),
            dim: self.dim.to_vec(),
        };
        for i in &self.d {
            if i <= &T::from(-37).expect("") {
                ret.d.push(i.exp());
            } else if i > &T::from(-37).expect("") && i <= &T::from(18).expect("") {
                ret.d.push(i.exp().ln_1p());
            } else if i > &T::from(-18).expect("") && i <= &T::from(33.3).expect("") {
                ret.d.push(*i + i.mul(T::from(-1).expect("")).exp());
            } else {
                ret.d.push(*i);
            }
        }
        ret
    }
    
    // log2
    pub fn log2(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.log2()
        })
    }
    // logical_and
    // logical_not
    // logical_or
    // logical_xor
    // mul, there is one
    // mvlgamma
    // neg
    pub fn neg(&self) -> GenTensor<T> {
        let mut ret = GenTensor {
            d: Vec::with_capacity(self.d.len()),
            dim: self.dim.to_vec(),
        };

        for i in &self.d {
            ret.d.push(i.mul(T::zero() - T::one()));
        }
        ret
    }
    
    // polygamma
    // pow
    pub fn pow(&self, n: T) -> GenTensor<T> {
        self._pointwise(|x| {
            x.powf(n)
        })
    }
    // real
    // reciprocal
    pub fn reciprocal(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.recip()
        })
    }
    // remainder
    // round
    pub fn round(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.round()
        })
    }
    // rsqrt
    pub fn rsqrt(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.sqrt()/(*x)
        })
    }
    
    pub fn sigmoid(&self) -> GenTensor<T> {
        let mut ret = GenTensor {
            d: self.d.to_vec(),
            dim: self.dim.to_vec(),
        };

        for i in 0..self.d.len() {
            if self.d[i] > T::zero() {
                ret.d[i] = T::one()/(T::one() + self.d[i].neg().exp());
            }
            else {
                ret.d[i] = self.d[i].exp()/(T::one() + self.d[i].exp());
            }
        }
        ret
    }

    // sign
    pub fn sign(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            if *x == T::zero() {
                T::zero()
            } else if *x > T::zero() {
                T::one()
            } else {
                T::zero() - T::one()
            }
        })
    }
    // sin
    pub fn sin(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.sin()
        })
    }
    // sinh
    pub fn sinh(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.sinh()
        })
    }
    // sqrt
    pub fn sqrt(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.sqrt()
        })
    }
    // square
    pub fn square(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            (*x)*(*x)
        })
    }
    // tan
    pub fn tan(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.tan()
        })
    }
    // tanh
    pub fn tanh(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.tanh()
        })
    }
    // true_divide
    // trunc
    pub fn trunc(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.trunc()
        })
    }
    
    pub fn _right_broadcast<F>(&self, o: &GenTensor<T>, closure: F) -> GenTensor<T>
    where F: Fn(&T, &T) -> T {
        let mut ret = GenTensor {
            d: Vec::with_capacity(self.d.len()),
            dim: self.dim.clone(),
        };

        if self.d.len() == o.d.len() {
            for (v1, v2) in self.d.iter().zip(o.d.iter()) {
                ret.d.push(closure(v1, v2));
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
                ret.d.push(closure(&self.d[i], &o.d[index]));
                index += 1;
                if index >= o.d.len() {
                    index = 0;
                }
            }
        }
        ret
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
        self._right_broadcast(o, |x, y| *x + *y)
     }
    pub fn sub(&self, o: &GenTensor<T>) -> GenTensor<T> {
        self._right_broadcast(o, |x, y| *x - *y)
    }
    pub fn mul(&self, o: &GenTensor<T>) -> GenTensor<T> {
        self._right_broadcast(o, |x, y| *x * *y)
    }
    pub fn div(&self, o: &GenTensor<T>) -> GenTensor<T> {
        self._right_broadcast(o, |x, y| *x / *y)
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
        if self.dim.len() != 2 || o.dim.len() != 2 {
            panic!("Not a matrix input.");
        }
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

    /// outer product of right-most dimensions.
    pub fn outer(&self, o: &GenTensor<T>) -> GenTensor<T> {
        let mut dim = Vec::new();
        let mut data;
        let mut cap = 1;
        let mut outer_size = 1;
        let left_dim;
        let right_dim;
        if self.dim.len() == o.dim.len()
            && self.dim[0..self.dim.len()-1] == o.dim[0..self.dim.len()-1] {
                left_dim = self.dim[self.dim.len()-1];
                right_dim = o.dim[self.dim.len()-1];
                for i in 0..self.dim.len()-1 {
                    dim.push(self.dim[i]);
                    cap *= self.dim[i];
                    outer_size *= self.dim[i];
                }
                dim.push(left_dim);
                cap *= left_dim;
                dim.push(right_dim);
                cap *= right_dim;
                data = Vec::with_capacity(cap);
            } else {
            panic!("bad size for outer: {:?}, {:?}", self.dim, o.dim);
            }

        for k in 0..outer_size {
            for i in 0..left_dim {
                for j in 0..right_dim {
                    data.push(self.d[i + k*left_dim] * o.d[j + k*right_dim]);
                }
            }
        }
        
        GenTensor {
            d: data,
            dim: dim,
        }
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
    pub fn permute(&self, dims: &[usize]) -> GenTensor<T> {
        let mut ret = GenTensor {
            d: self.d.to_vec(),
            dim: self.dim.to_vec(),
        };
        let dim_len = ret.dim.len();
        let mut target_dim = vec![0; dim_len];
        for i in 0..dim_len {
            target_dim[i] = ret.dim[dims[i]];
        }

        let mut new_d = ret.d.to_vec();
        let mut index = vec![0; dim_len];
        let mut old_index = vec![0; dim_len];
        let old_stride = ret.stride();
        ret.dim = target_dim.to_vec();
        let new_stride = ret.stride();
        for i in 0..ret.d.len() {
            for j in 0..dim_len {
                old_index[dims[j]] = index[j];
            }

            let mut item_index = 0;
            let mut new_item_index = 0;
            for j in 0..dim_len {
                item_index += old_stride[j]*old_index[j];
                new_item_index += new_stride[j]*index[j];
            }
            new_d[new_item_index] = ret.d[item_index];
            
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
        ret.d = new_d;
        ret
    }


    // Comparison Ops


    pub fn all_close(&self, o: &GenTensor<T>) -> GenTensor<T> {
        self.eq_t(o)
    }

    pub fn arg_sort(&self, dim: usize, descending: bool) -> GenTensor<T> {
        let mut d = self.d.to_vec();

        let mut outer_size = 1;
        let mut inner_size = 1;

        for (i, index) in self.dim.iter().zip(0..self.dim.len()) {
            if index < dim {
                outer_size *= i;
            } else if index > dim {
                inner_size *= i;
            }
        }

        let stride = self.stride()[dim];
        let size = self.dim[dim];

        for i in 0..outer_size {
            for j in 0..inner_size {
                let mut collected = Vec::<(T, usize)>::with_capacity(size);
                for k in 0..size {
                    collected.push((self.d[k*stride + j + i*inner_size*size], k));
                }
                collected.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let (left, right): (Vec<_>, Vec<_>) = collected.iter().cloned().unzip();
                for k in 0..size {
                    d[k*stride + j + i*inner_size*size] = T::from(right[k]).expect("");
                }
            }
        }

        GenTensor {
            d: d,
            dim: self.dim.to_vec()
        }
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
    /// assert_eq!(m1.equal(&m2), true)
    /// ```
    pub fn equal(&self, o: &GenTensor<T>) -> bool {
        let mut same = true;
        for (v1, v2) in self.d.iter().zip(o.d.iter()) {
            if (*v1-*v2).abs() > T::min_positive_value().sqrt() {
                same = false;
                break;
            }
        }
        same
    }

    pub fn ge(&self, o: &GenTensor<T>) -> GenTensor<T> {
        GenTensor::new()
    }

    pub fn gt(&self, o: &GenTensor<T>) -> GenTensor<T> {
        GenTensor::new()
    }

    //pub fn isfinite(&self, o: &GenTensor<T>) -> GenTensor<T> {
    //    GenTensor::new()
    //}
    //
    //pub fn isinf(&self, o: &GenTensor<T>) -> GenTensor<T> {
    //    GenTensor::new()
    //}
    //
    //pub fn isnan(&self, o: &GenTensor<T>) -> GenTensor<T> {
    //    GenTensor::new()
    //}
    //
    //pub fn kthvalue(&self, o: &GenTensor<T>) -> GenTensor<T> {
    //    GenTensor::new()
    //}
    pub fn le(&self, o: &GenTensor<T>) -> GenTensor<T> {
        GenTensor::new()
    }

    pub fn lt(&self, o: &GenTensor<T>) -> GenTensor<T> {
        GenTensor::new()
    }
    // max
    // min
    pub fn ne(&self, o: &GenTensor<T>) -> GenTensor<T> {
        GenTensor::new()
    }
    // sort
    // topk
    

}

/// ```
/// # use auto_diff::tensor::gen_tensor::*;
/// let m1 = GenTensor::<f64>::fill(1., &vec![3,5,2]);
/// let m2 = GenTensor::<f64>::fill(1., &vec![3,5,2]);
/// assert_eq!(m1==m2, true)
/// ```
impl<T> PartialEq for GenTensor<T> where T: num_traits::Float {
    fn eq(&self, other: &Self) -> bool {
        if self.equal(other) {
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

impl fmt::Debug for GenTensor<f32> {
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
impl fmt::Debug for GenTensor<f64> {
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

impl<T> Clone for GenTensor<T> where T: num_traits::Float {
    fn clone(&self) -> Self {
        GenTensor {
            d: self.d.to_vec(),
            dim: self.dim.to_vec(),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gentensor() {
        {
            let mut m2 = GenTensor::<f64>::new_raw(&vec![1., 2., 3., 4.,], &vec![2, 2]);
            *m2.get_mut(&vec![0,0]) = 5.;
            assert_eq!(m2.get_raw(), vec![5., 2., 3., 4.,])
        }
    }

    #[test]
    fn test_gen_tensor_get() {
        {
            let m1 = GenTensor::<f64>::fill(1., &vec![10, 3, 28, 30]);
            assert_eq!(m1.get_N().get_raw(), vec![10.]);
            assert_eq!(m1.get_C().get_raw(), vec![3.]);
            assert_eq!(m1.get_H().get_raw(), vec![28.]);
            assert_eq!(m1.get_W().get_raw(), vec![30.]);

            let result = std::panic::catch_unwind(
                ||
                    m1.get_D().get_raw()
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn mean() {
        let a = GenTensor::<f32>::fill(1., &vec![3, 4, 3]);
        let b = a.mean(1, false);
        assert_eq!(b.size(), vec![3, 3]);
        assert_eq!(b.numel(), 9);
        //println!("{}", b);
        let c = a.mean(1, true);
        assert_eq!(c.size(), vec![3, 1, 3]);
        assert_eq!(c.numel(), 9);
        //println!("{}", c);
    }

    #[test]
    fn var() {
        let a = GenTensor::<f32>::new_raw(&vec![1., 2., 3., 4., 5., 6., ], &vec![3, 2]);
        let b = a.var(0, false);
        assert_eq!(b.size(), vec![2]);
        assert_eq!(b.numel(), 2);
        assert_eq!(b, GenTensor::<f32>::new_raw(&vec![2.666667, 2.666666], &vec![2]));
        //println!("{}", b);
        let c = a.var(1, true);
        assert_eq!(c.size(), vec![3, 1]);
        assert_eq!(c.numel(), 3);
        assert_eq!(c, GenTensor::<f32>::new_raw(&vec![0.25, 0.25, 0.25], &vec![3, 1]));
        //println!("{}", c);
    }

    #[test]
    fn std() {
        let a = GenTensor::<f32>::new_raw(&vec![1., 2., 3., 4., 5., 6., ], &vec![3, 2]);
        let b = a.std(0, false);
        assert_eq!(b.size(), vec![2]);
        assert_eq!(b.numel(), 2);
        assert_eq!(b, GenTensor::<f32>::new_raw(&vec![1.6329932, 1.632993], &vec![2]));
        //println!("{}", b);
        let c = a.std(1, true);
        assert_eq!(c.size(), vec![3, 1]);
        assert_eq!(c.numel(), 3);
        assert_eq!(c, GenTensor::<f32>::new_raw(&vec![0.5, 0.5, 0.5], &vec![3, 1]));
        //println!("{}", c);
    }

    #[test]
    fn outer() {
        let a = GenTensor::<f32>::fill(1., &vec![10, 2]);
        let b = GenTensor::<f32>::fill(1., &vec![10, 3]);
        let c = a.outer(&b);
        assert_eq!(c.size(), vec![10, 2, 3]);
        //println!("{}", c);
        let d = b.outer(&a);
        assert_eq!(d.size(), vec![10, 3, 2]);
    }

    #[test]
    fn permute() {
        let m1 = GenTensor::<f64>::fill(1., &vec![2, 3, 5]);
        let m11 = m1.permute(&vec![2, 0, 1]);
        assert_eq!(m11.size(), vec![5, 2, 3]);

        let m2 = GenTensor::<f64>::new_raw(&vec![1., 2., 3., 4.,], &vec![2, 2]);
        let m22 = m2.permute(&vec![1, 0]);
        assert_eq!(m22.get_raw(), vec![1., 3., 2., 4.]);
    }

    // Pointwise Ops
    #[test]
    fn ceil() {
        let a = GenTensor::<f32>::new_raw(&vec![0.9213,  1.0887, -0.8858, -1.7683],
                                              &vec![4]);
        
        let ret = a.ceil();

        let expect = GenTensor::<f32>::new_raw(&vec![1., 2., 0., -1.], 
                                               &vec![4]);
        assert_eq!(ret, expect);
    }
    
    #[test]
    fn log1pexp() {
        let a = GenTensor::<f32>::new_raw(&vec![0.9213,  1.0887, -0.8858, -1.7683],
                                              &vec![4]);
        
        let ret = a.log1pexp();

        let expect = GenTensor::<f32>::new_raw(&vec![1.2563436, 1.3788694, 0.34527916, 0.15753591], 
                                               &vec![4]);
        assert_eq!(ret, expect);
    }
    
    #[test]
    fn sigmoid() {
        let a = GenTensor::<f32>::new_raw(&vec![0.9213,  1.0887, -0.8858, -1.7683],
                                              &vec![4]);
        
        let ret = a.sigmoid();

        let expect = GenTensor::<f32>::new_raw(&vec![0.71530694, 0.7481369, 0.29197732, 0.14575386], 
                                               &vec![4]);
        assert_eq!(ret, expect);
    }

    #[test]
    fn sign() {
        let a = GenTensor::<f32>::new_raw(&vec![0.9213,  0.0, -0.0, -1.7683],
                                              &vec![4]);
        
        let ret = a.sign();

        let expect = GenTensor::<f32>::new_raw(&vec![1.0, 0.0, 0.0, -1.0],
                                               &vec![4]);
        assert_eq!(ret, expect);
    }
    

    // Comparison Ops
    #[test]
    fn arg_sort() {
        let mut a = GenTensor::<f32>::new_raw(&vec![0.0785,  1.5267, -0.8521,  0.4065,
                                                    0.1598,  0.0788, -0.0745, -1.2700,
                                                    1.2208,  1.0722, -0.7064,  1.2564,
                                                    0.0669, -0.2318, -0.8229, -0.9280,],
                                              &vec![4, 4]);
        
        let index = a.arg_sort(1, true);

        let expect = GenTensor::<f32>::new_raw(&vec![2., 0., 3., 1., 
                                                     3., 2., 1., 0., 
                                                     2., 1., 0., 3., 
                                                     3., 2., 1., 0.], 
                                               &vec![4, 4]);
        assert_eq!(index, expect);
    }
    
}
