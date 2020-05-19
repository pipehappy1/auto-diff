use std::fmt;
use std::ops::Range;
use std::collections::BTreeMap;
use crate::op::PaddingMode;

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

    pub fn index2dimpos(&self, index: usize) -> Vec::<usize>{
        let mut ret = Vec::new();
        let mut reminder = index;
        for i in &self.stride() {
            println!("{}", reminder);
            ret.push(reminder/i);
            reminder %= i;
        }
        ret
    }

    // 
    // as_tensor
    // as_strided
    // from_ndarray
    // zeros
    pub fn zeros(size: &[usize]) -> GenTensor<T> {
        let cap = size.iter().product();
        GenTensor {
            d: vec![T::zero(); cap],
            dim: size.to_vec(),
        }
    }
    // zeros_like
    pub fn zeros_like(&self) -> GenTensor<T> {
        let new_data = vec![T::zero(); self.d.len()];
        let new_dim = self.dim.to_vec();
        GenTensor {
            d: new_data,
            dim: new_dim,
        }
    }

    // ones
    pub fn ones(size: &[usize]) -> GenTensor<T> {
        let cap = size.iter().product();
        GenTensor {
            d: vec![T::one(); cap],
            dim: size.to_vec(),
        }
    }
    // ones_like
    pub fn ones_like(&self) -> GenTensor<T> {
        let new_data = vec![T::one(); self.d.len()];
        let new_dim = self.dim.to_vec();
        GenTensor {
            d: new_data,
            dim: new_dim,
        }
    }
    // arange
    pub fn arange(end: usize) -> GenTensor<T> {
        let mut ret = GenTensor::<T>::empty(&vec![end]);
        for i in 0..end {
            ret.d[i] = T::from(i).expect("");
        }
        ret
    }
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
    /// assign a row.
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
    pub fn set(&mut self, o: &[usize], v: T) {
        let stride = self.stride();
        let dsize = o.len();
        let mut index = 0;
        for i in 0..dsize {
            index += stride[i]*o[i];
        }
        self.d[index] = v;
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
    pub fn get_n(&self) -> GenTensor<T> {
        GenTensor {
            d: vec![T::from(self.dim[0]).expect("N")],
            dim: vec![1],
        }
    }
    /// get NCHW elements, always return the size of second left most dimension.
    pub fn get_c(&self) -> GenTensor<T> {
        GenTensor {
            d: vec![T::from(self.dim[1]).expect("N")],
            dim: vec![1],
        }
    }
    /// get NCDHW elements, will require the self.dim has 5 dimensions.
    pub fn get_d(&self) -> GenTensor<T> {
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
    pub fn get_h(&self) -> GenTensor<T> {
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
    pub fn get_w(&self) -> GenTensor<T> {
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

    // cat
    pub fn cat(&self, tensors: &[&GenTensor<T>], dim: usize) -> GenTensor<T> {
        for i in tensors {
            if i.dim.len() != self.dim.len() {
                panic!("cat needs all have the same number of dimens {}, {}", self.dim.len(), i.dim.len());
            }
            for (j, index) in i.dim.iter().zip(0..self.dim.len()) {
                if index != dim && self.dim[index] != *j {
                    panic!("cat needs all but the cat dim to have the same dim {:?}, {:?}", self.dim, i.dim);
                }
            }
        }

        let mut cap = 1;
        let mut new_dim = Vec::new();
        let mut outer_size = 1;
        let mut inner_size = 1;
        for i in 0..self.dim.len() {
            if i != dim {
                cap *= self.dim[i];
                new_dim.push(self.dim[i]);                
            } else {
                let mut dim_total = self.dim[i];
                for j in tensors {
                    dim_total += j.dim[i];
                }
                cap *= dim_total;
                new_dim.push(dim_total);
            }
            if i < dim {
                outer_size *= self.dim[i];
            }
            if i > dim {
                inner_size *= self.dim[i];
            }
        }
        let mut data = Vec::with_capacity(cap);
        unsafe{ data.set_len(cap); }

        let mut ret_range = Range{start: 0, end: self.dim[dim]*inner_size};
        for i in 0..outer_size {
            data[ret_range.clone()].clone_from_slice(&self.d[i*self.dim[dim]*inner_size..(i+1)*self.dim[dim]*inner_size]);
            
            //println!("outer: {:?}", ret_range);
            
            for j in tensors {
                ret_range = Range{start: ret_range.end, end: ret_range.end + j.dim[dim]*inner_size};
                data[ret_range.clone()].clone_from_slice(&j.d[i*j.dim[dim]*inner_size..(i+1)*j.dim[dim]*inner_size]);
                //println!("inner: {:?}", ret_range);
            }

            ret_range = Range{start: ret_range.end, end: ret_range.end + self.dim[dim]*inner_size};
        }
        
        GenTensor {
            d: data,
            dim: new_dim,
        }
    }
    //pub fn chunk() {}
    //pub fn gather() {}
    //pub fn index_select() {}
    //pub fn masked_select() {}
    //pub fn narrow() {}
    //pub fn nonzero() {}
    //reshape() {}
    pub fn reshape(&self, new_shape: &[usize]) -> GenTensor<T> {
        if self.dim.iter().product::<usize>() != new_shape.iter().product::<usize>() {
            panic!("reshape expects the same number of elements {:?}, {:?}", self.dim, new_shape);
        }
        GenTensor {
            d: self.d.to_vec(),
            dim: new_shape.to_vec(),
        }
    }
    // split
    pub fn split(&self, sections: &[usize], dim: usize) -> Vec<GenTensor<T>> {
        if sections.iter().sum::<usize>() != self.dim[dim] {
            panic!("sum of sections should be the size on dim.");
        }

        let mut outer_size = 1;
        let mut inner_size = 1;
        for (i, index) in self.dim.iter().zip(0..self.dim.len()) {
            if index < dim {
                outer_size *= i;
            }
            if index > dim {
                inner_size *= i;
            }
        }
        
        let mut ret = Vec::new();
        for i in sections {
            let mut t_size = Vec::new();
            for (j, index) in self.dim.iter().zip(0..self.dim.len()) {
                if index == dim {
                    t_size.push(*i);
                } else {
                    t_size.push(*j);
                }
            }
            let t = GenTensor::empty(&t_size);
            ret.push(t);
        }

        for i in 0..outer_size {
            let mut start = 0;
            for (j, index) in ret.iter_mut().zip(0..sections.len()) {
                j.d[i*inner_size*sections[index]..(i+1)*inner_size*sections[index]].clone_from_slice(
                    &self.d[i*inner_size*self.dim[dim] + start..i*inner_size*self.dim[dim] + start + sections[index]*inner_size]);
                start += sections[index]*inner_size;
            }
        }
        
        ret
    }
    //pub fn squeeze() {}
    // stack
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

    //pub fn t() {}
    //pub fn take() {}
    //pub fn transpose() {}
    //pub fn unbind() {}
    // permute 
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
        for _i in 0..ret.d.len() {
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

    /////
    // unsqueeze
    pub fn unsqueeze(&self, dim: usize) -> GenTensor<T> {
        let mut new_dim = Vec::new();
        for i in 0..self.dim.len() {
            if i == dim {
                new_dim.push(1);
            }
            new_dim.push(self.dim[i]);
        }
        GenTensor {
            d: self.d.to_vec(),
            dim: new_dim,
        }
    }

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
        // with same shape.
        if self.d.len() == o.d.len() {
            for (v1, v2) in self.d.iter().zip(o.d.iter()) {
                ret.d.push(closure(v1, v2));
            }
        // right single scale
        } else if o.dim.len() == 1 && o.dim[0] == 1{
            for i in 0..self.d.len() {
                ret.d.push(closure(&self.d[i], &o.d[0]));
            }
        } else {
            if self.d.len() < o.d.len() {
                panic!("right-hand broadcast only.");
            }
            if self.dim.len() <= o.dim.len() {
                panic!("unmatched dimension. {}, {}", self.dim.len(), o.dim.len());
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
        if self.dim[self.dim.len()-1] != o.dim[0] {
            panic!("matmul expect matched size {:?}, {:?}", self.dim, o.dim);
        }
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
                collected.sort_unstable_by(|a, b| {
                    let porder = a.0.partial_cmp(&b.0).unwrap();
                    if descending {
                        porder
                    } else {
                        porder.reverse()
                    }
                });
                let (_left, right): (Vec<_>, Vec<_>) = collected.iter().cloned().unzip();
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
        if self.size() != o.size() {
            panic!("max needs two tensor have the same size, {:?}, {:?}", self.dim, o.dim);
        }
        let mut ret = GenTensor::empty(&self.dim);

        for ((a, b), c) in self.d.iter().zip(o.d.iter()).zip(ret.d.iter_mut()) {
            if a >= b {
                *c = T::one();
            } else {
                *c = T::zero();
            }
        }
        ret
    }

    pub fn gt(&self, o: &GenTensor<T>) -> GenTensor<T> {
        if self.size() != o.size() {
            panic!("max needs two tensor have the same size, {:?}, {:?}", self.dim, o.dim);
        }
        let mut ret = GenTensor::empty(&self.dim);

        for ((a, b), c) in self.d.iter().zip(o.d.iter()).zip(ret.d.iter_mut()) {
            if a > b {
                *c = T::one();
            } else {
                *c = T::zero();
            }
        }
        ret
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
    // le
    pub fn le(&self, o: &GenTensor<T>) -> GenTensor<T> {
        if self.size() != o.size() {
            panic!("max needs two tensor have the same size, {:?}, {:?}", self.dim, o.dim);
        }
        let mut ret = GenTensor::empty(&self.dim);

        for ((a, b), c) in self.d.iter().zip(o.d.iter()).zip(ret.d.iter_mut()) {
            if a <= b {
                *c = T::one();
            } else {
                *c = T::zero();
            }
        }
        ret
    }
    // lt
    pub fn lt(&self, o: &GenTensor<T>) -> GenTensor<T> {
        if self.size() != o.size() {
            panic!("max needs two tensor have the same size, {:?}, {:?}", self.dim, o.dim);
        }
        let mut ret = GenTensor::empty(&self.dim);

        for ((a, b), c) in self.d.iter().zip(o.d.iter()).zip(ret.d.iter_mut()) {
            if a < b {
                *c = T::one();
            } else {
                *c = T::zero();
            }
        }
        ret
    }
    // max, there are 3 versions.
    pub fn max_all(&self) -> GenTensor<T> {
        unimplemented!()
    }
    pub fn max_along(&self) -> (GenTensor<T>, GenTensor<T>) {
        unimplemented!()
    }
    pub fn max(&self, o: &GenTensor<T>) -> GenTensor<T> {
        if self.size() != o.size() {
            panic!("max needs two tensor have the same size, {:?}, {:?}", self.dim, o.dim);
        }
        let mut ret = GenTensor::empty(&self.dim);

        for ((a, b), c) in self.d.iter().zip(o.d.iter()).zip(ret.d.iter_mut()) {
            if a >= b {
                *c = *a;
            } else {
                *c = *b;
            }
        }
        ret
    }
    // min, there are 3 versions.
    pub fn min_all(&self) -> GenTensor<T> {
        unimplemented!()
    }
    pub fn min_along(&self) -> (GenTensor<T>, GenTensor<T>) {
        unimplemented!()
    }
    pub fn min(&self, o: &GenTensor<T>) -> GenTensor<T> {
        if self.size() != o.size() {
            panic!("max needs two tensor have the same size, {:?}, {:?}", self.dim, o.dim);
        }
        let mut ret = GenTensor::empty(&self.dim);

        for ((a, b), c) in self.d.iter().zip(o.d.iter()).zip(ret.d.iter_mut()) {
            if a >= b {
                *c = *b;
            } else {
                *c = *a;
            }
        }
        ret
    }
    // ne
    pub fn ne(&self, o: &GenTensor<T>) -> GenTensor<T> {
        if self.size() != o.size() {
            panic!("max needs two tensor have the same size, {:?}, {:?}", self.dim, o.dim);
        }
        

        let data = self.d.iter().zip(
            o.d.iter())
            .map(|(x, y)|
                 if *x != *y {
                     T::one()
                 } else {
                     T::zero()
                 }
        ).collect();
        GenTensor {
            d: data,
            dim: self.dim.to_vec(),
        }
    }
    // sort
    // topk
    

    // higher ops
    pub fn conv2d(&self, filter: &GenTensor<T>,
                  stride: (usize, usize),
                  padding: (usize, usize),
                  dilation: (usize, usize),
                  padding_mode: usize
    ) -> GenTensor<T> {
        if self.dim.len() < 4 {
            panic!("conv2d expects input data is 4 dim tensor NCHW, but get {:?}", self.dim);
        }
        if filter.dim.len() <4 {
            panic!("conv2d expects input data is 4 dim tensor NCHW, but get {:?}", filter.dim);
        }
        if self.dim.len() != filter.dim.len() {
            panic!("covn2d expects input and filter has the same dims, get {:?}, {:?}", self.dim, filter.dim);
        }
        
        let filter_size = filter.size();
        let out_channels = filter_size[0];
        let in_channels = filter_size[1];
        let sample_size = self.dim[0];
        let data_channels = self.dim[1];


        if in_channels != data_channels {
            panic!("covn2d expects input data channel size matches depth in filter {:?}, {:?}", self.dim, filter.dim);
        }
        
        // prepare the padded input
        let mut padded_dim = Vec::new();
        for i in 2..self.dim.len() {
            padded_dim.push(i);
        }
        
        //let row_range = Vec::new();
        //let height_range = Vec::new();
        //
        //for i in row_range {
        //    for j in height_range {
        //        
        //    }
        //}
        
        GenTensor::new()
    }

    // gneral convolutional operator, should work for 2d and 3d cases.
    pub fn conv_gen(&self, filter: &GenTensor<T>,
                    stride: &[usize],
                    padding: &[usize],
                    dilation: &[usize],
                    padding_mode: PaddingMode
    ) -> GenTensor<T> {

        if self.dim.len() != filter.dim.len() {
            panic!("covn2d expects input and filter has the same dims, get {:?}, {:?}", self.dim, filter.dim);
        }
        if stride.len() != padding.len() || stride.len() != dilation.len() {
            panic!("stride, padding, stride should have the same # of dims, {:?}, {:?}, {:?}", stride, padding, dilation);
        }

        let filter_size = filter.size();
        let out_channels = filter_size[0];
        let in_channels = filter_size[1];
        let sample_size = self.dim[0];
        let data_channels = self.dim[1];
        if in_channels != data_channels {
            panic!("covn2d expects input data channel size matches depth in filter {:?}, {:?}", self.dim, filter.dim);
        }
        
        // prepare the padded input
        let mut padded_dim = Vec::new();
        for i in 2..self.dim.len() {
            padded_dim.push(self.dim[i] + padding[i-2]*2);
        }
        // println!("{:?}", padded_dim);

        // find the start point in padded dimension
        let mut start_point = Vec::new();
        for i in 0..stride.len() {
            let half = filter.dim[2+i]/2;
            let dilated = half*dilation[i];
            start_point.push(dilated);
        }
        //println!("start_point: {:?}", start_point);

        let mut output_size = Vec::new();
        //println!("{:?}, {:?}", padded_dim, stride);
        for i in 0..stride.len() {
            let mut output_dim = (padded_dim[i] - start_point[i]*2)/stride[i];
            if padded_dim[i] % 2 == 0 {
                output_dim += 1;
            }
            output_size.push(output_dim);
        }
        let mut output_tensor_size = Vec::new();
        output_tensor_size.push(sample_size);
        output_tensor_size.push(filter.dim[0]);
        output_tensor_size.append(&mut output_size.clone()); // output_size moved.
        let output_inner_size = output_size.iter().product::<usize>();
        //println!("{:?}", output_size);
        //println!("{:?}", output_inner_size);
        //println!("{:?}", output_tensor_size);
        
        let mut ret = GenTensor::<T>::empty(&output_tensor_size);

        let conv_size = filter.dim.iter().product::<usize>()/out_channels; // this is Cin xd1xd2xd3...
        let mut data_block = Vec::<T>::with_capacity(conv_size);
        unsafe{ data_block.set_len(conv_size); }
        let mut filter_block = Vec::<T>::with_capacity(conv_size);
        unsafe{ filter_block.set_len(conv_size); }

        let inner_steps = output_inner_size*out_channels;
        let filter_step = conv_size;
        
        for i in 0..sample_size {
            for j in 0..out_channels {
                filter_block.copy_from_slice(&filter.d[(j)*filter_step..(j+1)*filter_step]);

                let mut left_upper = vec![0; stride.len()];
                for k in 0..output_inner_size {
                    //println!("left_upper: {:?}", left_upper);

                    // get_data_block
                    let mut current_data_elem = left_upper.to_vec();
                    for in_channel_index in 0..in_channels {
                        for inner_index in 0..conv_size/in_channels {

                            // assign single scale to the tmp tensor.
                            let mut push_value = T::zero();
                            let mut in_margin = false;
                            for i in 0..current_data_elem.len() {
                                if current_data_elem[i] < padding[i] || current_data_elem[i] >= (padding[i] + self.dim[i+2]){
                                    push_value = T::zero();
                                    in_margin = true;
                                    break
                                }
                            }
                            if ! in_margin {
                                let real_data_elem = current_data_elem.iter().zip(padding.iter()).map(|(x, y)| x - y).collect::<Vec::<usize>>();
                                let mut real_data_elem2 = vec![i, in_channel_index];
                                real_data_elem2.append(&mut real_data_elem.clone());
                                push_value = self.get(&real_data_elem2);
                            }

                            data_block[in_channel_index*(conv_size/in_channels) + inner_index] = push_value;


                            // update to the next position.
                            let mut current_pos = current_data_elem.len()-1;
                            loop {
                                current_data_elem[current_pos] += dilation[current_pos];
                                if current_data_elem[current_pos] >= dilation[current_pos]*filter.dim[current_pos+2] + left_upper[current_pos] {
                                    current_data_elem[current_pos] = left_upper[current_pos];
                                    if current_pos > 0 {
                                        current_pos -= 1;
                                    } else {
                                        break;
                                    }
                                } else {
                                    break;
                                }
                            };
                        }
                    };
                
                    //let value = data_block.iter().zip(&filter_block).map(|(x, y)|
                    //                                                     (*x)*(*y)
                    //).sum::<T>();
                    let mut value = T::zero();
                    for (x, y) in data_block.iter().zip(&filter_block) {
                        value = value + (*x)*(*y);
                    }
                    //println!("index: {}, {}, {}", i, j, k);
                    //println!("raw index: {}", i*inner_steps + j*output_inner_size + k);
                    ret.d[i*inner_steps + j*output_inner_size + k] = value;

                    // update for next prodsum position
                    let mut current_pos = left_upper.len()-1;
                    loop {
                        left_upper[current_pos] += stride[current_pos];
                        let mut compare_pos = padded_dim[current_pos] - start_point[current_pos]*2;
                        if filter.dim[current_pos+2] % 2 == 0 {
                            compare_pos += 1;
                        }
                        if left_upper[current_pos] >= compare_pos {
                            left_upper[current_pos] = 0;
                            if current_pos > 0 {
                                current_pos -= 1;
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    };

                }
            }
        }
        
        ret
    }

    // the 1st return is the gradient for w
    // the 2nd return is the gradient for the input, given the output_grad
    pub fn conv_grad_gen(&self, filter: &GenTensor<T>,
                         stride: &[usize],
                         padding: &[usize],
                         dilation: &[usize],
                         padding_mode: PaddingMode,
                         output_grad: &GenTensor<T>
    ) -> (GenTensor<T>, GenTensor<T>) {
        if self.dim.len() <= 2 {
            panic!("input data for conv has not enough dim {:?}.", self.dim);
        }
        if filter.dim.len() <= 2 {
            panic!("filter for conv has not enough dim {:?}.", filter.dim);
        }
        if output_grad.dim.len() <= 2 {
            panic!("output gradient for conv has not enough dim {:?}.", filter.dim);
        }
        if self.dim.len() != filter.dim.len() || self.dim.len() != output_grad.dim.len() {
            panic!("covn2d expects input, output gradient and filter has the same dims, get {:?}, {:?}, {:?}", self.dim, filter.dim, output_grad.dim);
        }
        if filter.dim[1] != self.dim[1] {
            panic!("covn2d expects input data channel size matches depth in filter {:?}, {:?}", self.dim, filter.dim);
        }
        if self.dim[0] != output_grad.dim[0] {
            panic!("conv2d expects input and output has the same N: {:?}, {:?}", self.dim, output_grad.dim);
        }
        if filter.dim[0] != output_grad.dim[1] {
            panic!("conv2d expects filter and output has the same Cout: {:?}, {:?}", filter.dim, output_grad.dim);
        }
        if stride.len() != padding.len() || stride.len() != dilation.len() {
            panic!("stride, padding, stride should have the same # of dims, {:?}, {:?}, {:?}", stride, padding, dilation);
        }
        if stride.len()+2 != filter.dim.len() {
            panic!("expect the same inner size, {:?}, {:?}", stride, filter.dim);
        }
        
        let filter_size = filter.size();
        let n_c_out = filter_size[0];
        let n_c_in = filter_size[1];
        let n_n = self.dim[0];
        let n_d_dd = self.dim.iter().product::<usize>()/n_n/n_c_in;
        let n_f_dd = filter.dim.iter().product::<usize>()/n_c_out/n_c_in;
        let d_inner = self.dim.len() - 2;

        for i in 0..n_n {
            for j in 0..n_c_out {
                // left_upper in padded dimension.
                let mut left_upper = vec![0; d_inner];

                
                loop {
                    println!("{:?}", left_upper);

                    // remember where to get data.
                    // let mut data_loc = BTreeMap::<Vec::<usize>, >::new();

                    for cin_index in 0..n_c_in {
                        for dd_index in 0..n_f_dd {

                            // get current position for filter elements.
                            let mut filter_elem = Vec::new();
                            let mut reminder = dd_index;
                            for dim_pos in 0..d_inner {
                                let left_product = filter_size[dim_pos+3..filter_size.len()]
                                    .iter()
                                    .product::<usize>();
                                filter_elem.push(reminder / left_product);
                                reminder = reminder % left_product;
                            }
                            //println!("filter_elem: {:?}", filter_elem);

                            
                            // get current position for data elements in padded dimension
                            let mut data_elem = left_upper.to_vec();
                            for dim_pos in 0..d_inner {
                                data_elem[dim_pos] += filter_elem[dim_pos]*dilation[dim_pos];
                            }
                            //println!("data_elem: {:?}", data_elem);


                            // find real current position from filter
                            let mut full_filter_elem = vec![j, cin_index];
                            full_filter_elem.append(&mut filter_elem.clone());
                            // println!("filter_value: {}", filter_value.to_f32().expect(""));
                            // println!("full_filter_elem: {:?}", full_filter_elem);

                            // find real current position from data
                            let mut zero_padded_flag = false;
                            let mut unpadded_elem = data_elem.clone();
                            for dim_pos in 0..d_inner {
                                if data_elem[dim_pos] < padding[dim_pos] {
                                    match padding_mode {
                                        PaddingMode::Zeros => {
                                            zero_padded_flag = true;
                                        },
                                        PaddingMode::Reflect => {
                                            unpadded_elem[dim_pos] = padding[dim_pos] - data_elem[dim_pos] - 1;
                                        },
                                        PaddingMode::Replicate => {
                                            unpadded_elem[dim_pos] = 0;
                                        },
                                        PaddingMode::Circular => {
                                            unpadded_elem[dim_pos] = self.dim[dim_pos+2] - (padding[dim_pos] - data_elem[dim_pos]);
                                        },
                                    }
                                } else if data_elem[dim_pos] >= self.dim[dim_pos + 2] + padding[dim_pos] {
                                    match padding_mode {
                                        PaddingMode::Zeros => {
                                            zero_padded_flag = true;
                                        },
                                        PaddingMode::Reflect => {
                                            unpadded_elem[dim_pos] = self.dim[dim_pos+2] - (data_elem[dim_pos] - (self.dim[dim_pos + 2] + padding[dim_pos]) + 1);
                                        },
                                        PaddingMode::Replicate => {
                                            unpadded_elem[dim_pos] = self.dim[dim_pos + 2]-1;
                                        },
                                        PaddingMode::Circular => {
                                            unpadded_elem[dim_pos] = data_elem[dim_pos] - (self.dim[dim_pos + 2] + padding[dim_pos]);
                                        },
                                    }
                                } else {
                                    unpadded_elem[dim_pos] -= padding[dim_pos];
                                }
                            }
                            let mut full_data_elem = vec![i, cin_index];
                            full_data_elem.append(&mut unpadded_elem.clone());
                            //println!("full_data_elem: {:?}", full_data_elem);

                            
                            let filter_value = filter.get(&full_filter_elem);
                            let data_value;
                            if zero_padded_flag {
                                data_value = T::zero();
                            } else {
                                data_value = self.get(&full_data_elem);
                            }

                            
                        }
                    }

                    // update left_upper to the next position.
                    for current_pos in 0..d_inner {
                        let real_pos = d_inner - current_pos - 1;
                        left_upper[real_pos] += stride[real_pos];
                        
                        let compare_pos = self.dim[real_pos+2]
                            + padding[real_pos]*2
                            - ((filter.dim[real_pos + 2]-1)*dilation[real_pos] + 1);
                        
                        if left_upper[real_pos] > compare_pos {
                            left_upper[real_pos] = 0;
                        } else {
                            break;
                        }
                    }
                    if left_upper.iter().sum::<usize>() == 0 {
                        break;
                    }
                };
            }
        }

        
        (GenTensor::new(), GenTensor::new())
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
            write!(f, "[")?;
            for i in 0..self.dim[0] {
                write!(f, "[")?;
                for j in 0..self.dim[1] {
                    write!(f, "{}, ", self.get(&vec![i, j]))?;
                }
                write!(f, "]\n")?;
            }
            write!(f, "]\n")
        } else {
            write!(f, "{:?}\n", self.dim)?;
            write!(f, "{:?}", self.d)            
        }
    }
}
impl fmt::Display for GenTensor<f64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.dim)?;
        write!(f, "{:?}", self.d)
    }
}

impl fmt::Debug for GenTensor<f32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.dim.len() == 2 {
            write!(f, "[")?;
            for i in 0..self.dim[0] {
                write!(f, "[")?;
                for j in 0..self.dim[1] {
                    write!(f, "{}, ", self.get(&vec![i, j]))?;
                }
                write!(f, "]\n")?;
            }
            write!(f, "]\n")
        } else {
            write!(f, "{:?}\n", self.dim)?;
            write!(f, "{:?}", self.d)            
        }
    }
}
impl fmt::Debug for GenTensor<f64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dim.len() == 2 {
            write!(f, "[")?;
            for i in 0..self.dim[0] {
                write!(f, "[")?;
                for j in 0..self.dim[1] {
                    write!(f, "{}, ", self.get(&vec![i, j]))?;
                }
                write!(f, "]\n")?;
            }
            write!(f, "]\n")
        } else {
            write!(f, "{:?}\n", self.dim)?;
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
    fn test_index2dimpos() {
        let a = GenTensor::<f32>::empty(&vec![10, 5, 3, 4]);

        let b = a.index2dimpos(10);
        assert_eq!(b, vec![0, 0, 2, 2]);
    }

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
            assert_eq!(m1.get_n().get_raw(), vec![10.]);
            assert_eq!(m1.get_c().get_raw(), vec![3.]);
            assert_eq!(m1.get_h().get_raw(), vec![28.]);
            assert_eq!(m1.get_w().get_raw(), vec![30.]);

            let result = std::panic::catch_unwind(
                ||
                    m1.get_d().get_raw()
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
    fn cat() {
        let a = GenTensor::<f32>::fill(1., &vec![5, 3, 3, 2]);
        let b = GenTensor::<f32>::fill(2., &vec![5, 3, 3, 2]);
        let c = GenTensor::<f32>::fill(3., &vec![5, 3, 3, 2]);

        let d = a.cat(&vec![&b, &c], 1);
        //println!("{}", d);
        assert_eq!(d, GenTensor::new_raw(&vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 
                                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 
                                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 
                                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 
                                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], &vec![5, 9, 3, 2]));
    }

    #[test]
    fn split() {
        let a = GenTensor::<f32>::fill(1., &vec![5, 3, 3, 2]);
        let b = GenTensor::<f32>::fill(2., &vec![5, 3, 3, 2]);
        let c = GenTensor::<f32>::fill(3., &vec![5, 3, 3, 2]);

        let d = a.cat(&vec![&b, &c], 1);

        let secs = vec![3, 3, 3];
        let tensors = d.split(&secs, 1);
        //println!("{}, \n{}, \n{}", tensors[0], tensors[1], tensors[2]);
        assert_eq!(tensors[0], a);
        assert_eq!(tensors[1], b);
        assert_eq!(tensors[2], c);
    }
    
    #[test]
    fn stack() {}

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
        let a = GenTensor::<f32>::new_raw(&vec![0.0785,  1.5267, -0.8521,  0.4065,
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

    #[test]
    fn max() {
        let a = GenTensor::<f32>::new_raw(&vec![1., 3., 10., 11.], &vec![2,2]);
        let b = GenTensor::<f32>::new_raw(&vec![2., 4., 5., 6.], &vec![2,2]);
        let c = a.max(&b);
        assert_eq!(c, GenTensor::<f32>::new_raw(&vec![2., 4., 10., 11.], &vec![2,2]));
    }

    #[test]
    fn min() {
        let a = GenTensor::<f32>::new_raw(&vec![1., 3., 10., 11.], &vec![2,2]);
        let b = GenTensor::<f32>::new_raw(&vec![2., 4., 5., 6.], &vec![2,2]);
        let c = a.min(&b);
        assert_eq!(c, GenTensor::<f32>::new_raw(&vec![1., 3., 5., 6.], &vec![2,2]));
    }

    #[test]
    fn ne() {
        let a = GenTensor::<f32>::new_raw(&vec![1., 3., 10., 11.], &vec![2,2]);
        let b = GenTensor::<f32>::new_raw(&vec![2., 3., 10., 6.], &vec![2,2]);
        let c = a.ne(&b);
        assert_eq!(c, GenTensor::<f32>::new_raw(&vec![1., 0., 0., 1.], &vec![2,2]));
    }
    
    #[test]
    fn conv_gen() {

        {
            let data = GenTensor::<f32>::arange(30).reshape(&vec![2, 3, 5]);
            let filter = GenTensor::<f32>::arange(18).reshape(&vec![2, 3, 3]);
            let stride = vec![1];
            let padding = vec![0];
            let dilation = vec![1];
            let padding_mode = PaddingMode::Zeros;
            let result = data.conv_gen(&filter, &stride, &padding, &dilation, padding_mode);
            println!("output size: {:?}", result.size());
            println!("output size: {:?}", result.d);
            assert_eq!(result, GenTensor::<f32>::new_raw(&vec![312.0, 348.0, 384.0, 798.0, 915.0, 1032.0, 852.0, 888.0, 924.0, 2553.0, 2670.0, 2787.0], &vec![2, 2, 3]));
        }

        {
            let mut raw_data = Vec::new();
            for i in 0..75 {
                raw_data.push(i as f32);
            }
            let data = GenTensor::<f32>::new_raw(&raw_data, &vec![1, 3, 5, 5]);
            let mut raw_data = Vec::new();
            for i in 0..54 {
                raw_data.push(i as f32);
            }
            let filter = GenTensor::<f32>::new_raw(&raw_data, &vec![2, 3, 3, 3]);
            
            let stride = vec![1, 1];
            let padding = vec![0, 0];
            let dilation = vec![1, 1];
            let padding_mode = PaddingMode::Zeros;
            
            let result = data.conv_gen(&filter, &stride, &padding, &dilation, padding_mode);
            
            println!("output size: {:?}", result.size());
            println!("output size: {:?}", result.d);
            assert_eq!(result, GenTensor::<f32>::new_raw(&vec![15219.0, 15570.0, 15921.0, 16974.0, 17325.0, 17676.0, 18729.0, 19080.0, 19431.0, 37818.0, 38898.0, 39978.0, 43218.0, 44298.0, 45378.0, 48618.0, 49698.0, 50778.0], &vec![1, 2, 3, 3]));    
        }
        
        {
            let mut raw_data = Vec::new();
            for i in 0..60 {
                raw_data.push(i as f32);
            }
            let data = GenTensor::<f32>::new_raw(&raw_data, &vec![1, 3, 5, 4]);
            let mut raw_data = Vec::new();
            for i in 0..36 {
                raw_data.push(i as f32);
            }
            let filter = GenTensor::<f32>::new_raw(&raw_data, &vec![2, 3, 3, 2]);
            
            let stride = vec![1, 1];
            let padding = vec![0, 0];
            let dilation = vec![1, 1];
            let padding_mode = PaddingMode::Zeros;
            
            let result = data.conv_gen(&filter, &stride, &padding, &dilation, padding_mode);
            
            println!("output size: {:?}", result.size());
            println!("output size: {:?}", result.d);
            assert_eq!(result, GenTensor::<f32>::new_raw(&vec![5289.0, 5442.0, 5595.0, 5901.0, 6054.0, 6207.0, 6513.0, 6666.0, 6819.0, 13227.0, 13704.0, 14181.0, 15135.0, 15612.0, 16089.0, 17043.0, 17520.0, 17997.0], &vec![1, 2, 3, 3]));    
        }

        {
            let data = GenTensor::<f32>::arange(375).reshape(&vec![1, 3, 5, 5, 5]);
            let filter = GenTensor::<f32>::arange(162).reshape(&vec![2, 3, 3, 3, 3]);
            let stride = vec![1, 1, 1];
            let padding = vec![0, 0, 0];
            let dilation = vec![1, 1, 1];
            let padding_mode = PaddingMode::Zeros;
            let result = data.conv_gen(&filter, &stride, &padding, &dilation, padding_mode);
            println!("output size: {:?}", result.size());
            println!("output size: {:?}", result.d);
            assert_eq!(result, GenTensor::<f32>::new_raw(&vec![700704.0, 703944.0, 707184.0, 716904.0, 720144.0, 723384.0, 733104.0, 736344.0, 739584.0, 781704.0, 784944.0, 788184.0, 797904.0, 801144.0, 804384.0, 814104.0, 817344.0, 820584.0, 862704.0, 865944.0, 869184.0, 878904.0, 882144.0, 885384.0, 895104.0, 898344.0, 901584.0, 1724220.0, 1734021.0, 1743822.0, 1773225.0, 1783026.0, 1792827.0, 1822230.0, 1832031.0, 1841832.0, 1969245.0, 1979046.0, 1988847.0, 2018250.0, 2028051.0, 2037852.0, 2067255.0, 2077056.0, 2086857.0, 2214270.0, 2224071.0, 2233872.0, 2263275.0, 2273076.0, 2282877.0, 2312280.0, 2322081.0, 2331882.0], &vec![1, 2, 3, 3, 3]));
        }
    }

    #[test]
    fn conv_grad_gen() {

        {/*
            let data = GenTensor::<f32>::arange(75).reshape(&vec![1, 3, 5, 5]);
            let filter = GenTensor::<f32>::arange(54).reshape(&vec![2, 3, 3, 3]);
            let output_grad = GenTensor::<f32>::arange(18).reshape(&vec![1, 2, 3, 3]);
            
            let stride = vec![1, 1];
            let padding = vec![0, 0];
            let dilation = vec![1, 1];
            let padding_mode = PaddingMode::Zeros;
            
            let result = data.conv_grad_gen(&filter, &stride, &padding, &dilation, padding_mode, &output_grad);
        
            assert_eq!(true, true);
        */}

        {

            let data = GenTensor::<f32>::arange(60).reshape(&vec![1, 3, 5, 4]);
            let filter = GenTensor::<f32>::arange(36).reshape(&vec![2, 3, 3, 2]);
            let output_grad = GenTensor::<f32>::arange(18).reshape(&vec![1, 2, 3, 3]);
            
            let stride = vec![1, 1];
            let padding = vec![0, 0];
            let dilation = vec![1, 1];
            let padding_mode = PaddingMode::Zeros;
            
            let result = data.conv_grad_gen(&filter, &stride, &padding, &dilation, padding_mode, &output_grad);
            
            assert_eq!(true, false);
        }

    }
}
