// extern crate ndarray;
use ndarray;

use std::fmt;
use num_traits;


/// Naive tensor implementation, single thread, no check.
pub struct GenTensor<T> {
    d: Vec<T>,
    dim: Vec<usize>,
}
impl<T> GenTensor<T> where T: num_traits::Float {
    fn new() -> GenTensor<T> {
        GenTensor { d: Vec::<T>::new(), dim: Vec::new() }
    }

    /// Create a tensor with given Vec.
    pub fn new_raw(data: &Vec<T>, shape: &Vec<usize>) -> GenTensor<T> {
        let mut new_data = data.to_vec();
        let mut new_dim = shape.to_vec();
        GenTensor {
            d: new_data,
            dim: new_dim,
        }
    }
    /// dump the underlying vec
    pub fn get_raw(&self) -> Vec<T> {
        self.d.to_vec()
    }

    /// Create a tensor filled with the same value d
    ///
    /// ```
    /// # use auto_diff::tensor::*;
    /// let m1 = GenTensor::<f64>::new_full(1., &vec![3,5,2]);
    /// ```
    pub fn new_full(d: T, shape: &Vec<usize>) -> GenTensor<T> {
        let mut dsize = 0;
        for i in shape {
            dsize *= (*i);
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
    /// # use auto_diff::tensor::*;
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
    /// # use auto_diff::tensor::*;
    /// let m1 = GenTensor::<f64>::new_raw(&vec![1.,2.,3.,4.,5.,6.], &vec![2,3]);
    /// assert_eq!(m1.get(&vec![1,1]), 5.);
    /// ```
    pub fn get(&self, o: &Vec<usize>) -> T {
        let stride = self.stride();
        let dsize = o.len();
        let mut index = 0;
        for i in 0..dsize {
            index += (stride[i]*o[i]);
        }
        self.d[index]
    }

    /// Returns the size of the self tensor.
    pub fn size(&self) -> Vec<usize> {
        self.dim.to_vec()
    }

    /// Returns the total number of elements in the input tensor.
    pub fn numel(&self) -> usize {
        self.d.len()
    }
    
    /// element-wise add.
    ///
    /// ```
    /// # use auto_diff::tensor::*;
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
        for (v1, v2) in self.d.iter().zip(o.d.iter()) {
            ret.d.push(*v1 + *v2);
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
    /// # use auto_diff::tensor::*;
    /// let m1 = GenTensor::<f64>::new_raw(&vec![1.,2.,3.,4.,5.,6.], &vec![3,2]);
    /// let m2 = GenTensor::<f64>::new_raw(&vec![2.,3.,4.,5.,6.,7.], &vec![2,3]);
    /// let mut result = m1.mm(&m2);
    /// assert!(result == GenTensor::<f64>::new_raw(&vec![12.,15.,18.,26.,33.,40.,40.,51.,62.,], &vec![3,3]), "");
    /// ```
    pub fn mm(&self, o: &GenTensor<T>) -> GenTensor<T>{
        let ls = self.dim[0];
        let rs = o.dim[1];
        let mut ret = GenTensor {
            d: Vec::with_capacity((ls*rs)),
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
            rloop *= self.dim[i];
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

    /// Concatenates sequence of tensors along a new dimension.
    ///
    /// All tensors need to be of the same size.
    /// ```
    /// # use auto_diff::tensor::*;
    /// let m1 = GenTensor::<f64>::new_raw(&vec![1.,2.,3.,4.,5.,6.], &vec![3,2]);
    /// let m2 = GenTensor::<f64>::new_raw(&vec![2.,3.,4.,5.,6.,7.], &vec![3,2]);
    /// let result = GenTensor::<f64>::stack(&vec![&m1, &m2], 1);
    /// let raw = result.get_raw();
    /// for i in raw {
    ///     println!("{}", i);
    /// }
    /// assert_eq!(result.size(), vec![3,2,2]);
    /// ```
    pub fn stack(tensors: &Vec<&Self>, dim: usize) -> GenTensor<T> {
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
        
        if dim == 0 {
            for i in 0..tensors.len() {
                for j in 0..tensors[0].dim.len() {
                    ret.d.push(tensors[i].d[j]);
                }
            }
        } else if dim == tensors[0].dim.len() {
            for j in 0..tensors[0].dim.len() {
                for i in 0..tensors.len() {
                    ret.d.push(tensors[i].d[j]);
                }
            }
        } else {
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
        }
        ret
    }

    /// Permute the dimensions of this tensor.
    pub fn permute(&mut self, dims: &Vec<usize>) {
        let stride = self.stride();
        let d = Vec::with_capacity(self.d.len());
        
    }

    /// Computes element-wise equality
    /// use eq_t instead, as eq is reserved for == overloading.
    ///
    /// ```
    /// # use auto_diff::tensor::*;
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
    /// # use auto_diff::tensor::*;
    /// let m1 = GenTensor::<f64>::new_full(1., &vec![3,5,2]);
    /// let m2 = GenTensor::<f64>::new_full(1., &vec![3,5,2]);
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
/// # use auto_diff::tensor::*;
/// let m1 = GenTensor::<f64>::new_full(1., &vec![3,5,2]);
/// let m2 = GenTensor::<f64>::new_full(1., &vec![3,5,2]);
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

impl<T> fmt::Display for GenTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0")
    }
}




enum TypedTensor {
    Typef32(GenTensor<f32>),
    Typef64(GenTensor<f64>),
}

macro_rules! typed_tensor_method {
    ($a:ident) => {
        fn $a(&self, o: &TypedTensor) -> TypedTensor {
            match (&self, o) {
                (TypedTensor::Typef32(v1), TypedTensor::Typef32(v2)) => {TypedTensor::Typef32(v1.$a(v2))},
                (TypedTensor::Typef64(v1), TypedTensor::Typef64(v2)) => {TypedTensor::Typef64(v1.$a(v2))},
                _ => {panic!("should have same tensor type!");},
            }
        }
    }
    
}

impl TypedTensor {
    fn new() -> TypedTensor {
        // Default value type is f32.
        TypedTensor::Typef32(GenTensor::new())
    }
    fn to_f32(i: TypedTensor) {}
    fn to_f64(i: TypedTensor) {}

    /// ```
    /// # use auto_diff::tensor::*;
    /// let m1 = TypedTensor::Typef64
    /// let m2 = GenTensor::<f64>::new_raw(&vec![1.,2.,3.,4.,], &vec![2,2]);
    /// let m3 = m1.add(&m2);
    /// assert_eq!(m3.get(&vec![0,0]), 2.);
    /// assert_eq!(m3.get(&vec![1,1]), 8.);
    /// ```
    typed_tensor_method!(add);
    typed_tensor_method!(sub);
    typed_tensor_method!(mul);
    typed_tensor_method!(div);

}
impl fmt::Display for TypedTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypedTensor::Typef32(v) => write!(f, "({}, )", v),
            TypedTensor::Typef64(v) => write!(f, "({}, )", v),
        }
    }
}


macro_rules! tensor_method {
    ($a:ident) => {
        pub fn $a(&self, o: &Tensor) -> Tensor {
            Tensor {
                v: self.v.$a(&o.v),
            }
        }
    }
}

pub struct Tensor {
    v: TypedTensor,
}
impl Tensor {
    pub fn new() -> Tensor {
        Tensor {
            v: TypedTensor::new(),
        }
    }

    /// Create a tensor from a Vec,
    /// ```
    /// # use auto_diff::tensor::*;
    /// let t1 = Tensor::from_vec_f32(&vec![0., 1., 2., 4.,], &vec![2,2]);
    /// ```
    pub fn from_vec_f32(input: &Vec<f32>, dim: &Vec<usize>) -> Tensor {
        let mut data = input.to_vec();
        let mut idim = dim.to_vec();

        Tensor {
            v: TypedTensor::Typef32(GenTensor { d: data, dim: idim }),
        }
    }
    pub fn to_vec_f32(&mut self) -> Vec<f32> {
        let mut data = Vec::<f32>::new();
        if let TypedTensor::Typef32(gt) = &self.v {
            for item in &gt.d {
                data.push(item.clone())
            }
        } else {
            ()
        }
        data
    }
    pub fn from_vec_f64(i: &Vec<f64>) -> Tensor {
        Tensor::new()
    }
    pub fn full() -> Tensor {
        Tensor::new()
    }
    pub fn full_like() -> Tensor {
        Tensor::new()
    }
    pub fn empty() -> Tensor {
        // <- this will no work.
        Tensor::new()
    }
    pub fn new_ones(dim: &Vec<u32>) -> Tensor {
        Tensor::new()
    }
    pub fn new_zeros(dim: &Vec<u32>) -> Tensor {
        Tensor::new()
    }
    pub fn zeros_like(o: &Tensor) -> Tensor {
        Tensor::new()
    }
    pub fn ones_like(o: &Tensor) -> Tensor {
        Tensor::new()
    }
    pub fn range(start: f64, step: f64) -> Tensor {
        Tensor::new()
    }
    pub fn linespace(start: f64, end: f64, steps: u32) -> Tensor {
        Tensor::new()
    }
    pub fn logspace(start: f64, end: f64, steps: u32, base: f64) -> Tensor {
        Tensor::new()
    }
    pub fn eye(n: u32, m: u32) -> Tensor {
        Tensor::new()
    }

    pub fn cat() {}
    pub fn chunk() {}
    pub fn gather() {}
    pub fn index_select() {}
    pub fn masked_select() {}
    pub fn narrow() {}
    pub fn nonzero() {}
    pub fn reshape() {}
    pub fn split() {}
    pub fn squeeze() {}
    pub fn stack() {}
    pub fn t() {}
    pub fn take() {}
    pub fn transpose() {}
    pub fn unbind() {}
    pub fn unsqueeze() {}
    pub fn condition() {} // this is pytorch where

    pub fn to_f64(&mut self) {}
    pub fn to_f32(&mut self) {}

    tensor_method!(add);
    tensor_method!(sub);
    tensor_method!(mul);
    tensor_method!(div);
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, )", self.v)
    }
}
