use std::fmt;
use num_traits;

pub struct GenTensor<T> {
    d: Vec<T>,
    dim: Vec<u32>,
}
impl<T> GenTensor<T> where T: num_traits::Float {
    fn new() -> GenTensor<T> {
        GenTensor { d: Vec::<T>::new(), dim: Vec::new() }
    }
    pub fn new_raw(data: &Vec<T>, shape: &Vec<u32>) -> GenTensor<T> {
        let mut new_data = data.to_vec();
        let mut new_dim = shape.to_vec();
        GenTensor {
            d: new_data,
            dim: new_dim,
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
    pub fn stride(&self) -> Vec<u32> {
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
    pub fn get(&self, o: &Vec<u32>) -> T {
        let stride = self.stride();
        let dsize = o.len();
        let mut index = 0;
        for i in 0..dsize {
            index += (stride[i]*o[i]) as usize;
        }
        self.d[index]
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
        for item in self.d.iter().zip(o.d.iter()) {
            let (v1, v2) = item;
            ret.d.push(*v1 + *v2);
        }
        ret
    }
    fn sub(&self, o: GenTensor<T>) -> GenTensor<T> {
        let mut ret = GenTensor {
            d: Vec::with_capacity(self.d.len()),
            dim: self.dim.clone(),
        };
        for item in self.d.iter().zip(o.d.iter()) {
            let (v1, v2) = item;
            ret.d.push(*v1 - *v2);
        }
        ret
    }
    fn mul(&self, o: GenTensor<T>) -> GenTensor<T> {
        let mut ret = GenTensor {
            d: Vec::with_capacity(self.d.len()),
            dim: self.dim.clone(),
        };
        for item in self.d.iter().zip(o.d.iter()) {
            let (v1, v2) = item;
            ret.d.push(*v1 * *v2);
        }
        ret
    }
    fn div(&self, o: GenTensor<T>) -> GenTensor<T> {
        let mut ret = GenTensor {
            d: Vec::with_capacity(self.d.len()),
            dim: self.dim.clone(),
        };
        for item in self.d.iter().zip(o.d.iter()) {
            let (v1, v2) = item;
            ret.d.push(*v1 / *v2);
        }
        ret
    }

    fn mm(&self, o: GenTensor<T>) {
        
    }
}

impl<T> fmt::Display for GenTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0")
    }
}

enum TypedTensor {
    Typef32(GenTensor<f32>),
    Typef64(GenTensor<f64>),
}
impl TypedTensor {
    fn new() -> TypedTensor {
        // Default value type is f32.
        TypedTensor::Typef32(GenTensor::new())
    }
    fn to_f32(i: TypedTensor) {}
    fn to_f64(i: TypedTensor) {}

    fn add(&self, o: &TypedTensor) -> TypedTensor {
        match (&self, o) {
            (TypedTensor::Typef32(v1), TypedTensor::Typef32(v2)) => {TypedTensor::Typef32(v1.add(v2))},
            (TypedTensor::Typef64(v1), TypedTensor::Typef64(v2)) => {TypedTensor::Typef64(v1.add(v2))},
            _ => {panic!("should have same tensor type!");},
        }
    }
    fn sub(&self, o: &TypedTensor) {}
    fn mul(&self, o: &TypedTensor) {}
    fn div(&self, o: &TypedTensor) {}
}
impl fmt::Display for TypedTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypedTensor::Typef32(v) => write!(f, "({}, )", v),
            TypedTensor::Typef64(v) => write!(f, "({}, )", v),
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
    pub fn from_vec_f32(input: &Vec<f32>, dim: &Vec<u32>) -> Tensor {
        let mut data = Vec::<f32>::new();
        for item in input {
            data.push(item.clone());
        }
        let mut idim = Vec::<u32>::new();
        for item in dim {
            idim.push(item.clone());
        }

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

    pub fn add(&self, o: &Tensor) {
        
    }
    pub fn sub() {}
    pub fn mul() {}
    pub fn div() {}
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, )", self.v)
    }
}
