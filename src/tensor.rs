// extern crate ndarray;
// Default value type is f32.
// Right dimension of the tensor changes fastest.
use std::rc::Rc;
use std::cell::RefCell;

use ndarray;

use std::fmt;
use num_traits;


pub mod gen_tensor;
use gen_tensor::*;

enum TypedTensor {
    Typef32(GenTensor<f32>),
    Typef64(GenTensor<f64>),
}

macro_rules! typed_tensor_method_single_same_return {
    ($a:ident, $b:ty) => {
        fn $a(&self) -> $b {
            match &self {
                TypedTensor::Typef32(v1) => {v1.$a()},
                TypedTensor::Typef64(v1) => {v1.$a()},
                _ => {panic!("should have same tensor type!");},
            }
        }
    }
}

macro_rules! typed_tensor_method_single_tensor_return {
    ($a:ident) => {
        fn $a(&self) -> TypedTensor {
            match &self {
                TypedTensor::Typef32(v1) => {TypedTensor::Typef32(v1.$a())},
                TypedTensor::Typef64(v1) => {TypedTensor::Typef64(v1.$a())},
                _ => {panic!("should have same tensor type!");},
            }
        }
    }
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

    typed_tensor_method_single_same_return!(size, Vec<usize>);
    typed_tensor_method_single_same_return!(numel, usize);

    typed_tensor_method_single_tensor_return!(sum);
    typed_tensor_method_single_tensor_return!(get_N);
    typed_tensor_method_single_tensor_return!(get_C);
    typed_tensor_method_single_tensor_return!(get_D);
    typed_tensor_method_single_tensor_return!(get_H);
    typed_tensor_method_single_tensor_return!(get_W);
    
    fn to_f32(i: TypedTensor) {}
    fn to_f64(i: TypedTensor) {}

    fn fill(size: &Vec<usize>, fill_value: f32) -> TypedTensor {
        TypedTensor::Typef32(GenTensor::fill(fill_value, size))
    }

    fn unsqueeze(&mut self, dim: &Vec<usize>) {
        match &self {
            TypedTensor::Typef32(v1) => {v1.unsqueeze(dim)},
            TypedTensor::Typef64(v1) => {v1.unsqueeze(dim)},
            _ => {panic!("should have same tensor type!");},
        }
    }

    // ```
    // # use auto_diff::tensor::*;
    // let m1 = TypedTensor::Typef64
    // let m2 = GenTensor::<f64>::new_raw(&vec![1.,2.,3.,4.,], &vec![2,2]);
    // let m3 = m1.add(&m2);
    // assert_eq!(m3.get(&vec![0,0]), 2.);
    // assert_eq!(m3.get(&vec![1,1]), 8.);
    // ```
    typed_tensor_method!(add);
    typed_tensor_method!(sub);
    typed_tensor_method!(mul);
    typed_tensor_method!(div);



    typed_tensor_method!(mm);
    typed_tensor_method!(matmul);

}
impl fmt::Display for TypedTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypedTensor::Typef32(v) => write!(f, "({}, )", v),
            TypedTensor::Typef64(v) => write!(f, "({}, )", v),
        }
    }
}
impl PartialEq for TypedTensor {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TypedTensor::Typef32(v), TypedTensor::Typef32(v2)) => v.eq(v2),
            (TypedTensor::Typef64(v), TypedTensor::Typef64(v2)) => v.eq(v2),
            _ => {panic!("should have same tensor type!");},
        }
    }
}
impl Eq for TypedTensor {}

impl Clone for TypedTensor {
    fn clone(&self) -> Self {
        match self {
            TypedTensor::Typef32(v) => TypedTensor::Typef32(v.clone()),
            TypedTensor::Typef64(v) => TypedTensor::Typef64(v.clone()),
            _ => {panic!("should have same tensor type!");},
        }
    }
}



macro_rules! tensor_method {
    ($a:ident) => {
        pub fn $a(&self, o: &Tensor) -> Tensor {
            Tensor {
                v: Rc::new(RefCell::new(self.v.borrow().$a(&o.v.borrow()))),
            }
        }
    }
}

macro_rules! tensor_method_single_same_return {
    ($a:ident, $b:ty) => {
        pub fn $a(&self) -> $b {
            self.v.borrow().$a()
        }
    }
}

macro_rules! tensor_method_single_tensor_return {
    ($a:ident) => {
        pub fn $a(&self) -> Tensor {
            Tensor {
                v: Rc::new(RefCell::new(self.v.borrow().$a())),
            }
        }
    }
}



pub struct Tensor {
    v: Rc<RefCell<TypedTensor>>,
}
impl Tensor {
    pub fn new() -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(TypedTensor::new())),
        }
    }
    pub fn is_empty() -> bool {
        true
    }

    tensor_method_single_same_return!(size, Vec<usize>);
    tensor_method_single_same_return!(numel, usize);
    
    tensor_method_single_tensor_return!(sum);
    tensor_method_single_tensor_return!(get_N);
    tensor_method_single_tensor_return!(get_C);
    tensor_method_single_tensor_return!(get_D);
    tensor_method_single_tensor_return!(get_H);
    tensor_method_single_tensor_return!(get_W);


    /// Create a tensor from a Vec,
    /// ```
    /// # use auto_diff::tensor::*;
    /// let t1 = Tensor::from_vec_f32(&vec![0., 1., 2., 4.,], &vec![2,2]);
    /// ```
    pub fn from_vec_f32(input: &Vec<f32>, dim: &Vec<usize>) -> Tensor {
        let data = input.to_vec();
        let idim = dim.to_vec();

        Tensor {
            //v: Rc::new(RefCell::new(TypedTensor::Typef32(GenTensor { d: data, dim: idim }))),
            v: Rc::new(RefCell::new(TypedTensor::Typef32(GenTensor::new_raw(&data, &idim) ))),
        }
    }
    pub fn to_vec_f32(&mut self) -> Vec<f32> {
        //let mut data = Vec::<f32>::new();
        //if let TypedTensor::Typef32(gt) = *self.v.borrow() {
        //    for item in &gt.d {
        //        data.push(item.clone())
        //    }
        //} else {
        //    ()
        //}
        //data
        Vec::new()
    }
    pub fn from_vec_f64(i: &Vec<f64>) -> Tensor {
        Tensor::new()
    }

    pub fn swap(&self, o: Tensor) {
        self.v.swap(&o.v);
    }
    
    /// Returns a tensor of size size filled with fill_value.
    pub fn fill(size: &Vec<usize>, fill_value: f32) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(TypedTensor::fill(size, fill_value))),
        }
    }
    pub fn fill_like() -> Tensor {
        Tensor::new()
    }
    pub fn empty() -> Tensor {
        // <- this will no work. As there must be sth.
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
    
    /// Returns a new tensor with a dimension of size one inserted at the specified position.
    /// 
    /// The returned tensor shares the same underlying data with this tensor.
    ///
    /// 
    pub fn unsqueeze(&mut self, dim: &Vec<usize>) -> &Tensor {
        self.v.borrow_mut().unsqueeze(dim);
        self
    }
    
    pub fn condition() {} // this is pytorch where

    
    pub fn to_f64(&mut self) {}
    pub fn to_f32(&mut self) {}

    tensor_method!(add);
    tensor_method!(sub);
    tensor_method!(mul);
    tensor_method!(div);

    tensor_method!(mm);
    tensor_method!(matmul);
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, )", self.v.borrow())
    }
}
impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, )", self.v.borrow())
    }
}
impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.v.eq(&other.v)
    }
}
impl Eq for Tensor {}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().clone())),
        }
    }
}
