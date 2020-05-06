//! 
//! A general tensor type.
//! 

// extern crate ndarray;
// Default value type is f32.
// Right dimension of the tensor changes fastest.
use std::rc::Rc;
use std::cell::RefCell;

use std::fmt;

pub mod gen_tensor;
pub mod typed_tensor;


use typed_tensor::TypedTensor;
use gen_tensor::GenTensor;


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

    pub fn get_scale_f32(&self) -> f32 {
        self.v.borrow().get_scale_f32()
    }

    tensor_method_single_tensor_return!(get_N);
    tensor_method_single_tensor_return!(get_C);
    tensor_method_single_tensor_return!(get_D);
    tensor_method_single_tensor_return!(get_H);
    tensor_method_single_tensor_return!(get_W);
    tensor_method_single_tensor_return!(numel_tensor);



    pub fn same_shape(&self, o: &Tensor) -> bool {
        let a = self.size();
        let b = o.size();
        a == b
    }


    /// Create a tensor from a Vec,
    /// ```
    /// # use auto_diff::tensor::*;
    /// let t1 = Tensor::from_vec_f32(&vec![0., 1., 2., 4.,], &vec![2,2]);
    /// ```
    pub fn from_vec_f32(input: &[f32], dim: &[usize]) -> Tensor {
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
    pub fn from_vec_f64(i: &[f64]) -> Tensor {
        Tensor::new()
    }
    pub fn from_record(&self, row: usize, record: &[f32]) -> Result<(), ()> {
        self.v.borrow_mut().from_record(row, record)
    }

    pub fn swap(&self, o: Tensor) {
        self.v.swap(&o.v);
    }
    
    /// Returns a tensor of size size filled with fill_value.
    pub fn fill(size: &[usize], fill_value: f32) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(TypedTensor::fill(size, fill_value))),
        }
    }
    pub fn fill_like() -> Tensor {
        Tensor::new()
    }
    pub fn empty(shape: &[usize]) -> Tensor {
        for i in shape {
            if *i == 0 {
                println!("");
            }
        }
        Tensor {
            v: Rc::new(RefCell::new(TypedTensor::empty(shape))),
        }
    }
    pub fn new_ones(dim: &[u32]) -> Tensor {
        Tensor::new()
    }
    pub fn new_zeros(dim: &[u32]) -> Tensor {
        Tensor::new()
    }
    tensor_method_single_tensor_return!(zeros_like);
    tensor_method_single_tensor_return!(ones_like);
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

    pub fn permute(&self, dim: &[usize]) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().permute(dim))),
        }
    }
    
    /// Returns a new tensor with a dimension of size one inserted at the specified position.
    /// 
    /// The returned tensor shares the same underlying data with this tensor.
    ///
    /// 
    pub fn unsqueeze(&mut self, dim: &[usize]) -> &Tensor {
        self.v.borrow_mut().unsqueeze(dim);
        self
    }
    
    pub fn condition() {} // this is pytorch where

    
    pub fn to_f64(&mut self) {}
    pub fn to_f32(&mut self) {}

    tensor_method_single_tensor_return!(neg);
    tensor_method_single_tensor_return!(log1pexp);
    tensor_method_single_tensor_return!(sigmoid);

    tensor_method!(add);
    tensor_method!(sub);
    tensor_method!(mul);
    tensor_method!(div);

    tensor_method!(mm);
    tensor_method!(matmul);
    tensor_method!(outer);

    // reduction ops
    //tensor_method_single_tensor_return!(argmax);
    //tensor_method_single_tensor_return!(argmin);
    //tensor_method_single_tensor_return!(dist);
    //tensor_method_single_tensor_return!(logsumexp);
    pub fn mean(&self, dim: usize, keepdim: bool) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().mean(dim, keepdim))),
        }
    }
    
    //tensor_method_single_tensor_return!(median);
    //tensor_method_single_tensor_return!(mode);
    //tensor_method_single_tensor_return!(norm);
    //tensor_method_single_tensor_return!(prod);
    pub fn std(&self, dim: usize, keepdim: bool) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().std(dim, keepdim))),
        }
    }
    //tensor_method_single_tensor_return!(std_mean);
    tensor_method_single_tensor_return!(sum);
    //tensor_method_single_tensor_return!(unique);
    //tensor_method_single_tensor_return!(unique_consecutive);
    pub fn var(&self, dim: usize, keepdim: bool) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().var(dim, keepdim))),
        }
    }
    //tensor_method_single_tensor_return!(var_mean);

    pub fn normalize(&self, mean: &[f32], std: &[f32]) -> Tensor {
        if self.size().len() != 2 {
            panic!("fn normalize is for two-dimensional data.");
        }
        let width = self.size()[1];
        let total = self.size()[0];
        if width != mean.len() {
            panic!("input mean has a different size. {}, {}", width, mean.len());
        }
        if width != std.len() {
            panic!("input std has a different size. {}, {}", width, std.len());
        }
        
        let data_mean = self.mean(0, false);
        let tmp1 = self.sub(&data_mean).add(&Tensor::from_vec_f32(mean, &vec![width]));
        //let tmp1 = Tensor::from_vec_f32(mean, &vec![width]).sub(&data_mean).add(self);

        let data_std = tmp1.std(0, false);
        //println!("data_std: {:?}, tmp1: {:?}", data_std, tmp1);
        let tmp2 = tmp1.div(&data_std);
        tmp2
    }
    pub fn normalize_unit(&self) -> Tensor {
        if self.size().len() != 2 {
            panic!("fn normalize is for two-dimensional data.");
        }
        self.normalize(&vec![0. ; self.size()[self.size().len()-1]],
                       &vec![1. ; self.size()[self.size().len()-1]])
    }
    
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.v.borrow())
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_equal() {
        let a = Tensor::from_vec_f32(&vec![1., 2., 3., ], &vec![3, 1]);
        let b = Tensor::from_vec_f32(&vec![1., 2., 3., ], &vec![3, 1]);
        assert_eq!(a.same_shape(&b), true);

        let a = Tensor::from_vec_f32(&vec![1., 2., 3., ], &vec![1, 3]);
        let b = Tensor::from_vec_f32(&vec![1., 2., 3., ], &vec![3, 1]);
        assert_eq!(a.same_shape(&b), false);
    }

    #[test]
    fn normalize() {
        let a = Tensor::from_vec_f32(&vec![1., 2., 3., 4., 5., 6., ], &vec![3, 2]);
        let b = a.normalize_unit();
        assert_eq!(b, Tensor::from_vec_f32(&vec![-1.2247448, -1.2247448, 0.,0., 1.2247448, 1.2247448], &vec![3, 2]));
    }
}
