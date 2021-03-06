//! 
//! A general tensor type.
//! 

// extern crate ndarray;
// Default value type is f32.
// Right dimension of the tensor changes fastest.
use std::rc::Rc;
use std::cell::RefCell;
//use std::ops::Index;

use std::fmt;

pub mod gen_tensor;
#[cfg(feature = "use-cuda")]
pub mod cuda_tensor;
#[cfg(feature = "use-cuda")]
pub mod cuda_helper;
pub mod blas;
pub mod typed_tensor;
pub mod compare_tensor;
pub mod elemwise;
pub mod index_slicing;
pub mod convolution;
pub mod reduction;


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

    pub fn index2dimpos(&self, index: usize) -> Vec::<usize> {
        self.v.borrow().index2dimpos(index)
    }
    pub fn dimpos2index(&self, dimpos: &[usize]) -> usize {
        self.v.borrow().dimpos2index(dimpos)
    }
    
    pub fn is_empty() -> bool {
        unimplemented!();
    }

    pub fn size(&self) -> Vec<usize> {
        self.v.borrow().size().clone()
    }
    tensor_method_single_same_return!(numel, usize);

    pub fn get_scale_f32(&self) -> f32 {
        self.v.borrow().get_scale_f32()
    }

    tensor_method_single_tensor_return!(get_n);
    tensor_method_single_tensor_return!(get_c);
    tensor_method_single_tensor_return!(get_d);
    tensor_method_single_tensor_return!(get_h);
    tensor_method_single_tensor_return!(get_w);
    tensor_method_single_tensor_return!(numel_tensor);

    pub fn get_patch(&self, range: &[(usize, usize)], step: Option<&[usize]>) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().get_patch(range, step)))
        }
    }

    pub fn same_shape(&self, o: &Tensor) -> bool {
        let a = self.size();
        let b = o.size();
        a == b
    }


    pub fn from_vec_usize(input: &[usize], dim: &[usize]) -> Tensor {
        let data: Vec<f32> = input.iter().map(|x| *x as f32).collect();
        Self::from_vec_f32(&data, dim)
    }
    
    /// Create a tensor from a Vec,
    /// ```
    /// # use tensor_rs::tensor::*;
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
    /// return the internal buffer
    /// May fail if the underlying data is f64
    pub fn get_raw_f32(&self) -> Vec<f32> {
        self.v.borrow().get_raw_f32()
    }
    pub fn from_vec_f64(input: &[f64], dim: &[usize]) -> Tensor {
        let data = input.to_vec();
        let idim = dim.to_vec();

        Tensor {
            //v: Rc::new(RefCell::new(TypedTensor::Typef32(GenTensor { d: data, dim: idim }))),
            v: Rc::new(RefCell::new(TypedTensor::Typef64(GenTensor::new_raw(&data, &idim) ))),
        }
    }
    /// return the internal buffer
    /// May fail if the underlying data is f32
    pub fn get_raw_f64(&self) -> Vec<f64> {
        self.v.borrow().get_raw_f64()
    }

    /// try convert to Vec<u8>, value should be between 0, 255
    pub fn get_u8(&self) -> Option<Vec<u8>> {
        self.v.borrow().get_u8()
    }

    
    pub fn from_record(&self, row: usize, record: &[f32]) -> Result<(), ()> {
        self.v.borrow_mut().from_record(row, record)
    }
    pub fn get_f32(&self, o: &[usize]) -> f32 {
        self.v.borrow().get_f32(o)
    }
    pub fn set_f32(&mut self, o: &[usize], v: f32) {
        self.v.borrow_mut().set_f32(o, v);
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
    // zeros
    pub fn zeros(dim: &[usize]) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(TypedTensor::zeros(dim))),
        }
    }
    // zeros_like
    tensor_method_single_tensor_return!(zeros_like);
    // ones
    pub fn ones(dim: &[usize]) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(TypedTensor::ones(dim))),
        }
    }
    // ones_like
    tensor_method_single_tensor_return!(ones_like);
    // range
    pub fn range(start: f32, end: f32, step: Option<f32>) -> Tensor {
        let real_step;
        if let Some(v) = step {
            real_step = v;
        } else {
            real_step = 1.;
        }

        let mut value = start;
        let mut index = 0;
        let mut data = Vec::new();
        while value <= end {
            value += real_step;
            data.push(value);
            index += 1;
        }
        
        Tensor::from_vec_f32(&data, &vec![index])
    }
    // linspace
    pub fn linspace(start: f32, end: f32, steps: usize) -> Tensor {
        let real_step = (end-start)/(steps as f32);

        let mut value = start;
        let mut index = 0;
        let mut data = Vec::new();
        while value <= end {
            value += real_step;
            data.push(value);
            index += 1;
        }
        
        Tensor::from_vec_f32(&data, &vec![index])
    }
    // logspace
    pub fn logspace(start: f32, end: f32, steps: usize, base: f32) -> Tensor {
        let linspace_data = Tensor::linspace(start, end, steps);
        let mut ret_data = Vec::new();
        for i in 0..linspace_data.numel() {
            ret_data.push(base.powf(linspace_data.get_f32(&vec![i])));
        }
        Tensor::from_vec_f32(&ret_data, &vec![ret_data.len()])
    }
    // eye
    pub fn eye(n: usize, m: usize) -> Tensor {
        let ret = Tensor::empty(&vec![n, m]);
        for i in 0..n.min(m) {
            ret.v.borrow_mut().set_f32(&vec![i, i], 1.);
        }
        ret
    }
    // empty
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

    
    // Indexing, Slicing, Joining, Mutating Ops
    pub fn cat(&self, tensors: &[&Tensor], dim: usize) -> Tensor {
        let mut concrete_tensor = Vec::new();
        
        for i in tensors {
            concrete_tensor.push(i.v.borrow().clone());
        }
        let mut converted_tensor = Vec::new();
        for i in &concrete_tensor {
            converted_tensor.push(i);
        }
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().cat(&converted_tensor[..], dim))),
        }
    }
    pub fn chunk() {
        unimplemented!();
    }
    pub fn gather(&self, dim: usize, index: &Tensor) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().gather(dim, &index.v.borrow()))),
        }
    }
    pub fn index_select(&self, dim: usize, index: &Tensor) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().index_select(dim, &index.v.borrow()))),
        }
    }
    pub fn masked_select() {
        unimplemented!();
    }
    pub fn narrow() {
        unimplemented!();
    }
    pub fn nonzero() {
        unimplemented!();
    }
    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().reshape(new_shape))),
        }
    }
    pub fn split(&self, sections: &[usize], dim: usize) -> Vec<Tensor> {
        let typts = self.v.borrow().split(sections, dim);
        let mut ret = Vec::new();
        for i in typts {
            ret.push(Tensor {
                v: Rc::new(RefCell::new(i)),
            });
        }
        ret
    }
    pub fn squeeze(&self, dim: Option<usize>) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().squeeze(dim))),
        }
    }
    pub fn stack() {
        unimplemented!();
    }
    pub fn t() {
        unimplemented!();
    }
    pub fn take() {
        unimplemented!();
    }
    pub fn transpose() {
        unimplemented!();
    }
    pub fn unbind() {
        unimplemented!();
    }

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
    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().unsqueeze(dim))),
        }
    }
    
    //pub fn condition() {} // this is pytorch where
    pub fn conditional_select(&self, x: &Tensor, y: &Tensor) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().conditional_select(&x.v.borrow(), &y.v.borrow()))),
        }
    }
    pub fn repeat(&self, dim: &[usize]) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().repeat(dim))),
        }
    }

    
    pub fn to_f64(&mut self) {}
    pub fn to_f32(&mut self) {}

    // Pointwise Ops
    tensor_method_single_tensor_return!(abs);
    tensor_method_single_tensor_return!(acos);
    tensor_method_single_tensor_return!(asin);
    tensor_method_single_tensor_return!(atan);
    tensor_method_single_tensor_return!(ceil);
    // clamp
    tensor_method_single_tensor_return!(cos);
    tensor_method_single_tensor_return!(cosh);
    tensor_method_single_tensor_return!(exp);
    tensor_method_single_tensor_return!(expm1);
    tensor_method_single_tensor_return!(floor);
    tensor_method_single_tensor_return!(frac);
    // lerp
    pub fn lerp(&self, end: &Tensor, weight: f32) -> Tensor {
        self.add(&Tensor::fill(&self.size(), weight).mul(&end.sub(self)))
    }
    tensor_method_single_tensor_return!(log);
    tensor_method_single_tensor_return!(log10);
    tensor_method_single_tensor_return!(log1p);
    tensor_method_single_tensor_return!(log1pexp);
    tensor_method_single_tensor_return!(log2);
    tensor_method_single_tensor_return!(neg);
    // pow
    pub fn pow_f32(&self, n: f32) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().pow_f32(n))),
        }
    }
    tensor_method_single_tensor_return!(reciprocal);
    tensor_method_single_tensor_return!(round);
    tensor_method_single_tensor_return!(rsqrt);
    tensor_method_single_tensor_return!(sigmoid);
    tensor_method_single_tensor_return!(sign);
    tensor_method_single_tensor_return!(sin);
    tensor_method_single_tensor_return!(sinh);
    tensor_method_single_tensor_return!(sqrt);
    tensor_method_single_tensor_return!(square);
    tensor_method_single_tensor_return!(tan);
    tensor_method_single_tensor_return!(tanh);
    tensor_method_single_tensor_return!(trunc);

    tensor_method!(add);
    tensor_method!(sub);
    tensor_method!(mul);
    tensor_method!(div);

    tensor_method!(mm);
    tensor_method!(matmul);
    pub fn outer(&self, o: &Tensor, avg: Option<bool>) -> Tensor {
            Tensor {
                v: Rc::new(RefCell::new(self.v.borrow().outer(&o.v.borrow(), avg))),
            }
        }

    // reduction ops
    pub fn argmax(&self, dim: Option<&[usize]>, keepdim: bool) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().argmax(dim, keepdim))),
        }
    }
    pub fn argmin(&self, dim: Option<&[usize]>, keepdim: bool) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().argmin(dim, keepdim))),
        }
    }
    //tensor_method_single_tensor_return!(dist);
    pub fn logsumexp(&self, dim: Option<&[usize]>, keepdim: bool) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().logsumexp(dim, keepdim))),
        }
    }
    pub fn mean(&self, dim: Option<&[usize]>, keepdim: bool) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().mean(dim, keepdim))),
        }
    }
    
    //tensor_method_single_tensor_return!(median);
    //tensor_method_single_tensor_return!(mode);
    //tensor_method_single_tensor_return!(norm);
    //tensor_method_single_tensor_return!(prod);
    pub fn std(&self, dim: Option<&[usize]>, keepdim: bool) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().std(dim, keepdim))),
        }
    }
    //tensor_method_single_tensor_return!(std_mean);
    pub fn sum(&self, dim: Option<&[usize]>, keepdim: bool) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().sum(dim, keepdim))),
        }
    }
    //tensor_method_single_tensor_return!(unique);
    //tensor_method_single_tensor_return!(unique_consecutive);
    pub fn var(&self, dim: Option<&[usize]>, keepdim: bool) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().var(dim, keepdim))),
        }
    }
    //tensor_method_single_tensor_return!(var_mean);
    pub fn max(&self, dim: Option<&[usize]>, keepdim: bool) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().max(dim, keepdim))),
        }
    }
    pub fn min(&self, dim: Option<&[usize]>, keepdim: bool) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().min(dim, keepdim))),
        }
    }

    pub fn normalize(&self, mean: &[f32], std: &[f32]) -> Tensor {
        if self.size().len() != 2 {
            panic!("fn normalize is for two-dimensional data.");
        }
        let width = self.size()[1];
        if width != mean.len() {
            panic!("input mean has a different size. {}, {}", width, mean.len());
        }
        if width != std.len() {
            panic!("input std has a different size. {}, {}", width, std.len());
        }
        
        let data_mean = self.mean(Some(&[0]), false);
        let tmp1 = self.sub(&data_mean).add(&Tensor::from_vec_f32(mean, &vec![width]));
        //let tmp1 = Tensor::from_vec_f32(mean, &vec![width]).sub(&data_mean).add(self);

        let data_std = tmp1.std(Some(&[0]), false);
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


    // Comparison Ops
    tensor_method!(all_close);
    tensor_method!(eq_t);
    tensor_method!(ge);
    tensor_method!(gt);
    tensor_method!(le);
    tensor_method!(lt);
    tensor_method!(max_pair);
    tensor_method!(min_pair);
    tensor_method!(ne);

    // conv ops
    pub fn conv2d(&self, o: &Tensor,
                  stride: (usize, usize),
                  padding: (usize, usize),
                  dilation: (usize, usize),
                  padding_mode: PaddingMode
    ) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().conv2d(&o.v.borrow(), stride, padding, dilation, padding_mode))),
        }
    }
    pub fn conv2d_grad(&self, o: &Tensor,
                       stride: (usize, usize),
                       padding: (usize, usize),
                       dilation: (usize, usize),
                       padding_mode: PaddingMode,
                       output_grad: &Tensor
    ) -> (Tensor, Tensor) {
        let (r1, r2) = self.v.borrow().conv2d_grad(&o.v.borrow(), stride, padding, dilation, padding_mode, &output_grad.v.borrow());
        (Tensor { v: Rc::new(RefCell::new(r1))},
         Tensor { v: Rc::new(RefCell::new(r2))},
        )
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


// index and slicing
//pub struct TensorView {
//    dim_index: usize,
//}
//
//impl Index<usize> for Tensor {
//    type Output = TensorView;
//
//    fn index(&self, dim_index: usize) -> &Self::Output {
//        TensorView {
//            dim_index: dim_index,
//        }
//    }
//}

#[derive(Clone, Copy, PartialEq)]
pub enum PaddingMode{
    Zeros,
    Reflect,
    Replicate,
    Circular,
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

    // test for basic ops
    #[test]
    fn test_add() {
        let m1 = Tensor::from_vec_f32(&vec![1.,2.,3.,4.,], &vec![2,2]);
        let m2 = Tensor::from_vec_f32(&vec![1.,2.,3.,4.,], &vec![2,2]);
        let m3 = m1.add(&m2);
        assert_eq!(m3.get_f32(&vec![0,0]), 2.);
        assert_eq!(m3.get_f32(&vec![1,1]), 8.);
    }

    #[test]
    fn test_mm() {
        let m1 = Tensor::from_vec_f32(&vec![1.,2.,3.,4.,5.,6.], &vec![3,2]);
        let m2 = Tensor::from_vec_f32(&vec![2.,3.,4.,5.,6.,7.], &vec![2,3]);
        let result = m1.mm(&m2);
        assert!(result == Tensor::from_vec_f32(&vec![12.,15.,18.,26.,33.,40.,40.,51.,62.,], &vec![3,3]), "");
    }

    #[test]
    fn test_matmul() {
        let m1 = Tensor::from_vec_f32(&vec![1.,2.,3.,4.,5.,6.], &vec![3,2]);
        let m2 = Tensor::from_vec_f32(&vec![2.,3.,4.,5.,6.,7.], &vec![2,3]);
        let result = m1.matmul(&m2);
        assert!(result == Tensor::from_vec_f32(&vec![12.,15.,18.,26.,33.,40.,40.,51.,62.,], &vec![3,3]), "");
    }

    #[test]
    fn test_outer() {
        let m1 = Tensor::from_vec_f32(&vec![1.,2.,3.,4.,5.,6.], &vec![3,2]);
        let m2 = Tensor::from_vec_f32(&vec![2.,3.,4.,5.,6.,7.], &vec![3,2]);
        let result = m1.outer(&m2, None);
        assert_eq!(result, Tensor::from_vec_f32(&vec![2.0, 3.0, 4.0, 6.0, 12.0, 15.0, 16.0, 20.0, 30.0, 35.0, 36.0, 42.0], &vec![3,2, 2]));
    }
}
