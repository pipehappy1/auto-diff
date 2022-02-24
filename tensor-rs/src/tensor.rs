//! 
//! A general tensor type.
//! 

// extern crate ndarray;
// Default value type is f32.
// Right dimension of the tensor changes fastest.
use std::rc::Rc;
use std::cell::RefCell;
//use std::ops::Index;
use ::rand::prelude::StdRng;

#[cfg(feature = "use-serde")]
use serde::{Serialize, Deserialize};

use std::fmt;




use super::typed_tensor::TypedTensor;
use crate::tensor_impl::gen_tensor::GenTensor;

/// 2-to-1
macro_rules! tensor_method {
    ($a:ident) => {
        pub fn $a(&self, o: &Tensor) -> Tensor {
            Tensor {
                v: Rc::new(RefCell::new(self.v.borrow().$a(&o.v.borrow()))),
            }
        }
    }
}

/// 2-to-1option
macro_rules! tensor_method_2_to_1option {
    ($a:ident) => {
        pub fn $a(&self, o: &Tensor) -> Option<Tensor> {
            self.v.borrow().$a(&o.v.borrow()).map(|v| Tensor {
                v: Rc::new(RefCell::new(v))})            
        }
    }
}

/// 1-to-other
macro_rules! tensor_method_single_same_return {
    ($a:ident, $b:ty) => {
        pub fn $a(&self) -> $b {
            self.v.borrow().$a()
        }
    }
}

/// 1-to-1
macro_rules! tensor_method_single_tensor_return {
    ($a:ident) => {
        pub fn $a(&self) -> Tensor {
            Tensor {
                v: Rc::new(RefCell::new(self.v.borrow().$a())),
            }
        }
    }
}

/// 1-to-1option
macro_rules! tensor_method_1_option_tensor_return {
    ($a:ident) => {
        pub fn $a(&self) -> Option<Tensor> {
            let r = self.v.borrow().$a();
            r.map(|r1| Tensor {
                v: Rc::new(RefCell::new(r1)),
            })            
        }
    }
}

/// 1-to-2option
macro_rules! tensor_method_2_option_tensor_return {
    ($a:ident) => {
        pub fn $a(&self) -> Option<[Tensor; 2]> {
            let r = self.v.borrow().$a();
            r.map(|[r1, r2]| [Tensor {
                v: Rc::new(RefCell::new(r1)),},
                              Tensor {
                                  v: Rc::new(RefCell::new(r2)),
                              }])
        }
    }
}

/// 1-to-3option
macro_rules! tensor_method_3_option_tensor_return {
    ($a:ident) => {
        pub fn $a(&self) -> Option<[Tensor; 3]> {
            let r = self.v.borrow().$a();
            r.map(|[r1, r2, r3]| [
                Tensor {
                        v: Rc::new(RefCell::new(r1)),},
                          Tensor {
                              v: Rc::new(RefCell::new(r2)),
                          },
                          Tensor {
                              v: Rc::new(RefCell::new(r3)),
                          }
            ])
        }
    }
}



pub struct Tensor {
    v: Rc<RefCell<TypedTensor>>,
}

impl Default for Tensor {
    fn default() -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(TypedTensor::new())),
        }
    }
}

impl Tensor {
    pub fn new() -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(TypedTensor::new())),
        }
    }

    pub fn data_copy(&self, o: &Tensor) {
        self.v.borrow_mut().data_copy(&o.v.borrow());
    }

    pub fn swap(&self, o: &Tensor) {
        self.v.swap(&o.v);
    }

    pub fn ref_copy(&self) -> Tensor {
        Tensor {
            v: self.v.clone(),
        }
    }

    /// Right most is the continous indexing,
    /// This method convert continuous index to index along each dimension.
    pub fn index2dimpos(&self, index: usize) -> Vec::<usize> {
        self.v.borrow().index2dimpos(index)
    }
    /// Right most is the continous indexing,
    /// This method convert index along each dimension to continuous index.
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
    pub fn get_scale_f64(&self) -> f64 {
        self.v.borrow().get_scale_f64()
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
    pub fn set_patch(&self, other: &Tensor,
                     range: &[(usize, usize)], step: Option<&[usize]>) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().set_patch(
                &other.v.borrow(), range, step)))
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

    
    pub fn from_record_f32(&self, row: usize, record: &[f32]) -> Result<(), &'static str> {
        self.v.borrow_mut().from_record_f32(row, record)
    }
    pub fn from_record_f64(&self, row: usize, record: &[f64]) -> Result<(), &'static str> {
        self.v.borrow_mut().from_record_f64(row, record)
    }
    pub fn get_f32(&self, o: &[usize]) -> f32 {
        self.v.borrow().get_f32(o)
    }
    pub fn set_f32(&mut self, o: &[usize], v: f32) {
        self.v.borrow_mut().set_f32(o, v);
    }

    pub fn get_f64(&self, o: &[usize]) -> f64 {
        self.v.borrow().get_f64(o)
    }
    pub fn set_f64(&mut self, o: &[usize], v: f64) {
        self.v.borrow_mut().set_f64(o, v);
    }


    
    /// Returns a tensor of size size filled with fill_value.
    pub fn fill(size: &[usize], fill_value: &Tensor) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(TypedTensor::fill(size, &fill_value.v.borrow()))),
        }
    }
    pub fn fill_f32(size: &[usize], fill_value: f32) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(TypedTensor::fill_f32(size, fill_value))),
        }
    }
    pub fn fill_f64(size: &[usize], fill_value: f64) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(TypedTensor::fill_f64(size, fill_value))),
        }
    }
    
    // zeros
    pub fn zeros(dim: &[usize]) -> Tensor {
        Tensor {
            #[cfg(feature = "use-f64")]
            v: Rc::new(RefCell::new(TypedTensor::zeros_f64(dim))),
            #[cfg(feature = "use-f32")]
            v: Rc::new(RefCell::new(TypedTensor::zeros_f32(dim))),
        }
    }
    // zeros_like
    tensor_method_single_tensor_return!(zeros_like);
    // ones
    pub fn ones(dim: &[usize]) -> Tensor {
        Tensor {
            #[cfg(feature = "use-f64")]
            v: Rc::new(RefCell::new(TypedTensor::ones_f64(dim))),
            #[cfg(feature = "use-f32")]
            v: Rc::new(RefCell::new(TypedTensor::ones_f32(dim))),
        }
    }
    // ones_like
    tensor_method_single_tensor_return!(ones_like);
    // range
    pub fn range(start: f32, end: f32, step: Option<f32>) -> Tensor {
        let real_step = if let Some(v) = step {
            v
        } else {
            1.
        };

        let mut value = start;
        let mut index = 0;
        let mut data = Vec::new();
        while value <= end {
            value += real_step;
            data.push(value);
            index += 1;
        }
        
        Tensor::from_vec_f32(&data, &[index])
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
        
        Tensor::from_vec_f32(&data, &[index])
    }
    // logspace
    pub fn logspace(start: f32, end: f32, steps: usize, base: f32) -> Tensor {
        let linspace_data = Tensor::linspace(start, end, steps);
        let mut ret_data = Vec::new();
        for i in 0..linspace_data.numel() {
            ret_data.push(base.powf(linspace_data.get_f32(&[i])));
        }
        Tensor::from_vec_f32(&ret_data, &[ret_data.len()])
    }
    // eye
    pub fn eye(n: usize, m: usize) -> Tensor {
        let ret = Tensor::zeros(&[n, m]);
        for i in 0..n.min(m) {
            ret.v.borrow_mut().set_f32(&[i, i], 1.);
        }
        ret
    }
    // empty
    pub fn empty(shape: &[usize]) -> Tensor {
        for i in shape {
            if *i == 0 {
                println!("empty: shape with zeros in it.");
            }
        }
        Tensor {
            #[cfg(feature = "use-f64")]
            v: Rc::new(RefCell::new(TypedTensor::zeros_f64(shape))),
            #[cfg(feature = "use-f32")]
            v: Rc::new(RefCell::new(TypedTensor::zeros_f32(shape))),
        }
    }

    pub fn log10_like(&self) -> Tensor {
	Tensor {
	    v: Rc::new(RefCell::new(self.v.borrow().log10_like())),
	}
    }

    pub fn log2_like(&self) -> Tensor {
	Tensor {
	    v: Rc::new(RefCell::new(self.v.borrow().log2_like())),
	}
    }

    
    // Indexing, Slicing, Joining, Mutating Ops
    pub fn cat(&self, tensors: &[Tensor], dim: usize) -> Tensor {
        let mut concrete_tensor = Vec::new();
        
        for i in tensors {
            concrete_tensor.push(i.v.borrow().clone());
        }
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().cat(&concrete_tensor, dim))),
        }
    }
    pub fn chunk(&self, chunks: usize, dim: usize) -> Vec<Tensor> {
        let mut result = self.v.borrow().chunk(chunks, dim);
        let mut ret = Vec::new();
        for i in result.drain(..) {
            ret.push(Tensor {
                v: Rc::new(RefCell::new(i))
            });
        }
        ret
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
    pub fn index_exclude(&self, dim: usize, index: &Tensor) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().index_exclude(dim, &index.v.borrow()))),
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
    pub fn stack(&self, tensors: &[Tensor], dim: usize) -> Tensor {
        let mut concrete_tensor = Vec::new();
        
        for i in tensors {
            concrete_tensor.push(i.v.borrow().clone());
        }
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().stack(&concrete_tensor, dim))),
        }
    }
    pub fn t(&self) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().t()))
        }
    }
    pub fn take(&self, index: &[usize]) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().take(index)))
        }
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
    pub fn lerp(&self, end: &Tensor, weight: &Tensor) -> Tensor {
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
    tensor_method!(mul); // element-wise
    tensor_method!(div);

    tensor_method!(mm); //  matrix-multiplication
    tensor_method!(matmul); // tensor-multiplication
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
    pub fn prod(&self, dim: Option<&[usize]>, keepdim: bool) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().prod(dim, keepdim))),
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

    // linalg
    /// mean and std are all scalars.
    pub fn normalize(&self, mean: &Tensor, std: &Tensor) -> Tensor {
        if self.size().len() != 2 {
            panic!("fn normalize is for two-dimensional data.");
        }
        //let width = self.size()[1];
        //if width != mean.len() {
        //    panic!("input mean has a different size. {}, {}", width, mean.len());
        //}
        //if width != std.len() {
        //    panic!("input std has a different size. {}, {}", width, std.len());
        //}
        
        self.sub(mean).div(std)
    }
    tensor_method_single_tensor_return!(normalize_unit);

    tensor_method_2_option_tensor_return!(lu);
    tensor_method_2_to_1option!(lu_solve);
    tensor_method_2_option_tensor_return!(qr);
    tensor_method_2_option_tensor_return!(eigen);
    tensor_method_1_option_tensor_return!(cholesky);
    tensor_method_1_option_tensor_return!(det);
    tensor_method_3_option_tensor_return!(svd);
    tensor_method_1_option_tensor_return!(inv);
    tensor_method_single_tensor_return!(pinv);
    tensor_method_single_tensor_return!(tr);


    // Comparison Ops
    tensor_method!(all_close);
    pub fn arg_sort(&self, dim: usize, descending: bool) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().arg_sort(dim, descending))),
        }
    }
    tensor_method!(eq_t);
    pub fn equal(&self, o: &Tensor) -> bool {
        self.v.borrow().equal(&o.v.borrow())
    }
    tensor_method!(ge);
    tensor_method!(gt);
    tensor_method!(le);
    tensor_method!(lt);
    tensor_method!(max_pair);
    tensor_method!(min_pair);
    tensor_method!(ne);

    // rand
    pub fn rand_usize(rng: &mut StdRng,
                      dim: &[usize],
                      left: usize, right: usize) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(TypedTensor::rand_usize(rng, dim, left, right))),
        }
    }
    pub fn normal_f64(rng: &mut StdRng,
                  dim: &[usize],
                  mean: f64, std: f64) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(TypedTensor::normal_f64(rng, dim, mean, std))),
        }
    }
    pub fn normal_f32(rng: &mut StdRng,
                  dim: &[usize],
                  mean: f32, std: f32) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(TypedTensor::normal_f32(rng, dim, mean, std))),
        }
    }
    pub fn uniform_f64(rng: &mut StdRng,
                   dim: &[usize],
                   from: f64, to: f64) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(TypedTensor::uniform_f64(rng, dim, from, to)))
        }
    }
    pub fn uniform_f32(rng: &mut StdRng,
                   dim: &[usize],
                   from: f32, to: f32) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(TypedTensor::uniform_f32(rng, dim, from, to)))
        }
    }
    

    // conv ops
    pub fn conv2d(&self, weight: &Tensor,
                  stride: (usize, usize),
                  padding: (usize, usize),
                  dilation: (usize, usize),
                  padding_mode: PaddingMode
    ) -> Tensor {
        Tensor {
            v: Rc::new(RefCell::new(self.v.borrow().conv2d(&weight.v.borrow(), stride, padding, dilation, padding_mode))),
        }
    }
    pub fn conv2d_grad(&self, weight: &Tensor,
                       stride: (usize, usize),
                       padding: (usize, usize),
                       dilation: (usize, usize),
                       padding_mode: PaddingMode,
                       output_grad: &Tensor
    ) -> (Tensor, Tensor) {
        let (r1, r2) = self.v.borrow().conv2d_grad(&weight.v.borrow(), stride, padding, dilation, padding_mode, &output_grad.v.borrow());
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
        write!(f, "({:?}, )", self.v.borrow())
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
        assert_eq!(b, Tensor::from_vec_f32(&vec![0.10482848, 0.20965695, 0.31448543, 0.4193139, 0.5241424, 0.62897086,], &vec![3, 2]));
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
