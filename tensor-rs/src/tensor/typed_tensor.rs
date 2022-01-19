use std::fmt;
use std::mem::discriminant;

#[cfg(feature = "use-serde")]
use serde::{Serialize, Deserialize};

use super::gen_tensor::*;
#[cfg(feature = "use-cuda")]
use super::cuda_tensor::*;
use crate::tensor::PaddingMode;
use super::compare_tensor::CompareTensor;
use super::elemwise::ElemwiseTensorOp;
use super::index_slicing::IndexSlicing;
use super::convolution::{Convolution};
#[cfg(feature = "use-blas")]
use super::convolution::{gemm_conv_f32, gemm_conv_f64};
use super::reduction::ReduceTensor;

#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub enum TypedTensor {
    Typef32(GenTensor<f32>),
    Typef64(GenTensor<f64>),
    #[cfg(feature = "use-cuda")]
    Cudaf32(CudaTensor),
}

// used for size alike ops
macro_rules! typed_tensor_method_single_same_return {
    ($a:ident, $b:ty) => {
        pub fn $a(&self) -> $b {
            match &self {
                TypedTensor::Typef32(v1) => {v1.$a()},
                TypedTensor::Typef64(v1) => {v1.$a()},
                #[cfg(feature = "use-cuda")]
                TypedTensor::Cudaf32(v1) => {v1.$a()},
                //_ => {panic!("should have same tensor type!");},
            }
        }
    }
}

// used for log alike ops
macro_rules! typed_tensor_method_single_tensor_return {
    ($a:ident) => {
        pub fn $a(&self) -> TypedTensor {
            match &self {
                TypedTensor::Typef32(v1) => {TypedTensor::Typef32(v1.$a())},
                TypedTensor::Typef64(v1) => {TypedTensor::Typef64(v1.$a())},
                #[cfg(feature = "use-cuda")]
                TypedTensor::Cudaf32(v1) => {TypedTensor::Cudaf32(v1.$a())},
                //_ => {panic!("should have same tensor type!");},
            }
        }
    }
}

// used for 2in1
macro_rules! typed_tensor_method {
    ($a:ident) => {
        pub fn $a(&self, o: &TypedTensor) -> TypedTensor {
            match (&self, o) {
                (TypedTensor::Typef32(v1), TypedTensor::Typef32(v2)) => {TypedTensor::Typef32(v1.$a(v2))},
                (TypedTensor::Typef64(v1), TypedTensor::Typef64(v2)) => {TypedTensor::Typef64(v1.$a(v2))},
                #[cfg(feature = "use-cuda")]
                (TypedTensor::Cudaf32(v1), TypedTensor::Cudaf32(v2)) => {TypedTensor::Cudaf32(v1.$a(v2))},
                _ => {panic!("should have same tensor type!");},
            }
        }
    }
}

impl Default for TypedTensor {
    fn default() -> TypedTensor {
        TypedTensor::Typef64(GenTensor::new())
    }
}

impl TypedTensor {
    pub fn new() -> TypedTensor {
        // Default value type is f32.
        TypedTensor::Typef32(GenTensor::new())
    }

    pub fn index2dimpos(&self, index: usize) -> Vec::<usize> {
        match self {
            TypedTensor::Typef32(v1) => {v1.index2dimpos(index)},
            TypedTensor::Typef64(v1) => {v1.index2dimpos(index)},
            //_ => {panic!("should have same tensor type!");},
        }
    }
    pub fn dimpos2index(&self, dimpos: &[usize]) -> usize {
        match self {
            TypedTensor::Typef32(v1) => {v1.dimpos2index(dimpos)},
            TypedTensor::Typef64(v1) => {v1.dimpos2index(dimpos)},
            //_ => {panic!("should have same tensor type!");},
        }
    }

    pub fn zeros(shape: &[usize]) -> TypedTensor {
        TypedTensor::Typef32(GenTensor::<f32>::zeros(shape))
    }
    typed_tensor_method_single_tensor_return!(zeros_like);
    pub fn ones(shape: &[usize]) -> TypedTensor {
        TypedTensor::Typef32(GenTensor::<f32>::ones(shape))
    }
    typed_tensor_method_single_tensor_return!(ones_like);
    pub fn empty(shape: &[usize]) -> TypedTensor {
        TypedTensor::Typef32(GenTensor::<f32>::zeros(shape))
    }
    pub fn fill(size: &[usize], fill_value: f32) -> TypedTensor {
        TypedTensor::Typef32(GenTensor::fill(fill_value, size))
    }
    pub fn from_record(&mut self, row: usize, record: &[f32]) -> Result<(), &'static str>{
        match self {
            TypedTensor::Typef32(v1) => {v1.from_record(row, record)},
            TypedTensor::Typef64(v1) => {v1.from_record(row, record)},
            //_ => {panic!("should have same tensor type!");},
        }
    }

    pub fn get_f32(&self, o: &[usize]) -> f32 {
        match &self {
            TypedTensor::Typef32(v1) => {v1.get(o)},
            TypedTensor::Typef64(v1) => {v1.get(o) as f32},
            //_ => {panic!("should have same tensor type!");},
        }
    }
    // pub fn get_f32() -> f32 {}
    pub fn set_f32(&mut self, o: &[usize], v: f32) {
        match self {
            TypedTensor::Typef32(v1) => {v1.set(o, v)},
            TypedTensor::Typef64(v1) => {v1.set(o, v as f64)},
            //_ => {panic!("should have same tensor type!");},
        }
    }

    typed_tensor_method_single_same_return!(size, &Vec<usize>);
    typed_tensor_method_single_same_return!(numel, usize);
    pub fn get_scale_f32(&self) -> f32 {
        match &self {
            TypedTensor::Typef32(v1) => {v1.get_scale()},
            TypedTensor::Typef64(v1) => {v1.get_scale() as f32},
            //_ => {panic!("should have same tensor type!");},
        }
    }

    
    typed_tensor_method_single_tensor_return!(get_n);
    typed_tensor_method_single_tensor_return!(get_c);
    typed_tensor_method_single_tensor_return!(get_d);
    typed_tensor_method_single_tensor_return!(get_h);
    typed_tensor_method_single_tensor_return!(get_w);
    typed_tensor_method_single_tensor_return!(numel_tensor);

    pub fn get_patch(&self, range: &[(usize, usize)], step: Option<&[usize]>) -> TypedTensor {
        match &self {
            TypedTensor::Typef32(v1) => {TypedTensor::Typef32(v1.get_patch(range, step))},
            TypedTensor::Typef64(v1) => {TypedTensor::Typef64(v1.get_patch(range, step))},
            //_ => {panic!("should have same tensor type!");},
        }
    }

    /// convert itself to f32
    pub fn to_gentensorf32(_i: &TypedTensor) -> TypedTensor {
        unimplemented!();
    }
    /// convert itself to f64
    pub fn to_gentensorf64(_i: &TypedTensor) -> TypedTensor {
        unimplemented!();
    }

    pub fn get_raw_f32(&self) -> Vec<f32> {
        match &self {
            TypedTensor::Typef32(v1) => v1.clone().get_raw(),
            _ => panic!("This is not f32 tensor"),
        }
    }
    pub fn get_raw_f64(&self) -> Vec<f64> {
        match &self {
            TypedTensor::Typef64(v1) => v1.clone().get_raw(),
            _ => panic!("This is not f64 tensor"),
        }
    }
    pub fn get_u8(&self) -> Option<Vec<u8>> {
        match &self {
            TypedTensor::Typef32(v1) => v1.get_u8(),
            TypedTensor::Typef64(v1) => v1.get_u8(),
            //_ => panic!("This is not f64 tensor"),
        }
    }

    // Indexing, Slicing, Joining, Mutating Ops
    pub fn cat(&self, tensors: &[&TypedTensor], dim: usize) -> TypedTensor {
        match &self {
            TypedTensor::Typef32(v1) => {
                let mut converted_tensor = Vec::new();
                for i in tensors {
                    if discriminant(*i) == discriminant(&TypedTensor::Typef32(GenTensor::<f32>::new())) {
                        let tmp_ref;
                        match i {
                            TypedTensor::Typef32(v1) => {tmp_ref = v1;},
                            TypedTensor::Typef64(_v1) => {panic!("");},
                            //_ => panic!("Other case"),
                        }
                        converted_tensor.push(tmp_ref);
                    } else {
                        unimplemented!();
                    }
                }
                TypedTensor::Typef32(v1.cat(&converted_tensor[..], dim))
            },
            TypedTensor::Typef64(v1) => {
                let mut converted_tensor = Vec::new();
                for i in tensors {
                    if discriminant(*i) == discriminant(&TypedTensor::Typef64(GenTensor::<f64>::new())) {
                        let tmp_ref;
                        match i {
                            TypedTensor::Typef64(v1) => {tmp_ref = v1;},
                            TypedTensor::Typef32(_v1) => {panic!("");},
                            //_ => panic!("Other case"),
                        }
                        converted_tensor.push(tmp_ref);
                    } else {
                        unimplemented!();
                    }
                }
                TypedTensor::Typef64(v1.cat(&converted_tensor[..], dim))
            },
            //_ => panic!("Other case"),
        }
    }

    pub fn gather(&self, dim: usize, index: &TypedTensor) -> TypedTensor {
        match (self, index) {
            (TypedTensor::Typef32(v1), TypedTensor::Typef32(v2)) => {
                TypedTensor::Typef32(v1.gather(dim, v2))
            },
            (TypedTensor::Typef64(v1), TypedTensor::Typef64(v2)) => {
                TypedTensor::Typef64(v1.gather(dim, v2))
            },
            _ => {panic!("should have same tensor type!");},
        }
    }
    pub fn index_select(&self, dim: usize, index: &TypedTensor) -> TypedTensor {
        match (self, index) {
            (TypedTensor::Typef32(v1), TypedTensor::Typef32(v2)) => {
                TypedTensor::Typef32(v1.index_select(dim, v2))
            },
            (TypedTensor::Typef64(v1), TypedTensor::Typef64(v2)) => {
                TypedTensor::Typef64(v1.index_select(dim, v2))
            },
            _ => {panic!("should have same tensor type!");},
        }
    }
    pub fn reshape(&self, new_shape: &[usize]) -> TypedTensor {
        match &self {
            TypedTensor::Typef32(v1) => {
                TypedTensor::Typef32(v1.reshape(new_shape))
            },
            TypedTensor::Typef64(v1) => {
                TypedTensor::Typef64(v1.reshape(new_shape))
            },
            //_ => {panic!("should have same tensor type!");},
        }
    }

    pub fn split(&self, sections: &[usize], dim: usize) -> Vec<TypedTensor> {

        match &self {
            TypedTensor::Typef32(v1) => {
                let gts = v1.split(sections, dim);
                let mut ret = Vec::new();
                for i in gts {
                   ret.push(TypedTensor::Typef32(i));
                }
                ret
            },
            TypedTensor::Typef64(v1) => {
                let gts = v1.split(sections, dim);
                let mut ret = Vec::new();
                for i in gts {
                   ret.push(TypedTensor::Typef64(i));
                }
                ret
            },
            //_ => {panic!("should have same tensor type!");},
        }

    }

    pub fn squeeze(&self, dim: Option<usize>) -> TypedTensor {
        match &self {
            TypedTensor::Typef32(v1) => {TypedTensor::Typef32(v1.squeeze(dim))},
            TypedTensor::Typef64(v1) => {TypedTensor::Typef64(v1.squeeze(dim))},
            //_ => {panic!("should have same tensor type!");},
        }
    }

    // Concatenates sequence of tensors along a new dimension.
    //pub fn stack(&self, tensors: &[&Self], dim: usize) -> TypedTensor {
    //}

    pub fn permute(&self, dim: &[usize]) -> TypedTensor {
        match &self {
            TypedTensor::Typef32(v1) => {TypedTensor::Typef32(v1.permute(dim))},
            TypedTensor::Typef64(v1) => {TypedTensor::Typef64(v1.permute(dim))},
            //_ => {panic!("should have same tensor type!");},
        }
    }
    pub fn unsqueeze(&self, dim: usize) -> TypedTensor {
        match self {
            TypedTensor::Typef32(v1) => {TypedTensor::Typef32(v1.unsqueeze(dim))},
            TypedTensor::Typef64(v1) => {TypedTensor::Typef64(v1.unsqueeze(dim))},
            //_ => {panic!("should have same tensor type!");},
        }
    }
    pub fn conditional_select(&self, x: &TypedTensor, y: &TypedTensor) -> TypedTensor {
        match (self, x, y) {
            (TypedTensor::Typef32(v1), TypedTensor::Typef32(v2), TypedTensor::Typef32(v3)) => {
                TypedTensor::Typef32(v1.conditional_select(v2, v3))
            },
            (TypedTensor::Typef64(v1), TypedTensor::Typef64(v2), TypedTensor::Typef64(v3)) => {
                TypedTensor::Typef64(v1.conditional_select(v2, v3))
            },
            _ => {panic!("should have same tensor type!");},
        }
    }
    pub fn repeat(&self, dim: &[usize]) -> TypedTensor {
        match &self {
            TypedTensor::Typef32(v1) => {TypedTensor::Typef32(v1.repeat(dim))},
            TypedTensor::Typef64(v1) => {TypedTensor::Typef64(v1.repeat(dim))},
            //_ => {panic!("should have same tensor type!");},
        }
    }
    
    
    // Pointwise Ops
    typed_tensor_method_single_tensor_return!(abs);
    typed_tensor_method_single_tensor_return!(acos);
    typed_tensor_method_single_tensor_return!(asin);
    typed_tensor_method_single_tensor_return!(atan);
    typed_tensor_method_single_tensor_return!(ceil);
    // clamp
    typed_tensor_method_single_tensor_return!(cos);
    typed_tensor_method_single_tensor_return!(cosh);
    typed_tensor_method_single_tensor_return!(exp);
    typed_tensor_method_single_tensor_return!(expm1);
    typed_tensor_method_single_tensor_return!(floor);
    typed_tensor_method_single_tensor_return!(frac);
    // lerp
    typed_tensor_method_single_tensor_return!(log);
    typed_tensor_method_single_tensor_return!(log10);
    typed_tensor_method_single_tensor_return!(log1p);
    typed_tensor_method_single_tensor_return!(log1pexp);
    typed_tensor_method_single_tensor_return!(log2);
    typed_tensor_method_single_tensor_return!(neg);
    // pow
    pub fn pow_f32(&self, n: f32) -> TypedTensor {
        match &self {
            TypedTensor::Typef32(v1) => {TypedTensor::Typef32(v1.pow(n))},
            TypedTensor::Typef64(v1) => {TypedTensor::Typef64(v1.pow(n as f64))},
            //_ => {panic!("should have same tensor type!");},
        }
    }
    typed_tensor_method_single_tensor_return!(reciprocal);
    typed_tensor_method_single_tensor_return!(round);
    typed_tensor_method_single_tensor_return!(rsqrt);
    typed_tensor_method_single_tensor_return!(sigmoid);
    typed_tensor_method_single_tensor_return!(sign);
    typed_tensor_method_single_tensor_return!(sin);
    typed_tensor_method_single_tensor_return!(sinh);
    typed_tensor_method_single_tensor_return!(sqrt);
    typed_tensor_method_single_tensor_return!(square);
    typed_tensor_method_single_tensor_return!(tan);
    typed_tensor_method_single_tensor_return!(tanh);
    typed_tensor_method_single_tensor_return!(trunc);

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
    pub fn outer(&self, o: &TypedTensor, avg: Option<bool>) -> TypedTensor {
        match (&self, o) {
            (TypedTensor::Typef32(v1), TypedTensor::Typef32(v2)) => {
                TypedTensor::Typef32(v1.outer(v2, avg))},
            (TypedTensor::Typef64(v1), TypedTensor::Typef64(v2)) => {
                TypedTensor::Typef64(v1.outer(v2, avg))},
            _ => {panic!("should have same tensor type!");},
        }
    }

    // reduction ops
    pub fn argmax(&self, dim: Option<&[usize]>, keepdim: bool) -> TypedTensor {
        match &self {
            TypedTensor::Typef32(v1) => {TypedTensor::Typef32(v1.argmax(dim, keepdim))},
            TypedTensor::Typef64(v1) => {TypedTensor::Typef64(v1.argmax(dim, keepdim))},
            //_ => {panic!("should have same tensor type!");},
        }
    }
    pub fn argmin(&self, dim: Option<&[usize]>, keepdim: bool) -> TypedTensor {
        match &self {
            TypedTensor::Typef32(v1) => {TypedTensor::Typef32(v1.argmin(dim, keepdim))},
            TypedTensor::Typef64(v1) => {TypedTensor::Typef64(v1.argmin(dim, keepdim))},
            //_ => {panic!("should have same tensor type!");},
        }
    }
    pub fn logsumexp(&self, dim: Option<&[usize]>, keepdim: bool) -> TypedTensor {
        match &self {
            TypedTensor::Typef32(v1) => {TypedTensor::Typef32(v1.logsumexp(dim, keepdim))},
            TypedTensor::Typef64(v1) => {TypedTensor::Typef64(v1.logsumexp(dim, keepdim))},
            //_ => {panic!("should have same tensor type!");},
        }
    }
    pub fn mean(&self, dim: Option<&[usize]>, keepdim: bool) -> TypedTensor {
        match &self {
            TypedTensor::Typef32(v1) => {TypedTensor::Typef32(v1.mean(dim, keepdim))},
            TypedTensor::Typef64(v1) => {TypedTensor::Typef64(v1.mean(dim, keepdim))},
            //_ => {panic!("should have same tensor type!");},
        }
    }
    pub fn std(&self, dim: Option<&[usize]>, keepdim: bool) -> TypedTensor {
        match &self {
            TypedTensor::Typef32(v1) => {TypedTensor::Typef32(v1.std(dim, keepdim))},
            TypedTensor::Typef64(v1) => {TypedTensor::Typef64(v1.std(dim, keepdim))},
            //_ => {panic!("should have same tensor type!");},
        }
    }
    pub fn sum(&self, dim: Option<&[usize]>, keepdim: bool) -> TypedTensor {
        match &self {
            TypedTensor::Typef32(v1) => {TypedTensor::Typef32(v1.sum(dim, keepdim))},
            TypedTensor::Typef64(v1) => {TypedTensor::Typef64(v1.sum(dim, keepdim))},
            //_ => {panic!("should have same tensor type!");},
        }
    }
    pub fn var(&self, dim: Option<&[usize]>, keepdim: bool) -> TypedTensor {
        match &self {
            TypedTensor::Typef32(v1) => {TypedTensor::Typef32(v1.var(dim, keepdim))},
            TypedTensor::Typef64(v1) => {TypedTensor::Typef64(v1.var(dim, keepdim))},
            //_ => {panic!("should have same tensor type!");},
        }
    }
    pub fn max(&self, dim: Option<&[usize]>, keepdim: bool) -> TypedTensor {
        match &self {
            TypedTensor::Typef32(v1) => {TypedTensor::Typef32(v1.max(dim, keepdim))},
            TypedTensor::Typef64(v1) => {TypedTensor::Typef64(v1.max(dim, keepdim))},
            //_ => {panic!("should have same tensor type!");},
        }
    }
    pub fn min(&self, dim: Option<&[usize]>, keepdim: bool) -> TypedTensor {
        match &self {
            TypedTensor::Typef32(v1) => {TypedTensor::Typef32(v1.max(dim, keepdim))},
            TypedTensor::Typef64(v1) => {TypedTensor::Typef64(v1.min(dim, keepdim))},
            //_ => {panic!("should have same tensor type!");},
        }
    }

    // linalg
    

    
    // Comparison Ops
    typed_tensor_method!(all_close);
    // arg_sort
    typed_tensor_method!(eq_t);
    typed_tensor_method!(ge);
    typed_tensor_method!(gt);
    typed_tensor_method!(le);
    typed_tensor_method!(lt);
    typed_tensor_method!(max_pair);
    typed_tensor_method!(min_pair);
    typed_tensor_method!(ne);

    // conv ops
    pub fn conv2d(&self, filter: &TypedTensor,
                  stride: (usize, usize),
                  padding: (usize, usize),
                  dilation: (usize, usize),
                  padding_mode: PaddingMode) -> TypedTensor {
        match (self, filter) {
            (TypedTensor::Typef32(v1), TypedTensor::Typef32(v2)) => {
                #[cfg(not(feature = "use-blas"))]
                return TypedTensor::Typef32(v1.conv2d(v2, stride, padding, dilation, padding_mode));
                #[cfg(feature = "use-blas")]
                return TypedTensor::Typef32(gemm_conv_f32(v1, v2,
                                                          &[stride.0, stride.1],
                                                          &[padding.0, padding.1],
                                                          &[dilation.0, dilation.1],
                                                          padding_mode));
            },
            (TypedTensor::Typef64(v1), TypedTensor::Typef64(v2)) => {
                #[cfg(not(feature = "use-blas"))]
                return TypedTensor::Typef64(v1.conv2d(v2, stride, padding, dilation, padding_mode));
                #[cfg(feature = "use-blas")]
                return TypedTensor::Typef64(gemm_conv_f64(v1, v2,
                                                          &[stride.0, stride.1],
                                                          &[padding.0, padding.1],
                                                          &[dilation.0, dilation.1],
                                                          padding_mode));
            },
            _ => {panic!("should have same tensor type!");},
        }
    }
    pub fn conv2d_grad(&self, filter: &TypedTensor,
                       stride: (usize, usize),
                       padding: (usize, usize),
                       dilation: (usize, usize),
                       padding_mode: PaddingMode,
                       output_grad: &TypedTensor
    ) -> (TypedTensor, TypedTensor) {
        match (self, filter, output_grad) {
            (TypedTensor::Typef32(v1), TypedTensor::Typef32(v2), TypedTensor::Typef32(v3)) => {
                let (r1, r2) = v1.conv2d_grad(v2, stride, padding, dilation, padding_mode, v3);
                (TypedTensor::Typef32(r1), TypedTensor::Typef32(r2))
            },
            (TypedTensor::Typef64(v1), TypedTensor::Typef64(v2), TypedTensor::Typef64(v3)) => {
                let (r1, r2) = v1.conv2d_grad(v2, stride, padding, dilation, padding_mode, v3);
                (TypedTensor::Typef64(r1), TypedTensor::Typef64(r2))
            },
            _ => {panic!("should have same tensor type!");},
        }
    }
}
impl fmt::Display for TypedTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypedTensor::Typef32(v) => write!(f, "{}", v),
            TypedTensor::Typef64(v) => write!(f, "({}, )", v),
            //_ => panic!("Other case"),
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
            //_ => {panic!("should have same tensor type!");},
        }
    }
}
