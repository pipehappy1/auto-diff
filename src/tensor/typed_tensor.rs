use std::fmt;


use super::gen_tensor::*;

pub enum TypedTensor {
    Typef32(GenTensor<f32>),
    Typef64(GenTensor<f64>),
}

macro_rules! typed_tensor_method_single_same_return {
    ($a:ident, $b:ty) => {
        pub fn $a(&self) -> $b {
            match &self {
                TypedTensor::Typef32(v1) => {v1.$a()},
                TypedTensor::Typef64(v1) => {v1.$a()},
                //_ => {panic!("should have same tensor type!");},
            }
        }
    }
}

macro_rules! typed_tensor_method_single_tensor_return {
    ($a:ident) => {
        pub fn $a(&self) -> TypedTensor {
            match &self {
                TypedTensor::Typef32(v1) => {TypedTensor::Typef32(v1.$a())},
                TypedTensor::Typef64(v1) => {TypedTensor::Typef64(v1.$a())},
                //_ => {panic!("should have same tensor type!");},
            }
        }
    }
}

macro_rules! typed_tensor_method {
    ($a:ident) => {
        pub fn $a(&self, o: &TypedTensor) -> TypedTensor {
            match (&self, o) {
                (TypedTensor::Typef32(v1), TypedTensor::Typef32(v2)) => {TypedTensor::Typef32(v1.$a(v2))},
                (TypedTensor::Typef64(v1), TypedTensor::Typef64(v2)) => {TypedTensor::Typef64(v1.$a(v2))},
                _ => {panic!("should have same tensor type!");},
            }
        }
    }
}


impl TypedTensor {
    pub fn new() -> TypedTensor {
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
    typed_tensor_method_single_tensor_return!(numel_tensor);
    
    pub fn to_f32(i: TypedTensor) {}
    pub fn to_f64(i: TypedTensor) {}

    pub fn fill(size: &[usize], fill_value: f32) -> TypedTensor {
        TypedTensor::Typef32(GenTensor::fill(fill_value, size))
    }

    pub fn unsqueeze(&mut self, dim: &[usize]) {
        match &self {
            TypedTensor::Typef32(v1) => {v1.unsqueeze(dim)},
            TypedTensor::Typef64(v1) => {v1.unsqueeze(dim)},
            //_ => {panic!("should have same tensor type!");},
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
            TypedTensor::Typef32(v) => write!(f, "{}", v),
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
            //_ => {panic!("should have same tensor type!");},
        }
    }
}
