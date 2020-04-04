use std::fmt;

struct GenTensor<T> {
    d: Vec<T>
}
impl<T> GenTensor<T> {
    fn new() -> GenTensor<T> {
	GenTensor {
	    d: Vec::<T>::new(),
	}
    }
}
impl<T> fmt::Display for GenTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
	write!(f, "0")
    }
}

enum TypedTensor {
    Typef32( GenTensor<f32>),
    Typef64( GenTensor<f64>),
}
impl TypedTensor {
    fn new() -> TypedTensor {
	// Default value type is f32.
	TypedTensor::Typef32(GenTensor::new())
    }
    fn to_f32(i: TypedTensor) {
    }
    fn to_f64(i: TypedTensor) {
	
    }
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
    pub fn _raw() {}
    pub fn from_vec_f32(i: &Vec<f32>) -> Tensor {
	let mut ret = Tensor::new();
	ret
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
    pub fn empty() -> Tensor { // <- this will no work.
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


    
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, )", self.v)
    }
}
