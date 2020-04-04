use std::fmt;

struct GenTensor<T> {
    d: Vec<T>
}

pub struct Tensor {
    v: f64
}
impl Tensor {
    pub fn new() -> Tensor {
	Tensor{v:0.}
    }
    pub fn full() -> Tensor {
	Tensor{v:0.}
    }
    pub fn full_like() -> Tensor {
	Tensor{v:0.}
    }
    pub fn empty() -> Tensor { // <- this will no work.
	Tensor{v:0.}
    }
    pub fn new_ones(dim: &Vec<u32>) -> Tensor {
	Tensor{v:0.}
    }
    pub fn new_zeros(dim: &Vec<u32>) -> Tensor {
	Tensor{v:0.}
    }
    pub fn zeros_like(o: &Tensor) -> Tensor {
	Tensor{v:0.}
    }
    pub fn ones_like(o: &Tensor) -> Tensor {
    	Tensor{v:0.}
    }
    pub fn range(start: f64, step: f64) -> Tensor {
	Tensor{v:0.}
    }
    pub fn linespace(start: f64, end: f64, steps: u32) -> Tensor {
	Tensor{v:0.}
    }
    pub fn logspace(start: f64, end: f64, steps: u32, base: f64) -> Tensor {
	Tensor{v:0.}
    }
    pub fn eye(n: u32, m: u32) -> Tensor {
	Tensor{v:0.}
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
