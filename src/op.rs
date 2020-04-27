/// Only NCWH format is supported.
use std::cell::RefCell;
use std::rc::Rc;

use super::tensor::Tensor;

/// All op is OpTrait
pub trait OpTrait {
    fn get_name(&self) -> String;
    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]);
    fn grad(&self, input: u32, output: u32);
}



pub struct Op {
    o: Rc<RefCell<Box<dyn OpTrait>>>,
}
impl Op {
    pub fn new(o: Box<dyn OpTrait>) -> Self {
        Op {
            o: Rc::new(RefCell::new(o)),
        }
    }

    pub fn swap() {}

    pub fn get_name(&self) -> String {
        self.o.borrow_mut().get_name()
    }
    pub fn apply(&self, input: &[&Tensor], output: &[&Tensor]) {
        self.o.borrow_mut().apply(input, output)
    }
    pub fn grad(&self, input: &[&Tensor], output: &[&Tensor]) {
        self.o.borrow_mut().grad(0, 0)
    }
}
impl Clone for Op {
    fn clone(&self) -> Self {
        Op {
            o: Rc::clone(&self.o),
        }
    }
}




macro_rules! new_binary_op {
    ($a:ident, $b:expr, $c:tt) => {
        pub struct $a {}
        impl $a {
            pub fn new() -> $a{
                $a{}
            }
        }
        impl OpTrait for $a {
            fn get_name(&self) -> String {
                ($b).to_string()
            }
            fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]) {
                $c(input, output)
            }
            fn grad(&self, input: u32, output: u32) {
                
            }       
        }
    }
}

new_binary_op!(Add, "add",
               (|a:&[&Tensor], b:&[&Tensor]|
                b[0].swap(a[0].add(&a[1]))
               )
);
new_binary_op!(Sub, "sub",
               (|a:&[&Tensor], b:&[&Tensor]|
                b[0].swap(a[0].sub(a[1])))
);
new_binary_op!(Mul, "mul",
               (|a:&[&Tensor], b:&[&Tensor]|
                b[0].swap(a[0].mul(a[1])))
);
new_binary_op!(Div, "div",
               (|a:&[&Tensor], b:&[&Tensor]|
                b[0].swap(a[0].div(a[1])))
);


// Identity

pub struct Linear {
    in_fea: Option<usize>,
    out_fea: Option<usize>,
    bias_option: bool,
    weight: Tensor,
    bias: Tensor,
}
impl Linear {
    pub fn new(in_features: Option<usize>, out_features: Option<usize>, bias: bool) -> Linear{
        let mut ret = Linear {
            in_fea: in_features,
            out_fea: out_features,
            bias_option: bias,
            weight: Tensor::new(),
            bias: Tensor::new(),
        };
        if ret.in_fea != Option::None && ret.out_fea != Option::None {
            ret._new();
        }
        ret
    }
    fn _new(&mut self) {
        self.weight = Tensor::fill(&vec![self.in_fea.unwrap(), self.out_fea.unwrap()], 0.);
        self.bias = Tensor::fill(&vec![self.out_fea.unwrap(),], 0.);
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }
}
impl OpTrait for Linear {
    fn get_name(&self) -> String {
        "Linear".to_string()
    }
    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]) {
        if self.in_fea == None || self.out_fea == None {
            if self.in_fea == None {
                let in_size = input[0].size();
                self.in_fea = Some(in_size[in_size.len()-1]);
            }
            if self.out_fea == None {
                let out_size = output[0].size();
                self.out_fea = Some(out_size[0]);
            }
            self._new();
        }

        let ret = input[0].matmul(&self.weight);
        output[0].swap(ret);

        if self.bias_option {
            let ret = output[0].add(&self.bias);
            output[0].swap(ret);
        }
    }
    fn grad(&self, input: u32, output: u32) {
        
    }

}

// Bilinear

//
// Common Cost function
//
pub enum Reduction{
    None,
    Mean,
    Sum,
}

/// MSELoss
/// The left-most dimension is the N.
pub struct MSELoss {
    reduction: Reduction,
}
impl MSELoss {
    pub fn new() -> MSELoss {
        MSELoss {
            reduction: Reduction::None,
        }
    }
}
impl OpTrait for MSELoss {
    fn get_name(&self) -> String {
        "MSE".to_string()
    }
    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]) {
        // TODO: wait for Tensor to have lazy evaluation for elemwise operation.
        let tmp = input[0].sub(input[1]);
        let tmp2 = tmp.mul(&tmp);
        let tmp3 = tmp2.sum();
        let ret = tmp3.div(&input[0].get_N().mul(&input[0].get_C()));
        output[0].swap(ret);
    }
    fn grad(&self, input: u32, output: u32) {
        
    }
}
