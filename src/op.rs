/// Only NCWH format is supported.
use std::cell::RefCell;
use std::rc::Rc;

use super::tensor::Tensor;

pub trait Op {
    fn get_name(&self) -> &str;
    fn apply(&self, input: &Vec<Rc<RefCell<Tensor>>>, output: &mut Vec<Rc<RefCell<Tensor>>>);
    fn grad(&self, input: u32, output: u32);
}

macro_rules! new_binary_op {
    ($a:ident, $b:expr, $c:tt) => {
        pub struct $a {}
        impl $a {
            pub fn new() -> $a{
                $a{}
            }
        }
        impl Op for $a {
            fn get_name(&self) -> &str {
                $b
            }
            fn apply(&self, input: &Vec<Rc<RefCell<Tensor>>>, output: &mut Vec<Rc<RefCell<Tensor>>>) {
                $c(input, output)
            }
            fn grad(&self, input: u32, output: u32) {
                
            }       
        }
    }
}

new_binary_op!(add, "add",
               (|a:&Vec<Rc<RefCell<Tensor>>>, b:&mut Vec<Rc<RefCell<Tensor>>>|{b[0].replace(a[0].borrow().add(&a[1].borrow()));}) );
new_binary_op!(sub, "sub",
               (|a:&Vec<Rc<RefCell<Tensor>>>, b:&mut Vec<Rc<RefCell<Tensor>>>|{b[0].replace(a[0].borrow().sub(&a[1].borrow()));}) );
new_binary_op!(mul, "mul",
               (|a:&Vec<Rc<RefCell<Tensor>>>, b:&mut Vec<Rc<RefCell<Tensor>>>|{b[0].replace(a[0].borrow().mul(&a[1].borrow()));}) );
new_binary_op!(div, "div",
               (|a:&Vec<Rc<RefCell<Tensor>>>, b:&mut Vec<Rc<RefCell<Tensor>>>|{b[0].replace(a[0].borrow().div(&a[1].borrow()));}) );


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
        Linear{
            in_fea: in_features,
            out_fea: out_features,
            bias_option: bias,
            weight: Tensor::new(),
            bias: Tensor::new(),
        }
    }
}
impl Op for Linear {
    fn get_name(&self) -> &str {
        "Linear"
    }
    fn apply(&self, input: &Vec<Rc<RefCell<Tensor>>>, output: &mut Vec<Rc<RefCell<Tensor>>>) {
        
    }
    fn grad(&self, input: u32, output: u32) {
        
    }       
}

// Bilinear
