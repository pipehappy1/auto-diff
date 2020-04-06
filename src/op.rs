use std::cell::RefCell;
use std::rc::Rc;

use super::tensor::Tensor;

pub trait Op {
    fn get_name(&self) -> &str;
    fn apply(&self, input: &Vec<Rc<RefCell<Tensor>>>, output: &mut Vec<Rc<RefCell<Tensor>>>);
    fn grad(&self, input: u32, output: u32);
}

/// add
pub struct add {}
impl add {
    pub fn new() -> add{
        add{}
    }
}
impl Op for add {
    fn get_name(&self) -> &str {
        "add"
    }
    fn apply(&self, input: &Vec<Rc<RefCell<Tensor>>>, output: &mut Vec<Rc<RefCell<Tensor>>>) {
        output[0].replace(input[0].borrow().add(&input[1].borrow()));
    }
    fn grad(&self, input: u32, output: u32) {
        
    }
}

macro_rules! new_op {
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


new_op!(sub, "sub", (|a:&Vec<Rc<RefCell<Tensor>>>, b:&mut Vec<Rc<RefCell<Tensor>>>|{b[0].replace(a[0].borrow().sub(&a[1].borrow()));}) );
new_op!(mul, "mul", (|a:&Vec<Rc<RefCell<Tensor>>>, b:&mut Vec<Rc<RefCell<Tensor>>>|{b[0].replace(a[0].borrow().mul(&a[1].borrow()));}) );
new_op!(div, "div", (|a:&Vec<Rc<RefCell<Tensor>>>, b:&mut Vec<Rc<RefCell<Tensor>>>|{b[0].replace(a[0].borrow().div(&a[1].borrow()));}) );
