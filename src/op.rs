use std::cell::RefCell;
use std::rc::Rc;

use super::tensor::Tensor;

pub trait Op {
    fn get_name(&self) -> &str;
    fn apply(&self, input: &Vec<Rc<RefCell<Tensor>>>, output: &mut Vec<Rc<RefCell<Tensor>>>);
    fn grad(&self, input: u32, output: u32);
}

pub struct add {}

impl Op for add {
    fn get_name(&self) -> &str {
        "Add"
    }
    fn apply(&self, input: &Vec<Rc<RefCell<Tensor>>>, output: &mut Vec<Rc<RefCell<Tensor>>>) {
        output[0].replace(input[0].borrow().add(&input[1].borrow()));
    }
    fn grad(&self, input: u32, output: u32) {
        
    }
}
impl add {
    pub fn new() -> add{
        add{}
    }
}
