/// Only NCWH format is supported.
use std::cell::{RefCell, Ref};
use std::rc::Rc;

use super::tensor::Tensor;

/// All op is OpTrait
pub trait OpTrait {
    
    fn get_name(&self) -> String;
    fn get_input_size(&self) -> usize;
    fn get_output_size(&self) -> usize;

    /// Forward pass
    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]);
    fn call(&mut self, input: &[&Tensor]) -> Vec<Tensor> {
        if input.len() < self.get_input_size() {
            panic!("{} expect {} input, get {}", self.get_name(), self.get_input_size(), input.len());
        }
        let ret = vec![Tensor::new(); self.get_input_size()];
        let mut ret_ref = Vec::new();
        for i in &ret {
            ret_ref.push(i);
        }
        self.apply(input, &ret_ref[..]);
        ret
    }
    
    /// Given the forward input value and backward output_grad,
    /// Update weight gradient.
    /// return backward input gradeint.
    fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]);

    /// access weight values
    fn get_values(&self) -> Vec<&Tensor>;
    fn set_values(&self, v: &[Tensor]);
    /// access gradient values
    fn get_grads(&self) -> Vec<&Tensor>;
}


/// Op is the Rc wrapper of OpTraint
pub struct Op {
    o: Rc<RefCell<Box<dyn OpTrait>>>,
}
impl Op {
    pub fn new(o: Box<dyn OpTrait>) -> Self {
        Op {
            o: Rc::new(RefCell::new(o)),
        }
    }

    pub fn get(&self) -> Ref<Box<dyn OpTrait>> {
        self.o.borrow()
    }

    pub fn get_name(&self) -> String {
        self.o.borrow_mut().get_name()
    }
    pub fn apply(&self, input: &[&Tensor], output: &[&Tensor]) {
        self.o.borrow_mut().apply(input, output)
    }
    pub fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]) {
        self.o.borrow_mut().grad(input, output_grad, input_grad);
    }

    pub fn get_values(&self) -> Vec<Tensor> {
        let mut ret = Vec::new();
        for i in self.o.borrow().get_values() {
            ret.push(i.clone());
        }
        ret
    }
    pub fn set_values(&self, v: &[Tensor]) {
        self.o.borrow_mut().set_values(v);
    }
    pub fn get_grads(&self) -> Vec<Tensor> {
        let mut ret = Vec::new();
        for i in self.o.borrow().get_grads() {
            ret.push(i.clone());
        }
        ret
    }
}
impl Clone for Op {
    fn clone(&self) -> Self {
        Op {
            o: Rc::clone(&self.o),
        }
    }
}

pub fn _gradient_check(x: &Tensor, op: &dyn OpTrait) -> bool {
    
    let epsilon = Tensor::fill(&x.size(), 0.01);
    let xp = x.add(&epsilon);

    //

    
    true
}


pub mod local;
pub use local::Add as Add;
pub use local::Sub as Sub;
pub use local::Mul as Mul;
pub use local::Div as Div;


pub mod linear;
pub use linear::Linear as Linear;


pub mod nonlinear;
pub use nonlinear::Sigmoid as ELU;
pub use nonlinear::Sigmoid as ReLU;
pub use nonlinear::Sigmoid as Sigmoid;


pub mod loss;
pub use loss::MSELoss as MSELoss;
pub use loss::BCEWithLogitsLoss as BCEWithLogitsLoss;
