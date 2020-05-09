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

pub fn _gradient_checker(x: &[&Tensor], op: &mut dyn OpTrait, step: f32, tolerance: f32) -> bool {

    let mut epsilon = Vec::new();
    for i in x {
        let delta = Tensor::fill(&i.size(), step);
        let xp = i.add(&delta);
        epsilon.push(xp);
    }

    let input_grad = vec![Tensor::new(); op.get_input_size()];
    let mut input_grad_ref = Vec::new();
    for i in &input_grad {
        input_grad_ref.push(i);
    }
    let mut output_grad = Vec::new();
    for i in 0..op.get_output_size() {
        output_grad.push(Tensor::ones_like(x[i]));
    }
    let mut output_grad_ref = Vec::new();
    for i in 0..op.get_output_size() {
        output_grad_ref.push(&output_grad[i]);
    }
    op.grad(x, &output_grad_ref, &input_grad_ref);
    
    let mut good_derivative = true;
    let output = op.call(x);
    for index in 0..x.len() {
        let mut new_input = Vec::new();
        for j in 0..x.len() {
            if j == index {
                new_input.push(&epsilon[index]);
            }
            else {
                new_input.push(&x[j]);
            }
        }
        let new_output = op.call(&new_input);
        
        let numeric_grad = new_output[0].sub(&output[0])
            .div(&Tensor::fill(&new_output[0].size(), step));

        if input_grad_ref[index].sub(&numeric_grad).sum().get_scale_f32() > tolerance {
            good_derivative = false;
        }
    }
    
    good_derivative
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
