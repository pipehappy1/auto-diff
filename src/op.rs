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
        let ret = vec![Tensor::new(); self.get_output_size()];
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

// check right gradient
pub fn _gradient_checker(op: &mut dyn OpTrait,
                         one_input: &[&Tensor], input_mask: Option<&[bool]>,
                         step: Option<f32>, tolerance: Option<f32>) -> bool {

    let x_mask;
    if input_mask.is_none() {
        x_mask = vec![true; one_input.len()];
    } else {
        x_mask = input_mask.unwrap().to_vec();
    }

    let delta;
    if step.is_none() {
        delta = 0.01;
    } else {
        delta = step.unwrap();
    }

    let tol;
    if tolerance.is_none() {
        tol = 0.01;
    } else {
        tol = tolerance.unwrap();
    }

    // system output
    let output = op.call(one_input);
    if output.len() > 1 || output[0].numel() > 1 {
        panic!("gradient checker only handle scale output case. {:?}, {:?}", output.len(), output[0].size());
    }
    let output = output[0].get_scale_f32();

    // get the system gradient
    let input_grad = vec![Tensor::new(); op.get_input_size()];
    let mut input_grad_ref = Vec::new();
    for i in &input_grad {
        input_grad_ref.push(i);
    }
    let output_grad = Tensor::from_vec_f32(&[1.], &[1]);
    op.grad(one_input, &[&output_grad], &input_grad_ref);

    // get the numeric gradient
    let mut numeric_gradient = Vec::new();
    for v in one_input {
        numeric_gradient.push(v.zeros_like())
    }

    let mut good_gradient = true;
    for (index, v) in one_input.iter().enumerate() {
        if !x_mask[index] {
            continue;
        }
        
        for i in 0..v.numel() {
            let dimpos = v.index2dimpos(i);
                
            let base_value = v.get_f32(&dimpos);
            let right_value = base_value + delta;
            let mut right_tensor = (*v).clone();
            right_tensor.set_f32(&dimpos, right_value);

            let mut right_input = one_input.to_vec();
            right_input[index] = &right_tensor;
            let right_output = op.call(&right_input)[0].get_scale_f32();

            let scale_gradient = (right_output - output)/delta;
            numeric_gradient[index].set_f32(&dimpos, scale_gradient);

            let system_gradient = input_grad[index].get_f32(&dimpos);

            //println!("left: {:?}, right: {:?}", scale_gradient, system_gradient);
            if (scale_gradient - system_gradient)*(scale_gradient - system_gradient) > tol {
                good_gradient = false;
            }
        }
    }
    good_gradient
}


pub mod local;
pub use local::Add;
pub use local::Sub;
pub use local::Mul;
pub use local::Div;


pub mod convolution;
pub use convolution::PaddingMode;
pub use convolution::Conv2d;


pub mod linear;
pub use linear::Linear;


pub mod nonlinear;
pub use nonlinear::ELU;
pub use nonlinear::ReLU;
pub use nonlinear::Sigmoid;


pub mod loss;
pub use loss::MSELoss;
pub use loss::BCEWithLogitsLoss;
pub use loss::CrossEntropyLoss;
