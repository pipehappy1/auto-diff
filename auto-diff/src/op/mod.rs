/// Only NCWH format is supported.
use std::cell::{RefCell, Ref};
use std::rc::Rc;

use tensor_rs::tensor::Tensor;

/// All op is OpTrait
pub trait OpTrait {

    /// A conventional name for the op
    fn get_name(&self) -> String;

    /// The number of input needs by this op.
    fn get_input_size(&self) -> usize;

    /// The number of output produced by this op.
    fn get_output_size(&self) -> usize;

    /// Forward pass
    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]);

    
    fn call(&mut self, input: &[&Tensor]) -> Result<Vec<Tensor>, &str> {
        if input.len() != self.get_input_size() {
            //return Err("{} expect {} input, get {}", self.get_name(), self.get_input_size(), input.len());
            return Err("Op expect correct number of inputs.");
        }
        let ret = vec![Tensor::new(); self.get_output_size()];
        let mut ret_ref = Vec::new();
        for i in &ret {
            ret_ref.push(i);
        }
        self.apply(input, &ret_ref[..]);
        Ok(ret)
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

pub struct Nop {
}
impl OpTrait for Nop {
    fn get_name(&self) -> String {
        "Nop".to_string()
    }
    fn get_input_size(&self) -> usize {
        0
    }
    fn get_output_size(&self) -> usize {
        0
    }

    /// Forward pass
    fn apply(&mut self, _input: &[&Tensor], _output: &[&Tensor]) {
        
    }
    fn grad(&self, _input: &[&Tensor], _output_grad: &[&Tensor], _input_grad: &[&Tensor]) {
        
    }

    /// access weight values
    fn get_values(&self) -> Vec<&Tensor> {
        Vec::new()
    }
    fn set_values(&self, _v: &[Tensor]) {
        
    }
    /// access gradient values
    fn get_grads(&self) -> Vec<&Tensor> {
        Vec::new()
    }
}

///
/// Op is the Rc wrapper of typed op trait
///
pub struct Op {
    o: Rc<RefCell<Box<dyn OpTrait>>>,
    update_counter: RefCell<usize>, // guard for the case there optim.step is called when .backward is not called yet.
}
impl Op {
    pub fn new(o: Box<dyn OpTrait>) -> Self {
        Op {
            o: Rc::new(RefCell::new(o)),
            update_counter: RefCell::new(0),
        }
    }

    pub fn nop() -> Self {
        Op {
            o: Rc::new(RefCell::new(Box::new(Nop{}))),
            update_counter: RefCell::new(0),
        }
    }

    pub fn get(&self) -> Ref<Box<dyn OpTrait>> {
        self.o.borrow()
    }

    pub fn get_name(&self) -> String {
        self.o.borrow_mut().get_name()
    }
    pub fn get_update_counter(&self) -> usize {
        *self.update_counter.borrow()
    }
    pub fn apply(&self, input: &[&Tensor], output: &[&Tensor]) {
        self.o.borrow_mut().apply(input, output)
    }
    pub fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]) {
        self.o.borrow_mut().grad(input, output_grad, input_grad);
        let new_counter = self.update_counter.borrow().overflowing_add(1).0;
        self.update_counter.replace(new_counter);
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
            update_counter: self.update_counter.clone(),
        }
    }
}

///
/// Verify the gradient implementation is right.
///
pub fn _gradient_checker(op: &mut dyn OpTrait,
                         one_input: &[&Tensor], input_mask: Option<&[bool]>,
                         step: Option<f32>, tolerance: Option<f32>) -> bool {

    let x_mask = if let Some(val) = input_mask {val.to_vec()} else {vec![true; one_input.len()]};
    let delta = if let Some(val) = step {val} else {0.01};
    let tol = if let Some(val) = tolerance {val} else {0.01};


    // system output
    let output = op.call(one_input).unwrap();
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
            let right_output = op.call(&right_input).unwrap()[0].get_scale_f32();

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

///
/// View op
///
pub struct View {
    shape: Vec<usize>,
}
impl View {
    pub fn new(new_shape: &[usize]) -> View {
        View {
            shape: new_shape.to_vec(),
        }
    }
}
impl OpTrait for View {
    fn get_name(&self) -> String {
        "view".to_string()
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }

    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]) {
        if input.len() > 1 {
            panic!("view only acceipt one input");
        }

        let total_numel: usize = self.shape.iter().product();
        if input[0].numel() != total_numel {
            panic!("view expect tensor has a total elem of {}, get {}", total_numel, input[0].numel());
        }

        output[0].swap(input[0].reshape(&self.shape));
    }

    fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]) {
        
        input_grad[0].swap(output_grad[0].reshape(&input[0].size()));
    }

    fn get_values(&self) -> Vec<&Tensor> {
        Vec::new()
    }
    fn set_values(&self, _v: &[Tensor]) {
    }
    /// access gradient values
    fn get_grads(&self) -> Vec<&Tensor> {
        Vec::new()
    }
}

pub mod local;
pub use local::{Add, Sub, Mul, Div};

pub mod convolution;
pub use convolution::{ Conv2d};

pub mod linear;
pub use linear::Linear;

pub mod nonlinear;
pub use nonlinear::{ELU, ReLU, Sigmoid};

pub mod loss;
pub use loss::{MSELoss, BCEWithLogitsLoss, CrossEntropyLoss};
