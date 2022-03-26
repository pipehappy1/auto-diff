/// Only NCWH format is supported.
use std::cell::{RefCell};
use std::rc::Rc;

use tensor_rs::tensor::Tensor;
use crate::var::Var;
use crate::err::AutoDiffError;
use crate::collection::generational_index::{GenKey};
use crate::compute_graph::Net;


#[cfg(feature = "use-serde")]
use serde::{Serializer, de, de::MapAccess, de::SeqAccess,};
#[cfg(feature = "use-serde")]
use serde::{Serialize, Deserialize};
#[cfg(feature = "use-serde")]
use std::any::Any;

/// Implement operator by this trait
/// to allow the operator be able to stored
/// in the computation graph.
pub trait OpTrait {
    /// A conventional name for the op
    fn get_name(&self) -> &'static str;

    /// The number of input needs by this op.
    fn get_input_size(&self) -> usize;

    /// The number of output produced by this op.
    fn get_output_size(&self) -> usize;

    /// Forward pass
    fn apply(&self, input: &[Tensor], output: &[Tensor]);

    /// Given the forward input value and backward output_grad,
    /// Update weight gradient.
    /// return backward input gradeint.
    fn grad(&self, input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]);

    /// access weight values
    fn get_values(&self) -> Vec<Tensor>;
    fn set_values(&self, v: &[Tensor]);
    /// access gradient values
    fn get_grads(&self) -> Vec<Tensor>;

    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any;
}

/// Ops that first created,
/// then called needs to follow this behavior.
pub trait OpCall {
    fn call(&mut self, inputs: &[&Var]) -> Result<Vec<Var>, AutoDiffError>;
}

pub struct OpHandle {
    id: GenKey,    
    net: Rc<RefCell<Net>>,
}
impl OpHandle {
    pub fn new() -> OpHandle {
        OpHandle {
            id: GenKey::new(0, 0),
            net: Rc::new(RefCell::new(Net::new()))
        }
    }
}
impl Default for OpHandle {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! handle_method {
    () => {
        fn get_handle(&self) -> &OpHandle {
            &self.handle
        }
    
        fn get_handle_mut(&mut self) -> &mut OpHandle {
            &mut self.handle
        }
    }
}


///
/// Op is the Rc wrapper of typed op trait
///
#[derive(Clone)]
pub struct Op {
    inner_op: Rc<RefCell<Box<dyn OpTrait>>>,
}
impl Op {
    pub fn new(op: Rc<RefCell<Box<dyn OpTrait>>>) -> Self {
        Op {
            inner_op: op.clone(),
        }
    }
    pub fn inner(&self) -> &Rc<RefCell<Box<dyn OpTrait>>> {
	&self.inner_op
    }

    pub fn ref_copy(&self) -> Self {
        Op {
            inner_op: self.inner_op.clone(),
        }
    }

    pub fn get_name(&self) -> String {
        self.inner_op.borrow().get_name().to_string()
    }
    pub fn get_input_size(&self) -> usize {
        self.inner_op.borrow().get_input_size()
    }
    pub fn get_output_size(&self) -> usize {
        self.inner_op.borrow().get_output_size()
    }
    /// Read the input, do the calculation and write result to output.
    /// Called by compute_grapyh.
    pub fn apply(&self, input: &[Tensor],
                 output: &[Tensor]) {
        self.inner_op.borrow().apply(input, output);
    }
    /// Given input and output_grad, return input_grad (forward view)
    /// Called by compute_grapyh.
    pub fn grad(&self, input: &[Tensor],
                output_grad: &[Tensor],
                input_grad: &[Tensor]) {

        self.inner_op.borrow().grad(input, output_grad, input_grad);
    }

    /// access weight/paramenters
    pub fn get_values(&self) -> Vec<Tensor> {
        self.inner_op.borrow().get_values()
    }

    /// set parameters
    pub fn set_values(&self, v: &[Tensor]) {
        self.inner_op.borrow_mut().set_values(v);
    }

    /// return gradient for weight/parameters.
    pub fn get_grads(&self) -> Vec<Tensor> {
        self.inner_op.borrow().get_grads()
    }
}
//impl Clone for Op {
//    fn clone(&self) -> Self {
//        Op {
//            update_counter: self.update_counter.clone(),
//            para_grad: self.para_grad.iter().map(|(a, b)| (a.clone(), b.clone())).collect(),
//            func_apply: self.func_apply.clone(),
//            func_gradient: self.func_gradient.clone(),
//            name: self.name.clone(),
//            input_size: self.input_size,
//            output_size: self.output_size,
//        }
//    }
//}



//pub struct Nop {
//}
//impl OpTrait for Nop {
//    fn get_name(&self) -> String {
//        "Nop".to_string()
//    }
//    fn get_input_size(&self) -> usize {
//        0
//    }
//    fn get_output_size(&self) -> usize {
//        0
//    }
//
//    /// Forward pass
//    fn apply(&mut self, _input: &[&Tensor], _output: &[&Tensor]) {
//        
//    }
//    fn grad(&self, _input: &[&Tensor], _output_grad: &[&Tensor], _input_grad: &[&Tensor]) {
//        
//    }
//
//    /// access weight values
//    fn get_values(&self) -> Vec<&Tensor> {
//        Vec::new()
//    }
//    fn set_values(&self, _v: &[Tensor]) {
//        
//    }
//    /// access gradient values
//    fn get_grads(&self) -> Vec<&Tensor> {
//        Vec::new()
//    }
//}



///
/// Verify the gradient implementation is right.
///
/// op: the tested operator.
/// one_input: test data points.
/// input_mask: may skip some data point if the element is false.
/// step: delta that is used for numeric difference.
/// tolerance: numeric tolerance for equality.
///
/// one_input and input_mask should have the same size.
/// step and tolerance are both scalar.
pub fn _gradient_checker(op: &mut dyn OpTrait,
                         one_input: &[Tensor], input_mask: Option<&[bool]>,
                         step: Option<Tensor>, tolerance: Option<Tensor>)
			 -> bool {

    let x_mask = if let Some(val) = input_mask {val.to_vec()} else {vec![true; one_input.len()]};
    let delta = if let Some(val) = step {val.get_scale_f64()} else {0.01};
    let tol = if let Some(val) = tolerance {val.get_scale_f64()} else {0.01};


    // system output
    let output = Tensor::new();
    op.apply(one_input, &[output.ref_copy()]);

    let output = output.get_scale_f64();

    // get the system gradient
    let input_grad = vec![Tensor::new(); op.get_input_size()];
    let mut input_grad_ref = Vec::new();
    for i in &input_grad {
        input_grad_ref.push(i.ref_copy());
    }
    let output_grad = Tensor::from_vec_f64(&[1.], &[1]);
    op.grad(one_input, &[output_grad], &input_grad_ref);

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
                
            let base_value = v.get_f64(&dimpos);
            let right_value = base_value + delta;
            let mut right_tensor = (*v).clone();
            right_tensor.set_f64(&dimpos, right_value);

            let mut right_input = one_input.to_vec();
            right_input[index] = right_tensor.ref_copy();
            let right_output = Tensor::new();
            op.apply(&right_input, &[right_output.ref_copy()]);
            let right_output = right_output.get_scale_f64();

            let scale_gradient = (right_output - output)/delta;
            numeric_gradient[index].set_f64(&dimpos, scale_gradient);

            let system_gradient = input_grad[index].get_f64(&dimpos);

            if (scale_gradient - system_gradient)*(scale_gradient - system_gradient) > tol {
                println!("input: {:?}, numeric: {:?}, imple: {:?}", one_input[0], scale_gradient, system_gradient);
                good_gradient = false;
            }
        }
    }
    good_gradient
}

///
/// View op
///
#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct View {
    shape: Vec<usize>,
    #[cfg_attr(feature = "use-serde", serde(skip))]
    handle: OpHandle,
}
impl View {
    pub fn new(new_shape: &[usize]) -> View {
        View {
            shape: new_shape.to_vec(),
            handle: OpHandle::new(),
        }
    }
    handle_method!();
}
impl OpCall for View {
    fn call(&mut self, inputs: &[&Var]) -> Result<Vec<Var>, AutoDiffError> {
        let new_one = View {
            shape: self.shape.clone(),
            handle: OpHandle::new(),
        };

        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

        inputs[0].called_with(op, &inputs[1..inputs.len()])
    }
}
impl OpTrait for View {
    fn get_name(&self) -> &'static str {
        "View"
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }

    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        if input.len() > 1 {
            panic!("view only acceipt one input");
        }

        let total_numel: usize = self.shape.iter().product();
        if input[0].numel() != total_numel {
            panic!("view expect tensor has a total elem of {}, get {}", total_numel, input[0].numel());
        }

        output[0].swap(&input[0].reshape(&self.shape));
    }

    fn grad(&self, input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]) {
        
        input_grad[0].swap(&output_grad[0].reshape(&input[0].size()));
    }

    fn get_values(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn set_values(&self, _v: &[Tensor]) {
    }
    /// access gradient values
    fn get_grads(&self) -> Vec<Tensor> {
        Vec::new()
    }

    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
	self
    }
}

pub mod macros;

pub mod local;
pub use local::{Add, Sub, Mul, Div, Matmul, Outer};

pub mod linear;
pub use linear::Linear;

pub mod nonlinear;
pub use nonlinear::{ELU, ReLU, };

pub mod convolution;
pub use convolution::{ Conv2d};

pub mod pooling;

pub mod loss;
pub use loss::{MSELoss, BCEWithLogitsLoss, CrossEntropyLoss};

pub mod element;
pub use element::{Abs, Acos, Asin, Atan, Ceil, Cos, Cosh, Exp, Expm1, Floor, Frac, Log, Log10, Log1p, Log1pexp, Log2, Neg, Reciprocal, Round, Rsqrt,Sigmoid, Sign, Sin, Sinh, Sqrt, Tan, Tanh, Trunc};

pub mod comparison;
pub use comparison::{MaxPair, MinPair, ArgSort, EqElem, Equal, Ge, Gt, Le, Lt, Ne};

pub mod index_slicing;
pub use index_slicing::{Cat, Chunk, ConditionalSelect, Gather, IndexSelect, IndexExclude, Reshape, Split, Squeeze, Stack, T, Take, Permute, Unsqueeze, Repeat};

pub mod linalg;
pub use linalg::{Det, Inv, NormalizeUnit, Tr};

pub mod reduction;
pub use reduction::{Argmax, Argmin, Logsumexp, Mean, Prod, Std, Sum, Variance, Max, Min};

pub mod vision;
pub use vision::{GetPatch, SetPatch};

#[cfg(feature = "use-serde")]
use auto_diff_macros::gen_serde_funcs;
#[cfg(feature = "use-serde")]
use serde::{ser};
#[cfg(feature = "use-serde")]
gen_serde_funcs!(View,
                 Add, Sub, Mul, Div, Matmul, Outer,
                 Linear,
                 ELU, ReLU,
                 Conv2d,
                 MSELoss, BCEWithLogitsLoss, CrossEntropyLoss,
                 Abs, Acos, Asin, Atan, Ceil, Cos, Cosh, Exp, Expm1, Floor, Frac, Log, Log10, Log1p, Log1pexp, Log2, Neg, Reciprocal, Round, Rsqrt,Sigmoid, Sign, Sin, Sinh, Sqrt, Tan, Tanh, Trunc,
                 MaxPair, MinPair, ArgSort, EqElem, Equal, Ge, Gt, Le, Lt, Ne,
                 Cat, Chunk, ConditionalSelect, Gather, IndexSelect, IndexExclude, Reshape, Split, Squeeze, Stack, T, Take, Permute, Unsqueeze, Repeat,
                 Det, Inv, NormalizeUnit, Tr,
                 Argmax, Argmin, Logsumexp, Mean, Prod, Std, Sum, Variance, Max, Min,
                 GetPatch, SetPatch);
