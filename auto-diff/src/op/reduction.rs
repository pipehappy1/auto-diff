use std::cell::{RefCell};
use std::rc::Rc;

use tensor_rs::tensor::Tensor;
use super::{OpTrait, OpCall, Op, OpHandle};
use crate::var::{Var};
use crate::err::AutoDiffError;
use super::macros::one_to_1_op_with_paras;

pub struct Sum {
    handle: OpHandle,
    dim: Option<Vec<usize>>,
    keepdim: bool
}
impl Sum {
    pub fn new(dim: Option<&[usize]>, keepdim: bool) -> Sum {
        Sum {
            handle: OpHandle::new(),
            dim: if let Some(v) = dim {Some(v.to_vec())} else {None},
            keepdim,
        }
    }
    fn get_handle(&self) -> &OpHandle {
        &self.handle
    }
    fn get_handle_mut(&mut self) -> &mut OpHandle {
        &mut self.handle
    }
}
impl OpCall for Sum {
    fn call(&mut self, inputs: &[&Var])
            -> Result<Vec<Var>, AutoDiffError> {
        let new_one = Sum {
            handle: OpHandle::new(),
            dim: if let Some(v) = self.dim {Some(v.to_vec())} else {None},
            keepdim: self.keepdim,
        };

        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

        Ok(inputs[0].called_with(op, &inputs[1..inputs.len()])?)
    }
}
impl OpTrait for Sum {

    fn get_name(&self) -> String {
        "sum".to_string()
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        // TODO
        //output[0].swap(&input[0].sum())
    }
    fn grad(&self, input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]) {
        unimplemented!();
    }
    fn get_values(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn get_grads(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn set_values(&self, _v: &[Tensor]) {
    }
}
