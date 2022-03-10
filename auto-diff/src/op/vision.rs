use tensor_rs::tensor::Tensor;
use super::{OpTrait, OpHandle, OpCall, Op};

use std::cell::{RefCell};
use std::rc::Rc;

use crate::var::{Var};
use crate::err::AutoDiffError;

#[cfg(feature = "use-serde")]
use std::any::Any;

pub struct GetPatch {
    handle: OpHandle,
    range: Vec<(usize, usize)>,
    step: Option<Vec<usize>>,
}
impl GetPatch {
    pub fn new(range: &[(usize, usize)], step: Option<&[usize]>)
               -> GetPatch{
        let new_range = range.to_vec();
        let new_step = step.map(|v| v.to_vec());
        GetPatch {
            handle: OpHandle::new(),
            range: new_range,
            step: new_step
        }
    }
    fn get_handle(&self) -> &OpHandle {
        &self.handle
    }
    fn get_handle_mut(&mut self) -> &mut OpHandle {
        &mut self.handle
    }
}
impl OpCall for GetPatch {
    fn call(&mut self, inputs: &[&Var])
            -> Result<Vec<Var>, AutoDiffError> {
        let new_one = GetPatch {
            handle: OpHandle::new(),
            range: self.range.clone(),
            step: self.step.clone(),
        };

        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

        inputs[0].called_with(op, &inputs[1..inputs.len()])
    }
}
impl OpTrait for GetPatch {

    fn get_name(&self) -> &'static str {
        "get_patch"
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        let step = self.step.as_ref().map(|v| &v[..]);
        output[0].swap(&input[0].get_patch(&self.range, step));
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
    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
	self
    }
}


pub struct SetPatch {
    handle: OpHandle,
    range: Vec<(usize, usize)>,
    step: Option<Vec<usize>>,
}
impl SetPatch {
    pub fn new(range: &[(usize, usize)],
               step: Option<&[usize]>)
               -> SetPatch {
        let new_range = range.to_vec();
        let new_step = step.map(|v| v.to_vec());
        SetPatch {
            handle: OpHandle::new(),
            range: new_range,
            step: new_step
        }
    }
    fn get_handle(&self) -> &OpHandle {
        &self.handle
    }
    fn get_handle_mut(&mut self) -> &mut OpHandle {
        &mut self.handle
    }
}
impl OpCall for SetPatch {
    fn call(&mut self, inputs: &[&Var])
            -> Result<Vec<Var>, AutoDiffError> {
        let new_one = SetPatch {
            handle: OpHandle::new(),
            range: self.range.clone(),
            step: self.step.clone(),
        };

        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

        inputs[0].called_with(op, &inputs[1..inputs.len()])
    }
}
impl OpTrait for SetPatch {

    fn get_name(&self) -> &'static str {
        "set_patch"
    }
    fn get_input_size(&self) -> usize {
        2
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        let step = self.step.as_ref().map(|v| &v[..]);
        output[0].swap(&input[0].set_patch(&input[1], &self.range, step));
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
    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
	self
    }
}


