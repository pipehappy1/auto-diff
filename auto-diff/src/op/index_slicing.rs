#![allow(clippy::redundant_closure_call)]
use super::{Op, OpCall, OpHandle, OpTrait};
use tensor_rs::tensor::Tensor;

use std::cell::RefCell;
use std::rc::Rc;

use super::macros::{
    many_to_1_op_with_paras, new_element_op, one_to_1_op_with_paras, one_to_vec_op_with_paras,
};
use crate::err::AutoDiffError;
use crate::var::Var;

#[cfg(feature = "use-serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "use-serde")]
use std::any::Any;

#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct Cat {
    #[cfg_attr(feature = "use-serde", serde(skip))]
    handle: OpHandle,
    dim: usize,
}
impl Cat {
    pub fn new(dim: usize) -> Cat {
        Cat {
            handle: OpHandle::new(),
            dim,
        }
    }
    fn get_handle(&self) -> &OpHandle {
        &self.handle
    }
    fn get_handle_mut(&mut self) -> &mut OpHandle {
        &mut self.handle
    }
}
impl OpCall for Cat {
    fn call(&mut self, inputs: &[&Var]) -> Result<Vec<Var>, AutoDiffError> {
        let new_one = Cat {
            handle: OpHandle::new(),
            dim: self.dim,
        };

        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

        inputs[0].called_with(op, &inputs[1..inputs.len()])
    }
}
impl OpTrait for Cat {
    fn get_name(&self) -> &'static str {
        "Cat"
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        let mut new_input = vec![];
        for item in input.iter().skip(1) {
            new_input.push(item.ref_copy());
        }
        output[0].swap(&input[0].cat(&new_input, self.dim));
    }
    fn grad(&self, input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]) {
        let mut splits = Vec::new();
        for i in input {
            splits.push(i.size()[self.dim]);
        }
        let result = output_grad[0].split(&splits, self.dim);
        for i in result {
            input_grad[0].swap(&i);
        }
    }
    fn get_values(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn get_grads(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn set_values(&self, _v: &[Tensor]) {}
    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
        self
    }
}

one_to_vec_op_with_paras!(
    Chunk,
    "Chunk",
    1,
    1, // TODO, this is dependent on the number of output.
    chunk,
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        unimplemented!();
    }),
    chunks: usize,
    dim: usize
);

// gather
#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct Gather {
    #[cfg_attr(feature = "use-serde", serde(skip))]
    handle: OpHandle,
    dim: usize,
}
impl Gather {
    pub fn new(dim: usize) -> Gather {
        Gather {
            handle: OpHandle::new(),
            dim,
        }
    }
    fn get_handle(&self) -> &OpHandle {
        &self.handle
    }
    fn get_handle_mut(&mut self) -> &mut OpHandle {
        &mut self.handle
    }
}
impl OpCall for Gather {
    fn call(&mut self, inputs: &[&Var]) -> Result<Vec<Var>, AutoDiffError> {
        let new_one = Gather {
            handle: OpHandle::new(),
            dim: self.dim,
        };

        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

        inputs[0].called_with(op, &inputs[1..inputs.len()])
    }
}
impl OpTrait for Gather {
    fn get_name(&self) -> &'static str {
        "Gather"
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        output[0].swap(&input[0].gather(self.dim, &input[1]));
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
    fn set_values(&self, _v: &[Tensor]) {}
    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// index_select
#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct IndexSelect {
    #[cfg_attr(feature = "use-serde", serde(skip))]
    handle: OpHandle,
    dim: usize,
}
impl IndexSelect {
    pub fn new(dim: usize) -> IndexSelect {
        IndexSelect {
            handle: OpHandle::new(),
            dim,
        }
    }
    fn get_handle(&self) -> &OpHandle {
        &self.handle
    }
    fn get_handle_mut(&mut self) -> &mut OpHandle {
        &mut self.handle
    }
}
impl OpCall for IndexSelect {
    fn call(&mut self, inputs: &[&Var]) -> Result<Vec<Var>, AutoDiffError> {
        let new_one = IndexSelect {
            handle: OpHandle::new(),
            dim: self.dim,
        };

        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

        inputs[0].called_with(op, &inputs[1..inputs.len()])
    }
}
impl OpTrait for IndexSelect {
    fn get_name(&self) -> &'static str {
        "Index_select"
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        output[0].swap(&input[0].index_select(self.dim, &input[1]));
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
    fn set_values(&self, _v: &[Tensor]) {}
    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// index_exclude
#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct IndexExclude {
    #[cfg_attr(feature = "use-serde", serde(skip))]
    handle: OpHandle,
    dim: usize,
}
impl IndexExclude {
    pub fn new(dim: usize) -> IndexExclude {
        IndexExclude {
            handle: OpHandle::new(),
            dim,
        }
    }
    fn get_handle(&self) -> &OpHandle {
        &self.handle
    }
    fn get_handle_mut(&mut self) -> &mut OpHandle {
        &mut self.handle
    }
}
impl OpCall for IndexExclude {
    fn call(&mut self, inputs: &[&Var]) -> Result<Vec<Var>, AutoDiffError> {
        let new_one = IndexExclude {
            handle: OpHandle::new(),
            dim: self.dim,
        };

        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

        inputs[0].called_with(op, &inputs[1..inputs.len()])
    }
}
impl OpTrait for IndexExclude {
    fn get_name(&self) -> &'static str {
        "Index_exclude"
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        output[0].swap(&input[0].index_exclude(self.dim, &input[1]));
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
    fn set_values(&self, _v: &[Tensor]) {}
    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// reshape
#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct Reshape {
    #[cfg_attr(feature = "use-serde", serde(skip))]
    handle: OpHandle,
    new_shape: Vec<usize>,
}
impl Reshape {
    pub fn new(new_shape: &[usize]) -> Reshape {
        Reshape {
            handle: OpHandle::new(),
            new_shape: new_shape.to_vec(),
        }
    }
    fn get_handle(&self) -> &OpHandle {
        &self.handle
    }
    fn get_handle_mut(&mut self) -> &mut OpHandle {
        &mut self.handle
    }
}
impl OpCall for Reshape {
    fn call(&mut self, inputs: &[&Var]) -> Result<Vec<Var>, AutoDiffError> {
        let new_one = Reshape {
            handle: OpHandle::new(),
            new_shape: self.new_shape.clone(),
        };

        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

        inputs[0].called_with(op, &inputs[1..inputs.len()])
    }
}
impl OpTrait for Reshape {
    fn get_name(&self) -> &'static str {
        "Reshape"
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        output[0].swap(&input[0].reshape(&self.new_shape));
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
    fn set_values(&self, _v: &[Tensor]) {}
    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// split
#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct Split {
    #[cfg_attr(feature = "use-serde", serde(skip))]
    handle: OpHandle,
    sections: Vec<usize>,
    dim: usize,
}
impl Split {
    pub fn new(sections: &[usize], dim: usize) -> Split {
        Split {
            handle: OpHandle::new(),
            sections: sections.to_vec(),
            dim,
        }
    }
    fn get_handle(&self) -> &OpHandle {
        &self.handle
    }
    fn get_handle_mut(&mut self) -> &mut OpHandle {
        &mut self.handle
    }
}
impl OpCall for Split {
    fn call(&mut self, inputs: &[&Var]) -> Result<Vec<Var>, AutoDiffError> {
        let new_one = Split {
            handle: OpHandle::new(),
            sections: self.sections.clone(),
            dim: self.dim,
        };

        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

        inputs[0].called_with(op, &inputs[1..inputs.len()])
    }
}
impl OpTrait for Split {
    fn get_name(&self) -> &'static str {
        "Split"
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        self.sections.len()
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        let mut result = input[0].split(&self.sections, self.dim);
        for (index, i) in result.drain(..).enumerate() {
            output[index].swap(&i);
        }
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
    fn set_values(&self, _v: &[Tensor]) {}
    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// squeeze
one_to_1_op_with_paras!(
    Squeeze,
    "Squeeze",
    1,
    1,
    squeeze,
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        unimplemented!();
    }),
    dim: Option<usize>
);

// stack
many_to_1_op_with_paras!(
    Stack,
    "Stack",
    2, // TODO, this is dependent on the number of input.
    1,
    stack,
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        unimplemented!();
    }),
    dim: usize
);
// t
new_element_op!(
    T,
    "T",
    t,
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        unimplemented!();
    })
);

// take
#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct Take {
    #[cfg_attr(feature = "use-serde", serde(skip))]
    handle: OpHandle,
    sizes: Vec<usize>,
}
impl Take {
    pub fn new(sizes: &[usize]) -> Take {
        Take {
            handle: OpHandle::new(),
            sizes: sizes.to_vec(),
        }
    }
    fn get_handle(&self) -> &OpHandle {
        &self.handle
    }
    fn get_handle_mut(&mut self) -> &mut OpHandle {
        &mut self.handle
    }
}
impl OpCall for Take {
    fn call(&mut self, inputs: &[&Var]) -> Result<Vec<Var>, AutoDiffError> {
        let new_one = Take {
            handle: OpHandle::new(),
            sizes: self.sizes.clone(),
        };

        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

        inputs[0].called_with(op, &inputs[1..inputs.len()])
    }
}
impl OpTrait for Take {
    fn get_name(&self) -> &'static str {
        "Take"
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        output[0].swap(&input[0].take(&self.sizes))
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
    fn set_values(&self, _v: &[Tensor]) {}
    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// permute
#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct Permute {
    #[cfg_attr(feature = "use-serde", serde(skip))]
    handle: OpHandle,
    sizes: Vec<usize>,
}
impl Permute {
    pub fn new(sizes: &[usize]) -> Permute {
        Permute {
            handle: OpHandle::new(),
            sizes: sizes.to_vec(),
        }
    }
    fn get_handle(&self) -> &OpHandle {
        &self.handle
    }
    fn get_handle_mut(&mut self) -> &mut OpHandle {
        &mut self.handle
    }
}
impl OpCall for Permute {
    fn call(&mut self, inputs: &[&Var]) -> Result<Vec<Var>, AutoDiffError> {
        let new_one = Permute {
            handle: OpHandle::new(),
            sizes: self.sizes.clone(),
        };

        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

        inputs[0].called_with(op, &inputs[1..inputs.len()])
    }
}
impl OpTrait for Permute {
    fn get_name(&self) -> &'static str {
        "Permute"
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        output[0].swap(&input[0].permute(&self.sizes))
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
    fn set_values(&self, _v: &[Tensor]) {}
    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// unsqueeze
one_to_1_op_with_paras!(
    Unsqueeze,
    "Unsqueeze",
    1,
    1,
    unsqueeze,
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        unimplemented!();
    }),
    dim: usize
);

// conditional_select
#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct ConditionalSelect {
    #[cfg_attr(feature = "use-serde", serde(skip))]
    handle: OpHandle,
}
impl ConditionalSelect {
    pub fn new() -> ConditionalSelect {
        ConditionalSelect {
            handle: OpHandle::new(),
        }
    }
    fn get_handle(&self) -> &OpHandle {
        &self.handle
    }
    fn get_handle_mut(&mut self) -> &mut OpHandle {
        &mut self.handle
    }
}
impl OpCall for ConditionalSelect {
    fn call(&mut self, inputs: &[&Var]) -> Result<Vec<Var>, AutoDiffError> {
        let new_one = ConditionalSelect {
            handle: OpHandle::new(),
        };

        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

        inputs[0].called_with(op, &inputs[1..inputs.len()])
    }
}
impl OpTrait for ConditionalSelect {
    fn get_name(&self) -> &'static str {
        "ConditionalSelect"
    }
    fn get_input_size(&self) -> usize {
        3
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        output[0].swap(&input[0].conditional_select(&input[0], &input[1]));
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
    fn set_values(&self, _v: &[Tensor]) {}
    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Default for ConditionalSelect {
    fn default() -> Self {
        Self::new()
    }
}

// repeat
#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct Repeat {
    #[cfg_attr(feature = "use-serde", serde(skip))]
    handle: OpHandle,
    sizes: Vec<usize>,
}
impl Repeat {
    pub fn new(sizes: &[usize]) -> Repeat {
        Repeat {
            handle: OpHandle::new(),
            sizes: sizes.to_vec(),
        }
    }
    fn get_handle(&self) -> &OpHandle {
        &self.handle
    }
    fn get_handle_mut(&mut self) -> &mut OpHandle {
        &mut self.handle
    }
}
impl OpCall for Repeat {
    fn call(&mut self, inputs: &[&Var]) -> Result<Vec<Var>, AutoDiffError> {
        let new_one = Repeat {
            handle: OpHandle::new(),
            sizes: self.sizes.clone(),
        };

        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

        inputs[0].called_with(op, &inputs[1..inputs.len()])
    }
}
impl OpTrait for Repeat {
    fn get_name(&self) -> &'static str {
        "Repeat"
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        output[0].swap(&input[0].repeat(&self.sizes))
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
    fn set_values(&self, _v: &[Tensor]) {}
    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
        self
    }
}
