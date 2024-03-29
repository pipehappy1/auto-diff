#![allow(clippy::redundant_closure_call)]
use super::macros::new_binary_op;
use super::{Op, OpCall, OpHandle, OpTrait};
use tensor_rs::tensor::Tensor;

use std::cell::RefCell;
use std::rc::Rc;

use crate::err::AutoDiffError;
use crate::var::Var;

#[cfg(feature = "use-serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "use-serde")]
use std::any::Any;

// max_pair
new_binary_op!(
    MaxPair,
    "Max_pair",
    (|a: &[Tensor], b: &[Tensor]| b[0].swap(&a[0].max_pair(&a[1]))),
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        unimplemented!();
    })
);
// max, in reduction
// min_pair
new_binary_op!(
    MinPair,
    "Min_pair",
    (|a: &[Tensor], b: &[Tensor]| b[0].swap(&a[0].min_pair(&a[1]))),
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        unimplemented!();
    })
);
// min, in reduction
// arg_sort
#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct ArgSort {
    #[cfg_attr(feature = "use-serde", serde(skip))]
    handle: OpHandle,
    dim: usize,
    descending: bool,
}
impl ArgSort {
    pub fn new(dim: usize, descending: bool) -> ArgSort {
        ArgSort {
            handle: OpHandle::new(),
            dim,
            descending,
        }
    }
    fn get_handle(&self) -> &OpHandle {
        &self.handle
    }
    fn get_handle_mut(&mut self) -> &mut OpHandle {
        &mut self.handle
    }
}
impl OpCall for ArgSort {
    fn call(&mut self, inputs: &[&Var]) -> Result<Vec<Var>, AutoDiffError> {
        let new_one = ArgSort {
            handle: OpHandle::new(),
            dim: self.dim,
            descending: self.descending,
        };

        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

        inputs[0].called_with(op, &inputs[1..inputs.len()])
    }
}
impl OpTrait for ArgSort {
    fn get_name(&self) -> &'static str {
        "Arg_sort"
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        output[0].swap(&input[0].arg_sort(self.dim, self.descending))
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
// eq_t (use eq_elem)
new_binary_op!(
    EqElem,
    "Eq_t",
    (|a: &[Tensor], b: &[Tensor]| b[0].swap(&a[0].eq_t(&a[1]))),
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        unimplemented!();
    })
);
// equal, 0 is == 1 is !=
new_binary_op!(
    Equal,
    "Equal",
    (|a: &[Tensor], b: &[Tensor]| if a[0].equal(&a[1]) {
        b[0].swap(&Tensor::zeros(&[1]))
    } else {
        b[0].swap(&Tensor::ones(&[1]))
    }),
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        unimplemented!();
    })
);
// ge
new_binary_op!(
    Ge,
    "Ge",
    (|a: &[Tensor], b: &[Tensor]| b[0].swap(&a[0].ge(&a[1]))),
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        unimplemented!();
    })
);
// gt
new_binary_op!(
    Gt,
    "Gt",
    (|a: &[Tensor], b: &[Tensor]| b[0].swap(&a[0].gt(&a[1]))),
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        unimplemented!();
    })
);
// le
new_binary_op!(
    Le,
    "Le",
    (|a: &[Tensor], b: &[Tensor]| b[0].swap(&a[0].le(&a[1]))),
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        unimplemented!();
    })
);
// lt
new_binary_op!(
    Lt,
    "Lt",
    (|a: &[Tensor], b: &[Tensor]| b[0].swap(&a[0].lt(&a[1]))),
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        unimplemented!();
    })
);
// ne
new_binary_op!(
    Ne,
    "Ne",
    (|a: &[Tensor], b: &[Tensor]| b[0].swap(&a[0].ne(&a[1]))),
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        unimplemented!();
    })
);
