#![allow(clippy::redundant_closure_call)]
use tensor_rs::tensor::Tensor;
use super::{OpTrait, OpCall, Op, OpHandle};

use std::cell::{RefCell};
use std::rc::Rc;

use crate::var::{Var};
use crate::err::AutoDiffError;
use super::macros::many_to_1_op_with_paras;


many_to_1_op_with_paras!(Cat,
                          "cat",
                          2,
                          1,
                          cat,
                          (|input: &[Tensor],
                           output_grad: &[Tensor],
                           input_grad: &[Tensor]| {
                               unimplemented!();
                           }),
                          dim: usize);

