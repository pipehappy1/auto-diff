#![allow(clippy::redundant_closure_call)]
use tensor_rs::tensor::Tensor;
use super::{OpTrait, OpCall, Op, OpHandle};

use std::cell::{RefCell};
use std::rc::Rc;

use crate::var::{Var};
use crate::err::AutoDiffError;
use super::macros::{many_to_1_op_with_paras,
                    one_to_vec_op_with_paras,
                    new_element_op};


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

one_to_vec_op_with_paras!(Chunk,
                          "chunk",
                          1,1,chunk,
                          (|input: &[Tensor],
                           output_grad: &[Tensor],
                           input_grad: &[Tensor]| {
                               unimplemented!();
                           }),
                          chunks: usize, dim: usize);
                          
// gather

// index_select
// index_exclude
// reshape
// split
// squeeze
// stack
many_to_1_op_with_paras!(Stack,
                          "stack",
                          2,
                          1,
                          stack,
                          (|input: &[Tensor],
                           output_grad: &[Tensor],
                           input_grad: &[Tensor]| {
                               unimplemented!();
                           }),
                          dim: usize);
// t
new_element_op!(T,
                "t",
                t,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));
// take
// permute
// unsqueeze
// conditional_select
// repeat
