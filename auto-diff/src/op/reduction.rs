#![allow(clippy::redundant_closure_call)]
use std::cell::{RefCell};
use std::rc::Rc;

use tensor_rs::tensor::Tensor;
use super::{OpTrait, OpCall, Op, OpHandle};
use crate::err::AutoDiffError;

macro_rules! reduce_macro {
    ($a:ident, $b:expr, $c:ident, $d: tt) => {
        pub struct $a {
            handle: OpHandle,
            dim: Option<Vec<usize>>,
            keepdim: bool
        }
        impl $a {
            pub fn new(dim: Option<&[usize]>, keepdim: bool) -> $a{
                $a{
                    handle: OpHandle::new(),
                    dim: dim.map(|v| v.to_vec()),
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
        impl OpCall for $a {
            fn call(&mut self, inputs: &[&crate::var::Var]) -> Result<Vec<crate::var::Var>, AutoDiffError> {
                let new_one = $a {
                    handle: OpHandle::new(),
                    dim: self.dim.as_ref().map(|v| v.to_vec()),
                    keepdim: self.keepdim,
                };

                let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

                inputs[0].called_with(op, &inputs[1..inputs.len()])
            }
        }
        impl OpTrait for $a {
     
            fn get_name(&self) -> String {
                ($b).to_string()
            }
            fn get_input_size(&self) -> usize {
                1
            }
            fn get_output_size(&self) -> usize {
                1
            }
            fn apply(&self, input: &[Tensor], output: &[Tensor]) {
                match &self.dim {
                    Some(v) => {
                        let v1 = v.clone();
                        output[0].swap(&input[0].$c(Some(&v1), self.keepdim));
                    },
                    None => {
                        output[0].swap(&input[0].$c(None, self.keepdim));
                    },
        }
            }
            fn grad(&self, input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]) {
                $d(input, output_grad, input_grad)
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
    }
}




reduce_macro!(Argmax, "argmax", argmax,
              (|input: &[Tensor],
               output_grad: &[Tensor],
               input_grad: &[Tensor]| {
                   unimplemented!();
               }));


reduce_macro!(Argmin, "argmin", argmin,
              (|input: &[Tensor],
               output_grad: &[Tensor],
               input_grad: &[Tensor]| {
                   unimplemented!();
               }));


reduce_macro!(Logsumexp, "logsumexp", logsumexp,
              (|input: &[Tensor],
               output_grad: &[Tensor],
               input_grad: &[Tensor]| {
                   unimplemented!();
               }));


reduce_macro!(Mean, "mean", mean,
              (|input: &[Tensor],
               output_grad: &[Tensor],
               input_grad: &[Tensor]| {
                   unimplemented!();
               }));

reduce_macro!(Prod, "prod", prod,
              (|input: &[Tensor],
               output_grad: &[Tensor],
               input_grad: &[Tensor]| {
                   unimplemented!();
               }));

reduce_macro!(Std, "std", std,
              (|input: &[Tensor],
               output_grad: &[Tensor],
               input_grad: &[Tensor]| {
                   unimplemented!();
               }));

reduce_macro!(Sum, "sum", sum,
              (|input: &[Tensor],
               output_grad: &[Tensor],
               input_grad: &[Tensor]| {
                   unimplemented!();
               }));

reduce_macro!(Variance, "var", var,
              (|input: &[Tensor],
               output_grad: &[Tensor],
               input_grad: &[Tensor]| {
                   unimplemented!();
               }));

reduce_macro!(Max, "max", max,
              (|input: &[Tensor],
               output_grad: &[Tensor],
               input_grad: &[Tensor]| {
                   unimplemented!();
               }));

reduce_macro!(Min, "min", min,
              (|input: &[Tensor],
               output_grad: &[Tensor],
               input_grad: &[Tensor]| {
                   unimplemented!();
               }));
