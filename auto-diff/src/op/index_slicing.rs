#![allow(clippy::redundant_closure_call)]
use tensor_rs::tensor::Tensor;
use super::{OpTrait, OpCall, Op, OpHandle};

use std::cell::{RefCell};
use std::rc::Rc;

use crate::var::{Var};
use crate::err::AutoDiffError;

macro_rules! new_1_to_1_op_with_paras {
    ($a:ident, $b:expr, $c:ident, $d: tt, $( $arg_name:ident : $ArgTy:ty ),* $(,)?) => {
        pub struct $a {
            handle: OpHandle,
            $( $arg_name : $ArgTy ),*
        }
        impl $a {
            pub fn new($( $arg_name : $ArgTy ),*) -> $a{
                $a{
                    handle: OpHandle::new(),
                    $( $arg_name ),*
                }
            }
            fn get_handle(&self) -> &OpHandle {
                &self.handle
            }
            fn get_handle_mut(&mut self) -> &mut OpHandle {
                &mut self.handle
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
                output[0].swap(&input[0].$c($( self.$arg_name ),*))
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

//new_1_to_1_op_with_paras!(Cat,
//                          "cat",
//                          cat,
//                          (|input: &[Tensor],
//                           output_grad: &[Tensor],
//                           input_grad: &[Tensor]| {
//                               unimplemented!();
//                           }),
//                          tensors: &[Tensor], dim: usize);

pub struct Cat {
    handle: OpHandle,
    dim: usize
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
            dim: self.dim,
            handle: OpHandle::new(),
        };

        let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

        Ok(inputs[0].called_with(op, &inputs[1..inputs.len()])?)
    }
}
impl OpTrait for Cat {

    fn get_name(&self) -> String {
        "cat".to_string()
    }
    fn get_input_size(&self) -> usize {
        2
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        output[0].swap(&input[0].cat(&input[1..input.len()], self.dim));
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
