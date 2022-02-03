#![allow(clippy::redundant_closure_call)]
use tensor_rs::tensor::Tensor;
use super::{OpTrait, OpHandle};

macro_rules! new_element_op {
    ($a:ident, $b:expr, $c:ident, $d: tt) => {
        pub struct $a {
            handle: OpHandle,
        }
        impl $a {
            pub fn new() -> $a{
                $a{
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
        impl OpTrait for $a {
     
            fn get_name(&self) -> String {
                ($b).to_string()
            }
            fn get_input_size(&self) -> usize {
                2
            }
            fn get_output_size(&self) -> usize {
                1
            }
            fn apply(&self, input: &[Tensor], output: &[Tensor]) {
                output[0].swap(&input[0].$c())
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

new_element_op!(Abs,
                "abs",
                abs,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Acos,
                "acos",
                acos,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Asin,
                "asin",
                asin,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Atan,
                "atan",
                atan,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Ceil,
                "ceil",
                ceil,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Cos,
                "cos",
                cos,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Cosh,
                "cosh",
                cosh,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Exp,
                "exp",
                exp,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));


new_element_op!(Expm1,
                "expm1",
                expm1,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Floor,
                "floor",
                floor,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Frac,
                "frac",
                frac,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Log,
                "log",
                log,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Log10,
                "log10",
                log10,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Log1p,
                "log1p",
                log1p,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Log1pexp,
                "log1pexp",
                log1pexp,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Log2,
                "log2",
                log2,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Neg,
                "neg",
                neg,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Reciprocal,
                "reciprocal",
                reciprocal,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Round,
                "round",
                round,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Rsqrt,
                "rsqrt",
                rsqrt,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Sigmoid,
                "sigmoid",
                sigmoid,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Sign,
                "sign",
                sign,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Sin,
                "sin",
                sin,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Sinh,
                "sinh",
                sinh,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Sqrt,
                "sqrt",
                sqrt,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Tan,
                "tan",
                tan,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Tanh,
                "tanh",
                tanh,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));

new_element_op!(Trunc,
                "trunc",
                trunc,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     unimplemented!();
                 }));
