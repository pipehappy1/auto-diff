use crate::tensor::Tensor;
use super::OpTrait;

macro_rules! new_binary_op {
    ($a:ident, $b:expr, $c:tt, $d: tt) => {
        pub struct $a {}
        impl $a {
            pub fn new() -> $a{
                $a{}
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
            fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]) {
                $c(input, output)
            }
            fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]) {
                $d(input, output_grad, input_grad)
            }
            fn get_values(&self) -> Vec<&Tensor> {
                Vec::new()
            }
            fn get_grads(&self) -> Vec<&Tensor> {
                Vec::new()
            }
            fn set_values(&self, _v: &[Tensor]) {
            }
        }
    }
}

new_binary_op!(Add, "add",
               (|a:&[&Tensor], b:&[&Tensor]|
                b[0].swap(a[0].add(&a[1]))
               ),
               (|input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]| {
                   let x = input[0].ones_like().mul(output_grad[0]);
                   let y = input[1].ones_like().mul(output_grad[0]);
                   input_grad[0].swap(x);
                   input_grad[1].swap(y);
               })
);
new_binary_op!(Sub, "sub",
               (|a:&[&Tensor], b:&[&Tensor]|
                b[0].swap(a[0].sub(a[1]))),
               (|input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]| {
                   let x = input[0].ones_like().mul(output_grad[0]);
                   let y = input[1].ones_like().neg().mul(output_grad[0]);
                   input_grad[0].swap(x);
                   input_grad[1].swap(y);
               })
);
new_binary_op!(Mul, "mul",
               (|a:&[&Tensor], b:&[&Tensor]|
                b[0].swap(a[0].mul(a[1]))),
               (|input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]| {
                   let x = input[1].mul(output_grad[0]);
                   let y = input[0].mul(output_grad[0]);
                   input_grad[0].swap(x);
                   input_grad[1].swap(y);
               })
);
new_binary_op!(Div, "div",
               (|a:&[&Tensor], b:&[&Tensor]|
                b[0].swap(a[0].div(a[1]))),
               (|input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]| {
                   let x = input[1].reciprocal().mul(output_grad[0]);
                   let y = input[0].neg().div(input[1]).div(input[1]).mul(output_grad[0]);
                   input_grad[0].swap(x);
                   input_grad[1].swap(y);
               })
);
