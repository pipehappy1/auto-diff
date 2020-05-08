use crate::tensor::Tensor;
use super::OpTrait;

macro_rules! new_binary_op {
    ($a:ident, $b:expr, $c:tt) => {
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
                println!("binary op grad");
            }
            fn get_values(&self) -> Vec<&Tensor> {
                Vec::new()
            }
            fn get_grads(&self) -> Vec<&Tensor> {
                Vec::new()
            }
            fn set_values(&self, v: &[Tensor]) {
            }
        }
    }
}

new_binary_op!(Add, "add",
               (|a:&[&Tensor], b:&[&Tensor]|
                b[0].swap(a[0].add(&a[1]))
               )
);
new_binary_op!(Sub, "sub",
               (|a:&[&Tensor], b:&[&Tensor]|
                b[0].swap(a[0].sub(a[1])))
);
new_binary_op!(Mul, "mul",
               (|a:&[&Tensor], b:&[&Tensor]|
                b[0].swap(a[0].mul(a[1])))
);
new_binary_op!(Div, "div",
               (|a:&[&Tensor], b:&[&Tensor]|
                b[0].swap(a[0].div(a[1])))
);
