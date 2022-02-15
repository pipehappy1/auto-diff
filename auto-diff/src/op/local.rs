#![allow(clippy::redundant_closure_call)]
use tensor_rs::tensor::Tensor;
use super::{OpTrait, OpHandle};
use super::macros::new_binary_op;


new_binary_op!(Add, "add",
               (|a:&[Tensor], b:&[Tensor]|
                b[0].swap(&a[0].add(&a[1]))
               ),
               (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
                   let x = input[0].ones_like().mul(&output_grad[0]);
                   let y = input[1].ones_like().mul(&output_grad[0]);
                   input_grad[0].swap(&x);
                   input_grad[1].swap(&y);
               })
);
new_binary_op!(Sub, "sub",
               (|a:&[Tensor], b:&[Tensor]|
                b[0].swap(&a[0].sub(&a[1]))),
               (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
                   let x = input[0].ones_like().mul(&output_grad[0]);
                   let y = input[1].ones_like().neg().mul(&output_grad[0]);
                   input_grad[0].swap(&x);
                   input_grad[1].swap(&y);
               })
);
new_binary_op!(Mul, "mul",
               (|a:&[Tensor], b:&[Tensor]|
                b[0].swap(&a[0].mul(&a[1]))),
               (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
                   let x = input[1].mul(&output_grad[0]);
                   let y = input[0].mul(&output_grad[0]);
                   input_grad[0].swap(&x);
                   input_grad[1].swap(&y);
               })
);
new_binary_op!(Div, "div",
               (|a:&[Tensor], b:&[Tensor]|
                b[0].swap(&a[0].div(&a[1]))),
               (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
                   let x = input[1].reciprocal().mul(&output_grad[0]);
                   let y = input[0].neg().div(&input[1]).div(&input[1]).mul(&output_grad[0]);
                   input_grad[0].swap(&x);
                   input_grad[1].swap(&y);
               })
);

new_binary_op!(Matmul, "matmul",
               (|a:&[Tensor], b:&[Tensor]|
                b[0].swap(&a[0].matmul(&a[1]))),
               (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
                   input_grad[0].swap(&input[1].outer(&output_grad[0], Some(true)));
                   input_grad[1].swap(&input[0].outer(&output_grad[0], Some(true)));
               })
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::_gradient_checker;

    #[test]
    fn matmul() {
        let mut op = Mul::new();

        for i in 0..10 {
            let zero = Tensor::from_vec_f64(&vec![(i - 5) as f64], &vec![1]);
            let zero2 = zero.clone();
            let good_grad = _gradient_checker(&mut op, &[zero, zero2], None, None, None);
            assert_eq!(good_grad, true);                        
        }
    }
}
