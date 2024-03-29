#![allow(clippy::redundant_closure_call)]
use super::macros::new_binary_op;
use super::{OpHandle, OpTrait};
use tensor_rs::tensor::Tensor;

#[cfg(feature = "use-serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "use-serde")]
use std::any::Any;

new_binary_op!(
    Add,
    "Add",
    (|a: &[Tensor], b: &[Tensor]| b[0].swap(&a[0].add(&a[1]))),
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        let x = input[0].ones_like().mul(&output_grad[0]);
        let y = input[1].ones_like().mul(&output_grad[0]);
        input_grad[0].swap(&x);
        input_grad[1].swap(&y);
    })
);
new_binary_op!(
    Sub,
    "Sub",
    (|a: &[Tensor], b: &[Tensor]| b[0].swap(&a[0].sub(&a[1]))),
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        let x = input[0].ones_like().mul(&output_grad[0]);
        let y = input[1].ones_like().neg().mul(&output_grad[0]);
        input_grad[0].swap(&x);
        input_grad[1].swap(&y);
    })
);
new_binary_op!(
    Mul,
    "Mul",
    (|a: &[Tensor], b: &[Tensor]| b[0].swap(&a[0].mul(&a[1]))),
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        let x = input[1].mul(&output_grad[0]);
        let y = input[0].mul(&output_grad[0]);
        input_grad[0].swap(&x);
        input_grad[1].swap(&y);
    })
);
new_binary_op!(
    Div,
    "Div",
    (|a: &[Tensor], b: &[Tensor]| b[0].swap(&a[0].div(&a[1]))),
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        let x = input[1].reciprocal().mul(&output_grad[0]);
        let y = input[0]
            .neg()
            .div(&input[1])
            .div(&input[1])
            .mul(&output_grad[0]);
        input_grad[0].swap(&x);
        input_grad[1].swap(&y);
    })
);

new_binary_op!(
    Matmul,
    "Matmul",
    (|a: &[Tensor], b: &[Tensor]| b[0].swap(&a[0].matmul(&a[1]))),
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        input_grad[0].swap(&input[1].outer(&output_grad[0], Some(true)));
        input_grad[1].swap(&input[0].outer(&output_grad[0], Some(true)));
    })
);

new_binary_op!(
    Outer,
    "Outer",
    (|a: &[Tensor], b: &[Tensor]| b[0].swap(&a[0].outer(&a[1], None))),
    (|input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]| {
        unimplemented!();
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
