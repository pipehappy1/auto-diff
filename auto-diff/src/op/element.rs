#![allow(clippy::redundant_closure_call)]
use tensor_rs::tensor::Tensor;
use super::{OpTrait, OpHandle};
use super::macros::new_element_op;

#[cfg(feature = "use-serde")]
use serde::{Serialize, Deserialize};
#[cfg(feature = "use-serde")]
use std::any::Any;


new_element_op!(Abs,
                "abs",
                abs,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     input_grad[0].swap(
                         &input[0].conditional_select(
                             &input[0].ones_like(),
                             &input[0].ones_like().neg())
                             .mul(&output_grad[0]));
                 }));

new_element_op!(Acos,
                "acos",
                acos,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     let ret = input[0].ones_like().sub(&input[0].mul(&input[0])).sqrt().reciprocal().neg();
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Asin,
                "asin",
                asin,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     let ret = input[0].ones_like().sub(&input[0].mul(&input[0])).sqrt().reciprocal();
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Atan,
                "atan",
                atan,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     let ret = input[0].ones_like().add(&input[0].mul(&input[0])).reciprocal();
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Ceil,
                "ceil",
                ceil,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     input_grad[0].swap(&input[0].zeros_like());
                 }));

new_element_op!(Cos,
                "cos",
                cos,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     let ret = input[0].sin().neg();
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Cosh,
                "cosh",
                cosh,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     let ret = input[0].sinh();
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Exp,
                "exp",
                exp,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     let ret = input[0].exp();
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));


new_element_op!(Expm1,
                "expm1",
                expm1,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     let ret = input[0].exp();
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Floor,
                "floor",
                floor,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     input_grad[0].swap(&input[0].zeros_like());
                 }));

new_element_op!(Frac,
                "frac",
                frac,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     input_grad[0].swap(&input[0].ones_like());
                 }));

new_element_op!(Log,
                "log",
                log,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     let ret = input[0].reciprocal();
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Log10,
                "log10",
                log10,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     let ret = input[0].reciprocal().div(&input[0].log10_like());
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Log1p,
                "log1p",
                log1p,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     let ret = input[0].add(&input[0].ones_like()).reciprocal();
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Log1pexp,
                "log1pexp",
                log1pexp,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     let ret = input[0].neg().exp().add(&input[0].ones_like()).reciprocal();
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Log2,
                "log2",
                log2,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     let ret = input[0].reciprocal().div(&input[0].log2_like());
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Neg,
                "neg",
                neg,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     let ret = input[0].ones_like().neg();
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Reciprocal,
                "reciprocal",
                reciprocal,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     let ret = input[0].square().reciprocal().neg();
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Round,
                "round",
                round,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     let ret = input[0].zeros_like();
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Rsqrt,
                "rsqrt",
                rsqrt,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     let ret = input[0].sqrt().reciprocal().
                         div(&input[0]).neg().div(
			 &input[0].ones_like().add(&input[0].ones_like()));
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Sigmoid,
                "sigmoid",
                sigmoid,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     let ret = input[0].sigmoid().mul(&input[0].sigmoid().neg().add(&input[0].ones_like()));
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Sign,
                "sign",
                sign,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     let ret = input[0].zeros_like();
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Sin,
                "sin",
                sin,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
                     let ret = input[0].cos();
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Sinh,
                "sinh",
                sinh,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     let ret = input[0].cosh();
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Sqrt,
                "sqrt",
                sqrt,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     let ret = input[0].sqrt().reciprocal().div(
			 &input[0].ones_like().add(&input[0].ones_like()));
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Tan,
                "tan",
                tan,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     let ret = input[0].tan().square().add(&input[0].ones_like());
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Tanh,
                "tanh",
                tanh,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     let ret = input[0].tanh().square().neg().add(&input[0].ones_like());
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));

new_element_op!(Trunc,
                "trunc",
                trunc,
                (|input: &[Tensor],
                 output_grad: &[Tensor],
                 input_grad: &[Tensor]| {
		     let ret = input[0].zeros_like();
		     input_grad[0].swap(&ret.mul(&output_grad[0]));
                 }));


#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::_gradient_checker;

    fn test_range_data(op: &mut dyn OpTrait) {
        for i in 0..10 {
            let zero = Tensor::from_vec_f64(&vec![(i as f64 / 10.0 - 0.51)], &vec![1]);
            let good_grad = _gradient_checker(op, &[zero], None, None, None);
            assert_eq!(good_grad, true);                        
        }
    }

    #[test]
    fn abs() {
        let mut op = Abs::new();
        test_range_data(&mut op);
    }

    #[test]
    fn acos() {
        let mut op = Acos::new();
        test_range_data(&mut op);
    }

    #[test]
    fn asin() {
        let mut op = Asin::new();
        test_range_data(&mut op);
    }

    #[test]
    fn atan() {
        let mut op = Atan::new();
        test_range_data(&mut op);
    }

    #[test]
    fn ceil() {
        let mut op = Ceil::new();
        test_range_data(&mut op);
    }

    #[test]
    fn cos() {
        let mut op = Cos::new();
        test_range_data(&mut op);
    }

    #[test]
    fn cosh() {
        let mut op = Cosh::new();
        test_range_data(&mut op);
    }

    #[test]
    fn exp() {
        let mut op = Exp::new();
        test_range_data(&mut op);
    }

    #[test]
    fn expm1() {
        let mut op = Expm1::new();
        test_range_data(&mut op);
    }

    #[test]
    fn floor() {
        let mut op = Floor::new();
        test_range_data(&mut op);
    }

    #[test]
    fn frac() {
        let mut op = Frac::new();
        test_range_data(&mut op);
    }

    #[test]
    fn log() {
        let mut op = Log::new();
        for i in 0..10 {
            let zero = Tensor::from_vec_f64(&vec![(i as f64 / 10.0 + 0.51)], &vec![1]);
            let good_grad = _gradient_checker(&mut op, &[zero], None, None, None);
            assert_eq!(good_grad, true);                        
        }
    }

    #[test]
    fn log10() {
        let mut op = Log10::new();
        for i in 0..10 {
            let zero = Tensor::from_vec_f64(&vec![(i as f64 / 10.0 + 0.51)], &vec![1]);
            let good_grad = _gradient_checker(&mut op, &[zero], None, None, None);
            assert_eq!(good_grad, true);                        
        }
    }

    #[test]
    fn log1p() {
        let mut op = Log1p::new();
        for i in 0..10 {
            let zero = Tensor::from_vec_f64(&vec![(i as f64 / 10.0 - 0.51)], &vec![1]);
            let good_grad = _gradient_checker(&mut op, &[zero], None, None, None);
            assert_eq!(good_grad, true);                        
        }
    }

    #[test]
    fn log1pexp() {
        let mut op = Log1pexp::new();
        for i in 0..10 {
            let zero = Tensor::from_vec_f64(&vec![(i as f64 / 10.0 - 0.51)], &vec![1]);
            let good_grad = _gradient_checker(&mut op, &[zero], None, None, None);
            assert_eq!(good_grad, true);                        
        }
    }

    #[test]
    fn log2() {
        let mut op = Log2::new();
        for i in 0..10 {
            let zero = Tensor::from_vec_f64(&vec![(i as f64 / 10.0 + 0.51)], &vec![1]);
            let good_grad = _gradient_checker(&mut op, &[zero], None, None, None);
            assert_eq!(good_grad, true);                        
        }
    }

    #[test]
    fn neg() {
        let mut op = Neg::new();
        test_range_data(&mut op);
    }

    #[test]
    fn reciprocal() {
        let mut op = Reciprocal::new();
        for i in 0..10 {
            let zero = Tensor::from_vec_f64(&vec![(i as f64 / 10.0 + 0.51)], &vec![1]);
            let good_grad = _gradient_checker(&mut op, &[zero], None, None, None);
            assert_eq!(good_grad, true);                        
        }
    }

    #[test]
    fn round() {
        let mut op = Round::new();
        test_range_data(&mut op);
    }

    #[test]
    fn rsqrt() {
        let mut op = Rsqrt::new();
        for i in 0..10 {
            let zero = Tensor::from_vec_f64(&vec![(i as f64 / 10.0 + 0.51)], &vec![1]);
            let good_grad = _gradient_checker(&mut op, &[zero], None, None, None);
            assert_eq!(good_grad, true);                        
        }
    }

    #[test]
    fn sigmoid() {
        let mut op = Sigmoid::new();
        test_range_data(&mut op);
    }

    #[test]
    fn sign() {
        let mut op = Sign::new();
        test_range_data(&mut op);
    }

    #[test]
    fn sinh() {
        let mut op = Sinh::new();
        test_range_data(&mut op);
    }

    #[test]
    fn sqrt() {
        let mut op = Sqrt::new();
        test_range_data(&mut op);
    }

    #[test]
    fn tan() {
        let mut op = Tan::new();
        test_range_data(&mut op);
    }

    #[test]
    fn tanh() {
        let mut op = Tanh::new();
        test_range_data(&mut op);
    }

    #[test]
    fn trunc() {
        let mut op = Trunc::new();
        test_range_data(&mut op);
    }
}
