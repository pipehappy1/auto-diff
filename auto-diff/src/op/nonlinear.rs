#![allow(clippy::new_without_default)]
use tensor_rs::tensor::Tensor;
use super::{OpTrait, OpHandle};


/// ELU
pub struct ELU {
    alpha: Tensor,
    handle: OpHandle,
}
impl ELU {

    pub fn new(alpha: Tensor) -> ELU {
        ELU {
            alpha,
            handle: OpHandle::new(),
        }
    }

    
    handle_method!();
}
impl OpTrait for ELU {
    fn get_name(&self) -> String {
        "ELU".to_string()
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }

    /// Forward pass
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        let positive = input[0].max_pair(&input[0].zeros_like());
        let negative = input[0].expm1().mul(&Tensor::fill(&input[0].size(), &self.alpha)).min_pair(&input[0].zeros_like());
        let ret = positive.add(&negative);
        output[0].swap(&ret);
    }
    
    /// Given the forward input value and backward output_grad,
    /// Update weight gradient.
    /// return backward input gradeint.
    fn grad(&self, input: &[Tensor],
            output_grad: &[Tensor],
            input_grad: &[Tensor]) {
        let positive = input[0].ge(&input[0].zeros_like());
        let negative = input[0].lt(&input[0].zeros_like()).mul(&Tensor::fill(&input[0].size(), &self.alpha)).mul(&input[0].exp());
        let g = positive.add(&negative);
        input_grad[0].swap(&g.mul(&output_grad[0]));
    }

    /// access weight values
    fn get_values(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn set_values(&self, _v: &[Tensor]) {
    }
    /// access gradient values
    fn get_grads(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

// Hardshrink
// Hardtanh
// LeakyReLU
// LogSigmoid
// MultiheadAttention
// PReLU

/// ReLU
pub struct ReLU {
    handle: OpHandle,
}
impl ReLU {
    pub fn new() -> ReLU {
        ReLU {
            handle: OpHandle::new(),
        }
    }

    handle_method!();
}
impl OpTrait for ReLU {
    fn get_name(&self) -> String {
        "ReLU".to_string()
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    /// Forward pass
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        let ret = input[0].max_pair(&input[0].zeros_like());
        output[0].swap(&ret);
    }
    
    /// Given the forward input value and backward output_grad,
    /// Update weight gradient.
    /// return backward input gradeint.
    fn grad(&self, input: &[Tensor],
            output_grad: &[Tensor],
            input_grad: &[Tensor]) {
        let ret = input[0].ge(&input[0].zeros_like()); // gradient at 0 is 1. thus use right gradient.
        input_grad[0].swap(&ret.mul(&output_grad[0]));
    }

    /// access weight values
    fn get_values(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn set_values(&self, _v: &[Tensor]) {
    }
    /// access gradient values
    fn get_grads(&self) -> Vec<Tensor> {
        Vec::new()
    }
}


// ReLU6
// RReLU
// SELU
// CELU
// Sigmoid
pub struct Sigmoid {
    handle: OpHandle,    
}
impl Sigmoid {
    pub fn new() -> Sigmoid {
        Sigmoid {
            handle: OpHandle::new(),
        }
    }
    handle_method!();
}
impl OpTrait for Sigmoid {
    
    fn get_name(&self) -> String {
        "Sigmoid".to_string()
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    /// The first is the prediction, the second input is the label
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        if input.is_empty() {
            panic!("{} expect two input, get {}", self.get_name(), input.len());
        }
        output[0].swap(&input[0].sigmoid());
    }
    
    /// Given the forward input value and backward output_grad,
    /// Update weight gradient.
    /// return backward input gradeint.
    fn grad(&self, input: &[Tensor],
            output_grad: &[Tensor],
            input_grad: &[Tensor]) {
        let tmp1 = input[0].sigmoid().mul(&input[0].neg().sigmoid());
        let tmp2 = tmp1.mul(&output_grad[0]);
        input_grad[0].swap(&tmp2);
    }

    /// access weight values
    fn get_values(&self) -> Vec<Tensor> {
        Vec::new()
    }
    
    fn set_values(&self, _v: &[Tensor]) {
        
    }
    
    /// access gradient values
    fn get_grads(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

// Softplus
// Softshrink
// Softsign
// Tanh
// Tanhshrink
// Threshold

pub struct Sine {
    handle: OpHandle,
}
impl Sine {
    pub fn new() -> Sine {
        Sine {
            handle: OpHandle::new(),
        }
    }
    handle_method!();
}
impl OpTrait for Sine {
    fn get_name(&self) -> String {
        "Sine".to_string()
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    /// Forward pass
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        let ret = input[0].sin();
        output[0].swap(&ret);
    }
    
    /// Given the forward input value and backward output_grad,
    /// Update weight gradient.
    /// return backward input gradeint.
    fn grad(&self, input: &[Tensor],
            output_grad: &[Tensor],
            input_grad: &[Tensor]) {
        let ret = input[0].cos();
        input_grad[0].swap(&ret.mul(&output_grad[0]));
    }

    /// access weight values
    fn get_values(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn set_values(&self, _v: &[Tensor]) {
    }
    /// access gradient values
    fn get_grads(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

// Softmin
// Softmax
// Softmax2d
// LogSoftmax
// AdaptiveLogSoftmaxWithLoss



#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::_gradient_checker;

    #[test]
    fn elu() {
        let mut op = ELU::new(1.);

        for i in 0..10 {
            let zero = Tensor::from_vec_f64(&vec![(i - 5) as f64], &vec![1]);
            let good_grad = _gradient_checker(&mut op, &[zero], None, None, None);
            assert_eq!(good_grad, true);                        
        }


    }

    #[test]
    fn relu() {
        let mut op = ReLU::new();

        for i in 0..10 {
            let zero = Tensor::from_vec_f64(&vec![(i - 5) as f64], &vec![1]);
            let good_grad = _gradient_checker(&mut op, &[zero], None, None, None);
            assert_eq!(good_grad, true);                        
        }
    }

    #[test]
    fn sigmoid() {
        let mut op = Sigmoid::new();

        for i in 0..10 {
            let zero = Tensor::from_vec_f64(&vec![(i - 5) as f64], &vec![1]);
            let good_grad = _gradient_checker(&mut op, &[zero], None, None, None);
            assert_eq!(good_grad, true);                        
        }
    }

    #[test]
    fn sine() {
        let mut op = Sine::new();

        for i in 0..10 {
            let zero = Tensor::from_vec_f64(&vec![(i - 5) as f64], &vec![1]);
            let good_grad = _gradient_checker(&mut op, &[zero], None, None, None);
            assert_eq!(good_grad, true);                        
        }
    }
}
