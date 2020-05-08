use crate::tensor::Tensor;
use super::OpTrait;

/// ELU
pub struct ELU {
    alpha: f32,
}
impl ELU {
    pub fn new(alpha: f32) -> ELU {
        ELU {
            alpha: alpha,
        }
    }
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
    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]) {
        
    }
    
    /// Given the forward input value and backward output_grad,
    /// Update weight gradient.
    /// return backward input gradeint.
    fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]) {
    }

    /// access weight values
    fn get_values(&self) -> Vec<&Tensor> {
        Vec::new()
    }
    fn set_values(&self, v: &[Tensor]) {
    }
    /// access gradient values
    fn get_grads(&self) -> Vec<&Tensor> {
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
}
impl ReLU {
    pub fn new() -> ReLU {
        ReLU {
        }
    }
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
    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]) {
        
    }
    
    /// Given the forward input value and backward output_grad,
    /// Update weight gradient.
    /// return backward input gradeint.
    fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]) {
    }

    /// access weight values
    fn get_values(&self) -> Vec<&Tensor> {
        Vec::new()
    }
    fn set_values(&self, v: &[Tensor]) {
    }
    /// access gradient values
    fn get_grads(&self) -> Vec<&Tensor> {
        Vec::new()
    }
}


// ReLU6
// RReLU
// SELU
// CELU
// Sigmoid
pub struct Sigmoid {
    
}
impl Sigmoid {
    pub fn new() -> Sigmoid {
        Sigmoid {
        }
    }
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
    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]) {
        if input.len() < 1 {
            panic!("{} expect two input, get {}", self.get_name(), input.len());
        }
        output[0].swap(input[0].sigmoid());
    }
    
    /// Given the forward input value and backward output_grad,
    /// Update weight gradient.
    /// return backward input gradeint.
    fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]) {
        let tmp1 = input[0].sigmoid().mul(&input[0].neg().sigmoid());
        let tmp2 = tmp1.mul(output_grad[0]);
        input_grad[0].swap(tmp2);
    }

    /// access weight values
    fn get_values(&self) -> Vec<&Tensor> {
        Vec::new()
    }
    
    fn set_values(&self, v: &[Tensor]) {
        
    }
    
    /// access gradient values
    fn get_grads(&self) -> Vec<&Tensor> {
        Vec::new()
    }
}

// Softplus
// Softshrink
// Softsign
// Tanh
// Tanhshrink
// Threshold

// Softmin
// Softmax
// Softmax2d
// LogSoftmax
// AdaptiveLogSoftmaxWithLoss


//pub struct ELU {
//}
//impl ELU {
//    pub fn new() -> ELU {
//        ELU {
//        }
//    }
//}
//impl OpTrait for ELU {
//    fn get_name(&self) -> String {
//    }
//
//    /// Forward pass
//    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]) {
//        
//    }
//    
//    /// Given the forward input value and backward output_grad,
//    /// Update weight gradient.
//    /// return backward input gradeint.
//    fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]) {
//    }
//
//    /// access weight values
//    fn get_values(&self) -> Vec<&Tensor> {
//    }
//    fn set_values(&self, v: &[Tensor]) {
//    }
//    /// access gradient values
//    fn get_grads(&self) -> Vec<&Tensor> {
//        
//    }
//}
