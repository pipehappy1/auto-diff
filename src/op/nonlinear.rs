use crate::tensor::Tensor;
use super::OpTrait;

/// ELU
pub struct ELU {
}
impl ELU {
    pub fn new() -> ELU {
        ELU {
        }
    }
}
impl OpTrait for ELU {
    fn get_name(&self) -> String {
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
    }
    fn set_values(&self, v: &[Tensor]) {
    }
    /// access gradient values
    fn get_grads(&self) -> Vec<&Tensor> {
        
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
    }
    fn set_values(&self, v: &[Tensor]) {
    }
    /// access gradient values
    fn get_grads(&self) -> Vec<&Tensor> {
        
    }
}


// ReLU6
// RReLU
// SELU
// CELU
// Sigmoid
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
