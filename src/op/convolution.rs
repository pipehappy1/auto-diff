use crate::tensor::Tensor;
use super::OpTrait;

pub struct Conv1d {
    alpha: f32,
}
impl Conv1d {
    pub fn new(alpha: f32) -> Conv1d {
        Conv1d {
            alpha: alpha,
        }
    }
}
impl OpTrait for Conv1d {
    fn get_name(&self) -> String {
        "ELU".to_string()
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

// Conv2d
// Conv3d
// ConvTranspose1d
// ConvTranspose2d
// ConvTranspose3d
// Unfold
// Fold
