use crate::tensor::Tensor;
use super::OpTrait;


//
// Common Cost function
//
pub enum Reduction{
    None,
    Mean,
    Sum,
}

/// MSELoss
/// The left-most dimension is the N.
pub struct MSELoss {
    reduction: Reduction,
}
impl MSELoss {
    pub fn new() -> MSELoss {
        MSELoss {
            reduction: Reduction::None,
        }
    }
}
impl OpTrait for MSELoss {
    fn get_name(&self) -> String {
        "MSE".to_string()
    }
    fn get_input_size(&self) -> usize {
        2
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]) {
        // TODO: wait for Tensor to have lazy evaluation for elemwise operation.
        let tmp = input[0].sub(input[1]);
        let tmp2 = tmp.mul(&tmp);
        let tmp3 = tmp2.sum();
        let ret = tmp3.div(&input[0].get_N().mul(&input[0].get_C()));
        output[0].swap(ret);
    }
    fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]) {
        
        if input.len() < 2 {
            panic!("MSELoss expect two input, get {}", input.len());
        }
        if input_grad.len() < 2 {
            panic!("MSELoss expect two input gradient tensor, get {}", input_grad.len());
        }
        if output_grad.len() < 1 {
            panic!("MSELoss expect one output gradient, get {}", output_grad.len());
        }
        if ! input[0].same_shape(input[1]) {
            panic!("MSELoss expect two input have the same shape, get {:?}, {:?}", input[0].size(), input[1].size());
        }


        let tmp1 = input[0].sub(input[1]);
        let tmp2 = tmp1.div(&input[0].numel_tensor());
        let tmp3 = tmp2.mul(output_grad[0]);
        input_grad[0].swap(tmp3);

        let tmp1 = input[1].sub(input[0]);
        let tmp2 = tmp1.div(&input[0].numel_tensor());
        let tmp3 = tmp2.mul(output_grad[0]);
        input_grad[1].swap(tmp3);
    }

    fn get_values(&self) -> Vec<&Tensor> {
        Vec::new()
    }
    fn set_values(&self, v: &[Tensor]) {
    }

    fn get_grads(&self) -> Vec<&Tensor> {
        Vec::new()
    }
}


/// This loss combines a Sigmoid layer and the BCELoss in one single class.
/// This version is more numerically stable than using a plain Sigmoid followed
/// by a BCELoss as, by combining the operations into one layer,
/// we take advantage of the log-sum-exp trick for numerical stability.
///
/// -y log (1/(1 + exp(-x))) - (1-y) log(1 - 1/(1 + exp(-x)))
/// 
/// Prediction comes first, label comes second.
pub struct BCEWithLogitsLoss {
    
}
impl BCEWithLogitsLoss {
    pub fn new() -> BCEWithLogitsLoss {
        BCEWithLogitsLoss {
        }
    }
}
impl OpTrait for BCEWithLogitsLoss {
    
    fn get_name(&self) -> String {
        "BCEWithLogitsLoss".to_string()
    }
    fn get_input_size(&self) -> usize {
        2
    }
    fn get_output_size(&self) -> usize {
        1
    }
    /// The first is the prediction, the second input is the label
    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]) {
        if input.len() < 2 {
            panic!("{} expect two input, get {}", self.get_name(), input.len());
        }
        let ret_all = input[1].mul(&input[0].neg().log1pexp())
            .add(&(input[1].neg().add(&input[1].ones_like())).mul(&input[0].log1pexp()));
        let tmp3 = ret_all.sum();
        let ret = tmp3.div(&input[0].get_N().mul(&input[0].get_C()));
        output[0].swap(ret);
    }
    
    /// Given the forward input value and backward output_grad,
    /// Update weight gradient.
    /// return backward input gradeint.
    fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]) {
        // ddx y log (1 + exp(-x)) = -y  / (1 + exp(x))
        // ddx (1-y) log (1 + exp(x)) = (1-y) / (1 + exp(-x))
        let ones = Tensor::ones_like(&input[0]);
        let tmp1 = input[1].neg().div(&input[0].exp().add(&ones));
        let tmp2 = input[1].neg().add(&ones).div(&input[0].neg().exp().add(&ones));
        let tmp3 = tmp1.add(&tmp2);
        let tmp4 = tmp3.mul(output_grad[0]);
        
        let zeros = Tensor::zeros_like(input[0]);
        input_grad[0].swap(tmp4);
        input_grad[1].swap(zeros);
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
