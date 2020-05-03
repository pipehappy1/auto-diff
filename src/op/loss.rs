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
