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

// L1Loss
// MSELoss
/// MSELoss
/// The left-most dimension is the N.
pub struct MSELoss {
 }
impl MSELoss {
    pub fn new() -> MSELoss {
        MSELoss {
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
        let tmp3 = tmp2.sum(None, false);
        let ret = tmp3.div(&input[0].get_n().mul(&input[0].get_c()));
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
    fn set_values(&self, _v: &[Tensor]) {
    }

    fn get_grads(&self) -> Vec<&Tensor> {
        Vec::new()
    }
}


// CrossEntropyLoss
pub struct CrossEntropyLoss {}
impl CrossEntropyLoss {
    pub fn new() -> CrossEntropyLoss {
        CrossEntropyLoss {}
    }
}
impl OpTrait for CrossEntropyLoss {
    fn get_name(&self) -> String {
        "CrossEntropyLoss".to_string()
    }
    fn get_input_size(&self) -> usize {
        2
    }
    fn get_output_size(&self) -> usize {
        1
    }
    /// The first is the prediction, the second input is the label
    /// ORDER IS IMPORTANT, SECOND ARGUMENT WON'T GET GRADEINT.
    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]) {
        if input.len() < 2 {
            panic!("{} expect two input, get {}", self.get_name(), input.len());
        }
        if input[0].size().len() != (input[1].size().len()+1) {
            panic!("{} expect dim+1 and dim, get {}, {}", self.get_name(), input[0].size().len(), input[1].size().len());
        }

        let class_index = input[1].unsqueeze(1);
        let class_score = input[0].gather(1, &class_index);
        let val = class_score.neg().add(&input[0].logsumexp(Some(&[1]), true)).mean(None, false);
        output[0].swap(val);
    }
    
    /// Given the forward input value and backward output_grad,
    /// Update weight gradient.
    /// return backward input gradeint.
    fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]) {
        let max = input[0].max(None, None, Some(false));
        let mut dim_order: Vec<usize> = (0..input[0].size().len()).collect();
        dim_order[0] = 1;
        dim_order[1] = 0;
        let smaller = input[0].sub(&max).permute(&dim_order);
        let new_label = input[1].permute(&dim_order);
        
        let denominator = smaller.exp().sum(Some(&[0]), true);
        
    }

    /// access weight values
    fn get_values(&self) -> Vec<&Tensor> {
        Vec::new()
    }
    fn set_values(&self, _v: &[Tensor]) {
    }
    /// access gradient values
    fn get_grads(&self) -> Vec<&Tensor> {
        Vec::new()
    }
}
// CTCLoss
// NLLLoss
// PoissonNLLLoss
// KLDivLoss
// BCELoss

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
    /// ORDER IS IMPORTANT, SECOND ARGUMENT WON'T GET GRADEINT.
    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]) {
        if input.len() < 2 {
            panic!("{} expect two input, get {}", self.get_name(), input.len());
        }
        let ret_all = input[1].mul(&input[0].neg().log1pexp())
            .add(&(input[1].neg().add(&input[1].ones_like())).mul(&input[0].log1pexp()));
        let tmp3 = ret_all.sum(None, false);
        let ret = tmp3.div(&input[0].get_n().mul(&input[0].get_c()));
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
    
    fn set_values(&self, _v: &[Tensor]) {
        
    }
    
    /// access gradient values
    fn get_grads(&self) -> Vec<&Tensor> {
        Vec::new()
    }
}

// MarginRankingLoss
// HingeEmbeddingLoss
// MultiLabelMarginLoss
// SmoothL1Loss
// SoftMarginLoss
// MultiLabelSoftMarginLoss
// CosineEmbeddingLoss
// MultiMarginLoss
// TripletMarginLoss


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_CrossEntropyLoss() {
        let a = Tensor::from_vec_f32(&vec![1., 2., 3., 4., 5., 6., ], &vec![3, 2]);
        let b = Tensor::from_vec_f32(&vec![0., 0., 1., ], &vec![3]);
        let mut c = CrossEntropyLoss::new();
        let d = Tensor::new();
        c.apply(&[&a, &b], &[&d]);
        assert_eq!(d.get_scale_f32(), 0.97992826);
    }
}
