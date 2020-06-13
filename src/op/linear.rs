use crate::tensor::Tensor;
use super::OpTrait;

// Identity

pub struct Linear {
    in_fea: Option<usize>,
    out_fea: Option<usize>,
    bias_option: bool,
    weight: Tensor,
    bias: Tensor,
    weight_grad: Tensor,
    bias_grad: Tensor,
}
impl Linear {
    pub fn new(in_features: Option<usize>, out_features: Option<usize>, bias: bool) -> Linear{
        let mut ret = Linear {
            in_fea: in_features,
            out_fea: out_features,
            bias_option: bias,
            weight: Tensor::new(),
            bias: Tensor::new(),
            weight_grad: Tensor::new(),
            bias_grad: Tensor::new(),
        };
        if ret.in_fea != Option::None && ret.out_fea != Option::None {
            ret._new();
        }
        ret
    }
    fn _new(&mut self) {
        self.weight = Tensor::fill(&vec![self.in_fea.unwrap(), self.out_fea.unwrap()], 0.);
        self.bias = Tensor::fill(&vec![self.out_fea.unwrap(),], 0.);
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }
}
impl OpTrait for Linear {
    fn get_name(&self) -> String {
        "Linear".to_string()
    }
    fn get_input_size(&self) -> usize {
        2
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]) {
        if self.in_fea == None || self.out_fea == None {
            if self.in_fea == None {
                let in_size = input[0].size();
                self.in_fea = Some(in_size[in_size.len()-1]);
            }
            if self.out_fea == None {
                let out_size = output[0].size();
                self.out_fea = Some(out_size[0]);
            }
            self._new();
        }

        //println!("left sie: {:?}, right size: {:?}", input[0].size(), self.weight.size());
        let ret = input[0].matmul(&self.weight);
        output[0].swap(ret);
        //println!("matmut done");
        if self.bias_option {
            let ret = output[0].add(&self.bias);
            output[0].swap(ret);
        }
    }
    fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]) {
        if input.len() < 1 {
            panic!("Expect one input tensor");
        }
        if input[0].size()[1] != self.weight.size()[0] {
            panic!("Expect input dimension matches weight dimension {:?}, {:?}",
                   input[0].size(), self.weight.size());
        }
        if input[0].size()[0] != output_grad[0].size()[0] {
            panic!("Expect input population matches output gradient population {:?}, {:?}",
                   input[0].size(), output_grad[0].size());
        }
        if output_grad[0].size()[1] != self.weight.size()[1] {
            panic!("Expect output gradient dimension matches weight dimension {:?}, {:?}",
                   output_grad[0].size(), self.weight.size());
        }

        input_grad[0].swap(output_grad[0].matmul(&self.weight.permute(&vec![1,0])));
        self.weight_grad.swap(input[0].outer(&output_grad[0], Some(true)));
        if self.bias_option {
            self.bias_grad.swap(output_grad[0].mean(Some(&[0]), false));
        }
    }

    fn get_values(&self) -> Vec<&Tensor> {
        let mut ret = Vec::new();
        ret.push(&self.weight);
        if self.bias_option {
            ret.push(&self.bias);
        }
        ret
    }
    fn set_values(&self, v: &[Tensor]) {
        self.weight.swap(v[0].clone());
        if self.bias_option {
            self.bias.swap(v[1].clone());
        }
    }
    fn get_grads(&self) -> Vec<&Tensor> {
        let mut ret = Vec::new();
        ret.push(&self.weight_grad);
        if self.bias_option {
            ret.push(&self.bias_grad);
        }
        ret
    }
}

// Bilinear
