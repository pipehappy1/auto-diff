/// Only NCWH format is supported.

use super::tensor::Tensor;

pub trait Op {
    fn get_name(&self) -> &str;
    fn apply(&mut self, input: &Vec<&Tensor>, output: &Vec<&Tensor>);
    fn grad(&self, input: u32, output: u32);
}

macro_rules! new_binary_op {
    ($a:ident, $b:expr, $c:tt) => {
        pub struct $a {}
        impl $a {
            pub fn new() -> $a{
                $a{}
            }
        }
        impl Op for $a {
            fn get_name(&self) -> &str {
                $b
            }
            fn apply(&mut self, input: &Vec<&Tensor>, output: &Vec<&Tensor>) {
                $c(input, output)
            }
            fn grad(&self, input: u32, output: u32) {
                
            }       
        }
    }
}

new_binary_op!(add, "add",
               (|a:&Vec<&Tensor>, b:& Vec<&Tensor>|
                b[0].swap(a[0].add(&a[1]))
               )
);
new_binary_op!(sub, "sub",
               (|a:&Vec<&Tensor>, b:&Vec<&Tensor>|
                b[0].swap(a[0].sub(a[1])))
);
new_binary_op!(mul, "mul",
               (|a:&Vec<&Tensor>, b:&Vec<&Tensor>|
                b[0].swap(a[0].mul(a[1])))
);
new_binary_op!(div, "div",
               (|a:&Vec<&Tensor>, b:&Vec<&Tensor>|
                b[0].swap(a[0].div(a[1])))
);


// Identity

pub struct Linear {
    in_fea: Option<usize>,
    out_fea: Option<usize>,
    bias_option: bool,
    weight: Tensor,
    bias: Tensor,
}
impl Linear {
    pub fn new(in_features: Option<usize>, out_features: Option<usize>, bias: bool) -> Linear{
        let mut ret = Linear {
            in_fea: in_features,
            out_fea: out_features,
            bias_option: bias,
            weight: Tensor::new(),
            bias: Tensor::new(),
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
impl Op for Linear {
    fn get_name(&self) -> &str {
        "Linear"
    }
    fn apply(&mut self, input: &Vec<&Tensor>, output: &Vec<&Tensor>) {
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
        let ret = input[0].matmul(&self.weight);
        output[0].swap(ret);

        if self.bias_option {
            let ret = output[0].add(&self.bias);
            output[0].swap(ret);
        }


    }
    fn grad(&self, input: u32, output: u32) {
        
    }

}

// Bilinear

//
// Common Cost function
//
enum Reduction{
    None,
    Mean,
    Sum,
}

pub struct MSELoss {
    reduction: Reduction,
}
impl MSELoss {
    
}
impl Op for MSELoss {
    fn get_name(&self) -> &str {
        "MSE"
    }
    fn apply(&mut self, input: &Vec<&Tensor>, output: &Vec<&Tensor>) {
        if input[0].size().iter().zip(input[1].size().iter()).all(|x|x.0==x.1) {
            
        } else {
            panic!("MSELoss sees two input differ in shape.");
        }
    }
    fn grad(&self, input: u32, output: u32) {
        
    }
}
