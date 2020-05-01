/// Only NCWH format is supported.
use std::cell::{RefCell, Ref};
use std::rc::Rc;

use super::tensor::Tensor;

/// All op is OpTrait
pub trait OpTrait {
    fn get_name(&self) -> String;

    /// Forward pass
    fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]);
    
    /// Given the forward input value and backward output_grad,
    /// return backward input gradeint.
    fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]);

    fn get_values(&self) -> Vec<&Tensor>;
}


/// Op is the Rc wrapper of OpTraint
pub struct Op {
    o: Rc<RefCell<Box<dyn OpTrait>>>,
}
impl Op {
    pub fn new(o: Box<dyn OpTrait>) -> Self {
        Op {
            o: Rc::new(RefCell::new(o)),
        }
    }

    pub fn get(&self) -> Ref<Box<dyn OpTrait>> {
        self.o.borrow()
    }

    pub fn get_name(&self) -> String {
        self.o.borrow_mut().get_name()
    }
    pub fn apply(&self, input: &[&Tensor], output: &[&Tensor]) {
        self.o.borrow_mut().apply(input, output)
    }
    pub fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]) {
        self.o.borrow_mut().grad(input, output_grad, input_grad);
    }

    pub fn get_values(&self) -> Vec<Tensor> {
        let mut ret = Vec::new();
        for i in self.o.borrow().get_values() {
            ret.push(i.clone());
        }
        ret
    }
}
impl Clone for Op {
    fn clone(&self) -> Self {
        Op {
            o: Rc::clone(&self.o),
        }
    }
}




macro_rules! new_binary_op {
    ($a:ident, $b:expr, $c:tt) => {
        pub struct $a {}
        impl $a {
            pub fn new() -> $a{
                $a{}
            }
        }
        impl OpTrait for $a {
            fn get_name(&self) -> String {
                ($b).to_string()
            }
            fn apply(&mut self, input: &[&Tensor], output: &[&Tensor]) {
                $c(input, output)
            }
            fn grad(&self, input: &[&Tensor], output_grad: &[&Tensor], input_grad: &[&Tensor]) {
                println!("binary op grad");
            }
            fn get_values(&self) -> Vec<&Tensor> {
                Vec::new()
            }
        }
    }
}

new_binary_op!(Add, "add",
               (|a:&[&Tensor], b:&[&Tensor]|
                b[0].swap(a[0].add(&a[1]))
               )
);
new_binary_op!(Sub, "sub",
               (|a:&[&Tensor], b:&[&Tensor]|
                b[0].swap(a[0].sub(a[1])))
);
new_binary_op!(Mul, "mul",
               (|a:&[&Tensor], b:&[&Tensor]|
                b[0].swap(a[0].mul(a[1])))
);
new_binary_op!(Div, "div",
               (|a:&[&Tensor], b:&[&Tensor]|
                b[0].swap(a[0].div(a[1])))
);


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

        let ret = input[0].matmul(&self.weight);
        output[0].swap(ret);

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
        self.weight_grad.swap(input[0].outer(&output_grad[0]).mean(0, false));
        if self.bias_option {
            self.bias_grad.swap(output_grad[0].mean(0, false));
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
}

// Bilinear

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
}
