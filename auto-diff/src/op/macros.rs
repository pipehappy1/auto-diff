#![allow(clippy::redundant_closure_call)]

macro_rules! one_to_1_op_with_paras {
    ($a:ident, $b:expr, $is:expr,$os:expr, $c:ident, $d: tt, $( $arg_name:ident : $ArgTy:ty ),* $(,)?) => {
        pub struct $a {
            handle: OpHandle,
            $( $arg_name : $ArgTy ),*
        }
        impl $a {
            pub fn new($( $arg_name : $ArgTy ),*) -> $a{
                $a{
                    handle: OpHandle::new(),
                    $( $arg_name ),*
                }
            }
            fn get_handle(&self) -> &OpHandle {
                &self.handle
            }
            fn get_handle_mut(&mut self) -> &mut OpHandle {
                &mut self.handle
            }
        }
        impl OpCall for $a {
            fn call(&mut self, inputs: &[&Var]) -> Result<Vec<Var>, AutoDiffError> {
                let new_one = $a {
                    handle: OpHandle::new(),
                    $( $arg_name : self.$arg_name ),*
                };

                let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

                inputs[0].called_with(op, &inputs[1..inputs.len()])
            }
        }
        impl OpTrait for $a {
     
            fn get_name(&self) -> String {
                ($b).to_string()
            }
            fn get_input_size(&self) -> usize {
                $is
            }
            fn get_output_size(&self) -> usize {
                $os
            }
            fn apply(&self, input: &[Tensor], output: &[Tensor]) {
                output[0].swap(&input[0].$c($( self.$arg_name ),*))
            }
            fn grad(&self, input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]) {
                $d(input, output_grad, input_grad)
            }
            fn get_values(&self) -> Vec<Tensor> {
                Vec::new()
            }
            fn get_grads(&self) -> Vec<Tensor> {
                Vec::new()
            }
            fn set_values(&self, _v: &[Tensor]) {
            }
        }
    }
}

macro_rules! many_to_1_op_with_paras {
    ($a:ident, $b:expr, $is:expr,$os:expr, $c:ident, $d: tt, $( $arg_name:ident : $ArgTy:ty ),* $(,)?) => {
        pub struct $a {
            handle: OpHandle,
            $( $arg_name : $ArgTy ),*
        }
        impl $a {
            pub fn new($( $arg_name : $ArgTy ),*) -> $a{
                $a{
                    handle: OpHandle::new(),
                    $( $arg_name ),*
                }
            }
            fn get_handle(&self) -> &OpHandle {
                &self.handle
            }
            fn get_handle_mut(&mut self) -> &mut OpHandle {
                &mut self.handle
            }
        }
        impl OpCall for $a {
            fn call(&mut self, inputs: &[&Var]) -> Result<Vec<Var>, AutoDiffError> {
                let new_one = $a {
                    handle: OpHandle::new(),
                    $( $arg_name : self.$arg_name ),*
                };

                let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

                inputs[0].called_with(op, &inputs[1..inputs.len()])
            }
        }
        impl OpTrait for $a {
     
            fn get_name(&self) -> String {
                ($b).to_string()
            }
            fn get_input_size(&self) -> usize {
                $is
            }
            fn get_output_size(&self) -> usize {
                $os
            }
            fn apply(&self, input: &[Tensor], output: &[Tensor]) {
                output[0].swap(&input[0].$c(&input[1..input.len()], $( self.$arg_name ),*))
            }
            fn grad(&self, input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]) {
                $d(input, output_grad, input_grad)
            }
            fn get_values(&self) -> Vec<Tensor> {
                Vec::new()
            }
            fn get_grads(&self) -> Vec<Tensor> {
                Vec::new()
            }
            fn set_values(&self, _v: &[Tensor]) {
            }
        }
    }
}

macro_rules! one_to_vec_op_with_paras {
    ($a:ident, $b:expr, $is:expr,$os:expr, $c:ident, $d: tt, $( $arg_name:ident : $ArgTy:ty ),* $(,)?) => {
        pub struct $a {
            handle: OpHandle,
            $( $arg_name : $ArgTy ),*
        }
        impl $a {
            pub fn new($( $arg_name : $ArgTy ),*) -> $a{
                $a{
                    handle: OpHandle::new(),
                    $( $arg_name ),*
                }
            }
            fn get_handle(&self) -> &OpHandle {
                &self.handle
            }
            fn get_handle_mut(&mut self) -> &mut OpHandle {
                &mut self.handle
            }
        }
        impl OpCall for $a {
            fn call(&mut self, inputs: &[&Var]) -> Result<Vec<Var>, AutoDiffError> {
                let new_one = $a {
                    handle: OpHandle::new(),
                    $( $arg_name : self.$arg_name ),*
                };

                let op = Op::new(Rc::new(RefCell::new(Box::new(new_one))));

                inputs[0].called_with(op, &inputs[1..inputs.len()])
            }
        }
        impl OpTrait for $a {
     
            fn get_name(&self) -> String {
                ($b).to_string()
            }
            fn get_input_size(&self) -> usize {
                $is
            }
            fn get_output_size(&self) -> usize {
                $os
            }
            fn apply(&self, input: &[Tensor], output: &[Tensor]) {
                let result = input[0].$c($( self.$arg_name ),*);
                for (i, j) in output.iter().zip(result.iter()) {
                    i.swap(j);
                }
            }
            fn grad(&self, input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]) {
                $d(input, output_grad, input_grad)
            }
            fn get_values(&self) -> Vec<Tensor> {
                Vec::new()
            }
            fn get_grads(&self) -> Vec<Tensor> {
                Vec::new()
            }
            fn set_values(&self, _v: &[Tensor]) {
            }
        }
    }
}

macro_rules! new_binary_op {
    ($a:ident, $b:expr, $c:tt, $d: tt) => {
        pub struct $a {
            handle: OpHandle,
        }
        impl $a {
            pub fn new() -> $a{
                $a{
                    handle: OpHandle::new(),
                }
            }
            fn get_handle(&self) -> &OpHandle {
                &self.handle
            }
            fn get_handle_mut(&mut self) -> &mut OpHandle {
                &mut self.handle
            }
        }
        impl OpTrait for $a {
     
            fn get_name(&self) -> String {
                ($b).to_string()
            }
            fn get_input_size(&self) -> usize {
                2
            }
            fn get_output_size(&self) -> usize {
                1
            }
            fn apply(&self, input: &[Tensor], output: &[Tensor]) {
                $c(input, output)
            }
            fn grad(&self, input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]) {
                $d(input, output_grad, input_grad)
            }
            fn get_values(&self) -> Vec<Tensor> {
                Vec::new()
            }
            fn get_grads(&self) -> Vec<Tensor> {
                Vec::new()
            }
            fn set_values(&self, _v: &[Tensor]) {
            }
        }
        impl Default for $a {
            fn default() -> Self {
                Self::new()
            }
        }
    }
}

macro_rules! new_element_op {
    ($a:ident, $b:expr, $c:ident, $d: tt) => {
        pub struct $a {
            handle: OpHandle,
        }
        impl $a {
            pub fn new() -> $a{
                $a{
                    handle: OpHandle::new(),
                }
            }
            fn get_handle(&self) -> &OpHandle {
                &self.handle
            }
            fn get_handle_mut(&mut self) -> &mut OpHandle {
                &mut self.handle
            }
        }
        impl OpTrait for $a {
     
            fn get_name(&self) -> String {
                ($b).to_string()
            }
            fn get_input_size(&self) -> usize {
                2
            }
            fn get_output_size(&self) -> usize {
                1
            }
            fn apply(&self, input: &[Tensor], output: &[Tensor]) {
                output[0].swap(&input[0].$c())
            }
            fn grad(&self, input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]) {
                $d(input, output_grad, input_grad)
            }
            fn get_values(&self) -> Vec<Tensor> {
                Vec::new()
            }
            fn get_grads(&self) -> Vec<Tensor> {
                Vec::new()
            }
            fn set_values(&self, _v: &[Tensor]) {
            }
        }
        impl Default for $a {
            fn default() -> Self {
                Self::new()
            }
        }
    }
}

pub(crate) use one_to_1_op_with_paras;
pub(crate) use many_to_1_op_with_paras;
pub(crate) use one_to_vec_op_with_paras;
pub(crate) use new_binary_op;
pub(crate) use new_element_op;
