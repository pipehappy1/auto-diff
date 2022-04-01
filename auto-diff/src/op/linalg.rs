use super::{OpHandle, OpTrait};
use tensor_rs::tensor::Tensor;

#[cfg(feature = "use-serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "use-serde")]
use std::any::Any;

#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct NormalizeUnit {
    #[cfg_attr(feature = "use-serde", serde(skip))]
    handle: OpHandle,
}
impl NormalizeUnit {
    pub fn new() -> NormalizeUnit {
        NormalizeUnit {
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
impl OpTrait for NormalizeUnit {
    fn get_name(&self) -> &'static str {
        "NormalizeUnit"
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        output[0].swap(&input[0].normalize_unit());
    }
    fn grad(&self, input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]) {
        unimplemented!();
    }
    fn get_values(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn get_grads(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn set_values(&self, _v: &[Tensor]) {}
    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Default for NormalizeUnit {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct Det {
    #[cfg_attr(feature = "use-serde", serde(skip))]
    handle: OpHandle,
}
impl Det {
    pub fn new() -> Det {
        Det {
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
impl OpTrait for Det {
    fn get_name(&self) -> &'static str {
        "Det"
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        output[0].swap(&input[0].det().expect("det() does not get a result."));
    }
    fn grad(&self, input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]) {
        unimplemented!();
    }
    fn get_values(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn get_grads(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn set_values(&self, _v: &[Tensor]) {}
    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Default for Det {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct Inv {
    #[cfg_attr(feature = "use-serde", serde(skip))]
    handle: OpHandle,
}
impl Inv {
    pub fn new() -> Inv {
        Inv {
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
impl OpTrait for Inv {
    fn get_name(&self) -> &'static str {
        "Inv"
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        output[0].swap(&input[0].inv().expect("inv() does not get a result."));
    }
    fn grad(&self, input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]) {
        unimplemented!();
    }
    fn get_values(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn get_grads(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn set_values(&self, _v: &[Tensor]) {}
    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Default for Inv {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct Tr {
    #[cfg_attr(feature = "use-serde", serde(skip))]
    handle: OpHandle,
}
impl Tr {
    pub fn new() -> Tr {
        Tr {
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
impl OpTrait for Tr {
    fn get_name(&self) -> &'static str {
        "Tr"
    }
    fn get_input_size(&self) -> usize {
        1
    }
    fn get_output_size(&self) -> usize {
        1
    }
    fn apply(&self, input: &[Tensor], output: &[Tensor]) {
        output[0].swap(&input[0].tr());
    }
    fn grad(&self, input: &[Tensor], output_grad: &[Tensor], input_grad: &[Tensor]) {
        unimplemented!();
    }
    fn get_values(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn get_grads(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn set_values(&self, _v: &[Tensor]) {}
    #[cfg(feature = "use-serde")]
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Default for Tr {
    fn default() -> Self {
        Self::new()
    }
}
