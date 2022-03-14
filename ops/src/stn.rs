use tensor_rs::tensor::Tensor;
use auto_diff::op::{OpTrait, OpHandle};
use auto_diff_macros::add_op_handle;

#[add_op_handle]
pub struct AffineGrid {
}
impl AffineGrid {
    
}

#[add_op_handle]
pub struct AffineSample {
}
impl AffineSample {
    
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn it_works() {
        //let demo = H{};
    }
}
