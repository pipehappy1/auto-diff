use std::collections::BTreeMap;
use super::gen_tensor::*;
use crate::tensor::PaddingMode;

#[cfg(feature = "use-cuda")]
use crate::tensor::cuda_tensor::CudaTensor;
use crate::tensor_trait::convolution::Convolution;







#[cfg(test)]
mod tests {
    use crate::tensor::gen_tensor::GenTensor;
    use crate::tensor_trait::index_slicing::IndexSlicing;
    use super::*;


    

    
}


//////////////
// for cuda tensor
/////////
#[cfg(feature = "use-cuda")]
impl Convolution for CudaTensor {

    fn conv2d(&self, filter: &Self,
                  stride: (usize, usize),
                  padding: (usize, usize),
                  dilation: (usize, usize),
                  padding_mode: PaddingMode
    ) -> Self {
        todo!();
    }

    fn conv2d_grad(&self, filter: &Self,
                       stride: (usize, usize),
                       padding: (usize, usize),
                       dilation: (usize, usize),
                       padding_mode: PaddingMode,
                       output_grad: &Self
    ) -> (Self, Self) {
        todo!();
    }

    fn conv_gen(&self, filter: &Self,
                    stride: &[usize],
                    padding: &[usize],
                    dilation: &[usize],
                    padding_mode: PaddingMode
    ) -> Self {
        todo!();
    }

    fn conv_grad_gen(&self, filter: &Self,
                         stride: &[usize],
                         padding: &[usize],
                         dilation: &[usize],
                         padding_mode: PaddingMode,
                         output_grad: &Self,
    ) -> (Self, Self) {
        todo!();
    }
}
