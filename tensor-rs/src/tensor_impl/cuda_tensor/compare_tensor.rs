#[cfg(feature = "use-cuda")]
use crate::tensor::cuda_tensor::CudaTensor;
use crate::tensor_trait::compare_tensor::CompareTensor;




#[cfg(feature = "use-cuda")]
impl CompareTensor for CudaTensor {
    type TensorType = CudaTensor;
    
    fn max_pair(&self, o: &Self::TensorType) -> Self::TensorType {
        unimplemented!();
    }
    fn min_pair(&self, o: &Self::TensorType) -> Self::TensorType {
        unimplemented!();
    }
}

