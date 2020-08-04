use crate::tensor::gen_tensor::GenTensor;
#[cfg(feature = "use-cuda")]
use crate::tensor::cuda_tensor::CudaTensor;

pub trait CompareTensor {
    type TensorType;

    fn max_pair(&self, o: &Self::TensorType) -> Self::TensorType;
    fn min_pair(&self, o: &Self::TensorType) -> Self::TensorType;
    
}

impl<T> CompareTensor for GenTensor<T> where T: num_traits::Float {
    type TensorType = GenTensor<T>;
    
    fn max_pair(&self, o: &GenTensor<T>) -> GenTensor<T> {
        if self.size() != o.size() {
            panic!("max needs two tensor have the same size, {:?}, {:?}", self.size(), o.size());
        }
        let mut ret = GenTensor::empty(&self.size());

        for ((a, b), c) in self.get_data().iter().zip(o.get_data().iter()).zip(ret.get_data_mut().iter_mut()) {
            if a >= b {
                *c = *a;
            } else {
                *c = *b;
            }
        }
        ret
    }
    // min, 
    fn min_pair(&self, o: &GenTensor<T>) -> GenTensor<T> {
        if self.size() != o.size() {
            panic!("max needs two tensor have the same size, {:?}, {:?}", self.size(), o.size());
        }
        let mut ret = GenTensor::empty(&self.size());

        for ((a, b), c) in self.get_data().iter().zip(o.get_data().iter()).zip(ret.get_data_mut().iter_mut()) {
            if a >= b {
                *c = *b;
            } else {
                *c = *a;
            }
        }
        ret
    }
}

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

#[cfg(test)]
mod tests {
    use crate::tensor::gen_tensor::GenTensor;
    use super::*;
    
    #[test]
    fn max_pair() {
        let a = GenTensor::<f32>::new_raw(&vec![1., 3., 10., 11.], &vec![2,2]);
        let b = GenTensor::<f32>::new_raw(&vec![2., 4., 5., 6.], &vec![2,2]);
        let c = a.max_pair(&b);
        assert_eq!(c, GenTensor::<f32>::new_raw(&vec![2., 4., 10., 11.], &vec![2,2]));
    }

    #[test]
    fn min_pair() {
        let a = GenTensor::<f32>::new_raw(&vec![1., 3., 10., 11.], &vec![2,2]);
        let b = GenTensor::<f32>::new_raw(&vec![2., 4., 5., 6.], &vec![2,2]);
        let c = a.min_pair(&b);
        assert_eq!(c, GenTensor::<f32>::new_raw(&vec![1., 3., 5., 6.], &vec![2,2]));
    }
}
