pub trait CompareTensor {
    type TensorType;

    fn max_pair(&self, o: &Self::TensorType) -> Self::TensorType;
    fn min_pair(&self, o: &Self::TensorType) -> Self::TensorType;
    
}
