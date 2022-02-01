pub trait CompareTensor {
    type TensorType;
    type ElementType;

    fn max_pair(&self, o: &Self::TensorType) -> Self::TensorType;
    fn min_pair(&self, o: &Self::TensorType) -> Self::TensorType;
    fn all(&self, f: &dyn Fn(Self::ElementType) -> bool) -> bool;
    fn any(&self, f: &dyn Fn(Self::ElementType) -> bool) -> bool;
}
