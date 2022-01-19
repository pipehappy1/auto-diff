pub trait ElemwiseTensorOp {
    type TensorType;
    type ElementType;

    fn abs(&self) -> Self::TensorType;
    fn acos(&self) -> Self::TensorType;
    fn asin(&self) -> Self::TensorType;
    fn atan(&self) -> Self::TensorType;
    fn ceil(&self) -> Self::TensorType;
    fn clamp(&self, min: Self::ElementType, max: Self::ElementType) -> Self::TensorType;
    fn cos(&self) -> Self::TensorType;
    fn cosh(&self) -> Self::TensorType;
    fn exp(&self) -> Self::TensorType;
    fn expm1(&self) -> Self::TensorType;
    fn floor(&self) -> Self::TensorType;
    fn frac(&self) -> Self::TensorType ;
    fn log(&self) -> Self::TensorType;
    fn log10(&self) -> Self::TensorType;
    fn log1p(&self) -> Self::TensorType;
    fn log1pexp(&self) -> Self::TensorType;
    fn log2(&self) -> Self::TensorType;
    fn neg(&self) -> Self::TensorType;
    fn pow(&self, n: Self::ElementType) -> Self::TensorType;
    fn reciprocal(&self) -> Self::TensorType;
    fn round(&self) -> Self::TensorType;
    fn rsqrt(&self) -> Self::TensorType ;
    fn sigmoid(&self) -> Self::TensorType;
    fn sign(&self) -> Self::TensorType;
    fn sin(&self) -> Self::TensorType;
    fn sinh(&self) -> Self::TensorType;
    fn sqrt(&self) -> Self::TensorType;
    fn square(&self) -> Self::TensorType;
    fn tan(&self) -> Self::TensorType;
    fn tanh(&self) -> Self::TensorType;
    fn trunc(&self) -> Self::TensorType;
    
}
