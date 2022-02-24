pub trait LinearAlgbra {
    type TensorType;
    type ElementType;

    fn norm(&self) -> Self::TensorType;
    /// Assuming the input is 2 dimensional array,
    /// normalize_unit 
    fn normalize_unit(&self) -> Self::TensorType;
    fn lu(&self) -> Option<[Self::TensorType; 2]>;
    fn lu_solve(&self, y: &Self::TensorType) -> Option<Self::TensorType>;
    fn qr(&self) -> Option<[Self::TensorType; 2]>;
    fn eigen(&self) -> Option<[Self::TensorType; 2]>;
    fn cholesky(&self) -> Option<Self::TensorType>;
    fn det(&self) -> Option<Self::TensorType>;
    fn svd(&self) -> Option<[Self::TensorType; 3]>;
    fn inv(&self) -> Option<Self::TensorType>;
    fn pinv(&self) -> Self::TensorType;
    fn tr(&self) -> Self::TensorType;
}
