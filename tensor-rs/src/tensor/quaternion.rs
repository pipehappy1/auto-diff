
use crate::tensor::gen_tensor::GenTensor;


pub struct Quaternion<T> {
    pub coords: GenTensor<T>,
}

impl<T> Default for Quaternion<T> where T: num_traits::Float {
    fn default() -> Self {
        Quaternion {
            coords: GenTensor::zeros(&[4, 1]),
        }
    }
}

impl<T> Quaternion<T>
where
    T: num_traits::Float {

    pub fn normalize(&self) -> Self {
        Self::from(self.coords.normalize())
    }
}
