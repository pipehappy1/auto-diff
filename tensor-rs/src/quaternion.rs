
use crate::tensor::Tensor;


pub struct Quaternion<T> {
    d: (T, T, T, T),
}

impl<T> Default for Quaternion<T>
where T: num_traits::Float {
    fn default() -> Self {
        Quaternion {
            d: (T::one(), T::one(), T::one(), T::one())
        }
    }
}

impl<T> Quaternion<T>
where T: num_traits::Float {

    /// Make it a unit quaternion
    pub fn normalize(&self) -> Self {
        Quaternion {
            d: (T::one(), T::one(), T::one(), T::one())
        }
    }

    /// Quaternion multiplication
    pub fn qm(&self, o: &Quaternion<T>) -> Self {
        Quaternion {
            d: (T::one(), T::one(), T::one(), T::one())
        }
    }
}
