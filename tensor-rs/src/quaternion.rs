use std::ops::Div;
use crate::tensor::Tensor;

#[derive(PartialEq, Debug)]
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

impl<T> Div<T> for Quaternion<T>
where T: num_traits::Float {
    type Output = Self;
    
    fn div(self, rhs: T) -> Self::Output {
        Quaternion {
            d: (self.d.0/rhs, self.d.1/rhs, self.d.2/rhs, self.d.3/rhs)
        }
    }
}

impl<T> Quaternion<T>
where T: num_traits::Float {

    pub fn new(a: T, b: T, c: T, d: T) -> Self {
        Quaternion {
            d: (a, b, c, d)
        }
    }

    pub fn scalar_part(&self) -> T {
        self.d.0
    }

    pub fn vector_part(&self) -> (T, T, T) {
        (self.d.1, self.d.2, self.d.3)
    }

    pub fn conjugate(&self) -> Self {
        Quaternion {
            d: (self.d.0, -self.d.1, -self.d.2, -self.d.3)
        }
    }

    pub fn len(&self) -> T {
        T::sqrt(self.d.0*self.d.0
                + self.d.1*self.d.1
                + self.d.1*self.d.1
                + self.d.1*self.d.1)
    }

    pub fn norm(&self) -> T {
        self.len()
    }

    pub fn inverse(&self) -> Self {
        self.conjugate()/(self.d.0*self.d.0
                          + self.d.1*self.d.1
                          + self.d.1*self.d.1
                          + self.d.1*self.d.1)
    }

    /// Make it a unit quaternion
    pub fn normalize(&self) -> Self {
        let n = self.len();
        
        Quaternion {
            d: (self.d.0/n, self.d.1/n, self.d.2/n, self.d.3/n)
        }
    }

    /// Quaternion multiplication
    pub fn qm(&self, o: &Quaternion<T>) -> Self {
        Quaternion {
            d: (self.d.0*o.d.0 - self.d.1*o.d.1 - self.d.2*o.d.2 - self.d.3*o.d.3,
                self.d.0*o.d.1 + self.d.1*o.d.0 + self.d.2*o.d.3 - self.d.3*o.d.2,
                self.d.0*o.d.2 - self.d.1*o.d.3 + self.d.2*o.d.0 + self.d.3*o.d.1,
                self.d.0*o.d.3 + self.d.1*o.d.2 - self.d.2*o.d.1 + self.d.3*o.d.0,)
        }
    }

    pub fn to_tensor(&self) -> Tensor {
        Tensor::ones(&[4])
    }

    pub fn from_tensor(t: &Tensor) -> Self {
        Quaternion {
            d: (T::one(), T::one(), T::one(), T::one())
        }
    }

    pub fn slerp(p: &Self, q: &Self, t: T) -> Self {
        Quaternion {
            d: (T::one(), T::one(), T::one(), T::one())
        }
    }

    pub fn rotation_around_axis(axis: (T, T, T), theta: T) -> Self {
        Quaternion {
            d: (T::one(), T::one(), T::one(), T::one())
        }
    }

    pub fn rotate_around_x() -> Self {
        Quaternion {
            d: (T::one(), T::one(), T::one(), T::one())
        }
    }

    pub fn rotate_around_y() -> Self {
        Quaternion {
            d: (T::one(), T::one(), T::one(), T::one())
        }
    }

    pub fn rotate_around_z() -> Self {
        Quaternion {
            d: (T::one(), T::one(), T::one(), T::one())
        }
    }
    
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_qm() {
        let a = Quaternion::<f64>::new(1., 2., 3., 4.);
        let b = Quaternion::<f64>::new(2., 3., 4., 5.);

        let c = a.qm(&b);
        assert_eq!(c, Quaternion::<f64>::new(-36., 6., 12., 12.,))
    }
}
