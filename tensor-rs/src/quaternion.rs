use std::ops::Div;

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

    pub fn dot(&self, o: &Quaternion<T>) -> T {
        self.d.0*o.d.0
            + self.d.1*o.d.1
            + self.d.2*o.d.2
            + self.d.3*o.d.3
    }

    pub fn len(&self) -> T {
        T::sqrt(self.dot(self))
    }

    pub fn norm(&self) -> T {
        self.len()
    }

    pub fn inverse(&self) -> Self {
        self.conjugate()/self.dot(self)
    }

    /// Make it a unit quaternion
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        
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

    /// Create a quaternion ready to apply to vector for rotation.
    pub fn rotation_around_axis(axis: (T, T, T), theta: T) -> Self {
        let a = T::cos(theta/(T::one() + T::one()));
        let coef = T::sin(theta/(T::one() + T::one()));
        let norm = T::sqrt(axis.0*axis.0 + axis.1*axis.1 + axis.2*axis.2);
        
        Quaternion {
            d: (a, coef*axis.0/norm,
                coef*axis.1/norm,
                coef*axis.2/norm)
        }
    }

    pub fn rotate_around_x(theta: T) -> Self {
        Self::rotation_around_axis((T::one(), T::zero(), T::zero()), theta)
    }

    pub fn rotate_around_y(theta: T) -> Self {
        Self::rotation_around_axis((T::zero(), T::one(), T::zero()), theta)
    }

    pub fn rotate_around_z(theta: T) -> Self {
        Self::rotation_around_axis((T::zero(), T::zero(), T::one()), theta)
    }

    /// Apply unit quaternion to 3d vector for rotation.
    pub fn apply_rotation(&self, v: (T, T, T)) -> (T, T, T) {
        if self.norm() != T::one() {
            println!("Apply a non unit quaternion for rotation!");
        }
        
        let x = Quaternion {
            d: (T::zero(), v.0, v.1, v.2)
        };
        let xp = self.qm(&x).qm(&self.conjugate());
        (xp.d.1, xp.d.2, xp.d.3)
    }

    pub fn rotate_around_axis(axis: (T, T, T), theta: T, v: (T, T, T)) -> (T, T, T) {
        let q = Self::rotation_around_axis(axis, theta);
        q.apply_rotation(v)
    }

    pub fn slerp(p: &Self, q: &Self, t: T) -> Self {
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
