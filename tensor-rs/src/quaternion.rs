use std::ops::Div;

#[derive(PartialEq, Debug)]
pub struct Quaternion<T> {
    d: (T, T, T, T),
}

impl<T> Default for Quaternion<T>
where T: num_traits::Float {
    fn default() -> Self {
        Quaternion {
            d: (T::one(), T::zero(), T::zero(), T::zero())
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

    pub fn unit_exp(&self, t: T) -> Self {
        if self.norm() != T::one() {
            println!("unit_exp needs unit quaternion!");
        }

        let omega = T::acos(self.d.0);
        Quaternion {
            d: (T::cos(t*omega), T::sin(t*omega)*self.d.1,
                T::sin(t*omega)*self.d.2, T::sin(t*omega)*self.d.3)
        }
    }

    pub fn slerp(p: &Self, q: &Self, t: T) -> Self {
        if p.norm() != T::one() || q.norm() != T::one() {
            println!("slerp need unit quaternion!");
        }

        let p1 = p.normalize();
        let q1 = q.normalize();

        p1.qm(&p1.inverse().qm(&q1).unit_exp(t))
    }

    
    
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qm() {
        let a = Quaternion::<f64>::new(1., 2., 3., 4.);
        let b = Quaternion::<f64>::new(2., 3., 4., 5.);

        let c = a.qm(&b);
        assert_eq!(c, Quaternion::<f64>::new(-36., 6., 12., 12.,))
    }

    #[test]
    fn test_rotate_around_axis() {
        let a = Quaternion::<f64>::rotate_around_x(3.1415/2.);
        let v = (0., 0., 1.);
        let r = a.apply_rotation(v);

        assert!(f64::abs(r.0-0.) + f64::abs(r.1 + 1.) + f64::abs(r.2-0.) < 0.001);
    }

    #[test]
    fn test_unit_exp() {
        let b = Quaternion::<f64>::default();
        let b1 = b.unit_exp(1.);
        assert_eq!(b1, Quaternion::<f64>::new(1., 0., 0., 0.));

        //let b = Quaternion::<f64>::new(0.5, 0.5, 0.5, 0.5);
        //let b1 = b.unit_exp(1.);
        //assert_eq!(b1, Quaternion::<f64>::new(1., 0., 0., 0.));
        
    }

    #[test]
    fn test_slerp() {
        let a = Quaternion::<f64>::rotate_around_x(3.1415/2.);
        println!("{:?}", a);
        let b = Quaternion::<f64>::default();

        let c = Quaternion::<f64>::slerp(&a, &b, 1.);

        let v = (0., 0., 1.);
        let r = c.apply_rotation(v);

        assert_eq!(r, (1., 1., 1.,));

        assert_eq!(c, Quaternion::<f64>::default());
    }
}
