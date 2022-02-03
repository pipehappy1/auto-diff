use super::GenTensor;
use crate::tensor_trait::elemwise::ElemwiseTensorOp;

impl<T> ElemwiseTensorOp for GenTensor<T> where T: num_traits::Float {
    type TensorType = GenTensor<T>;
    type ElementType = T;

    // Pointwise Ops
    // abs
    fn abs(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.abs()
        })
    }
    // acos
    fn acos(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.acos()
        })
    }
    // add, there is one.
    // addcdiv
    // addcmul
    // angle
    // asin
    fn asin(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.asin()
        })
    }
    // atan
    fn atan(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.atan()
        })
    }
    // atan2
    // bitwise_not
    // bitwise_and
    // bitwise_or
    // bitwise_xor
    // ceil
    fn ceil(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.ceil()
        })
    }
    // clamp
    fn clamp(&self, min: T, max: T) -> GenTensor<T> {
        let mut ret = GenTensor::new_move(Vec::with_capacity(self.get_data().len()),
                                          self.get_size().to_vec());

        for i in self.get_data() {
            let value;
            if *i < min {
                value = min;
            } else if *i <= max {
                value = *i;
            } else {
                value = max;
            }
            ret.get_data_mut().push(value);
        }
        ret
    }
    // conj
    // cos
    fn cos(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.cos()
        })
    }
    // cosh
    fn cosh(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.cosh()
        })
    }
    // div, there is one.
    // digamma
    //fn digamma(&self) -> GenTensor<T> {
    //    self._pointwise(|x| {
    //        x.digamma()
    //    })
    //}
    // erf
    // erfc
    // erfinv
    // exp
    fn exp(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.exp()
        })
    }
    // expm1
    fn expm1(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.exp_m1()
        })
    }
    // floor
    fn floor(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.floor()
        })
    }
    // floor_divide
    // fmod
    // frac
    fn frac(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.fract()
        })
    }
    // imag
    // lerp, this is on Tensor.
    // lgamma
    // log
    fn log(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.ln()
        })
    }
    // log10
    fn log10(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.log10()
        })
    }
    // log1p
    fn log1p(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.ln_1p()
        })
    }

    /// Better log(1 + exp(x))
    /// see <https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf>
    fn log1pexp(&self) -> GenTensor<T> {
        let mut ret = GenTensor::new_move(Vec::with_capacity(self.get_data().len()),
                                          self.get_size().to_vec());
        for i in self.get_data() {
            if i <= &T::from(-37).expect("") {
                ret.get_data_mut().push(i.exp());
            } else if i > &T::from(-37).expect("") && i <= &T::from(18).expect("") {
                ret.get_data_mut().push(i.exp().ln_1p());
            } else if i > &T::from(-18).expect("") && i <= &T::from(33.3).expect("") {
                ret.get_data_mut().push(*i + i.mul(T::from(-1).expect("")).exp());
            } else {
                ret.get_data_mut().push(*i);
            }
        }
        ret
    }
    
    // log2
    fn log2(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.log2()
        })
    }
    // logical_and
    // logical_not
    // logical_or
    // logical_xor
    // mul, there is one
    // mvlgamma
    // neg
    fn neg(&self) -> GenTensor<T> {
        let mut ret = GenTensor::new_move(Vec::with_capacity(self.get_data().len()),
                                          self.get_size().to_vec());

        for i in self.get_data() {
            ret.get_data_mut().push(i.mul(T::zero() - T::one()));
        }
        ret
    }
    
    // polygamma
    // pow
    fn pow(&self, n: T) -> GenTensor<T> {
        self._pointwise(|x| {
            x.powf(n)
        })
    }
    // real
    // reciprocal
    fn reciprocal(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.recip()
        })
    }
    // remainder
    // round
    fn round(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.round()
        })
    }
    // rsqrt
    fn rsqrt(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.sqrt()/(*x) // TODO ?
        })
    }
    
    fn sigmoid(&self) -> GenTensor<T> {
        let mut ret = GenTensor::new_move(self.get_data().to_vec(),
                                          self.get_size().to_vec());

        for i in 0..self.get_data().len() {
            if self.get_data()[i] > T::zero() {
                ret.get_data_mut()[i] = T::one()/(T::one() + self.get_data()[i].neg().exp());
            }
            else {
                ret.get_data_mut()[i] = self.get_data()[i].exp()/(T::one() + self.get_data()[i].exp());
            }
        }
        ret
    }

    // sign
    fn sign(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            if *x == T::zero() {
                T::zero()
            } else if *x > T::zero() {
                T::one()
            } else {
                T::zero() - T::one()
            }
        })
    }
    // sin
    fn sin(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.sin()
        })
    }
    // sinh
    fn sinh(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.sinh()
        })
    }
    // sqrt
    fn sqrt(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.sqrt()
        })
    }
    // square
    fn square(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            (*x)*(*x)
        })
    }
    // tan
    fn tan(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.tan()
        })
    }
    // tanh
    fn tanh(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.tanh()
        })
    }
    // true_divide
    // trunc
    fn trunc(&self) -> GenTensor<T> {
        self._pointwise(|x| {
            x.trunc()
        })
    }
}

