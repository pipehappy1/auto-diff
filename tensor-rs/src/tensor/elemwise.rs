use crate::tensor::gen_tensor::GenTensor;
#[cfg(feature = "use-cuda")]
use crate::tensor::cuda_tensor::CudaTensor;

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
    /// see https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
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
            x.sqrt()/(*x)
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

/****************/
// Cuda element wise ops
/****************/
#[cfg(feature = "use-cuda")]
impl ElemwiseTensorOp for CudaTensor {
    type TensorType = CudaTensor;
    type ElementType = f32;

    // Pointwise Ops
    // abs
    fn abs(&self) -> CudaTensor {
        unimplemented!();
    }
    // acos
    fn acos(&self) -> CudaTensor {
        unimplemented!();
    }
    // add, there is one.
    // addcdiv
    // addcmul
    // angle
    // asin
    fn asin(&self) -> CudaTensor {
        unimplemented!();
    }
    // atan
    fn atan(&self) -> CudaTensor {
        unimplemented!();
    }
    // atan2
    // bitwise_not
    // bitwise_and
    // bitwise_or
    // bitwise_xor
    // ceil
    fn ceil(&self) -> CudaTensor {
        unimplemented!();
    }
    // clamp
    fn clamp(&self, min: Self::ElementType, max: Self::ElementType) -> CudaTensor {
        unimplemented!();
    }
    // conj
    // cos
    fn cos(&self) -> CudaTensor {
        unimplemented!();
    }
    // cosh
    fn cosh(&self) -> CudaTensor {
        unimplemented!();
    }
    // div, there is one.
    // digamma
    //fn digamma(&self) -> CudaTensor {
    //    self._pointwise(|x| {
    //        x.digamma()
    //    })
    //}
    // erf
    // erfc
    // erfinv
    // exp
    fn exp(&self) -> CudaTensor {
        unimplemented!();
    }
    // expm1
    fn expm1(&self) -> CudaTensor {
        unimplemented!();
    }
    // floor
    fn floor(&self) -> CudaTensor {
        unimplemented!();
    }
    // floor_divide
    // fmod
    // frac
    fn frac(&self) -> CudaTensor {
        unimplemented!();
    }
    // imag
    // lerp, this is on Tensor.
    // lgamma
    // log
    fn log(&self) -> CudaTensor {
        unimplemented!();
    }
    // log10
    fn log10(&self) -> CudaTensor {
        unimplemented!();
    }
    // log1p
    fn log1p(&self) -> CudaTensor {
        unimplemented!();
    }

    /// Better log(1 + exp(x))
    /// see https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    fn log1pexp(&self) -> CudaTensor {
        unimplemented!();
    }
    
    // log2
    fn log2(&self) -> CudaTensor {
        unimplemented!();
    }
    // logical_and
    // logical_not
    // logical_or
    // logical_xor
    // mul, there is one
    // mvlgamma
    // neg
    fn neg(&self) -> CudaTensor {
        unimplemented!();
    }
    
    // polygamma
    // pow
    fn pow(&self, n: Self::ElementType) -> CudaTensor {
        unimplemented!();
    }
    // real
    // reciprocal
    fn reciprocal(&self) -> CudaTensor {
        unimplemented!();
    }
    // remainder
    // round
    fn round(&self) -> CudaTensor {
        unimplemented!();
    }
    // rsqrt
    fn rsqrt(&self) -> CudaTensor {
        unimplemented!();
    }
    
    fn sigmoid(&self) -> CudaTensor {
        unimplemented!();
    }

    // sign
    fn sign(&self) -> CudaTensor {
        unimplemented!();
    }
    // sin
    fn sin(&self) -> CudaTensor {
        unimplemented!();
    }
    // sinh
    fn sinh(&self) -> CudaTensor {
        unimplemented!();
    }
    // sqrt
    fn sqrt(&self) -> CudaTensor {
        unimplemented!();
    }
    // square
    fn square(&self) -> CudaTensor {
        unimplemented!();
    }
    // tan
    fn tan(&self) -> CudaTensor {
        unimplemented!();
    }
    // tanh
    fn tanh(&self) -> CudaTensor {
        unimplemented!();
    }
    // true_divide
    // trunc
    fn trunc(&self) -> CudaTensor {
        unimplemented!();
    }
}
