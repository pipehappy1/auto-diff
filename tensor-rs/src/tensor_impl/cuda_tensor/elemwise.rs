#[cfg(feature = "use-cuda")]
use crate::tensor::cuda_tensor::CudaTensor;
#[cfg(feature = "use-cuda")]
use cuda11_cutensor_sys::{self, cutensorHandle_t, check_cutensor_status, cutensorInit, cudaDataType_t, cutensorTensorDescriptor_t, cutensorInitTensorDescriptor, cutensorPermutation,
                          cutensorOperator_t_CUTENSOR_OP_IDENTITY,
                          cutensorOperator_t_CUTENSOR_OP_SQRT,
                          cutensorOperator_t_CUTENSOR_OP_RELU,
                          cutensorOperator_t_CUTENSOR_OP_CONJ,
                          cutensorOperator_t_CUTENSOR_OP_RCP,
                          cutensorOperator_t_CUTENSOR_OP_SIGMOID,
                          cutensorOperator_t_CUTENSOR_OP_TANH,
                          cutensorOperator_t_CUTENSOR_OP_EXP,
                          cutensorOperator_t_CUTENSOR_OP_LOG,
                          cutensorOperator_t_CUTENSOR_OP_ABS,
                          cutensorOperator_t_CUTENSOR_OP_NEG,
                          cutensorOperator_t_CUTENSOR_OP_SIN,
                          cutensorOperator_t_CUTENSOR_OP_COS,
                          cutensorOperator_t_CUTENSOR_OP_TAN,
                          cutensorOperator_t_CUTENSOR_OP_SINH,
                          cutensorOperator_t_CUTENSOR_OP_COSH,
                          cutensorOperator_t_CUTENSOR_OP_ASIN,
                          cutensorOperator_t_CUTENSOR_OP_ACOS,
                          cutensorOperator_t_CUTENSOR_OP_ATAN,
                          cutensorOperator_t_CUTENSOR_OP_ASINH,
                          cutensorOperator_t_CUTENSOR_OP_ACOSH,
                          cutensorOperator_t_CUTENSOR_OP_ATANH,
                          cutensorOperator_t_CUTENSOR_OP_CEIL,
                          cutensorOperator_t_CUTENSOR_OP_FLOOR,
                          cutensorOperator_t_CUTENSOR_OP_ADD,
                          cutensorOperator_t_CUTENSOR_OP_MUL,
                          cutensorOperator_t_CUTENSOR_OP_MAX,
                          cutensorOperator_t_CUTENSOR_OP_MIN,
};
#[cfg(feature = "use-cuda")]
use cuda11_cudart_sys::{self, cudaMalloc, cudaStreamCreate, cudaMemcpy, cudaStreamSynchronize, cudaFree, cudaStreamDestroy, cudaMemcpyKind, check_cuda_status, cudaStream_t, cudaMemcpyAsync};

use crate::tensor_trait::elemwise::ElemwiseTensorOp;



// macro for cuda element
#[cfg(feature = "use-cuda")]
macro_rules! unary_cuda_ops {
    ($a: ident, $b: ident) => {
        fn $a(&self) -> CudaTensor {
            //let mut ret = CudaTensor::empty(self.size());
            let mut ret = self.zeros_like();
        
            unsafe {
                let mut stream: cudaStream_t = self._get_stream();
        
                let mut handle:cutensorHandle_t = std::mem::uninitialized();
                check_cutensor_status(cutensorInit(&mut handle as *mut _));
            
                let alpha: f32 = 1.0;
            
                let extent: Vec<i64> = self.size().iter().map(|x| *x as i64).collect();
            
                let mut descA: cutensorTensorDescriptor_t = std::mem::uninitialized();
                let mut descB: cutensorTensorDescriptor_t = std::mem::uninitialized();
            
                check_cutensor_status(cutensorInitTensorDescriptor( &mut handle,
                                               &mut descA,
                                               self.size().len() as _,
                                               extent.as_ptr(),
                                               std::ptr::null(),/*stride*/
                                               cudaDataType_t::CUDA_R_32F,
                                               $b));
                check_cutensor_status(cutensorInitTensorDescriptor( &mut handle,
                                               &mut descB,
                                               self.size().len() as _,
                                               extent.as_ptr(),
                                               std::ptr::null(),/*stride*/
                                               cudaDataType_t::CUDA_R_32F,
                                               cutensorOperator_t_CUTENSOR_OP_IDENTITY));
    
                let mut modeA: Vec<i32> = vec![32; self.size().len()];
                let mut modeB: Vec<i32> = vec![32; self.size().len()];
                for i in 0..self.size().len() {
                    modeA[i] = modeA[i] + i as i32;
                    modeB[i] = modeB[i] + i as i32;
                }
                
                check_cutensor_status(cutensorPermutation(&handle,
                                    &alpha as *const _ as _,
                                    self._get_device_data() as _,
                                    &descA as _,
                                    modeA.as_ptr(),
                                    ret._get_device_data() as _,
                                    &descB as _,
                                    modeB.as_ptr(),
                                    cudaDataType_t::CUDA_R_32F,
                                    stream as _
                ));

                // this is called in CudaTensor::_flush() !!!
                //cudaStreamSynchronize(stream as _);
            }
            ret
        }
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
    unary_cuda_ops!(abs, cutensorOperator_t_CUTENSOR_OP_ABS);

    // acos
    unary_cuda_ops!(acos, cutensorOperator_t_CUTENSOR_OP_ACOS);
    // add, there is one.
    // addcdiv
    // addcmul
    // angle
    // asin
    unary_cuda_ops!(asin, cutensorOperator_t_CUTENSOR_OP_ASIN);
    // atan
    unary_cuda_ops!(atan, cutensorOperator_t_CUTENSOR_OP_ATAN);
    // atan2
    // bitwise_not
    // bitwise_and
    // bitwise_or
    // bitwise_xor
    // ceil
    unary_cuda_ops!(ceil, cutensorOperator_t_CUTENSOR_OP_CEIL);
    // clamp
    fn clamp(&self, min: Self::ElementType, max: Self::ElementType) -> CudaTensor {
        unimplemented!();
    }
    // conj
    // cos
    unary_cuda_ops!(cos, cutensorOperator_t_CUTENSOR_OP_COS);
    // cosh
    unary_cuda_ops!(cosh, cutensorOperator_t_CUTENSOR_OP_COSH);
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
    unary_cuda_ops!(exp, cutensorOperator_t_CUTENSOR_OP_EXP);
    // expm1
    fn expm1(&self) -> CudaTensor {
        unimplemented!();
    }
    // floor
    unary_cuda_ops!(floor, cutensorOperator_t_CUTENSOR_OP_FLOOR);
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
    unary_cuda_ops!(log, cutensorOperator_t_CUTENSOR_OP_LOG);
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
    unary_cuda_ops!(neg, cutensorOperator_t_CUTENSOR_OP_NEG);
    
    // polygamma
    // pow
    fn pow(&self, n: Self::ElementType) -> CudaTensor {
        unimplemented!();
    }
    // real
    // reciprocal
    unary_cuda_ops!(reciprocal, cutensorOperator_t_CUTENSOR_OP_RCP);
    // remainder
    // round
    fn round(&self) -> CudaTensor {
        unimplemented!();
    }
    // rsqrt
    fn rsqrt(&self) -> CudaTensor {
        unimplemented!();
    }
    // sigmoid
    unary_cuda_ops!(sigmoid, cutensorOperator_t_CUTENSOR_OP_SIGMOID);

    // sign
    fn sign(&self) -> CudaTensor {
        unimplemented!();
    }
    // sin
    unary_cuda_ops!(sin, cutensorOperator_t_CUTENSOR_OP_SIN);
    // sinh
    unary_cuda_ops!(sinh, cutensorOperator_t_CUTENSOR_OP_SINH);
    // sqrt
    unary_cuda_ops!(sqrt, cutensorOperator_t_CUTENSOR_OP_SQRT);
    // square
    fn square(&self) -> CudaTensor {
        unimplemented!();
    }
    // tan
    unary_cuda_ops!(tan, cutensorOperator_t_CUTENSOR_OP_TAN);
    // tanh
    unary_cuda_ops!(tanh, cutensorOperator_t_CUTENSOR_OP_TANH);
    // true_divide
    // trunc
    fn trunc(&self) -> CudaTensor {
        unimplemented!();
    }
}

#[cfg(all(test, feature = "use-cuda"))]
mod tests {
    use super::*;

    #[test]
    fn cuda_abs() {
        let mut input = CudaTensor::new_raw(&vec![1., 2., 3., 4., 5., 6., 7., 8., -9.],
                                            &vec![1, 1, 3, 3]);
        let output = input.abs();

        let mut input_gen = GenTensor::new_raw(&vec![1., 2., 3., 4., 5., 6., 7., 8., -9.],
                                               &vec![1, 1, 3, 3]);
        let output_gen = input_gen.abs();
        assert_eq!(output.to_GenTensor(), output_gen);
        //println!("{:?}", output.to_GenTensor());
    }

    
}
