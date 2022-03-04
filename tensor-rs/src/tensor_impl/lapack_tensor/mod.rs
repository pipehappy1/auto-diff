pub mod compare_tensor;
pub mod convolution;
pub mod elemwise;
pub mod index_slicing;
pub mod linalg;
pub mod reduction;
pub mod blas_api;
pub mod lapack_api;

use crate::tensor_impl::gen_tensor::GenTensor;
use crate::tensor_impl::lapack_tensor::blas_api::BlasAPI;

macro_rules! blas_matmul {
    ($a:ty, $b: ident) => {
        pub fn $b(
            x: &GenTensor<$a>,
            y: &GenTensor<$a>,
        ) -> GenTensor<$a> {
            if x.size()[x.size().len()-1] != y.size()[0] {
                panic!("matmul expect matched size {:?}, {:?}", x.size(), y.size());
            }
            if x.size().len() == 1 && y.size().len() == 1 {
                panic!("Two vector have not matched size for matmul! {:?}, {:?}", x.numel(), y.numel());
            }
            let inner = y.size()[0];
            let mut cap = 1;
            let mut odim = Vec::new();
            let mut lloop = 1;
            let mut rloop = 1;
            for i in 0..x.size().len()-1 {
                cap *= x.size()[i];
                odim.push(x.size()[i]);
                lloop *= x.size()[i];
            }
            for i in 1..y.size().len() {
                cap *= y.size()[i];
                odim.push(y.size()[i]);
                rloop *= y.size()[i];
            }
            
            let mut ret = GenTensor::<$a>::new_move(
                vec![0.; cap], odim);
            
            BlasAPI::<$a>::gemm(false, false,
                                rloop, lloop, inner,
                                1., y.get_data(), rloop,
                                x.get_data(), inner,
                                1., ret.get_data_mut(), rloop,);
            ret
        }
    }
}

blas_matmul!(f32, matmul_f32);
blas_matmul!(f64, matmul_f64);

#[cfg(test)]
mod tests {
    use crate::tensor_impl::gen_tensor::GenTensor;
    use super::*;

    #[test]
    fn test_matmul() {
        let v1 = GenTensor::<f32>::new_raw(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
        let v2 = GenTensor::<f32>::new_raw(&[11., 12., 13., 14., 15., 16., 17., 18., 19.], &[3, 3]);
        let v3 = matmul_f32(&v1, &v2);
        let em = GenTensor::<f32>::new_raw(&[90.0, 96.0, 102.0, 216.0, 231.0, 246.0], &[2, 3]);
        assert_eq!(v3, em);
    }
}
