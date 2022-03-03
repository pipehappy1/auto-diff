use crate::tensor_impl::gen_tensor::GenTensor;
#[cfg(feature = "use-blas-lapack")]
use super::blas_api::BlasAPI;


#[cfg(feature = "use-blas-lapack")]
macro_rules! blas_add {
    ($a:ty, $b: ident) => {
        pub fn $b(
            x: &GenTensor<$a>,
            y: &GenTensor<$a>,
        ) -> GenTensor<$a> {
            let real_x;
            let mut real_y = y.get_data().clone();
            let mut real_size = x.numel();
            let real_x_vec;
            if x.numel() == 1 && y.numel() > 1 {
                real_x_vec = vec![x.get_data()[0]; y.numel()];
                real_x = &real_x_vec;
                real_size = y.numel();
            } else if x.numel() > 1 && y.numel() == 1 {
                real_x = x.get_data();
                real_y = vec![real_y[0]; x.numel()];
                real_size = x.numel();
            } else if x.numel() == y.numel() {
                real_x = x.get_data();
            } else {
                panic!("x and y need the same size.");
            }
            
            BlasAPI::<$a>::axpy(real_size,
                                1.0 as $a,
                                real_x, 1,
                                &mut real_y, 1);
            GenTensor::<$a>::new_raw(&real_y, y.size())
        }
    }
}

#[cfg(feature = "use-blas-lapack")]
blas_add!(f32, add_f32);

#[cfg(feature = "use-blas-lapack")]
blas_add!(f64, add_f64);

#[cfg(test)]
mod tests {
    use crate::tensor_impl::gen_tensor::GenTensor;
    use super::*;

    #[test]
    #[cfg(feature = "use-blas-lapack")]
    fn test_add() {
        let a = GenTensor::<f32>::ones(&[1, 2, 3]);
        let b = GenTensor::<f32>::ones(&[1, 2, 3]);
        let c = add_f32(&a, &b);
        let em = GenTensor::<f32>::new_raw(&[2.0, 2.0, 2.0, 2.0, 2.0, 2.0], &[1, 2, 3]);
        assert_eq!(c, em);
    }
}
