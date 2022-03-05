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
                if x.numel() < y.numel() {
                    panic!("right-hand broadcast only.");
                }
                if x.size().len() <= y.size().len() {
                    panic!("unmatched dimension. {}, {}", x.size().len(), y.size().len());
                }
                for i in 0..y.size().len() {
                    if y.size()[y.size().len()-i-1] != x.size()[x.size().len()-i-1] {
                        panic!("unmatched size.");
                    }
                }
                real_x = x.get_data();
                real_y = real_y.repeat(x.numel()/y.numel());
            }
            
            BlasAPI::<$a>::axpy(real_size,
                                1.0 as $a,
                                real_x, 1,
                                &mut real_y, 1);
            GenTensor::<$a>::new_move(real_y, x.size().clone())
        }
    }
}

#[cfg(feature = "use-blas-lapack")]
blas_add!(f32, add_f32);

#[cfg(feature = "use-blas-lapack")]
blas_add!(f64, add_f64);


#[cfg(feature = "use-blas-lapack")]
macro_rules! blas_sub {
    ($a:ty, $b: ident) => {
        pub fn $b(
            x: &GenTensor<$a>,
            y: &GenTensor<$a>,
        ) -> GenTensor<$a> {
            if x.numel() == 1 && y.numel() > 1 {
                let mut real_x_vec = vec![x.get_data()[0]; y.numel()];
                let real_size = y.numel();
                BlasAPI::<$a>::axpy(real_size,
                                    -1.0 as $a,
                                    y.get_data(), 1,
                                    &mut real_x_vec, 1);
                return GenTensor::<$a>::new_move(real_x_vec, y.size().clone());
            } else if x.numel() > 1 && y.numel() == 1 {
                let mut real_x_vec = x.get_data().clone();
                let real_size = x.numel();
                BlasAPI::<$a>::axpy(real_size,
                                    -1.0 as $a,
                                    y.get_data(), 1,
                                    &mut real_x_vec, 1);
                return GenTensor::<$a>::new_move(real_x_vec, y.size().clone());
            } else if x.size() == y.size() {
                let mut real_x_vec = x.get_data().clone();
                let real_size = x.numel();
                BlasAPI::<$a>::axpy(real_size,
                                    -1.0 as $a,
                                    y.get_data(), 1,
                                    &mut real_x_vec, 1);
                return GenTensor::<$a>::new_move(real_x_vec, y.size().clone());
            } else {
                if x.numel() < y.numel() {
                    panic!("right-hand broadcast only.");
                }
                if x.size().len() <= y.size().len() {
                    panic!("unmatched dimension and right-hand broadcast only. {}, {}",
			   x.size().len(), y.size().len());
                }
                for i in 0..y.size().len() {
                    if y.size()[y.size().len()-i-1] != x.size()[x.size().len()-i-1] {
                        panic!("unmatched size.");
                    }
                }
                let mut real_x_vec = x.get_data().clone();
                let real_y_vec = y.get_data().repeat(x.numel()/y.numel());
                let real_size = x.numel();
                BlasAPI::<$a>::axpy(real_size,
                                    -1.0 as $a,
                                    &real_y_vec, 1,
                                    &mut real_x_vec, 1);
                return GenTensor::<$a>::new_move(real_x_vec, x.size().clone());
            }
        }
    }
}

#[cfg(feature = "use-blas-lapack")]
blas_sub!(f32, sub_f32);

#[cfg(feature = "use-blas-lapack")]
blas_sub!(f64, sub_f64);

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

	let a = GenTensor::<f64>::ones(&[1, 2, 3]);
        let b = GenTensor::<f64>::ones(&[1, 2, 3]);
        let c = add_f64(&a, &b);
        let em = GenTensor::<f64>::new_raw(&[2.0, 2.0, 2.0, 2.0, 2.0, 2.0], &[1, 2, 3]);
        assert_eq!(c, em);

	let a = GenTensor::<f64>::ones(&[1, 2, 3]);
        let b = GenTensor::<f64>::ones(&[3]);
        let c = add_f64(&a, &b);
        let em = GenTensor::<f64>::new_raw(&[2.0, 2.0, 2.0, 2.0, 2.0, 2.0], &[1, 2, 3]);
        assert_eq!(c, em);
    }

    #[test]
    #[cfg(feature = "use-blas-lapack")]
    fn test_sub() {
        let a = GenTensor::<f32>::ones(&[1, 2, 3]);
        let b = GenTensor::<f32>::ones(&[1, 2, 3]);
        let c = sub_f32(&a, &b);
        let em = GenTensor::<f32>::new_raw(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[1, 2, 3]);
        assert_eq!(c, em);

	let a = GenTensor::<f64>::ones(&[1, 2, 3]);
        let b = GenTensor::<f64>::ones(&[1, 2, 3]);
        let c = sub_f64(&a, &b);
        let em = GenTensor::<f64>::new_raw(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[1, 2, 3]);
        assert_eq!(c, em);

	let a = GenTensor::<f64>::ones(&[1, 2, 3]);
        let b = GenTensor::<f64>::ones(&[3]);
        let c = sub_f64(&a, &b);
        let em = GenTensor::<f64>::new_raw(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[1, 2, 3]);
        assert_eq!(c, em);
    }
}
