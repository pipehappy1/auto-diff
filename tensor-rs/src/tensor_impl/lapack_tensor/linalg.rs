//
use crate::tensor_impl::gen_tensor::GenTensor;
use crate::tensor_trait::index_slicing::IndexSlicing;
#[cfg(feature = "use-blas-lapack")]
use super::lapack_api::LapackAPI;
use std::cmp;

#[cfg(feature = "use-blas-lapack")]
macro_rules! lapack_svd {
    ($a:ty, $b: ident) => {
        pub fn $b(
            x: &GenTensor<$a>,
        ) -> (GenTensor<$a>, GenTensor<$a>, GenTensor<$a>) {
	    if x.size().len() != 2 {
		panic!("lapack_svd expects 2d matrix.");
	    }
	    let n = x.size()[0];
	    let m = x.size()[1];
	    let mmn = cmp::min(m, n);
	    
            let mut ma = x.get_data().clone();
	    let mut s: Vec<$a> = vec![0.; mmn];
	    let mut u: Vec<$a> = vec![0.; mmn*m];
	    let mut vt: Vec<$a> = vec![0.; mmn*n];
	    let mut info: i32 = 0;
	    LapackAPI::<$a>::gesdd(&'S', m, n,
	    			    &mut ma, m,
				    &mut s,
				    &mut u, m,
				    &mut vt, mmn,
	 			    &mut info);
            let ret_u = GenTensor::<$a>::new_move(vt, vec![n, mmn]);
	    let ret_s = GenTensor::<$a>::new_move(s, vec![mmn]);
	    let ret_v = GenTensor::<$a>::new_move(u, vec![mmn, m]).t();
	    if info != 0 {
		panic!("svd return inf ononzero!");
	    }
	    (ret_u, ret_s, ret_v)
        }
    }
}

#[cfg(feature = "use-blas-lapack")]
lapack_svd!(f32, svd_f32);

#[cfg(feature = "use-blas-lapack")]
lapack_svd!(f64, svd_f64);


#[cfg(test)]
mod tests {
    use crate::tensor_impl::gen_tensor::GenTensor;
    use super::*;

    #[test]
    #[cfg(feature = "use-blas-lapack")]
    fn test_svd() {
	let m = GenTensor::<f64>::new_raw(&[4., 12., -16., 12., 37., -43., -16., -43., 98.], &[3, 3]);
	let (u, s, v) = svd_f64(&m);
	println!("{:?}, {:?}, {:?}", u, s, v);
    }
}
