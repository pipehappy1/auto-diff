use std::fmt;
#[cfg(feature = "use-blas")]
use blas::*;
use std::marker::PhantomData;


pub struct BlasTensor<T> {
    d: Vec<T>,
    dim: Vec<usize>,
}
impl<T> BlasTensor<T> where T: num_traits::Float {
    pub fn mm(&self, o: &BlasTensor<T>) -> BlasTensor<T> {
        BlasTensor {
            d: Vec::new(),
            dim: Vec::new()
        }
    }
}

pub struct BlasAPI<T> {
    d: PhantomData<T>,
}
#[cfg(feature = "use-blas")]
impl BlasAPI<f32> {
    pub fn axpy(n: usize, alpha: f32, x: &[f32], incx: usize, y: &mut [f32], incy: usize) {
        unsafe {
            saxpy(n as i32, alpha, x, incx as i32, y, incy as i32);
        }
    }
    
    pub fn gemm(transa: bool, transb: bool, m: usize, n: usize, k: usize,
            alpha: f32, a: &[f32], lda: usize,
            b: &[f32], ldb: usize,
            beta: f32, c: &mut [f32], ldc: usize,
    ) {
        let mut transa_flag = [0; 1];
        if !transa {
            'N'.encode_utf8(&mut transa_flag);
        } else {
            'T'.encode_utf8(&mut transa_flag);
        }

        let mut transb_flag = [0; 1];
        if !transb {
            'N'.encode_utf8(&mut transb_flag);
        } else {
            'T'.encode_utf8(&mut transb_flag);
        }
        unsafe {
            sgemm(
                transa_flag[0],
                transb_flag[0],
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a,
                lda as i32,
                b,
                ldb as i32,
                beta,
                c,
                ldc as i32,
            );
        }
    }
}
#[cfg(feature = "use-blas")]
impl BlasAPI<f64> {
    pub fn axpy(n: usize, alpha: f64, x: &[f64], incx: usize, y: &mut [f64], incy: usize) {
        unsafe {
            daxpy(n as i32, alpha, x, incx as i32, y, incy as i32);
        }
    }
    
    pub fn gemm() {
        
    }
}


#[cfg(all(test, feature = "use-blas"))]
mod tests {
    use super::*;
    extern crate openblas_src;

    #[test]
    fn test_blas_axpy() {
        let mut v1 = [1., 2.];
        let mut v2 = [1., 2.];
        BlasAPI::<f32>::axpy(2, 1., &v1, 1, &mut v2, 1);
        println!("{:?}", v2);

        let mut v1 = [1., 2.];
        let mut v2 = [1., 2.];
        BlasAPI::<f64>::axpy(2, 1., &v1, 1, &mut v2, 1);
        println!("{:?}", v2);
    }

    #[test]
    fn test_blas_gemm() {
        let mut v1 = [1., 2., 3., 4., 5., 6., 7., 8.];
        let mut v2 = [1., 2., 3., 4., 5., 6.,];
        let mut v3 = [0.; 12];
        BlasAPI::<f32>::gemm(false, false, 4, 3, 2, 1., &v1, 4, &v2, 2, 1., &mut v3, 4);
        println!("{:?}", v3);

        //let mut v1 = [1., 2.];
        //let mut v2 = [1., 2.];
        //BlasAPI::<f64>::axpy(2, 1., &v1, 1, &mut v2, 1);
        //println!("{:?}", v2);
    }
}
