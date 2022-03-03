
#[cfg(feature = "use-blas-lapack")]
use blas::*;
use std::marker::PhantomData;


pub struct BlasAPI<T> {
    d: PhantomData<T>,
}
#[cfg(feature = "use-blas-lapack")]
impl BlasAPI<f32> {
    // level 1
    pub fn rotg() {unimplemented!();}
    pub fn rotmg() {unimplemented!();}
    pub fn rot() {unimplemented!();}
    pub fn rotm() {unimplemented!();}
    pub fn swap(n: usize, x: &mut [f32], incx: usize, y: &mut [f32], incy: usize) {
        unsafe {
            sswap(n as i32, x, incx as i32, y, incy as i32);
        }
    }
    pub fn scal(n: usize, a: f32, x: &mut [f32], incx: usize) {
        unsafe {
            sscal(n as i32, a, x, incx as i32);
        }
    }
    pub fn copy(n: usize, x: & [f32], incx: usize, y: &mut [f32], incy: usize) {
        unsafe {
            scopy(n as i32, x, incx as i32, y, incy as i32);
        }
    }
    pub fn axpy(n: usize,
                alpha: f32,
                x: &[f32], incx: usize,
                y: &mut [f32], incy: usize) {
        unsafe {
            saxpy(n as i32, alpha, x, incx as i32, y, incy as i32);
        }
    }
    pub fn dot() {unimplemented!();}
    pub fn dsdot(n: usize, sb: &[f32], x: &[f32], incx: usize, y: &[f32], incy: usize) -> f32 {
        unsafe {
            sdsdot(n as i32, sb, x, incx as i32, y, incy as i32)
        }
    }
    pub fn nrm2(n: usize, x: &[f32], incx: usize) -> f32 {
        unsafe {
            snrm2(n as i32, x, incx as i32)
        }
    }
    pub fn cnrm2() {unimplemented!();}
    pub fn asum() {unimplemented!();}
    pub fn iamax() {unimplemented!();}
    

    // level 2
    pub fn gemv(trans: bool, m: usize, n: usize, alpha: f32,
                a: &[f32], lda: usize,
                x: &[f32], incx: usize, beta: f32,
                y: &mut [f32], incy: usize
    ) {
        let mut trans_flag = [0; 1];
        if !trans {
            'N'.encode_utf8(&mut trans_flag);
        } else {
            'T'.encode_utf8(&mut trans_flag);
        }
        
        unsafe {
            sgemv(
                trans_flag[0],
                m as i32,
                n as i32,
                alpha,
                a,
                lda as i32,
                x,
                incx as i32,
                beta as f32,
                y,
                incy as i32,
            )
        }
    }
    pub fn gbmv() {unimplemented!();}
    pub fn symv() {unimplemented!();}
    pub fn sbmv() {unimplemented!();}
    pub fn spmv() {unimplemented!();}
    pub fn trmv() {unimplemented!();}
    pub fn tbmv() {unimplemented!();}
    pub fn tpmv() {unimplemented!();}
    pub fn trsv() {unimplemented!();}
    pub fn tbsv() {unimplemented!();}
    pub fn tpsv() {unimplemented!();}
    pub fn ger() {unimplemented!();}
    pub fn syr() {unimplemented!();}
    pub fn spr() {unimplemented!();}
    pub fn syr2() {unimplemented!();}
    pub fn spr2 () {unimplemented!();}

    // level 3
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

    pub fn symm() {unimplemented!();}
    pub fn syrk() {unimplemented!();}
    pub fn syr2k() {unimplemented!();}
    pub fn trmm() {unimplemented!();}
    pub fn trsm() {unimplemented!();}
}


#[cfg(feature = "use-blas-lapack")]
impl BlasAPI<f64> {
    pub fn axpy(n: usize, alpha: f64, x: &[f64], incx: usize, y: &mut [f64], incy: usize) {
        unsafe {
            daxpy(n as i32, alpha, x, incx as i32, y, incy as i32);
        }
    }
    
    pub fn gemm(transa: bool, transb: bool, m: usize, n: usize, k: usize,
            alpha: f64, a: &[f64], lda: usize,
            b: &[f64], ldb: usize,
            beta: f64, c: &mut [f64], ldc: usize,) {
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
            dgemm(
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


#[cfg(all(test, feature = "use-blas-lapack"))]
mod tests {
    use super::*;
    
    extern crate openblas_src;

    #[test]
    fn test_swap() {
        let mut v1 = [1., 2.];
        let mut v2 = [3., 4.];
        BlasAPI::<f32>::swap(2, &mut v1, 1, &mut v2, 1);
        println!("{:?}, {:?}", v1, v2);
        assert_eq!(v1, [3., 4.]);
        assert_eq!(v2, [1., 2.]);
    }

    #[test]
    fn test_scal() {
        let mut v1 = [1., 2.];
        BlasAPI::<f32>::scal(2, 2., &mut v1, 1);
        println!("{:?}", v1);
        assert_eq!(v1, [2., 4.]);
    }

    #[test]
    fn test_copy() {
        let v1 = [1., 2.];
        let mut v2 = [3., 4.];
        BlasAPI::<f32>::copy(2, &v1, 1, &mut v2, 1);
        println!("{:?}, {:?}", v1, v2);
        assert_eq!(v1, [1., 2.]);
        assert_eq!(v2, [1., 2.]);
    }

    #[test]
    fn test_blas_axpy() {
        let v1 = [1., 2.];
        let mut v2 = [1., 2.];
        BlasAPI::<f32>::axpy(2, 1., &v1, 1, &mut v2, 1);
        println!("{:?}", v2);

        let v1 = [1., 2.];
        let mut v2 = [1., 2.];
        BlasAPI::<f64>::axpy(2, 1., &v1, 1, &mut v2, 1);
        println!("{:?}", v2);
    }

    #[test]
    fn test_dot() {
        let v1 = [1., 2.];
        let v2 = [3., 4.];
        let v3 = [5., 6.];
        let result = BlasAPI::<f32>::dsdot(2, &v1, &v2, 1, &v3, 1);
        println!("{:?}", result);
        assert_eq!(result, 40.);
    }

    #[test]
    fn test_nrm2() {
        let v1 = [1., 2.];
        let result = BlasAPI::<f32>::nrm2(2, &v1, 1);
        println!("{:?}", result);
        assert_eq!(result, 2.236068);
    }

    #[test]
    fn test_gemv() {
        let v1 = [1., 2., 3., 4., 5., 6.,];
        let v2 = [3., 4.];
        let mut v3 = [5., 6., 7.];
        BlasAPI::<f32>::gemv(false, 3, 2, 1., &v1, 3, &v2, 1, 1., &mut v3, 1);
        println!("{:?}", v3);
        assert_eq!(v3, [24.0, 32.0, 40.0]);
    }

    #[test]
    fn test_blas_gemm() {
        let v1 = [1., 2., 3., 4., 5., 6., 7., 8.];
        let v2 = [1., 2., 3., 4., 5., 6.,];
        let mut v3 = [0.; 12];
        BlasAPI::<f32>::gemm(false, false, 4, 3, 2, 1., &v1, 4, &v2, 2, 1., &mut v3, 4);
        println!("{:?}", v3);

        //let mut v1 = [1., 2.];
        //let mut v2 = [1., 2.];
        //BlasAPI::<f64>::axpy(2, 1., &v1, 1, &mut v2, 1);
        //println!("{:?}", v2);
    }
}
