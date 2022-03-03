#[cfg(feature = "use-blas-lapack")]
use lapack::*;
use std::marker::PhantomData;
use std::cmp;


pub struct LapackAPI<T> {
    d: PhantomData<T>,
}

#[cfg(feature = "use-blas-lapack")]
impl LapackAPI<f32> {
    pub fn gesdd(jobz: &char, m: usize, n: usize, 
                 a: &mut [f32], lda: usize, 
                 s: &mut [f32], 
                 u: &mut [f32], ldu: usize, 
                 vt: &mut [f32], ldvt: usize, 
                 info: &mut i32) {
        let (mx, mn) = if m > n {(m, n)} else {(n, m)};
        let (jobz, mini_work): (u8, usize) = match jobz {
            'A' => {
                (b'A', 4*mn*mn + 6*mn + mx)
            },
            'S' => {
                (b'S', 4*mn*mn + 7*mn)
            },
            'O' => {
                (b'O', 3*mn + cmp::max( mx, 5*mn*mn + 4*mn ))
            },
            'N' => {
                (b'N', 3*mn + cmp::max( mx, 7*mn ))
            },
            _ => panic!("unknown jobz: {}", jobz),
        };
        let mut work: Vec<f32> = vec![0.; mini_work];
        let lwork = mini_work as i32;
        let mut iwork: Vec<i32> = vec![0; 8*mn];
        unsafe {
            sgesdd(jobz, m as i32, n as i32,
                   a, lda as i32, 
                   s, 
                   u, ldu as i32, 
                   vt, ldvt as i32, 
                   &mut work, lwork, &mut iwork,
                   info);
        }
    }
}

#[cfg(feature = "use-blas-lapack")]
impl LapackAPI<f64> {
    pub fn gesdd(jobz: &char, m: usize, n: usize,
                 a: &mut [f64], lda: usize,
                 s: &mut [f64],
                 u: &mut [f64], ldu: usize,
                 vt: &mut [f64], ldvt: usize,
                 info: &mut i32) {
        let (mx, mn) = if m > n {(m, n)} else {(n, m)};
        let (jobz, mini_work): (u8, usize) = match jobz {
            'A' => {
                (b'A', 4*mn*mn + 6*mn + mx)
            },
            'S' => {
                (b'S', 4*mn*mn + 7*mn)
            },
            'O' => {
                (b'O', 3*mn + cmp::max( mx, 5*mn*mn + 4*mn ))
            },
            'N' => {
                (b'N', 3*mn + cmp::max( mx, 7*mn ))
            },
            _ => panic!("unknown jobz: {}", jobz),
        };
        let mut work: Vec<f64> = vec![0.; mini_work];
        let lwork = mini_work as i32;
        let mut iwork: Vec<i32> = vec![0; 8*mn];
        unsafe {
            dgesdd(jobz, m as i32, n as i32,
                   a, lda as i32,
                   s,
                   u, ldu as i32,
                   vt, ldvt as i32,
                   &mut work, lwork, &mut iwork,
                   info);
        }
    }
}

#[cfg(all(test, feature = "use-blas-lapack"))]
mod tests {
    //use super::*;

    #[test]
    fn test_svd() {
        let v1 = [1., 2.];
        let v2 = [3., 4.];
        //LapackAPI::<f32>::svd(2, &mut v1, 1, &mut v2, 1);
        println!("{:?}, {:?}", v1, v2);
        assert_eq!(v1, [1., 2.]);
        assert_eq!(v2, [3., 4.]);
    }
}
