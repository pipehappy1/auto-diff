#[cfg(feature = "use-blas-lapack")]
use lapack::*;
use std::marker::PhantomData;
use std::cmp;


pub struct LapackAPI<T> {
    d: PhantomData<T>,
}

#[cfg(feature = "use-blas-lapack")]
impl LapackAPI<f32> {
    /// = 'A':  all M columns of U and all N rows of V**T are
    ///         returned in the arrays U and VT;
    /// = 'S':  the first min(M,N) columns of U and the first
    ///         min(M,N) rows of V**T are returned in the arrays U
    ///         and VT;
    /// = 'O':  If M >= N, the first N columns of U are overwritten
    ///         on the array A and all rows of V**T are returned in
    ///         the array VT;
    ///         otherwise, all columns of U are returned in the
    ///         array U and the first M rows of V**T are overwritten
    ///         in the array A;
    /// = 'N':  no columns of U or rows of V**T are computed.
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
    use super::*;

    #[test]
    fn test_svd() {
        let mut m: Vec<f64> = vec![4., 12., -16., 12., 37., -43., -16., -43., 98.];
	let mut s = vec![0. ; 9];
	let mut u = vec![0. ; 9];
	let mut vt = vec![0. ; 9];
	let mut info: i32 = 0;
        LapackAPI::<f64>::gesdd(&'S', 3, 3,
				&mut m, 3,
				&mut s,
				&mut u, 3,
				&mut vt, 3,
				&mut info);
        //println!("{:?}, {:?}, {:?}", u, s, vt);
        let es: Vec<f64> = vec![123.47723179013161, 15.503963229407585, 0.018804980460810704];
        assert!((s[0] - es[0]).abs() < 1e-6);
	assert!((s[5] - es[1]).abs() < 1e-6);
	assert!((s[8] - es[2]).abs() < 1e-1);
    }
}
