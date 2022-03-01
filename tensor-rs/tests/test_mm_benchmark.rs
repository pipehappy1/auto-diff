//use tensor_rs::tensor::*;
//use tensor_rs::tensor::blas::*;


#[cfg(test)]
mod tests {

    extern crate openblas_src;
    use tensor_rs::tensor_impl::lapack_tensor::blas::BlasAPI;
    use tensor_rs::tensor::Tensor;

    #[test]
    fn test_gemm1() {

    for _i in 0..100000 {
        let mut v1: Vec<f32> = (0..128*256).map(|x| x as f32).collect();
        let mut v2: Vec<f32> = (0..128*256).map(|x| x as f32).collect();
        let mut v3: [f32; 65536] = [0.; 65536];

        let trans = false;
        BlasAPI::<f32>::gemm(trans, trans, 256, 256, 128, 1., &v1, 256, &v2, 128, 1., &mut v3, 256);
        //println!("{:?}", v3);        
    }

}    
    #[test]
    fn test_mm1() {

        for _i in 0..1000 {
            let v1 = Tensor::ones(&[256, 128]);
            let v2 = Tensor::ones(&[128, 256]);

            v1.mm(&v2);
        }           
    }
}

