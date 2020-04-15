use auto_diff::tensor::gen_tensor::*;

#[test]
fn test_gentensor() {
    {
        let mut m2 = GenTensor::<f64>::new_raw(&vec![1., 2., 3., 4.,], &vec![2, 2]);
        *m2.get_mut(&vec![0,0]) = 5.;
        assert_eq!(m2.get_raw(), vec![5., 2., 3., 4.,])
    }

    {
        let mut m1 = GenTensor::<f64>::fill(1., &vec![2, 3, 5]);
        m1.permute(&vec![2, 0, 1]);
        assert_eq!(m1.size(), vec![5, 2, 3]);

        let mut m2 = GenTensor::<f64>::new_raw(&vec![1., 2., 3., 4.,], &vec![2, 2]);
        m2.permute(&vec![1, 0]);
        assert_eq!(m2.get_raw(), vec![1., 3., 2., 4.]);
    }

}

#[test]
fn test_gen_tensor_get() {
    {
        let m1 = GenTensor::<f64>::fill(1., &vec![10, 3, 28, 30]);
        assert_eq!(m1.get_N().get_raw(), vec![10.]);
        assert_eq!(m1.get_C().get_raw(), vec![3.]);
        assert_eq!(m1.get_H().get_raw(), vec![28.]);
        assert_eq!(m1.get_W().get_raw(), vec![30.]);

        let result = std::panic::catch_unwind(
            ||
                m1.get_D().get_raw()
        );
        assert!(result.is_err());
    }
}
