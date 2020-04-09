use auto_diff::tensor::*;

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
