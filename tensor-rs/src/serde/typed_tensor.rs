

#[cfg(all(test, feature = "use-serde"))]
mod tests {
    use crate::typed_tensor::TypedTensor;
    use crate::tensor_impl::gen_tensor::GenTensor;

    #[test]
    fn test_serde() {
        let m1 = GenTensor::<f64>::new_raw(&vec![1.,2.,3.,4.,5.,6.], &vec![3,2]);
	let m1 = TypedTensor::Typef64(m1);

        let serialized = serde_pickle::to_vec(&m1, true).unwrap();
        let deserialized = serde_pickle::from_slice(&serialized).unwrap();
        //println!("{:?}", deserialized);
        assert_eq!(m1, deserialized);
    }
}
