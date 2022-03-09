

#[cfg(all(test, feature = "use-serde"))]
mod tests {
    use crate::collection::directed_graph::Graph;
    use crate::collection::generational_index::GenKey;

    #[test]
    fn test_serde_graph() {
	let mut m1 = Graph::<GenKey, GenKey>::new();
	let data1 = GenKey::new(1, 1);
	let data2 = GenKey::new(2, 6);
	let op1 = GenKey::new(3, 8);
	m1.add_data(&data1).unwrap();
	m1.add_data(&data2).unwrap();
	m1.add_op(&op1).unwrap();
	m1.connect(&[data1], &[data2], &op1).unwrap();
	
        let serialized = serde_pickle::to_vec(&m1, true).unwrap();
        let deserialized = serde_pickle::from_slice(&serialized).unwrap();
        //println!("{:?}", deserialized);
        assert_eq!(m1, deserialized);
    }
}
