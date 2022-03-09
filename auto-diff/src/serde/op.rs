

#[cfg(all(test, feature = "use-serde"))]
mod tests {
    use crate::op::linear::Linear;
    
    #[test]
    fn test_serde_op() {
	let mut m1 = Linear::new(None, None, true);
	
        let serialized = serde_pickle::to_vec(&m1, true).unwrap();
        let deserialized = serde_pickle::from_slice(&serialized).unwrap();
        //println!("{:?}", deserialized);
        assert_eq!(m1, deserialized);
    }
}
