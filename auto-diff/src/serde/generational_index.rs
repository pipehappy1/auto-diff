#[cfg(all(test, feature = "use-serde"))]
mod tests {
    use crate::collection::generational_index::{GenIndex, GenKey};

    #[test]
    fn test_serde_genkey() {
        let m1 = GenKey::new(1, 3);

        let serialized = serde_pickle::to_vec(&m1, true).unwrap();
        let deserialized = serde_pickle::from_slice(&serialized).unwrap();
        //println!("{:?}", deserialized);
        assert_eq!(m1, deserialized);
    }

    #[test]
    fn test_serde_genindex() {
        let mut m1 = GenIndex::<f32>::new();
        let key = m1.insert(10.);
        m1.remove(&key).unwrap();
        m1.insert(12.);

        let serialized = serde_pickle::to_vec(&m1, true).unwrap();
        let deserialized = serde_pickle::from_slice(&serialized).unwrap();
        //println!("{:?}", deserialized);
        assert_eq!(m1, deserialized);
    }
}
