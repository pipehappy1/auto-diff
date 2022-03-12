

#[cfg(all(test, feature = "use-serde"))]
mod tests {
    use crate::compute_graph::Net;
    use crate::var::Var;
    use rand::prelude::*;

    #[test]
    fn test_serde_net() {
	let mut rng = StdRng::seed_from_u64(671);
	let n = 10;
	let data = Var::normal(&mut rng, &vec![n, 2], 0., 2.);
	let result = data.matmul(&Var::new(&vec![2., 3.], &vec![2, 1])).unwrap() + Var::new(&vec![1.], &vec![1]);

        let serialized = serde_pickle::to_vec(&*result.dump_net().borrow(), true).unwrap();
        let deserialized: Net = serde_pickle::from_slice(&serialized).unwrap();
        //println!("{:?}", deserialized);
        //assert_eq!(*result.dump_net().borrow(), deserialized);
    }

}
