use auto_diff::rand::*;
use auto_diff::tensor::*;


fn main() {

    let a = Tensor::fill(&vec![3,4], 0.2);
    let mut rng = RNG::new();
    rng.normal_(&a, 0., 1.);
    println!("{}", a);
}
