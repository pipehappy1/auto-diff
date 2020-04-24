//use auto_diff::tensor::*;

use rand::prelude::*;
use rand::distributions::Uniform;
use rand_distr::{Normal, Distribution, LogNormal};

//use ndarray::*;
//use ndarray_linalg::*;
//
//// Solve `Ax=b`
//fn solve() -> Result<(), error::LinalgError> {
//    let a: Array2<f64> = random((3, 3));
//    let b: Array1<f64> = random(3);
//    let _x = a.solve(&b)?;
//    Ok(())
//}
//
//// Solve `Ax=b` for many b with fixed A
//fn factorize() -> Result<(), error::LinalgError> {
//    let a: Array2<f64> = random((3, 3));
//    let f = a.factorize_into()?; // LU factorize A (A is consumed)
//    for _ in 0..10 {
//        let b: Array1<f64> = random(3);
//        let _x = f.solve_into(b)?; // solve Ax=b using factorized L, U
//    }
//    Ok(())
//}



fn main() {
//    let a = &1;
//    let b = &2;
//
//    match (a, b) {
//        (1, 2) => println!(""),
//        _ => println!(""),
//    }
//    
//    solve().unwrap();
//    factorize().unwrap();

    let mut rng = StdRng::seed_from_u64(671);

    //let geometric = ::new(2., 3.).expect("");
    //println!("{}", geometric.sample(&mut rng));
    
    let lognormal = LogNormal::new(2., 3.).expect("");
    println!("{}", lognormal.sample(&mut rng));

    let normal = Normal::new(2., 3.).expect("");
    println!("{}", normal.sample(&mut rng));

    let random = Uniform::from(0..10);
    println!("{}", random.sample(&mut rng));

    let uniform = Uniform::from(-0.1..10.);
    println!("{}", uniform.sample(&mut rng));
}
