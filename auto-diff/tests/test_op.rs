use tensor_rs::tensor::*;
use auto_diff::op::*;



#[test]
fn test_linear() {
    let mut op = Linear::new(None, Some(5), false);
    let input = Tensor::fill(&vec![3,2], 1.);
    let output = Tensor::new();
    op.apply(&mut vec![&input],
             &mut vec![&output]);
    //println!("{}", output);


}
