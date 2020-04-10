use auto_diff::tensor::*;
use auto_diff::op::*;



#[test]
fn test_linear() {
    let mut op = Linear::new(None, None, true);
    let input = Tensor::fill(&vec![3,2], 1.);
    let output = Tensor::fill(&vec![3,1], 1.);
    op.apply(&mut vec![input],
             &mut vec![output]);
}
