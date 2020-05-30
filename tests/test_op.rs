use auto_diff::tensor::*;
use auto_diff::op::*;



#[test]
fn test_linear() {
    let mut op = Linear::new(None, Some(5), false);
    let input = Tensor::fill(&vec![3,2], 1.);
    let output = Tensor::new();
    op.apply(&mut vec![&input],
             &mut vec![&output]);
    //println!("{}", output);

    let mut op1 = Linear::new(Some(2), Some(5), true);
    op1.weight().swap(Tensor::from_vec_f32(&vec![1.,2.,3.,4.,5.,6.,7.,8.,9.,10.], &vec![2, 5]));
    op1.bias().swap(Tensor::from_vec_f32(&vec![1.,2.,3.,4.,5.], &vec![5]));
    let input = Tensor::fill(&vec![3,2], 1.);
    let output = Tensor::new();
    op1.apply(&mut vec![&input],
             &mut vec![&output]);
    assert_eq!(output, Tensor::from_vec_f32(&vec![8.0, 11.0, 14.0, 17.0, 20.0, 8.0, 11.0, 14.0, 17.0, 20.0, 8.0, 11.0, 14.0, 17.0, 20.0],
                                            &vec![3, 5]));
}


#[test]
fn test_mse() {
    let mut op = MSELoss::new();
    let input1 = Tensor::fill(&vec![3, 2], 1.);
    let input2 = Tensor::fill(&vec![3, 2], 2.);
    let output = Tensor::fill(&vec![1], 1.);
    op.apply(&mut vec![&input1, &input2], &mut vec![&output]);
    assert_eq!(output, Tensor::from_vec_f32(&vec![1.], &vec![1]));
}
