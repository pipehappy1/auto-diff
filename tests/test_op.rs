use auto_diff::graph::*;
use auto_diff::tensor::*;
use auto_diff::op::*;

use std::rc::Rc;
use std::cell::RefCell;

#[test]
fn test_Linear() {
    let mut op = Linear::new(None, None, true);
    let mut input = Tensor::fill(&vec![3,2], 1.);
    let mut output = Tensor::fill(&vec![3,1], 1.);
    op.apply(&mut vec![Rc::new(RefCell::new(input))],
             &mut vec![Rc::new(RefCell::new(output))]);
}
