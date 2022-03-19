//use auto_diff::Var;
use auto_diff::optim::Optimizer;
use auto_diff::compute_graph::Net;
use std::rc::Rc;
use std::cell::RefCell;

pub struct Momentum {

}

impl Momentum {
    
}

impl Optimizer for Momentum {
    fn step(&mut self, _net: Rc<RefCell<Net>>) {
	unimplemented!()
    }
}
