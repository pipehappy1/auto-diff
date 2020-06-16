use std::cell::RefCell;
use std::collections::{BTreeMap};
use std::fmt;
use std::rc::Rc;
use std::mem::drop;

use crate::tensor::Tensor;
use crate::op::*;
use crate::compute_graph::*;
use crate::collection::generational_index::*;

pub struct Module {
    net: Rc<RefCell<Net>>,
}

/// Network holder.
impl Module {
    /// Create an empty module.
    /// A module is mainly used to create new variables.
    pub fn new() -> Module {
        Module {
            net: Rc::new(RefCell::new(Net::new())),
        }
    }

    /// Create a new variable.
    pub fn var(&mut self) -> Var {
        let mut new_var = Var::new();

        // The following two lines need to go together.
        {
            self.net.borrow_mut().init_var(&mut new_var);
            new_var.net = Rc::clone(&self.net);
        }
        new_var
    }
    pub fn rm_var(&mut self, var: &Var) {
        self.net.borrow_mut().del_var(var);
        drop(var);
    }

    /// Try best evaluation of the computation graph.
    pub fn eval(&self) {
        self.net.borrow_mut().eval().expect("");
    }
    
    /// Same as eval
    pub fn forward(&self) { 
        self.net.borrow_mut().eval().expect("");
    }

    /// Back propagation
    pub fn backward_vector(&self, og: &BTreeMap<NetIndex, Tensor>) {
        self.net.borrow_mut().bptt(og);
    }

    /// Back propgation with a single value.
    pub fn backward(&self, og: f32) {
        self.net.borrow_mut().bptt_scale(og);
    }

    /// iterator over all data node.
    pub fn _visit_data<F>(&self, closure: F)
    where F: Fn(&Op) {
        self.net.borrow_mut().visit_data(closure);
    }
    /// iterator over all op node.
    pub fn _visit_op<F>(&self, closure: F)
    where F: Fn(&Op) {
        self.net.borrow_mut().visit_op(closure);
    }
}


/// Introduce variable to the system by creating Var
pub struct Var {
    id: NetIndex,
    net: Rc<RefCell<Net>>,
}

macro_rules! var_op_method {
    ($a:ident, $b:ident) => {
        pub fn $a(&self, o: &Var) -> Var {
            let result = self.new_attached();
            self.net
                .borrow_mut()
                .connect(&vec![self.id, o.id], Op::new(Box::new($b::new())), &vec![result.id]);
            result
        }
    }
    
}

impl Var {
    pub fn new() -> Var {
        Var {
            id: NetIndex::new(0, 0),
            net: Rc::new(RefCell::new(Net::new())),
        }
    }

    pub fn new_attached(&self) -> Var {
        let mut new_var = Var::new();

        // The following two lines need to go together.
        {
            self.net.borrow_mut().init_var(&mut new_var);
            new_var.net = Rc::clone(&self.net);
        }
        new_var
    }

    pub fn get_id(&self) -> &NetIndex {
        &self.id
    }

    pub fn set_id(&mut self, new_id: NetIndex) {
        self.id = new_id;
    }

    /// Give the variable a value
    ///
    /// ```
    /// # use auto_diff::var::*;
    /// # use auto_diff::tensor::*;
    /// let mut m = Module::new();
    /// let a = m.var();
    /// a.set(Tensor::new());
    /// ```
    pub fn set(&self, v: Tensor) {
        self.net
            .borrow_mut()
            .get_data_mut()
            .replace(&self.id, v).expect("");

        self.net.borrow_mut().set_mark(&self.id);
    }
    pub fn unset(&self) {
        self.net.borrow_mut().unset_mark(&self.id);
    }

    /// Get the underlying tensor.
    pub fn get(&self) -> Tensor {
        self.net.borrow().get_data().get(&self.id).expect("").clone()
    }

    /// Get the underlying gradient tensor.
    pub fn get_grad(&self) -> Tensor {
        self.net.borrow().get_grad().get(&self.id).expect("").clone()
    }

    /// apply the var to pre-faburacated op.
    pub fn to(&self, op: &Op) -> Var {
        let result = self.new_attached();
        self.net.borrow_mut().connect(&vec![self.id], op.clone(), &vec![result.id]);
        result
    }

    // uplift method from Tensor to Var
    pub fn size(&self) -> Vec<usize> {
        self.net.borrow().get_data().get(&self.id).expect("").size()
    }
    pub fn numel(&self) -> usize {
        self.net.borrow().get_data().get(&self.id).expect("").numel()
    }

    // Convient method definition.
    var_op_method!(add, Add);
    var_op_method!(sub, Sub);
    var_op_method!(mul, Mul);
    var_op_method!(div, Div);
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}, {})",
            self.id,
            self.net.borrow().get_data().get(&self.id).expect("")
        )
    }
}

// uplift loss function from op to here.
pub fn mseloss(a: &Var, b: &Var) -> Var {
    let result = a.new_attached();
    a.net.borrow_mut().connect(&vec![a.id, b.id], Op::new(Box::new(MSELoss::new())), &vec![result.id]);
    result
}
pub fn bcewithlogitsloss(predict: &Var, label: &Var) -> Var {
    let result = predict.new_attached();
    predict.net.borrow_mut().connect(&vec![predict.id, label.id], Op::new(Box::new(BCEWithLogitsLoss::new())), &vec![result.id]);
    result
}
pub fn crossentropyloss(predict: &Var, label: &Var) -> Var {
    let result = predict.new_attached();
    predict.net.borrow_mut().connect(&vec![predict.id, label.id], Op::new(Box::new(CrossEntropyLoss::new())), &vec![result.id]);
    result
}





#[cfg(test)]
mod tests {
    //use super::*;

    //#[test]
    //fn genindex_new_add_del() {
    //    let mut m = Module::new();
    //    let va = m.var();
    //
    //}
}
