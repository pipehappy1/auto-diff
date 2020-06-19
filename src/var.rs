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
        let mut new_var = Var::_default();

        // The following two lines need to go together.
        {
            self.net.borrow_mut().init_var(&mut new_var);
            new_var.net = Rc::clone(&self.net);
        }
        new_var
    }

    pub fn var_value(&mut self, v: Tensor) -> Var {
        let ret = self.var();
        ret.set(v);
        ret
    }

    ///
    /// Create a composed function with the given closure.
    ///
    pub fn func<F>(&self, ops: &[&Func], closure: F) -> Func
    where F: Fn(&[&Var]) -> Var {
        
        let sub_ops = Vec::new();
        let id = self.net.borrow_mut().init_func(&sub_ops);
        let ret = Func::_new(id, self.net.clone(), None);
        ret
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
        self.net.borrow_mut().visit_op(closure, None, None);
    }

    //
    // concrete Func
    //
    pub fn linear(&self, in_features: Option<usize>,
                  out_features: Option<usize>,
                  bias: bool) -> Func {
        let op = Linear::new(in_features, out_features, bias);
        let id = self.net.borrow_mut().init_op(Op::new(Box::new(op)));
        let ret = Func::_new(id, self.net.clone(), None);
        ret
    }
    pub fn mseloss(&self) -> Func {
        let op = MSELoss::new();
        let id = self.net.borrow_mut().init_op(Op::new(Box::new(op)));
        let ret = Func::_new(id, self.net.clone(), None);
        ret
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
    pub fn _default() -> Var {
        println!("Var::new() create a unattached variable, This is usually not what we want.");
        Var {
            id: NetIndex::new(0, 0),
            net: Rc::new(RefCell::new(Net::new())),
        }
    }

    // return a var with association with the network.
    ///
    /// Create a variable using an existing variable.
    ///
    pub fn new_attached(&self) -> Var {
        println!("Deprecated! Var::new_attached");
        let mut new_var = Var::_default();

        // The following two lines need to go together.
        {
            self.net.borrow_mut().init_var(&mut new_var);
            new_var.net = Rc::clone(&self.net);
        }
        new_var
    }

    ///
    /// Create a variable given the network.
    /// This is for whoever has a network handler.
    ///
    pub fn new_variable(net: Rc<RefCell<Net>>) -> Var {
        let mut ret = Var::_default();
        ret.net = net.clone();
        net.borrow_mut().init_var(&mut ret);
        ret
    }

    ///
    /// Getter for the id member.
    ///
    pub fn get_id(&self) -> &NetIndex {
        &self.id
    }

    ///
    /// Setter for the id member.
    ///
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

    pub fn backward(&self, og: f64) {
        
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

impl Drop for Var {
    fn drop(&mut self) {
        self.net.borrow_mut().del_var(&self);
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

///
/// User facing struct representing concrete ops and composed functions.
///
pub struct Func {
    id: NetIndex,
    net: Rc<RefCell<Net>>,
    closure: Option<Box<dyn Fn(&[&Var]) -> Var>>,
}
impl Func {
    pub fn _default() -> Func {
        Func {
            id: NetIndex::new(0, 0),
            net: Rc::new(RefCell::new(Net::new())),
            closure: None,
        }
    }

    pub fn _new(id: NetIndex,
                net: Rc<RefCell<Net>>,
                closure: Option<Box<dyn Fn(&[&Var]) -> Var>>) -> Func {
        Func {
            id: id,
            net: net,
            closure: closure,
        }
    }

    pub fn get_id(&self) -> &NetIndex {
        &self.id
    }

    pub fn call(&self, input: &[&Var]) -> Var {
        //
        // Ask graph to see is this Func is concrete or composed.
        // If this is concrete,
        //     1. connect the input and output variable in graph.
        //     2. do real computation.
        // If this is composed.
        //     1. call the closure.
        //
        // If there is already a output variable, it SHOULD be reused.

        //
        // If there is already an output variable associated with this
        // Func, just use it.
        //
        

        if self.closure.is_some() {
            return (self.closure.as_ref().unwrap())(input);
        } else {
            let mut ret;
            let existing_output = self.net.borrow().get_func_output(&self);
            if existing_output.is_some() {
                ret = Var::_default();
                ret.set_id(existing_output.unwrap());
            } else {
                ret = Var::new_variable(self.net.clone());
            }

            self.net.borrow_mut().connect2(input, &self, &[&ret]);

            self.net.borrow().eval_op(input, &self, &[&ret]);

            return ret;
        }
    }

    // This is for optimizer call over concrete ops
    pub fn _visit_op<F>(&self, closure: F)
    where F: Fn(&Op) {
        let mut todo_funcs = vec![self.id];
        let mut all_ops: Vec<NetIndex> = Vec::new();
        while todo_funcs.len() > 0 {
            let todo_func = todo_funcs.pop().expect("");
            let sub_funcs = self.net.borrow().get_sub_func(todo_func);
            if sub_funcs.len() == 0 { // this is a concrete Func
                all_ops.push(todo_func);
            } else { // this is a composed Func
                todo_funcs.extend(&sub_funcs);
            }
        }

        self.net.borrow_mut().visit_op(closure, Some(all_ops), None);
    }
}

impl fmt::Display for Func {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(Func: {}, {})",
            self.id,
            1
        )
    }
}

impl Drop for Func {
    fn drop(&mut self) {
        self.net.borrow_mut().del_func_or_op(&self);
    }
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
