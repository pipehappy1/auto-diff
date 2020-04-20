use std::cell::RefCell;
use std::collections::BTreeSet;
use std::fmt;
use std::rc::Rc;

use super::collection::generational_index::*;
use super::tensor::Tensor;
use super::op::*;


pub struct Module {
    net: Rc<RefCell<Net>>,
}

/// Network holder.
impl Module {
    pub fn new() -> Module {
        Module {
            net: Rc::new(RefCell::new(Net::new())),
        }
    }

    pub fn var(&mut self) -> Var {
        let mut new_var = Var::new();

        // The following two lines need to go together.
        {
            self.net.borrow_mut().init_var(&mut new_var);
            new_var.net = Rc::clone(&self.net);
        }
        new_var
    }

    /// Try best evaluation of the computation graph.
    pub fn eval(&self) {
        self.net.borrow_mut().eval();
    }
    
    /// 
    pub fn forward(&self) { 
        self.net.borrow_mut().eval();
    }

    /// Back propagation
    pub fn grad(&self, og: &[Tensor]) -> Result<u32, &'static str> {
	Ok(0)
    }

    /// Back propagation
    pub fn backward(&self, og: &[Tensor]) -> Result<u32, &'static str> {
	Ok(0)
    }
}

macro_rules! var_op_method {
    ($a:ident) => {
        pub fn $a(&self, o: &Var) -> Var {
            let result = self.new_attached();
            self.net
                .borrow_mut()
                .connect(&vec![self.id, o.id], Box::new($a::new()), &vec![result.id]);
            result
        }
    }
    
}

/// Introduce variable to the system by creating Var
pub struct Var {
    id: NetIndex,
    net: Rc<RefCell<Net>>,
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

    pub fn _id(&self) -> &NetIndex {
        &self.id
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
            .data
            .replace(&self.id, v);

        self.net.borrow_mut().set_mark(&self.id);
    }
    pub fn get(&self) -> Tensor {
        self.net.borrow().data.get(&self.id).expect("").clone()
    }

    // Convient method definition.
    var_op_method!(add);
    var_op_method!(sub);
    var_op_method!(mul);
    var_op_method!(div);
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}, {})",
            self.id,
            self.net.borrow().data.get(&self.id).expect("")
        )
    }
}

pub fn MSELoss(a: &Var, b: &Var) -> Var {
    let result = a.new_attached();
    a.net.borrow_mut().connect(&vec![a.id, b.id], Box::new(MSELoss::new()), &vec![result.id]);
    result
}



/// The computation network.
/// Connection has duplication.
struct Net {
    data: GenIndex<Tensor>,
    ops: GenIndex<RefCell<Box<dyn Op>>>,
    forward_data2op: GenIndex<Vec<NetIndex>>,
    forward_op2data: GenIndex<Vec<NetIndex>>,
    backward_data2op: GenIndex<Vec<NetIndex>>,
    backward_op2data: GenIndex<Vec<NetIndex>>,
    // the NetIndex of var which have been set input value.
    set_mark: BTreeSet<NetIndex>,
    // cache of output nodes
    cache_output: BTreeSet<NetIndex>,
    cache_grad: GenIndex<Rc<RefCell<Tensor>>>,
}

impl Net {
    fn new() -> Net {
        Net {
            data: GenIndex::new(),
            ops: GenIndex::new(),
            forward_data2op: GenIndex::new(),
            forward_op2data: GenIndex::new(),
            backward_data2op: GenIndex::new(),
            backward_op2data: GenIndex::new(),
            set_mark: BTreeSet::new(),
            cache_output: BTreeSet::new(),
            cache_grad: GenIndex::new(),
        }
    }

    /// Insert an empty var into the network.
    fn init_var(&mut self, var: &mut Var) {
        let id = self.data.insert(Tensor::new());
        let fid = self.forward_data2op.insert(Vec::new());
        let bid = self.backward_data2op.insert(Vec::new());
        assert!(id == fid);
        assert!(id == bid);
        var.id = id;
    }

    fn del_var(&mut self, var: &NetIndex) {}

    /// Insert operator into the network.
    fn init_op(&mut self, op: Box<dyn Op>) -> NetIndex {
        let id = self.ops.insert(RefCell::new(op));
        let fid = self.forward_op2data.insert(Vec::new());
        let bid = self.backward_op2data.insert(Vec::new());
        assert!(id == fid);
        assert!(id == bid);
        id
    }

    /// Build input-operator-output relation, with given components.
    fn connect(&mut self, input: &[NetIndex], op: Box<dyn Op>, output: &Vec<NetIndex>) {
        let opid = self.init_op(op);
        for val in input {
            self.backward_op2data.get_mut(&opid).expect("").push(*val);
            self.forward_data2op.get_mut(val).expect("").push(opid);
        }
        for val in output {
            self.forward_op2data.get_mut(&opid).expect("").push(*val);
            self.backward_data2op.get_mut(val).expect("").push(opid);
        }
    }

    /// set the set_mark, set_mark is used to label var with input value with it.
    fn set_mark(&mut self, did: &NetIndex) {
        self.set_mark.insert(*did);
    }
    fn unset_mark(&mut self, did: &NetIndex) {
        self.set_mark.remove(did);
    }

    /// Merge
    fn merge(&self, o: &Net) -> Net {
        Net::new()
    }

    /// Forward evaluate the computaiton graph.
    fn eval(&mut self) {
        // vars has a value and
        let mut jobs = BTreeSet::<NetIndex>::new();
        let mut done = BTreeSet::<NetIndex>::new(); // ops done.

        for index in self.set_mark.iter() {
            jobs.insert(*index);
        }

        while jobs.len() > 0 {
            let job = jobs.iter().next().expect("").clone();
            // println!("current job: {}", job);

            let undone_ops: Vec<&NetIndex> = self
                .forward_data2op
                .get(&job)
                .expect("")
                .iter()
                .filter(|op| !done.contains(op))
                .collect();

            if undone_ops.len() <= 0 {
                jobs.remove(&job);
            } else {
                for op in undone_ops {
                    if self
                        .backward_op2data
                        .get(op)
                        .expect("")
                        .iter()
                        .all(|dt| jobs.contains(dt))
                    {
                        // do real stuff
                        let mut inputs: Vec<&Tensor> = Vec::new();
                        for input in self.backward_op2data.get(op).expect("").iter() {
                            let a = self.data.get(input).expect("");
                            inputs.push(a);
                        }

                        let mut outputs: Vec<&Tensor> = Vec::new();
                        for output in self.forward_op2data.get(op).expect("").iter() {
                            let a = self.data.get(output).expect("");
                            outputs.push(a);
                        }

                        self.ops
                            .get_mut(op)
                            .expect("")
                            .borrow_mut()
                            .apply(&mut inputs, &mut outputs);

                        for output in self.forward_op2data.get(op).expect("").iter() {
                            jobs.insert(*output);
                        }
                        done.insert(*op);
                    }
                }
            }
        }
    }

    /// build output node cache
    pub fn build_output_cache(&mut self) {
        
    }
}
