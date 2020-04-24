use std::cell::RefCell;
use std::collections::BTreeSet;
use std::fmt;
use std::rc::Rc;

use rand::prelude::*;
use rand_distr::{Normal, Distribution, LogNormal};

use super::collection::generational_index::*;
use super::collection::graph::Graph;
use super::tensor::Tensor;
use super::op::*;


pub struct Module {
    net: Rc<RefCell<Net>>,
    rng: StdRng,
}

/// Network holder.
impl Module {
    pub fn new() -> Module {
        Module {
            net: Rc::new(RefCell::new(Net::new())),
            rng: StdRng::seed_from_u64(671),
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
    pub fn backward(&self, og: &[Tensor]) -> Result<u32, &'static str> {
	Ok(0)
    }

    pub fn backward_scale(&self, og: f32) -> Result<u32, &'static str> {
	Ok(0)
    }


    // random init
    
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }
    
    pub fn bernoulli() {}
    pub fn cauchy() {}
    pub fn exponential() {}
    pub fn geometric() {}
    pub fn log_normal() {}
    
    pub fn normal(&mut self, dim: &[usize], mean: f32, std: f32) -> Tensor {
        let mut elem = 1;
        for i in dim {
            elem *= i;
        }
        let mut dta = Vec::<f32>::with_capacity(elem);
        let normal = Normal::new(mean, std).expect("");
        for i in 0..elem {
            dta.push(normal.sample(&mut self.rng));
        }
        Tensor::from_vec_f32(&dta, dim)
    }
    
    //pub fn random() {}
    
    pub fn uniform<F>(dim: &[usize], from: F, to: F) -> Tensor
    where F: num_traits::Float {
        Tensor::new()
    }
}

macro_rules! var_op_method {
    ($a:ident) => {
        pub fn $a(&self, o: &Var) -> Var {
            let result = self.new_attached();
            self.net
                .borrow_mut()
                .connect(&vec![self.id, o.id], Op::new(Box::new($a::new())), &vec![result.id]);
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

    /// Get the underlying tensor.
    pub fn get(&self) -> Tensor {
        self.net.borrow().data.get(&self.id).expect("").clone()
    }

    /// apply the var to pre-faburacated op.
    pub fn to(&self, op: &Op) -> Var {
        let result = self.new_attached();
        self.net.borrow_mut().connect(&vec![self.id], op.clone(), &vec![result.id]);
        result
    }

    // uplift method from Tensor to Var
    pub fn size(&self) -> Vec<usize> {
        self.net.borrow().data.get(&self.id).expect("").size()
    }
    pub fn numel(&self) -> usize {
        self.net.borrow().data.get(&self.id).expect("").numel()
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

// uplift loss function from op to here.
pub fn MSELoss(a: &Var, b: &Var) -> Var {
    let result = a.new_attached();
    a.net.borrow_mut().connect(&vec![a.id, b.id], Op::new(Box::new(MSELoss::new())), &vec![result.id]);
    result
}



/// The computation network.
/// Connection has duplication.
struct Net {
    data: GenIndex<Tensor>,
    ops: GenIndex<Op>,
    set_mark: BTreeSet<NetIndex>,
    graph: Graph,
}

impl Net {
    fn new() -> Net {
        Net {
            data: GenIndex::new(),
            ops: GenIndex::new(),
            set_mark: BTreeSet::new(),
            graph: Graph::new(),
        }
    }

    /// Insert an empty var into the network.
    fn init_var(&mut self, var: &mut Var) {
        let id = self.data.insert(Tensor::new());
        self.graph.add_data(&id);
        var.id = id;
    }

    fn del_var(&mut self, var: &NetIndex) {}

    /// Insert operator into the network.
    fn init_op(&mut self, op: Op) -> NetIndex {
        let id = self.ops.insert(op.clone());
        self.graph.add_op(&id);
        id
    }

    /// Build input-operator-output relation, with given components.
    fn connect(&mut self, input: &[NetIndex], op: Op, output: &[NetIndex]) {
        let opid = self.init_op(op);
        self.graph.connect(input, output, &opid);
    }

    /// set the set_mark, set_mark is used to label var with input value with it.
    fn set_mark(&mut self, did: &NetIndex) {
        self.set_mark.insert(*did);
    }
    fn unset_mark(&mut self, did: &NetIndex) {
        self.set_mark.remove(did);
    }

    /// Merge two computation graph
    fn merge(&self, o: &Net) -> Net {
        Net::new()
    }

    /// Forward evaluate the computaiton graph.
    fn eval(&mut self) -> Result<(), BTreeSet<NetIndex>> {
        let mut all_input = Vec::new();
        for i in &self.set_mark {
            all_input.push(i.clone());
        }
        
        self.graph
            .walk(
                &all_input[..],
                true,
                |input, output, op| {
                    println!("op: {}", self.ops.get(op).expect("").get_name());
                    
                    let mut inputs: Vec<&Tensor> = Vec::new();
                    for input_id in input {
                        let a = self.data.get(input_id).expect("");
                        inputs.push(a);
                    }

                    let mut outputs: Vec<&Tensor> = Vec::new();
                    for output_id in output {
                        let a = self.data.get(output_id).expect("");
                        outputs.push(a);
                    }

                    self.ops
                        .get(op)
                        .expect("")
                        .apply(&inputs, &outputs);
                    
                    println!("var.rs: {:?}", outputs[0].size());
                    
                }
            )?;

        Ok(())
    }

}
