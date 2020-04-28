use std::cell::RefCell;
use std::collections::{BTreeSet, BTreeMap};
use std::fmt;
use std::rc::Rc;


use super::collection::generational_index::*;
use super::collection::graph::Graph;
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
        self.net.borrow_mut().eval().expect("");
    }
    
    /// 
    pub fn forward(&self) { 
        self.net.borrow_mut().eval().expect("");
    }

    /// Back propagation
    pub fn backward_vector(&self, og: &BTreeMap<NetIndex, Tensor>) {
	self.net.borrow_mut().bptt(og);
    }

    pub fn backward(&self, og: f32) {
        
	self.net.borrow_mut().bptt_scale(og);
    }
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
            .replace(&self.id, v).expect("");

        self.net.borrow_mut().set_mark(&self.id);
    }

    /// Get the underlying tensor.
    pub fn get(&self) -> Tensor {
        self.net.borrow().data.get(&self.id).expect("").clone()
    }

    /// Get the underlying gradient tensor.
    pub fn get_grad(&self) -> Tensor {
        self.net.borrow().data_grad.get(&self.id).expect("").clone()
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
            self.net.borrow().data.get(&self.id).expect("")
        )
    }
}

// uplift loss function from op to here.
pub fn mseloss(a: &Var, b: &Var) -> Var {
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
    data_grad: BTreeMap<NetIndex, Tensor>,
}

impl Net {
    fn new() -> Net {
        Net {
            data: GenIndex::new(),
            ops: GenIndex::new(),
            set_mark: BTreeSet::new(),
            graph: Graph::new(),
            data_grad: BTreeMap::new(),
        }
    }

    /// Insert an empty var into the network.
    fn init_var(&mut self, var: &mut Var) {
        let id = self.data.insert(Tensor::new());
        self.graph.add_data(&id).expect("");
        var.id = id;
    }

    fn del_var(&mut self, var: &Var) {
        self.data.remove(&var.id).expect("");
        self.graph.del_data(&var.id).expect("");
    }

    /// Insert operator into the network.
    fn init_op(&mut self, op: Op) -> NetIndex {
        let id = self.ops.insert(op.clone());
        self.graph.add_op(&id).expect("");
        id
    }

    /// Build input-operator-output relation, with given components.
    fn connect(&mut self, input: &[NetIndex], op: Op, output: &[NetIndex]) {
        let opid = self.init_op(op);
        self.graph.connect(input, output, &opid).expect("");
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
                    //println!("op: {}", self.ops.get(op).expect("").get_name());
                    
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
                    
                    //println!("var.rs: {:?}", outputs[0].size());
                    
                }
            )?;

        Ok(())
    }

    fn bptt_scale(&mut self, r: f32) {
        let output = self.graph.get_output_cache();
        let mut output_grad = BTreeMap::new();
        for i in &output {
            output_grad.insert(i.clone(),
                               Tensor::fill(&self.data.get(i).expect("").size(),
                                            r));
        }
        self.bptt(&output_grad);
    }

    fn bptt(&mut self, output_grad: &BTreeMap<NetIndex, Tensor>) {
        let mut output = Vec::new();
        self.data_grad.clear();
        for (k, v) in output_grad {
            output.push(k.clone());
            self.data_grad.insert(k.clone(), v.clone());
        }

        for i in self.graph.list_data() {
            self.data_grad.insert(i, Tensor::new());
        }
        
        self.graph
            .walk(
                &output[..],
                false,
                |output_grads, input_grads, op| {
                    //println!("op: {}", self.ops.get(op).expect("").get_name());

                    // collect input tensor.
                    let mut inputs: Vec<&Tensor> = Vec::new();
                    for input_id in input_grads {
                        let a = self.data.get(input_id).expect("");
                        inputs.push(a);
                    }

                    // collect the output tensor ready (forward view).
                    let mut output_grad: Vec<&Tensor> = Vec::new();
                    for output_id in output_grads {
                        let a = self.data_grad.get(output_id).expect("");
                        output_grad.push(a);
                    }
                    // collect the input tensor ready (forward view).
                    let mut input_grad: Vec<&Tensor> = Vec::new();
                    for input_id in input_grads {
                        let a = self.data_grad.get(input_id).expect("");
                        input_grad.push(a);
                    }

                    self.ops
                        .get(op)
                        .expect("")
                        .grad(&inputs, &output_grad, &input_grad);
                    
                    //println!("var.rs: {:?}", 1);
                    
                }
            ).expect("");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genindex_new_add_del() {
    }
}
