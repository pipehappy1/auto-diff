use std::collections::{BTreeSet, BTreeMap};

use crate::collection::generational_index::*;
use crate::collection::graph::Graph;
use crate::tensor::Tensor;
use crate::op::*;
use crate::var::*;

/// The computation network.
/// Connection has duplication.
pub struct Net {
    data: GenIndex<Tensor>,
    ops: GenIndex<Op>,
    set_mark: BTreeSet<NetIndex>,
    graph: Graph,
    data_grad: BTreeMap<NetIndex, Tensor>,
}

impl Net {
    pub fn new() -> Net {
        Net {
            data: GenIndex::new(),
            ops: GenIndex::new(),
            set_mark: BTreeSet::new(),
            graph: Graph::new(),
            data_grad: BTreeMap::new(),
        }
    }

    pub fn get_data(&self) -> &GenIndex<Tensor> {
        &self.data
    }

    pub fn get_data_mut(&mut self) -> &mut GenIndex<Tensor> {
        &mut self.data
    }

    pub fn get_grad(&self)  -> &BTreeMap<NetIndex, Tensor> {
        &self.data_grad
    }

    /// Insert an empty var into the network.
    pub fn init_var(&mut self, var: &mut Var) {
        let id = self.data.insert(Tensor::new());
        self.graph.add_data(&id).expect("");
        var.set_id(id);
    }

    pub fn del_var(&mut self, var: &Var) {
        self.data.remove(var.get_id()).expect("");
        self.graph.del_data(var.get_id()).expect("");
    }

    /// Insert operator into the network.
    pub fn init_op(&mut self, op: Op) -> NetIndex {
        let id = self.ops.insert(op.clone());
        self.graph.add_op(&id).expect("");
        id
    }

    /// Build input-operator-output relation, with given components.
    pub fn connect(&mut self, input: &[NetIndex], op: Op, output: &[NetIndex]) {
        let opid = self.init_op(op);
        self.graph.connect(input, output, &opid).expect("");
    }

    /// set the set_mark, set_mark is used to label var with input value with it.
    pub fn set_mark(&mut self, did: &NetIndex) {
        self.set_mark.insert(*did);
    }
    pub fn unset_mark(&mut self, did: &NetIndex) {
        self.set_mark.remove(did);
    }

    /// Merge two computation graph
    //fn merge(&self, o: &Net) -> Net {
    //    Net::new()
    //}

    /// Forward evaluate the computaiton graph.
    pub fn eval(&mut self) -> Result<(), BTreeSet<NetIndex>> {
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

    pub fn bptt_scale(&mut self, r: f32) {
        let output = self.graph.get_output_cache();
        let mut output_grad = BTreeMap::new();
        for i in &output {
            output_grad.insert(i.clone(),
                               Tensor::fill(&self.data.get(i).expect("").size(),
                                            r));
        }
        self.bptt(&output_grad);
    }

    pub fn bptt(&mut self, output_grad: &BTreeMap<NetIndex, Tensor>) {
        let mut output = Vec::new();
        self.data_grad.clear();
        for (k, v) in output_grad {
            output.push(k.clone());
            self.data_grad.insert(k.clone(), v.clone());
        }

        for i in self.graph.list_data() {
            if ! self.data_grad.contains_key(&i) {
                self.data_grad.insert(i, Tensor::new());                
            }
        }
        
        self.graph
            .walk(
                &output[..],
                false,
                |output_grads, input_grads, op| {
                    // println!("op, bptt: {}", self.ops.get(op).expect("").get_name());

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

    /// Iterate over all ops, no order guarantee
    pub fn visit_op<F>(&mut self, closure: F)
    where F: Fn(&Op) {
        for i in self.graph.list_op() {
            closure(self.ops.get(&i).expect(""));
        }
    }

    pub fn visit_data<F>(&mut self, closure: F)
    where F: Fn(&Op) {
        for i in self.graph.list_data() {
            closure(self.ops.get(&i).expect(""));
        }
    }
}

