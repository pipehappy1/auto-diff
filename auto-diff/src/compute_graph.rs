#![allow(clippy::redundant_closure)]
use std::collections::{BTreeSet, BTreeMap};
use std::fmt;

use crate::collection::generational_index::{GenIndex, GenKey};
use crate::collection::graph::Graph;
use tensor_rs::tensor::Tensor;
use crate::op::Op;
use crate::err::AutoDiffError;

/// The computation network.
/// Connection has duplication.
pub struct Net {
    data: GenIndex<Tensor>,
    ops: GenIndex<Op>,
    set_mark: BTreeSet<GenKey>,
    graph: Graph<GenKey, GenKey>,
    data_grad: BTreeMap<GenKey, Tensor>,
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
    pub fn get_ops(&self) -> &GenIndex<Op> {
        &self.ops
    }
    pub fn get_ops_mut(&mut self) -> &mut GenIndex<Op> {
        &mut self.ops
    }

    pub fn add_tensor(&mut self, t: Tensor) -> GenKey {
        let id = self.data.insert(t);
        self.graph.add_data(&id).expect("");
        id
    }

    pub fn get_tensor(&self, id: GenKey) -> Result<Tensor, AutoDiffError> {
        match self.data.get(&id) {
            Ok(v) => {Ok(v.ref_copy())}, // shallow copy a tensor.
            Err(v) => {Err(v)}
        }
    }
    pub fn set_tensor(&mut self, id: GenKey, val: Tensor) -> Result<(), ()> {
        self.data.replace(&id, val)?;
        Ok(())
    }

    /// Insert operator into the network.
    pub fn add_op(&mut self, op: Op) -> GenKey {
        let id = self.ops.insert(op);
        self.graph.add_op(&id).expect("");
        id
    }
    pub fn get_op(&self, id: GenKey) -> Result<Op, AutoDiffError> {
        Ok(self.ops.get(&id)?.ref_copy())
    }

    pub fn get_grad(&self, id: GenKey) -> Result<Tensor, AutoDiffError> {
        match self.data_grad.get(&id) {
            Some(v) => {Ok(v.ref_copy())},
            None => {Err(AutoDiffError::new(&format!("Data {:?} doesn't ahave gradient yet.", id)))}
        }
    }
    


//    pub fn is_dangling_var(&self, var: &Var) -> Result<bool, ()> {
//        if !self.data.contains(var.get_id()) {
//            Err(())
//        } else if self.graph.iter_op_given_input(var.get_id()).expect("").count() == 0 &&
//            self.graph.iter_op_given_output(var.get_id()).expect("").count() == 0{
//                Ok(true)
//            } else {
//                Ok(false)
//            }
//
//    }


//    ///
//    /// Remove a concrete op or composed func from the graph.
//    ///
//    pub fn del_func_or_op(&mut self, func: &Func) {
//        let _ = self.ops.remove(func.get_id());
//        let _ = self.graph.drop_op(func.get_id());
//
//        // ignore the result as to allow duplicate delete
//
//        //
//        // The following dosen't work 
//        // because if the composed goes out of scope doesn't mean
//        //     its member ops go out of scope.
//        //
//        // Check to see the func type.
//        // If it is a op, delete it
//        // If it is a func, find all the underlying op
//        //     and var in between and remove them.
//        //
//
//    }

//    ///
//    /// Disconnect the variable and the function the variable is the input.
//    /// Delete the variable if it becomes the dangling variable.
//    ///
//    pub fn decouple_input(&mut self, func: &Func) -> Vec<GenKey> {
//        let mut decoupled_inputs = Vec::new();
//        let inputs: Vec<GenKey> = self.graph.iter_input_given_op(func.get_id())
//            .expect("").map(|x| x.clone()).collect();
//        for i in inputs {
//            self.graph.decouple_data_func(&i, func.get_id()).expect("");
//            decoupled_inputs.push(i);
//        }
//        decoupled_inputs
//    }


    ///
    /// Build input-operator-output relation, with given components.
    ///
    pub fn connect(&mut self, input: &[GenKey], op: GenKey, output: &[GenKey]) {

        self.graph.connect(input, output, &op).expect("");
    }


    /// set the set_mark, set_mark is used to label var with input value with it.
    pub fn set_mark(&mut self, did: &GenKey) {
        self.set_mark.insert(*did);
    }
    pub fn unset_mark(&mut self, did: &GenKey) {
        self.set_mark.remove(did);
    }

    /// Forward evaluate the computaiton graph.
    /// This is for static graph run.
    pub fn eval(&mut self) -> Result<(), BTreeSet<GenKey>> {
        println!("Deprecated! no more whole network forward pass.");
        let mut all_input = Vec::new();
        for i in &self.set_mark {
            all_input.push(*i);
        }
        
        self.graph
            .walk(
                &all_input[..],
                true,
                |input, output, op| {
                    //println!("op: {}", self.ops.get(op).expect("").get_name());
                    
                    let mut inputs: Vec<Tensor> = Vec::new();
                    for input_id in input {
                        let a = self.data.get(input_id).expect("").ref_copy();
                        inputs.push(a);
                    }

                    let mut outputs: Vec<Tensor> = Vec::new();
                    for output_id in output {
                        let a = self.data.get(output_id).expect("").ref_copy();
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

//    pub fn eval_op(&self, input: &[&Var], func: &Func, output: &[&Var]) {
//        let mut inputs: Vec<&Tensor> = Vec::new();
//        for input_var in input {
//            let a = self.data.get(input_var.get_id()).expect("");
//            inputs.push(a);
//        }
//
//        let mut outputs: Vec<&Tensor> = Vec::new();
//        for output_var in output {
//            let a = self.data.get(output_var.get_id()).expect("");
//            outputs.push(a);
//        }
//
//        self.ops
//            .get(func.get_id())
//            .expect("")
//            .apply(&inputs, &outputs);
//    }

    pub fn bptt_scale(&mut self, r: f32) {
        let output = self.graph.get_output_edge_data();
        let mut output_grad = BTreeMap::new();
        for i in &output {
            output_grad.insert(*i,
                               Tensor::fill(&self.data.get(i).expect("").size(),
                                            r));
        }
        self.bptt(&output_grad);
    }

    pub fn bptt(&mut self, output_grad: &BTreeMap<GenKey, Tensor>) {
        let mut output = Vec::new();
        self.data_grad.clear();
        for (k, v) in output_grad {
            output.push(*k);
            self.data_grad.insert(*k, v.clone());
        }

        for i in self.graph.iter_data() {
            self.data_grad.entry(*i).or_insert(Tensor::new());
        }

        self.graph
            .walk(
                &output[..],
                false,
                |output_grads, input_grads, op| {
                    //println!("op, bptt: {}", self.ops.get(op).expect("").get_name());

                    // collect input tensor.
                    let mut inputs: Vec<Tensor> = Vec::new();
                    for input_id in input_grads {
                        //println!("bptt {:?}", input_id);
                        let a = self.data.get(input_id).expect("").ref_copy();
                        inputs.push(a);
                    }
                    //println!("input: size {:?}", inputs.len());

                    // collect the output tensor gradient (forward view).
                    let mut output_grad: Vec<Tensor> = Vec::new();
                    for output_id in output_grads {
                        //println!("bptt 2 {:?}", output_id);
                        let a = self.data_grad.get(output_id).expect("").ref_copy();
                        output_grad.push(a);
                    }
                    //println!("output grad: size {:?}", output_grad.len());
                    
                    // collect the input tensor gradient (forward view).
                    let mut input_grad: Vec<Tensor> = Vec::new();
                    for input_id in input_grads {
                        //println!("bptt 3 {:?}", input_id);
                        let a = self.data_grad.get(input_id).expect("").ref_copy();
                        input_grad.push(a);
                    }
                    //println!("input grad: size {:?}", input_grad.len());

                    self.ops
                        .get(op)
                        .expect("")
                        .grad(&inputs, &output_grad, &input_grad);
                    
                    //println!("var.rs: {:?}", 1);
                    
                }
            ).expect("");
    }

    /// Iterate over all ops, no order guarantee
    pub fn visit_op<F>(&mut self, closure: F,
                       allow: Option<Vec<GenKey>>,
                       skip: Option<Vec<GenKey>>)
    where F: Fn(&Op) {
        let allow_list = if let Some(val) = allow { val } else {Vec::new()};
        let skip_list = if let Some(val) = skip {val} else {Vec::new()};
        
        for i in self.graph.iter_op() {
            if (allow_list.is_empty() && skip_list.is_empty()) ||
                (!allow_list.is_empty() && allow_list.contains(i)) ||
                (!skip_list.is_empty() && !skip_list.contains(i) ) {
                    closure(self.ops.get(i).expect(""));
            }
        }
    }

    pub fn visit_data<F>(&mut self, closure: F)
    where F: Fn(GenKey, &Tensor) {
        for i in self.graph.iter_data() {
            closure(*i, self.data.get(i).expect(""));
        }
    }

    /// Move content in other network into self.
    /// Return new ids for those have origianl_keys in the old network.
    pub fn append(&mut self, other: &Self,
                  original_keys: &[GenKey]) -> Result<Vec<GenKey>, AutoDiffError> {

        let mut data_key_map = BTreeMap::new();
        let mut ret_keys = Vec::new();
        for key in other.get_data().iter_key() {
            let new_key = self.add_tensor(other.get_tensor(key)?);
            if original_keys.contains(&key) {
                ret_keys.push(new_key);
            }
            data_key_map.insert(key, new_key);
        }
        
        let mut op_key_map = BTreeMap::new();
        for key in other.get_ops().iter_key() {
            let new_key = self.add_op(other.get_op(key)?);
            op_key_map.insert(key, new_key);
        }

        self.graph.append(&other.graph, data_key_map, op_key_map)?;

        Ok(ret_keys)
    }
}

impl fmt::Debug for Net {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Dumping Net")?;
        for i in self.data.iter_key() {
            write!(f, "id: {:?} data: {:?}", i, self.data.get(&i))?;
        }
        writeln!(f, "data_grad")?;
        for (k, v) in self.data_grad.iter() {
            write!(f, "id: {:?} data: {:?}", k, v)?;
        }
        write!(f, "graph: {:?}", self.graph)
    }
}

