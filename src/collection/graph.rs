use std::collections::{BTreeMap, BTreeSet};
use super::generational_index::*;


pub struct Graph {
    data: BTreeSet<NetIndex>,
    op: BTreeSet<NetIndex>,
    forward_dt_op: BTreeMap<NetIndex, BTreeSet<NetIndex>>,
    forward_op_dt: BTreeMap<NetIndex, BTreeSet<NetIndex>>,
    backward_dt_op: BTreeMap<NetIndex, BTreeSet<NetIndex>>,
    backward_op_dt: BTreeMap<NetIndex, BTreeSet<NetIndex>>,
}
impl Graph {
    /// Create a graph with defaults
    pub fn new() -> Graph {
        Graph{
            data: BTreeSet::new(),
            op: BTreeSet::new(),
            forward_dt_op: BTreeMap::new(),
            forward_op_dt: BTreeMap::new(),
            backward_dt_op: BTreeMap::new(),
            backward_op_dt: BTreeMap::new(),
        }
    }
    pub fn add_data(&mut self, id: &NetIndex) -> Result<NetIndex, &str> {
        if !self.data.contains(id) {
            self.data.insert(*id);
            self.forward_dt_op.insert(id.clone(), BTreeSet::new());
            self.backward_dt_op.insert(id.clone(), BTreeSet::new());
            Ok(id.clone())            
        } else {
            Err("data is exits!")
        }
    }
    pub fn del_data(&mut self, id: &NetIndex) -> Result<NetIndex, &str> {
        if self.data.contains(id) {
            self.data.remove(id);
            for i in self.forward_dt_op.get_mut(id).expect("").iter() {
                self.backward_op_dt.get_mut(i).expect("").remove(id);
            }
            self.forward_dt_op.remove(id);
            for i in self.backward_dt_op.get_mut(id).expect("").iter() {
                self.forward_op_dt.get_mut(i).expect("").remove(id);
            }
            self.backward_dt_op.remove(id);

            Ok(id.clone())
        } else {
            Err("data id is not found!")
        }
    }
    
    pub fn add_op(&mut self, id: &NetIndex) -> Result<NetIndex, &str> {
        if !self.op.contains(id) {
            self.op.insert(*id);
            self.forward_op_dt.insert(id.clone(), BTreeSet::new());
            self.backward_op_dt.insert(id.clone(), BTreeSet::new());
            Ok(id.clone())
        } else {
            Err("op id exists.")
        }
    }
    pub fn del_op(&mut self, id: &NetIndex) -> Result<NetIndex, &str> {
        if self.op.contains(id) {
            self.op.remove(id);
            for i in self.forward_op_dt.get_mut(id).expect("").iter() {
                self.backward_dt_op.get_mut(i).expect("").remove(id);
            }
            self.forward_op_dt.remove(id);
            for i in self.backward_op_dt.get_mut(id).expect("").iter() {
                self.forward_dt_op.get_mut(i).expect("").remove(id);
            }
            self.backward_op_dt.remove(id);
            Ok(id.clone())
        } else {
            Err("op id is not found!")
        }

    }
    
    pub fn connect(&mut self, dti: &Vec<NetIndex>,
                   dto: &Vec<NetIndex>,
                   op: &NetIndex) -> Result<NetIndex, &str> {
        let mut valid_ids = true;

        if !self.op.contains(op) {
            valid_ids = false;
        }
        for i in dti {
            if !self.data.contains(i) {
                valid_ids = false;
            }
        }
        for i in dto {
            if !self.data.contains(i) {
                valid_ids = false;
            }
        }
        
        if valid_ids {
            for i in dti {
                self.forward_dt_op.get_mut(i).expect("").insert(op.clone());
                self.backward_op_dt.get_mut(op).expect("").insert(i.clone());
            }
            for i in dto {
                self.forward_op_dt.get_mut(op).expect("").insert(i.clone());
                self.backward_dt_op.get_mut(i).expect("").insert(op.clone());
            }
            Ok(op.clone())
        } else {
            Err("Invalid id!")
        }
    }

    pub fn walk(&mut self, start_set: &Vec<NetIndex>,
                forward: bool,
                closure: &Fn(&Vec<NetIndex>, &Vec<NetIndex>, &NetIndex)) {
        let mut fdo = &self.forward_dt_op;
        let mut fod = &self.forward_op_dt;
        let mut bdo = &self.backward_dt_op;
        let mut bod = &self.backward_op_dt;
        if !forward {
            fdo = &self.backward_dt_op;
            fod = &self.backward_op_dt;
            bdo = &self.forward_dt_op;
            bod = &self.forward_op_dt;
        }

        // data id has a value
        let mut jobs = BTreeSet::<NetIndex>::new();
        // op is done.
        let mut done = BTreeSet::<NetIndex>::new(); // ops done.

        for index in start_set {
            jobs.insert(*index);
        }

        while jobs.len() > 0 {
            let job = jobs.iter().next().expect("").clone();

            let undone_ops: Vec<&NetIndex> = fdo
                .get(&job)
                .expect("")
                .iter()
                .filter(|op| !done.contains(op))
                .collect();

            if undone_ops.len() <= 0 {
                jobs.remove(&job);
            } else {
                for op in undone_ops {
                    if bod
                        .get(op)
                        .expect("")
                        .iter()
                        .all(|dt| jobs.contains(dt)) {
                            // do real stuff
                            let mut inputs: Vec<NetIndex> = Vec::new();
                            for input in bod.get(op).expect("").iter() {
                                inputs.push(input.clone());
                            }

                            let mut outputs: Vec<NetIndex> = Vec::new();
                            for output in fod.get(op).expect("").iter() {
                                outputs.push(output.clone());
                            }

                            closure(&inputs, &outputs, &op);

                            for output in fod.get(op).expect("").iter() {
                                jobs.insert(*output);
                            }
                            done.insert(*op);
                        }
                }
            }
        }
    }
}
