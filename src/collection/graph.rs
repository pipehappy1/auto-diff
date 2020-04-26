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

    /// Add a data node.
    /// ```
    /// # use auto_diff::collection::graph::*;
    /// # use auto_diff::collection::generational_index::*;
    /// let mut g = Graph::new();
    /// let data1 = NetIndex::new(0,0);
    /// let data2 = NetIndex::new(1,0);
    /// g.add_data(&data1);
    /// g.add_data(&data2);
    /// ```
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

    /// Remove a data node, op node and downstream data/op node are removed.
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

    /// Add a danglging op node.
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

    /// Remvoe an op node, input data node and downstream data/op node are removed.
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

    /// list data node without upstream op node in a set.
    pub fn get_input_cache(&self) -> BTreeSet<NetIndex> {
        let mut jobs = BTreeSet::<NetIndex>::new();
        for i in &self.data {
            if self.backward_dt_op.get(i).expect("").len() <= 0 {
                jobs.insert(i.clone());
            }
        }
        jobs
    }

    /// list data node without downstream op node in a set.
    pub fn get_output_cache(&self) -> BTreeSet<NetIndex> {
        let mut jobs = BTreeSet::<NetIndex>::new();
        for i in &self.data {
            if self.forward_dt_op.get(i).expect("").len() <= 0 {
                jobs.insert(i.clone());
            }
        }
        jobs
    }

    /// Connect input data, output data and operation
    pub fn connect(&mut self, dti: &[NetIndex],
                   dto: &[NetIndex],
                   op: &NetIndex) -> Result<NetIndex, &str> {
        let mut valid_ids = true;

        // make sure pre-exist
        if !self.op.contains(op) {
            valid_ids = false;
        }
        // make sure input data pre-exist
        for i in dti {
            if !self.data.contains(i) {
                valid_ids = false;
            }
        }
        // make sure output data pre-exist
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

    /// Walk through the graph with a starting set of data nodes.
    /// Go through backwards if forward is false.
    pub fn walk<F>(&self, start_set: &[NetIndex],
                   forward: bool,
                   closure: F) -> Result<(), BTreeSet<NetIndex>>
    where F: Fn(&[NetIndex], &[NetIndex], &NetIndex)  {
        let mut fdo = &self.forward_dt_op;
        let mut fod = &self.forward_op_dt;
        //let mut bdo = &self.backward_dt_op;
        let mut bod = &self.backward_op_dt;
        if !forward {
            fdo = &self.backward_dt_op;
            fod = &self.backward_op_dt;
            //bdo = &self.forward_dt_op;
            bod = &self.forward_op_dt;
        }

        // data id has a value
        let mut jobs = BTreeSet::<NetIndex>::new();
        // op is done.
        let mut done = BTreeSet::<NetIndex>::new(); // ops done.

        for index in start_set {
            jobs.insert(*index);
        }
        
        loop {
            let mut made_progress = false;

            // collect ops needs to do given the data in jobs.
            let mut edge_op = BTreeSet::<NetIndex>::new();
            for dt in &jobs {
                for op_candidate in &fdo[dt] {
                    edge_op.insert(op_candidate.clone());
                }
            }

            // process op if possible
            for op_candidate in edge_op {
                if bod[&op_candidate]
                    .iter()
                    .all(|dt| jobs.contains(dt)) {

                        // collect input ids.
                        let mut inputs = Vec::<NetIndex>::new();
                        for input in bod[&op_candidate].iter() {
                            inputs.push(input.clone());
                        }
                        // collect output ids.
                        let mut outputs = Vec::<NetIndex>::new();
                        for output in fod[&op_candidate].iter() {
                            outputs.push(output.clone());
                        }

                        // all the closure
                        closure(&inputs, &outputs, &op_candidate);

                        // maintain the list
                        // the following line should go before the rest.
                        done.insert(op_candidate);
                        // remove the data from jobs if all its downstream op is done.
                        for input in bod[&op_candidate].iter() {
                            if fdo[input]
                                .iter()
                                .all(|op| done.contains(op)) {
                                    jobs.remove(input);
                                }
                        }
                        // add the output back to the jobs.
                        for output in fod[&op_candidate].iter() {
                            // don't add to jobs if it's the final data node.
                            if fdo[output].len() > 0 {
                                jobs.insert(*output);                                
                            }
                        }

                        // flag there is sth done.
                        made_progress = true;
                    }
            }

            if ! made_progress {
                break;
            }
        }

        if jobs.len() > 0 {
            Err(jobs)
        } else {
            Ok(())
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn new() {
        let _g = Graph::new();
    }

    // A   B
    //  \ /
    //   Op
    //   |
    //   C
    fn setup_y(g: &mut Graph) {
        let data_a = NetIndex::new(0,0);
        let data_b = NetIndex::new(1,0);
        let data_c = NetIndex::new(2,0);
        g.add_data(&data_a).expect("");
        g.add_data(&data_b).expect("");
        g.add_data(&data_c).expect("");
        
        let op_a = NetIndex::new(0,0);
        g.add_op(&op_a).expect("");

        g.connect(&[data_a, data_b], &[data_c,], &op_a).expect("");
    }

    // A   B
    //  \ /
    //   Op1
    //   |
    //   C   D
    //    \ /
    //     Op2
    //     |
    //     E
    fn setup_yy(g: &mut Graph) {
        let data_a = NetIndex::new(0,0);
        let data_b = NetIndex::new(1,0);
        let data_c = NetIndex::new(2,0);
        let data_d = NetIndex::new(3,0);
        let data_e = NetIndex::new(4,0);
        g.add_data(&data_a).expect("");
        g.add_data(&data_b).expect("");
        g.add_data(&data_c).expect("");
        g.add_data(&data_d).expect("");
        g.add_data(&data_e).expect("");
        
        let op1 = NetIndex::new(0,0);
        g.add_op(&op1).expect("");
        let op2 = NetIndex::new(1,0);
        g.add_op(&op2).expect("");

        g.connect(&[data_a, data_b], &[data_c,], &op1).expect("");
        g.connect(&[data_c, data_d], &[data_e,], &op2).expect("");
    }

    #[test]
    fn test_get_input_cache() {
        let mut g = Graph::new();
        setup_y(&mut g);
        assert_eq!(g.get_input_cache().len(), 2);

        let mut g = Graph::new();
        setup_yy(&mut g);
        assert_eq!(g.get_input_cache().len(), 3);
    }

    #[test]
    fn test_get_output_cache() {
        let mut g = Graph::new();
        setup_y(&mut g);
        assert_eq!(g.get_output_cache().len(), 1);

        let mut g = Graph::new();
        setup_yy(&mut g);
        assert_eq!(g.get_output_cache().len(), 1);
    }
}
