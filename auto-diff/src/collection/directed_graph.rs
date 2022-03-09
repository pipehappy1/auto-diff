//! A directed graph implementation with interleave op node and data node
//! and all the edges are data node.
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use crate::err::AutoDiffError;
use super::generational_index::GenKey;

#[cfg(feature = "use-serde")]
use serde::{Serialize, Deserialize};

/// Graph
#[cfg_attr(feature = "use-serde", derive(Serialize, Deserialize))]
pub struct Graph<TData: Ord, TOp: Ord> {
    data: BTreeSet<TData>,
    op: BTreeSet<TOp>,
    forward_dt_op: BTreeMap<TData, BTreeSet<TOp>>,
    forward_op_dt: BTreeMap<TOp, BTreeSet<TData>>,
    backward_dt_op: BTreeMap<TData, BTreeSet<TOp>>,
    backward_op_dt: BTreeMap<TOp, BTreeSet<TData>>,
}

impl<TData: Clone + Copy + Ord, TOp: Clone + Copy + Ord> Default for Graph<TData, TOp> {
    fn default() -> Graph<TData, TOp> {
        Graph{
            data: BTreeSet::new(),
            op: BTreeSet::new(),
            forward_dt_op: BTreeMap::new(),
            forward_op_dt: BTreeMap::new(),
            backward_dt_op: BTreeMap::new(),
            backward_op_dt: BTreeMap::new(),
        }
    }
}

impl<TData: Clone + Copy + Ord, TOp: Clone + Copy + Ord> Graph<TData, TOp> {
    /// Create a graph with defaults
    pub fn new() -> Graph<TData, TOp> {
        Graph{
            data: BTreeSet::new(),
            op: BTreeSet::new(),
            forward_dt_op: BTreeMap::new(),
            forward_op_dt: BTreeMap::new(),
            backward_dt_op: BTreeMap::new(),
            backward_op_dt: BTreeMap::new(),
        }
    }

    /// iterator over data node.
    pub fn iter_data(&self) -> NodeIterator<TData> {
        NodeIterator {
            iter: self.data.iter()
        }
    }
    /// iterator over op node.
    pub fn iter_op(&self) -> NodeIterator<TOp> {
        NodeIterator {
            iter: self.op.iter()
        }
    }

    ///
    /// Return the list of ops that the given variable is the input of the func.
    ///
    pub fn iter_op_given_input(&self, var: &TData) -> Result<NodeIterator<TOp>, &str> {
        if !self.data.contains(var) {
            Err("Not a valid variable/data")
        } else {
            Ok(NodeIterator {
                iter: self.forward_dt_op.get(var).expect("").iter()
            })
        }
    }

    ///
    /// Return the list of ops that the given variable is the output.
    ///
    pub fn iter_op_given_output(&self, var: &TData) -> Result<NodeIterator<TOp>, &str> {
        if !self.data.contains(var) {
            Err("Not a valid variable/data")
        } else {
            Ok(NodeIterator {
                iter: self.backward_dt_op.get(var).expect("").iter()
            })
        }
    }

    ///
    /// Return the list of input given the func.
    ///
    pub fn iter_input_given_op(&self, func: &TOp) -> Result<NodeIterator<TData>, &str> {
        if !self.op.contains(func) {
            Err("Bad func id.")
        } else {
            Ok(NodeIterator {
                iter: self.backward_op_dt.get(func).expect("").iter()
            })
        }
    }

    ///
    /// Return a list of data as the output of the op.
    ///
    pub fn iter_output_given_op(&self, func: &TOp) -> Result<NodeIterator<TData>, &str> {
        if !self.op.contains(func) {
            Err("Bad func id.")
        } else {
            Ok(NodeIterator {
                iter: self.forward_op_dt.get(func).expect("").iter()
            })
        }
    }

    /// Add a data node.
    pub fn add_data(&mut self, id: &TData) -> Result<TData, &str> {
        if !self.data.contains(id) {
            self.data.insert(*id);
            self.forward_dt_op.insert(*id, BTreeSet::new());
            self.backward_dt_op.insert(*id, BTreeSet::new());
            Ok(*id)
        } else {
            Err("data is exits!")
        }
    }

    /// Remove a data node, op node and downstream data/op node are removed.
    pub fn drop_data(&mut self, id: &TData) -> Result<TData, &str> {
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

            Ok(*id)
        } else {
            Err("data id is not found!")
        }
    }

    /// Add a danglging op node.
    pub fn add_op(&mut self, id: &TOp) -> Result<TOp, &str> {
        if !self.op.contains(id) {
            self.op.insert(*id);
            self.forward_op_dt.insert(*id, BTreeSet::new());
            self.backward_op_dt.insert(*id, BTreeSet::new());
            Ok(*id)
        } else {
            Err("op id exists.")
        }
    }

    /// Remvoe an op node, input data node and downstream data/op node are removed.
    pub fn drop_op(&mut self, id: &TOp) -> Result<TOp, &str> {
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
            Ok(*id)
        } else {
            Err("op id is not found!")
        }

    }

    ///
    /// Decouple input variable and op
    ///
    pub fn decouple_data_func(&mut self, var: &TData, func: &TOp) -> Result<(), AutoDiffError> {
        if self.data.contains(var) && self.op.contains(func) {
            self.forward_dt_op.get_mut(var).expect("").remove(func);
            self.backward_op_dt.get_mut(func).expect("").remove(var);
            Ok(())
        } else {
            Err(AutoDiffError::new("invalid var or func"))
        }
    }

    ///
    /// Decouple op and output variable
    ///
    pub fn decouple_func_data(&mut self, func: &TOp, var: &TData) -> Result<(), AutoDiffError> {
        if self.data.contains(var) && self.op.contains(func) {
            self.forward_op_dt.get_mut(func).expect("").remove(var);
            self.backward_dt_op.get_mut(var).expect("").remove(func);
            Ok(())
        } else {
            Err(AutoDiffError::new("invalid var or func"))
        }
    }

    /// list data node without upstream op node in a set.
    pub fn get_input_edge_data(&self) -> BTreeSet<TData> {
        let mut jobs = BTreeSet::new();
        for i in &self.data {
            if self.backward_dt_op.get(i).expect("").is_empty() {
                jobs.insert(*i);
            }
        }
        jobs
    }

    /// list data node without downstream op node in a set.
    pub fn get_output_edge_data(&self) -> BTreeSet<TData> {
        let mut jobs = BTreeSet::new();
        for i in &self.data {
            if self.forward_dt_op.get(i).expect("").is_empty() {
                jobs.insert(*i);
            }
        }
        jobs
    }

    /// Connect input data, output data and operation
    pub fn connect(&mut self, dti: &[TData],
                   dto: &[TData],
                   op: &TOp) -> Result<TOp, &str> {
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
                self.forward_dt_op.get_mut(i).expect("").insert(*op);
                self.backward_op_dt.get_mut(op).expect("").insert(*i);
            }
            for i in dto {
                self.forward_op_dt.get_mut(op).expect("").insert(*i);
                self.backward_dt_op.get_mut(i).expect("").insert(*op);
            }
            Ok(*op)
        } else {
            Err("Invalid id!")
        }
    }

    /// Auxilary connect, This allows the graph to support loop.
    pub fn connect_aux(&mut self, input_data: &[TData],
                       output_data: &[TData],
                       op: &TOp) -> Result<TOp, &str> {
        if !self.op.contains(op) ||
            input_data.iter().any(|x| !self.data.contains(x)) ||
            output_data.iter().any(|x| !self.data.contains(x)) {
                return Err("Invalid id!");
            }
        unimplemented!();
        //return Ok(*op);
    }

    ///
    /// Walk through the graph with a starting set of data nodes.
    /// Go through backwards if forward is false.
    /// The closure call provides the side-effect.
    ///
    /// This Walk() guarantee the input of visiting op is already visited
    /// or it's an input.
    ///
    pub fn walk<F>(&self, start_set: &[TData],
                   forward: bool,
                   closure: F) -> Result<(), BTreeSet<TData>>
    where F: Fn(&[TData], &[TData], &TOp)  {
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
        let mut jobs = BTreeSet::<TData>::new();
        // op is done.
        let mut done = BTreeSet::<TOp>::new(); // ops done.

        for index in start_set {
            jobs.insert(*index);
        }
        
        loop {
            let mut made_progress = false;

            // collect ops needs to do given the data in jobs.
            let mut edge_op = BTreeSet::<TOp>::new();
            for dt in &jobs {
                for op_candidate in &fdo[dt] {
                    edge_op.insert(*op_candidate);
                }
            }

            // process op if possible
            for op_candidate in edge_op {
                if bod[&op_candidate]
                    .iter()
                    .all(|dt| jobs.contains(dt)) {

                        // collect input ids.
                        let mut inputs = Vec::<TData>::new();
                        for input in bod[&op_candidate].iter() {
                            inputs.push(*input);
                        }
                        // collect output ids.
                        let mut outputs = Vec::<TData>::new();
                        for output in fod[&op_candidate].iter() {
                            outputs.push(*output);
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
                            if !fdo[output].is_empty() {
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

        if !jobs.is_empty() {
            Err(jobs)
        } else {
            Ok(())
        }
    }

    /////
    ///// Walk through the graph with a starting set of data nodes.
    ///// Go through backwards if forward is false.
    /////
    //pub fn walk_dyn(&self, start_set: &[TData],
    //               forward: bool,
    //                closure: dyn Calling) -> Result<(), BTreeSet<TData>> {
    //    Ok(())
    //}

    /// 
    pub fn append(&mut self, other: &Self,
                  data_key_map: BTreeMap<TData, TData>,
                  op_key_map: BTreeMap<TOp, TOp>) -> Result<(), AutoDiffError> {

        for key in other.iter_data() {
            self.data.insert(data_key_map[key]);
        }
        for key in other.iter_op() {
            self.op.insert(op_key_map[key]);
        }
        for (key, value) in other.forward_dt_op.iter() {
            let mut new_set = BTreeSet::new();
            for key in value.iter() {
                new_set.insert(op_key_map[key]);
            }
            self.forward_dt_op.insert(data_key_map[key], new_set);
        }
        for (key, value) in other.backward_dt_op.iter() {
            let mut new_set = BTreeSet::new();
            for key in value.iter() {
                new_set.insert(op_key_map[key]);
            }
            self.backward_dt_op.insert(data_key_map[key], new_set);
        }
        for (key, value) in other.forward_op_dt.iter() {
            let mut new_set = BTreeSet::new();
            for key in value.iter() {
                new_set.insert(data_key_map[key]);
            }
            self.forward_op_dt.insert(op_key_map[key], new_set);
        }
        for (key, value) in other.backward_op_dt.iter() {
            let mut new_set = BTreeSet::new();
            for key in value.iter() {
                new_set.insert(data_key_map[key]);
            }
            self.backward_op_dt.insert(op_key_map[key], new_set);
        }

        
        Ok(())
    }
}

// iterator
pub struct NodeIterator<'a, TNode> {
    iter: std::collections::btree_set::Iter<'a, TNode>,
}
impl<'a, TNode> Iterator for NodeIterator<'a, TNode> {
    type Item = &'a TNode;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl fmt::Debug for Graph<GenKey, GenKey> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Dumping graph")?;
        writeln!(f, "data: {:?}", self.data)?;
        writeln!(f, "op: {:?}", self.op)?;
        writeln!(f, "dt 2 op: {:?}", self.forward_dt_op)?;
        writeln!(f, "op 2 dt: {:?}", self.forward_op_dt)
    }
}

impl<T1: Ord, T2: Ord> PartialEq for Graph<T1, T2> {
    fn eq(&self, other: &Self) -> bool {
	self.data.eq(&other.data) &&
	    self.op.eq(&other.op) &&
	    self.forward_dt_op.eq(&other.forward_dt_op) &&
	    self.forward_op_dt.eq(&other.forward_op_dt) &&
	    self.backward_dt_op.eq(&other.backward_dt_op) &&
	    self.backward_op_dt.eq(&other.backward_op_dt)
    }
}

impl<T1: Ord, T2: Ord> Eq for Graph<T1, T2> {}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::collection::generational_index::{GenKey};
    
    #[test]
    fn new() {
        let _g = Graph::<GenKey, GenKey>::new();
    }

    // A   B
    //  \ /
    //   Op
    //   |
    //   C
    fn setup_y(g: &mut Graph<GenKey, GenKey>) {
        let data_a = GenKey::new(0,0);
        let data_b = GenKey::new(1,0);
        let data_c = GenKey::new(2,0);
        g.add_data(&data_a).expect("");
        g.add_data(&data_b).expect("");
        g.add_data(&data_c).expect("");
        
        let op_a = GenKey::new(0,0);
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
    fn setup_yy(g: &mut Graph<GenKey, GenKey>) {
        let data_a = GenKey::new(0,0);
        let data_b = GenKey::new(1,0);
        let data_c = GenKey::new(2,0);
        let data_d = GenKey::new(3,0);
        let data_e = GenKey::new(4,0);
        g.add_data(&data_a).expect("");
        g.add_data(&data_b).expect("");
        g.add_data(&data_c).expect("");
        g.add_data(&data_d).expect("");
        g.add_data(&data_e).expect("");
        
        let op1 = GenKey::new(0,0);
        g.add_op(&op1).expect("");
        let op2 = GenKey::new(1,0);
        g.add_op(&op2).expect("");

        g.connect(&[data_a, data_b], &[data_c,], &op1).expect("");
        g.connect(&[data_c, data_d], &[data_e,], &op2).expect("");
    }

    #[test]
    fn iter() {
        let mut g = Graph::new();
        setup_yy(&mut g);
        
        for i in g.iter_data() {
            println!("{:?}", i);
        }

        for i in g.iter_op() {
            println!("{:?}", i);
        }
    }

    #[test]
    fn test_get_input_cache() {
        let mut g = Graph::new();
        setup_y(&mut g);
        assert_eq!(g.get_input_edge_data().len(), 2);

        let mut g = Graph::<GenKey, GenKey>::new();
        setup_yy(&mut g);
        assert_eq!(g.get_input_edge_data().len(), 3);
    }

    #[test]
    fn test_get_output_cache() {
        let mut g = Graph::new();
        setup_y(&mut g);
        assert_eq!(g.get_output_edge_data().len(), 1);

        let mut g = Graph::<GenKey, GenKey>::new();
        setup_yy(&mut g);
        assert_eq!(g.get_output_edge_data().len(), 1);
    }

    #[test]
    fn add_data() {

        let mut g = Graph::<GenKey, GenKey>::new();
        let data1 = GenKey::new(0,0);
        let data2 = GenKey::new(1,0);
        g.add_data(&data1).expect("");
        g.add_data(&data2).expect("");
    }
}

