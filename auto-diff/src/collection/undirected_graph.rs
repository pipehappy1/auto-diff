use std::collections::{BTreeSet, BTreeMap};
use crate::err::AutoDiffError;


/// Graph
pub struct UnDirectedGraph<TNode, TEdge> {
    node: BTreeSet<TNode>,
    edge: BTreeSet<TEdge>,
    edge2node: BTreeMap<TEdge, BTreeSet<TNode>>,
    node2edige: BTreeMap<TNode, BTreeSet<TEdge>>,
}

impl<TNode:Clone + Copy + Ord,
     TEdge:Clone + Copy + Ord>
    Default for UnDirectedGraph<TNode, TEdge> {
    fn default() -> UnDirectedGraph<TNode, TEdge> {
	UnDirectedGraph {
	    node: BTreeSet::new(),
	    edge: BTreeSet::new(),
	    edge2node: BTreeMap::new(),
	    node2edige: BTreeMap::new(),
        }
    }
}

impl<TNode, TEdge> UnDirectedGraph<TNode, TEdge> {
    pub fn new() -> UnDirectedGraph<TNode, TEdge> {
        UnDirectedGraph {
	    node: BTreeSet::new(),
	    edge: BTreeSet::new(),
	    edge2node: BTreeMap::new(),
	    node2edige: BTreeMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collection::generational_index::{GenKey};

    #[test]
    fn new() {
        let _g = UnDirectedGraph::<GenKey, GenKey>::new();
    }
}
