/// A generational index implementation.
/// A generational index is a map-a-like container, which
/// invalidate index/key when the item is removed,
/// even the container itself don't have the access to that index/key.
use std::fmt;

/// NetIndex index used for generational index.
#[derive(Debug, PartialEq, Eq, Ord, PartialOrd, Copy, Clone)]
pub struct NetIndex {
    id: usize,
    gen: usize,
}

impl NetIndex {
    pub fn new(id: usize, gen: usize) -> NetIndex {
        NetIndex { id, gen }
    }
}

impl fmt::Display for NetIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.id, self.gen)
    }
}

/// A simple generational index implementation.
/// The data is stored in a read-only manner,
/// Use RefCell to get mutability.
/// Not secure, no index validity check.
pub struct GenIndex<T> {
    data: Vec<T>,
    generation: Vec<usize>,
    available: Vec<usize>,
}

impl<T> GenIndex<T> {
    /// Create an empty GenIndex
    pub fn new() -> GenIndex<T> {
        GenIndex::<T> {
            data: Vec::new(),
            generation: Vec::new(),
            available: Vec::new(),
        }
    }

    /// Clear the GenIndex
    pub fn clear(&mut self) {
        self.data = Vec::new();
        self.generation = Vec::new();
        self.available = Vec::new();
    }

    ///
    /// Check if a key is in the collection
    ///
    pub fn contains(&self, index: &NetIndex) -> bool {
        index.id < self.generation.len() && self.generation[index.id] == index.gen
    }

    /// Return the registered item
    pub fn get(&self, index: &NetIndex) -> Option<&T> {
        if index.id < self.generation.len() && self.generation[index.id] == index.gen {
            Option::Some(&self.data[index.id])
        } else {
            Option::None
        }
    }

    /// Return a mut reference.
    pub fn get_mut(&mut self, index: &NetIndex) -> Option<&mut T> {
        if index.id < self.generation.len() && self.generation[index.id] == index.gen {
            Option::Some(&mut self.data[index.id])
        } else {
            Option::None
        }
    }

    /// Number of item in the list.
    pub fn len(&self) -> usize {
        self.data.len() - self.available.len()
    }
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Add a new item to the list.
    pub fn insert(&mut self, val: T) -> NetIndex {
        let mut ret = NetIndex::new(0, 0);
        if self.available.is_empty() {
            ret.id = self.data.len();
            self.data.push(val);
            self.generation.push(0);
            ret.gen = 0;
        } else {
            ret.id = self.available.pop().expect("id in available");
            self.data[ret.id] = val;
            ret.gen = self.generation[ret.id];
        }
        ret
    }

    /// Remove an item from the list.
    pub fn remove(&mut self, index: &NetIndex) -> Result<(), ()> {
        if index.id < self.generation.len() && self.generation[index.id] == index.gen {
            self.generation[index.id] += 1;
            self.available.push(index.id);
            Ok(())
        } else {
            Err(())
        }
    }

    /// Replace the item of the index with a new one.
    pub fn replace(&mut self, index: &NetIndex, val: T) -> Result<(), ()> {
        if index.id < self.data.len() && self.generation[index.id] == index.gen {
            self.data[index.id] = val;
            Ok(())
        } else {
            Err(())
        }
    }

    pub fn append(&mut self, other: &mut GenIndex<T>) {
        
    }
}


pub struct GenIndexIter<'a, T> {
    index: usize,
    gen_index_ref: &'a GenIndex<T>,
}
impl<'a, T> GenIndexIter<'a, T> {
    pub fn new(index_ref: &GenIndex<T>) -> GenIndexIter<T> {
        if index_ref.available.is_empty() {
            GenIndexIter {
                index: 0,
                gen_index_ref: index_ref,
            }
        } else {
            
            GenIndexIter {
                index: index_ref.available[0],
                gen_index_ref: index_ref,
            }
        }
        
    }
}
impl<'a, T> Iterator for GenIndexIter<'a, T> {
    type Item = NetIndex;
    
    fn next(&mut self) -> Option<NetIndex> {
        if self.gen_index_ref.available.is_empty() {
            if self.data.len() == self.index {
                return None
            } else {
                let ret = NetIndex::new(self.index, self.generation[self.index]);
                self.index += 1;
                return ret;
            }
        } else {
            
        }
        Some(NetIndex::new(0,0))
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genindex_new_add_del() {
        let mut g = GenIndex::<f32>::new();
        assert_eq!(g.len(), 0);
        let index1 = g.insert(1.);
        assert_eq!(g.len(), 1);
        assert_eq!(g.remove(&index1).expect(""), ());

        let index2 = g.insert(2.);
        let index3 = g.insert(3.);
        assert_eq!(g.len(), 2);
        assert_eq!(*g.get(&index2).expect(""), 2.);
        assert_eq!(*g.get(&index3).expect(""), 3.);

        g.clear();
    }

    #[test]
    fn test_gen_index() {
        #[derive(Debug, Copy, Clone)]
        struct A {
            v: u32,
        }
        let mut a = GenIndex::<A>::new();
    
        let index1 = a.insert(A { v: 10 });
        assert_eq!(index1, NetIndex::new(0, 0));
        let index2 = a.insert(A { v: 20 });
        assert_eq!(index2, NetIndex::new(1, 0));
    
        let tv1 = a.get(&index1).unwrap().v;
        assert_eq!(tv1, 10);
        let tv2 = a.get(&index2).unwrap().v;
        assert_eq!(tv2, 20);
        let tv_none = a.get(&NetIndex::new(0, 1));
        assert_eq!(tv_none.is_none(), true);
    
        let a2 = a.remove(&index2);
        let tv_none = a.get(&index2);
        assert_eq!(tv_none.is_none(), true);
        assert_eq!(a2.expect(""), ());
    
        let index3 = a.insert(A { v: 30 });
        assert_eq!(index3, NetIndex::new(1, 1));
    }
}
