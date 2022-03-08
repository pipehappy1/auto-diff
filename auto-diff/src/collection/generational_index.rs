//! A generational index implementation.
//! A generational index is a map-a-like container, which
//! invalidate index/key when the item is removed,
//! even the container itself don't have the access to that index/key.
use std::fmt;

use crate::err::AutoDiffError;

/// GenKey index used for generational index.
#[derive(Debug, PartialEq, Eq, Ord, PartialOrd, Copy, Clone)]
pub struct GenKey {
    id: usize,
    gen: usize,
}
    
impl GenKey {
    pub fn new(id: usize, gen: usize) -> GenKey {
        GenKey { id, gen }
    }
}

impl fmt::Display for GenKey {
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
    pub fn contains(&self, index: &GenKey) -> bool {
        index.id < self.generation.len() && self.generation[index.id] == index.gen
    }

    /// Return the registered item
    pub fn get(&self, index: &GenKey) -> Result<&T, AutoDiffError> {
        if index.id < self.generation.len() && self.generation[index.id] == index.gen {
            Ok(&self.data[index.id])
        } else {
            Err(AutoDiffError::new(&format!("GenIndex cannot find the item by key {:?}!", index)))
        }
    }

    /// Return a mut reference.
    pub fn get_mut(&mut self, index: &GenKey) -> Option<&mut T> {
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
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add a new item to the list.
    pub fn insert(&mut self, val: T) -> GenKey {
        let mut ret = GenKey::new(0, 0);
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
    pub fn remove(&mut self, index: &GenKey) -> Result<(), ()> {
        if index.id < self.generation.len() && self.generation[index.id] == index.gen {
            self.generation[index.id] += 1;
            self.available.push(index.id);
            Ok(())
        } else {
            Err(())
        }
    }

    /// Replace the item of the index with a new one.
    pub fn replace(&mut self, index: &GenKey, val: T) -> Result<(), ()> {
        if index.id < self.data.len() && self.generation[index.id] == index.gen {
            self.data[index.id] = val;
            Ok(())
        } else {
            Err(())
        }
    }

    pub fn iter_key(&self) -> GenIndexIter<T> {
        GenIndexIter::<T>::new(self)
    }
}


pub struct GenIndexIter<'a, T> {
    index: usize,
    gen_index_ref: &'a GenIndex<T>,
}
impl<'a, T> GenIndexIter<'a, T> {
    pub fn new(index_ref: &GenIndex<T>) -> GenIndexIter<T> {
        GenIndexIter {
            index: 0,
            gen_index_ref: index_ref,
        }
    }
}
impl<'a, T> Iterator for GenIndexIter<'a, T> {
    type Item = GenKey;
    
    fn next(&mut self) -> Option<GenKey> {
        let ret: GenKey;
        if self.gen_index_ref.data.is_empty() {
            return None;
        }
        if self.gen_index_ref.data.len() == self.index {
            None
        } else {
            if self.gen_index_ref.available.is_empty() {
                ret = GenKey::new(self.index,
                                    self.gen_index_ref.generation[self.index]);
            } else {
                loop {
                    if self.gen_index_ref.data.len() == self.index {
                        return None;
                    }
                    if self.gen_index_ref.available.contains(&self.index) {
                        self.index += 1;
                    } else {
                        ret = GenKey::new(self.index,
                                            self.gen_index_ref.generation[self.index]);
                        break;
                    }
                }
            }
            
            self.index += 1;
            Some(ret)
        }
    }
}

impl<T> Default for GenIndex<T> {
    fn default() -> Self {
        Self::new()
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
        assert_eq!(index1, GenKey::new(0, 0));
        let index2 = a.insert(A { v: 20 });
        assert_eq!(index2, GenKey::new(1, 0));
    
        let tv1 = a.get(&index1).unwrap().v;
        assert_eq!(tv1, 10);
        let tv2 = a.get(&index2).unwrap().v;
        assert_eq!(tv2, 20);
        //let tv_none = a.get(&GenKey::new(0, 1));
        //assert_eq!(tv_none.unwrap().is_none(), true);
    
        let a2 = a.remove(&index2);
        let tv_none = a.get(&index2);
        //assert_eq!(tv_none.unwrap().is_none(), true);
        assert_eq!(a2.expect(""), ());
    
        let index3 = a.insert(A { v: 30 });
        assert_eq!(index3, GenKey::new(1, 1));
    }

    #[test]
    fn iter() {
        #[derive(Debug, Copy, Clone)]
        struct A {
            v: u32,
        }
        let mut a = GenIndex::<A>::new();

        let index1 = a.insert(A { v: 10 });
        let index2 = a.insert(A { v: 20 });
        let index3 = a.insert(A { v: 30 });

        let keys: Vec<GenKey> = a.iter_key().collect();
        assert_eq!(keys, vec![GenKey::new(0, 0), GenKey::new(1, 0), GenKey::new(2, 0)]);

        a.remove(&index2).expect("");
        let keys: Vec<GenKey> = a.iter_key().collect();
        assert_eq!(keys, vec![GenKey::new(0, 0), GenKey::new(2, 0)]);

        a.remove(&index3).expect("");
        let keys: Vec<GenKey> = a.iter_key().collect();
        assert_eq!(keys, vec![GenKey::new(0, 0)]);

        a.remove(&index1).expect("");
        let keys: Vec<GenKey> = a.iter_key().collect();
        assert_eq!(keys, vec![]);
    }
}
