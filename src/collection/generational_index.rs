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
    pub fn new() -> GenIndex<T> {
        GenIndex::<T> {
            data: Vec::new(),
            generation: Vec::new(),
            available: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.data = Vec::new();
        self.generation = Vec::new();
        self.available = Vec::new();
    }

    pub fn get(&self, index: &NetIndex) -> Option<&T> {
        if index.id < self.generation.len() && self.generation[index.id] == index.gen {
            Option::Some(&self.data[index.id])
        } else {
            Option::None
        }
    }

    pub fn get_mut(&mut self, index: &NetIndex) -> Option<&mut T> {
        if index.id < self.generation.len() && self.generation[index.id] == index.gen {
            Option::Some(&mut self.data[index.id])
        } else {
            Option::None
        }
    }

    pub fn insert(&mut self, val: T) -> NetIndex {
        let mut ret = NetIndex::new(0, 0);
        if self.available.len() <= 0 {
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

    pub fn remove(&mut self, index: &NetIndex) -> bool {
        if index.id < self.generation.len() && self.generation[index.id] == index.gen {
            self.generation[index.id] += 1;
            self.available.push(index.id);
            true
        } else {
            false
        }
    }

    pub fn replace(&mut self, index: &NetIndex, val: T) {
        self.data[index.id] = val;
    }
}

impl<T> Iterator for GenIndex<T> {
    type Item = NetIndex;
    
    fn next(&mut self) -> Option<NetIndex> {
        Some(NetIndex::new(0,0))
    }
}
