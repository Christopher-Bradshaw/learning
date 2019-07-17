use std::collections::hash_map::DefaultHasher;
use std::convert::TryInto;
use std::hash::{Hash, Hasher};

#[derive(Debug)]
pub struct HashTable {
    slots: Box<[HashChain]>,
    n_slots: usize,
    n_items: usize,
}

type HashChain = Vec<HashItem>;

#[derive(Debug)]
struct HashItem {
    key: String,
    val: i32,
}

impl HashTable {
    pub fn new() -> HashTable {
        HashTable::new_with_size(2)
    }

    pub fn new_with_size(n_slots: usize) -> HashTable {
        let mut slots = Vec::new();
        for _ in 0..n_slots {
            slots.push(HashChain::new());
        }
        return HashTable {
            slots: slots.into_boxed_slice(),
            n_slots: n_slots,
            n_items: 0,
        };
    }

    pub fn find(&self, key: String) -> Option<i32> {
        for item in self.slots[self.hash(&key)].iter() {
            if item.key == key {
                return Some(item.val);
            }
        }
        None
    }

    // This needs to take ownership of self, I think, because of resizes and thread safety
    pub fn insert(mut self, key: String, val: i32) -> HashTable {
        let slot = self.hash(&key);

        for (i, item) in self.slots[slot].iter().enumerate() {
            if item.key == key {
                // We don't need to worry about resizing as we didn't change the length
                self.slots[slot][i].val = val;
                return self;
            }
        }
        self.slots[self.hash(&key)].push(HashItem { key, val });
        self.n_items += 1;
        if self.n_items == self.n_slots {
            return self.resize();
        }
        return self;
    }

    fn hash(&self, key: &String) -> usize {
        let mut s = DefaultHasher::new();
        key.hash(&mut s);
        (s.finish() % self.n_slots as u64).try_into().unwrap()
    }

    fn resize(self) -> HashTable {
        println!("resizing!");
        let mut n = HashTable::new_with_size(self.n_slots * 2);
        for slot in self.slots.iter() {
            for item in slot.iter() {
                n = n.insert(item.key.clone(), item.val);
            }
        }
        n
    }
}

#[cfg(test)]
mod test {
    use crate::hash_table::HashTable;

    #[test]
    fn basic_functionality_ints() {
        let mut h = HashTable::new();
        h = h.insert(String::from("b"), 1);
        h = h.insert(String::from("b"), 2);
        h = h.insert(String::from("d"), 2);
        h = h.insert(String::from("e"), 2);
        println!("{:?}", h);

        println!("{:?}", h.find(String::from("b")));
    }
}
