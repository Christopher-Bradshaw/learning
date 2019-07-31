use std::collections::HashMap;

#[derive(Debug)]
pub struct Trie<T> {
    next: HashMap<char, Trie<T>>,
    val: Option<T>,
}

impl<T> Trie<T>
where
    // If the type doesn't implement copy, we run into trouble returning it out of find.
    // We could maybe return a reference out of find to get around this?
    // But I like this illustration of generics with restrictions!
    T: Copy,
{
    pub fn new() -> Trie<T> {
        Trie {
            next: HashMap::new(),
            val: None,
        }
    }

    pub fn insert(&mut self, key: String, val: T) {
        let (nxt, rst) = nxt_rst(key);

        match nxt {
            None => self.val = Some(val),
            Some(v) => {
                // Create node if necessary
                if !self.next.contains_key(&v) {
                    self.next.insert(v, Trie::new());
                }
                // Insert the rest into that node
                self.next.get_mut(&v).unwrap().insert(rst, val);
            }
        };
    }

    pub fn find(&self, key: String) -> Option<T> {
        let (nxt, rst) = nxt_rst(key);

        match nxt {
            None => self.val,
            Some(v) => match self.next.get(&v) {
                None => None,
                Some(v) => v.find(rst),
            },
        }
    }

    pub fn keys(&self) {
        for key in self.next.keys() {
            println!("{}", key);
        }
    }
}

fn nxt_rst(s: String) -> (Option<char>, String) {
    if s.len() == 0 {
        return (None, String::from(""));
    } else {
        return (s[0..1].chars().next(), String::from(&s[1..]));
    }
}

#[cfg(test)]
mod test {
    use crate::trie::Trie;

    #[test]
    fn printable() {
        let mut t = Trie::new();
        t.insert(String::from("b"), 1);
        t.insert(String::from("v"), 2);

        println!("{}", t.find(String::from("b")).unwrap());
        println!("{:?}", t);
    }

    #[test]
    fn basic_functionality_ints() {
        let mut t = Trie::new();

        t.insert(String::from("bear"), 1);
        t.insert(String::from("beat"), 2);
        t.insert(String::from("pear"), 3);

        assert_eq!(t.find(String::from("bear")).unwrap(), 1);
        assert_eq!(t.find(String::from("beat")).unwrap(), 2);
        assert_eq!(t.find(String::from("pear")).unwrap(), 3);

        assert_eq!(t.find(String::from("bears")), None);
        assert_eq!(t.find(String::from("bea")), None);
    }

    #[test]
    fn basic_functionality_floats() {
        let mut t = Trie::new();

        t.insert(String::from("bear"), 1.);
        t.insert(String::from("pear"), 2.);

        assert_eq!(t.find(String::from("bear")).unwrap(), 1.);
        assert_eq!(t.find(String::from("pear")).unwrap(), 2.);
        assert_eq!(t.find(String::from("bears")), None);
        assert_eq!(t.find(String::from("bea")), None);
    }
}
