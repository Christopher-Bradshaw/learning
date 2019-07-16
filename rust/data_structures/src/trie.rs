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

        if !self.next.contains_key(&nxt) {
            self.next.insert(nxt, Trie::new());
        }

        if rst == "".to_string() {
            self.val = Some(val);
        } else {
            self.next
                .get_mut(&nxt)
                .unwrap()
                .insert(rst.to_string(), val);
        }
    }

    pub fn find(&self, key: String) -> Option<T> {
        let (nxt, rst) = nxt_rst(key);

        if rst == "".to_string() {
            return self.val;
        }

        return match self.next.get(&nxt) {
            Some(v) => v.find(rst),
            None => None,
        };
    }

    pub fn keys(&self) {
        for key in self.next.keys() {
            println!("{}", key);
        }
    }
}

fn nxt_rst(s: String) -> (char, String) {
    let (nxt, rst) = (&s[0..1], &s[1..]);
    let nxt = nxt.chars().next().unwrap();

    (nxt, rst.to_string())
}

#[cfg(test)]
mod test {
    use crate::trie::Trie;

    #[test]
    fn basic_functionality_ints() {
        let mut t = Trie::new();

        t.insert("bear".to_string(), 1);
        t.insert("beat".to_string(), 1);
        t.insert("pear".to_string(), 2);

        assert_eq!(t.find("bear".to_string()).unwrap(), 1);
        assert_eq!(t.find("beat".to_string()).unwrap(), 1);
        assert_eq!(t.find("pear".to_string()).unwrap(), 2);

        assert_eq!(t.find("bears".to_string()), None);
        assert_eq!(t.find("bea".to_string()), None);
    }

    #[test]
    fn basic_functionality_floats() {
        let mut t = Trie::new();

        t.insert("bear".to_string(), 1.);
        t.insert("pear".to_string(), 2.);

        assert_eq!(t.find("bear".to_string()).unwrap(), 1.);
        assert_eq!(t.find("pear".to_string()).unwrap(), 2.);
        assert_eq!(t.find("bears".to_string()), None);
        assert_eq!(t.find("bea".to_string()), None);
    }
}
