use std::collections::HashMap;

#[derive(Debug)]
pub struct Trie {
    next: HashMap<char, Trie>,
}

impl Trie {
    pub fn new() -> Trie {
        Trie {
            next: HashMap::new(),
        }
    }

    fn nxt_rst(s: String) -> (char, String) {
        let (nxt, rst) = (&s[0..1], &s[1..]);
        let nxt = nxt.chars().next().unwrap();

        (nxt, rst.to_string())
    }

    pub fn insert(&mut self, val: String) {
        let (nxt, rst) = Trie::nxt_rst(val);

        if !self.next.contains_key(&nxt) {
            self.next.insert(nxt, Trie::new());
        }

        if rst != "".to_string() {
            self.next.get_mut(&nxt).unwrap().insert(rst.to_string());
        }
    }

    pub fn contains(&self, val: String) -> bool {
        let (nxt, rst) = Trie::nxt_rst(val);

        match self.next.contains_key(&nxt) {
            true => {
                if rst == "".to_string() {
                    return true;
                } else {
                    return self.next.get(&nxt).unwrap().contains(rst);
                }
            }
            false => {
                return false;
            }
        }
    }

    pub fn keys(&self) {
        for key in self.next.keys() {
            println!("{}", key);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::trie::Trie;

    #[test]
    fn basic_functionality() {
        let mut t = Trie::new();

        t.insert("bear".to_string());
        t.insert("beat".to_string());
        t.insert("pear".to_string());

        assert!(t.contains("bear".to_string()));
        assert!(t.contains("beat".to_string()));
        assert!(!t.contains("bean".to_string()));
    }
}
