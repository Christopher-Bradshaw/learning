// Heavily inspired by https://rust-unofficial.github.io/too-many-lists/first-final.html

type More<T> = Option<Box<Node<T>>>;

struct Node<T> {
    next: More<T>,
    val: T,
}

pub struct Stack<T> {
    head: More<T>,
}

impl<T> Stack<T> {
    pub fn new() -> Stack<T> {
        Stack { head: None }
    }

    // We pass a reference to the stack - we borrow the stack (don't take ownership)
    // This reference needs to be mutable (as we will change the value of the stack's head)
    pub fn push(&mut self, val: T) {
        self.head = Some(Box::new(Node {
            next: self.head.take(),
            val: val,
        }));
    }

    pub fn pop(&mut self) -> Option<T> {
        match self.head.take() {
            None => None,
            Some(v) => {
                self.head = v.next;
                Some(v.val)
            }
        }
    }
}

#[cfg(test)]
mod test {
    // This is run from the root of the library and we import this as the stack module
    use crate::stack::Stack;

    #[test]
    fn basic_functionality() {
        // The stack needs to be mutable as its methods mutate it!
        let mut s = Stack::new();

        // Both push and pop just borrow the stack. So it is still owned in this function.
        // They mutate it though.
        assert_eq!(s.pop(), None);

        s.push(1);
        assert_eq!(s.pop(), Some(1));
        assert_eq!(s.pop(), None);

        s.push(1);
        s.push(2);
        assert_eq!(s.pop(), Some(2));
        assert_eq!(s.pop(), Some(1));
        assert_eq!(s.pop(), None);
        // And now s will be freed as we get to the end of its scope
    }

    #[test]
    fn is_generic() {
        let mut s = Stack::new();

        assert_eq!(s.pop(), None);

        s.push("blah");
        assert_eq!(s.pop(), Some("blah"));
        assert_eq!(s.pop(), None);
    }

}
