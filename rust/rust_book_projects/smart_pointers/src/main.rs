mod deref_example;
use std::rc::Rc;

fn main() {
    let x = 10;
    basic_pointers(&x);
    boxes();
    multiple_ownership();
}

fn multiple_ownership() {
    #[derive(Debug)]
    struct Node {
        val: i32,
        next: Option<Rc<Node>>,
    }

    let c = Rc::new(Node { val: 2, next: None });
    println!("{}", Rc::strong_count(&c));
    let p1 = Node {
        val: 1,
        next: Some(Rc::clone(&c)),
    };
    println!("{}", Rc::strong_count(&c));

    // We could have done this with Box. We have `p(arent)1 -> c(hild)`
    // But what if the child has two parents?
    // We usually can't do that because the ownership of `c` has been moved
    // into `p1`. This is why we needed Rc here.
    let p2 = Node {
        val: 1,
        next: Some(Rc::clone(&c)),
    };

    // Rc::clone doesn't do a clone. In just increments a counter
    println!("{}", Rc::strong_count(&c));

    drop(p1);
    println!("{}", Rc::strong_count(&c));

    drop(p2);
    println!("{}", Rc::strong_count(&c));

    // Note Rc only allows immutable references.
}

// This function borrows the value of x
fn basic_pointers(x: &i32) {
    println!("This is a pointer to an int with value: {}", *x);
    println!("But printing the reference also gives: {}", x);
}

fn boxes() {
    // x is a pointer (stored on the stack) to some data on the heap
    let x = Box::new(1);
    println!("This is a box containing an int of value: {}", *x);
    println!("But printing the reference also gives: {}", x);

    // Sticking an int in a box isn't really useful...
    // However there are times where you really need boxes

    // Recursive typing.
    // Imagine you have a type that has as part of it another version of itself
    // E.g. Node { Option<Node> }
    // We have no idea at compile time how much memory to allocate for this, and
    // so it can't live on the stack.
    #[derive(Debug)]
    struct Node {
        val: i32,
        // If we didn't box this, this size of a node would be infinite!
        next: Option<Box<Node>>,
    }

    let nodes = Node {
        val: 1,
        next: Some(Box::new(Node { val: 2, next: None })),
    };
    println!("{:?}", nodes);

    // What makes this box a smart pointer?
    // Pointer: It implements `Deref`.
    // Smart: It implements `Drop`. This is how, when the box goes out of scope and it is
    // cleaned up, it knows to clean up the values it points to too.

    let d = deref_example::MyDeref::new(1);
    deref_example::deref_func(*d);
}
