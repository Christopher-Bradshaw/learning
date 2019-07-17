use std::ops::Deref;

pub struct MyDeref {
    value: i32,
}

impl MyDeref {
    pub fn new(value: i32) -> MyDeref {
        MyDeref { value }
    }
}

impl Deref for MyDeref {
    type Target = i32;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

pub fn deref_func(x: i32) {
    println!("In here with the dereferenced int, val: {}", x);
}
