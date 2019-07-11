# Rust

Find all the docs using `rustup docs`. These include the book, std library docs, reference, and much more...

## Todo
* Continue reading the book
* Read some good code (something in https://github.com/rust-lang (rustlings, rls), ripgrep)?
* interop with python?

## Cargo

The rust build/dependency system. Useful commands,

* cargo new: create a new project (with cargo.toml + src/main.rs)
* cargo build: build
* cargo check: check if it can be built (often a lot faster than actually building)
* cargo run: build and run

## Simple Language features

### Branching

* A pretty normal `if`/`else if`/`else`
* `match`


### Loops
* `loop { }` - infinite loop over the block
* `while <expression>`
* `for element in <generator>`


## Ownership

Two rules:
* Each **value** has a single **variable** that is its owner
* Once the owner goes out of scope, the memory for that value can be freed.

### Scope

It looks like rust is block scoped

### GC

For things allocated on the stack, I think that GC is pretty easy. When a block goes out of scope, you just move the stack pointer to before that block?

For things allocated on the heap, we need to keep track of the variables that are going out of scope and free their memory. But this means that each block of memory can only be owned by a single variable. This means that when we, for example, pass a value to a function, we need to be clear whether we are just giving access to that value or also passing ownership. In the first case, when the function ends nothing should be done with the memory. In the second, it needs to be freed.

It's actually more compilated than that!

```
let mut s1 = String::from("hello");
let mut s2 = s1;
```

Who owns the memory?
