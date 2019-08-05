# Rust

Find all the docs using `rustup docs`. These include the book, std library docs, reference, and much more...

## Todo
* Continue reading the book
* Read some good code (something in https://github.com/rust-lang (rustlings, rls), ripgrep)?
* interop with python?
* Check https://arveknudsen.com/posts/practical-networked-applications-in-rust/module-1/

## Cargo

The rust build/dependency system. Useful commands,

* `cargo new`: Create a new project (with cargo.toml + src/main.rs)
* `cargo build`: Build
* `cargo check`: Check if it can be built (often a lot faster than actually building)
* `cargo run`: Build and run
* `cargo test <search_pattern>`: Run all tests (that match the optional pattern). Add `-- --nocapture` to see stdout.

## New Language Features

### Traits

A collection of methods for some unknown type. E.g. you can define a `trait Animal` that can `fn talk(&self)` and `fn name(&self)` (with optional default implementation).
Methods defined in the same trait can access other methods in that trait (e.g. the `Animal.talk` can hit `Animal.name`).
A data type (e.g. `Sheep`) can then implement a trait by implementing these methods in a `impl Animal for Sheep` block.
See [here](https://doc.rust-lang.org/rust-by-example/trait.html) for this full example.

For a more practicle, but very simple, use of traits, checkout out [Deref](https://doc.rust-lang.org/std/ops/trait.Deref.html)

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

A third rule?
* Only one mutable borrow

### Scope

It looks like rust is block scoped.

### GC

For things allocated on the stack, I think that GC is pretty easy. When a block goes out of scope, you just move the stack pointer to before that block?

For things allocated on the heap, we need to keep track of the variables that are going out of scope and free their memory. But this means that each block of memory can only be owned by a single variable. This means that when we, for example, pass a value to a function, we need to be clear whether we are just giving access to that value or also passing ownership. In the first case, when the function ends nothing should be done with the memory. In the second, it needs to be freed.

It's actually more compilated than that!

```
let mut s1 = String::from("hello");
let mut s2 = s1;
```

Who owns the memory?

## Debugging

Rust makes writing and running tests very easy. But what do you do when you can't work out why a specific test, let's call it `failing_test`, keeps failing?

When running tests you will see something like,
```
(ins)s:some_lib $ cargo test
    Finished dev [unoptimized + debuginfo] target(s) in 0.01s
     Running target/debug/deps/some_lib-04c82099363c591d

     ...
failures:
    some_file::test::failing_test
```

The file in the *Running* line is the binary that cargo has created to run the tests. We can re-run these tests, optionally selecting only a single test, with,

```
(ins)s:some_lib $ ./target/debug/deps/some_lib-04c82099363c591d [--test failing_test]
```

But because we have a standalone binary, we can also run them using our favourite debugging utility,

```
(ins)s:some_lib $ gdb
    ...
@(gdb) run --test failing_test
```

Before running, we can include a breakpoint wherever rust panics,
```
@(gdb) break rust_panic
Breakpoint 1 at 0x5555555c1683: file src/libstd/panicking.rs, line 523.
@(gdb) run
```

