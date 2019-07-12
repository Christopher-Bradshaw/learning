# Rust as python accelerator

In the same way that we can use C to speed up python, we can also use rust.

Go back and read that section before trying this!

## Create rust library

See [here](https://doc.rust-lang.org/1.2.0/book/rust-inside-other-languages.html#a-rust-library)

* Generate a new rust library `cargo new engine --lib`
* Modify `engine/src/lib.rs` to contain some publically exported functions. These need to be `#[no_mangle]` and `pub extern`.
* Add `crate-type = ["dylib"]` under `[lib]` in cargo.toml
* `cargo build --release`


This gives us `engine/target/release/libengine.so`. Which we can `objdump -t engine/target/release/libengine.so  | grep doubler` to see!

## Non cython

See [the rust example](https://doc.rust-lang.org/1.2.0/book/rust-inside-other-languages.html#python) and [the ctypes docs](https://docs.python.org/3/library/ctypes.html).

Then look at [non_cython.py](./non_cython.py).
