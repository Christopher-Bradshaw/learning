fn main() {
    basic_closures();
    capturing_details();
}

fn basic_closures() {
    // The way to define an anonymous function (which we then
    // bind to a variable. Note that some of the type annotations
    // here are not necessary - they can be inferred.
    let add_one = |i: i32| -> i32 { i + 1 };
    println!("adding one to 2 gives you {}.", add_one(2));

    // The body of the closure captures the enclosing environment
    // Also see the type inference.
    let x = 1;
    let add_one_to_x = || x + 1;
    println!(
        "We get x from the enviroment and add one to get {}.",
        add_one_to_x()
    );

    // The simplest closure possible
    let one = || 1;
    println!("One = {}.", one());
    // Without assigning! Note that we shadow the external x.
    println!("Two = {}.", (|x| x)(2));
}

fn capturing_details() {
    // This needs to be mutable as the closure will change it.
    let mut count = 0;
    // This closure needs to be mutable because it stores a mutable (count)
    // inside! An so calling it changes it!
    // Technically, we could increment count by either taking ownership, or
    // passing it as a mutable reference. The latter is less restrictive and so
    // rust does that.
    let mut inc = || {
        count += 1;
        println!("Count now at {}", count);
    };

    inc();
    inc();

    count += 1;
    println!("Count now at {} (outside closure)", count);
    // If you uncomment this, the code won't compile, because:
    // We have borrowed a mutable count in inc. Without this last call to inc,
    // inc has gone out of scope and so that borrow is gone.
    // With this call, we would then be mutating something that is mutably borrowed
    // This is because we would then have two mutable references which is bad bad bad
    // inc();

    use std::mem;
    let box_val = Box::new(1);
    let take_ownership = || {
        mem::drop(box_val);
    };

    take_ownership();
    // While before we could get away with a mutable borrow. This closure has to take ownership.
    // Thus subsequent calls are illegal, as the main scope (from which the closure will capture
    // the value) no longer has it!
    //take_ownership();
}
