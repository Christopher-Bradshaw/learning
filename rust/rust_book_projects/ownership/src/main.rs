fn main() {
    stack_scopes();
    pvar_vs_fvars();
    heap_scopes();

    println!("\nOWNERSHIP");
    let s1 = gives_ownership();
    takes_ownership(s1);
    // s1 is no longer available here
    let mut s2 = String::from("hi");
    s2 = takes_and_gives_ownership(s2);
    println!("{}", s2);

    println!("\nBORROWING");
    // It would be pretty annoying if we needed to return any argument
    // that we wanted to continue using in the parent function. The way around this
    // is to pass a reference to an object. This allows a function to refer to an object
    // without taking ownership of it. This is sometimes called borrowing.
    // Think of this reference just as you would think of a pointer - the syntax is the same as
    // C: & to get the reference (the address), * to dereference.
    let mut s2 = String::from("hi again");
    let s2_len = get_len(&s2);
    println!("Length of '{}' is {}", s2, s2_len);

    // By default references are immutable (you can't change what is being referred to).
    // But you can also create (one) mutable reference. You can't create more than one else
    // weird things might happen if they concurrently try to change things! You can create as many
    // immutable references as you like. But you can't have both immutable and a mutable reference.
    // Whew!
    cat(&mut s2, &String::from(" and again"));
    println!("{}", s2);
}

fn get_len(s: &String) -> usize {
    s.len()
} // s goes out of scope here, but what s points to does not.

// Borrow s1 as mutable. Append s2 to it.
fn cat(s1: &mut String, s2: &String) {
    s1.push_str(s2)
}

fn gives_ownership() -> String {
    String::from("hello")
}

fn takes_ownership(s1: String) {
    println!("{}", s1);
} // s1 is deallocated here because it is no longer in scope!

fn takes_and_gives_ownership(s1: String) -> String {
    s1
} // s1 is given back to the calling function and so is still in scope.

fn heap_scopes() {
    println!("\nHEAP_SCOPES");

    // String is a variable length, optionally mutable, string type.
    // Because of the variable length, it must be allocated on the heap.
    // This is different to string literals, which are fixed length.
    let mut s = String::from("hello");
    println!("{}", s);

    s.push_str(", world!");
    println!("{}", s);
}

// Pointer variables vs fixed variables
// I'm not entirely sure what to call this. I think it could also be,
// heap vs stack allocated variables. But basically, some variables store
// the value (fvars) and others store a pointer to a value. (Think python
// lists vs ints).
fn pvar_vs_fvars() {
    println!("\nPVARS_VS_FVARS");
    let mut x = 1;
    let y = x;
    x = x + 1;
    // Numbers are allocated on the stack and so y is a copy of x
    println!("x is {}, y is still {}", x, y);

    let s1 = String::from("hello");
    let mut s2 = s1;
    // The string type is really a struct with a len, capacity and a pointer to data
    // So both s1 and s2 point to the same data, *but who owns it*?
    // To combat this, s1 is *invalidated*. You can't use it, can't ever read from it.
    println!("s2 is {}, s1 has been invalidated", s2);
    s2.push_str("!");
    println!("s2 is {}, s1 has (still) been invalidated", s2);
}

// This is simple because all the variables are defined on the stack.
// They just get popped off when they are no longer in scope.
fn stack_scopes() {
    println!("\nSTACK_SCOPES");
    // x is not yet in scope here - it hasn't been defined!

    // Let's bring x into scope
    let x = 2;
    println!("{}: x in scope", x);

    {
        println!("{}: x still in scope in a sub-block", x);
        let y = 3;
        println!("{}: y in scope as it is defined in a sub-block", y);
    }
    // Actually rust's scopes are more detailed than block scope - as
    // x is not used below here, it is already out of scope! Pretty cool.

    println!("But y is not in scope here! Rust is block scoped");
    // This will generate a compiler error.
    // println!("{}: y in scope as it is defined in a sub-block", y);
}
