fn main() {
    casting()
}

fn casting() {
    let x1: i32 = 2;
    let x2 = x1 as i64;

    // We can't just do x2 == x1 because they are different types!
    // But we can cast the RHS into the same type as the LHS (as long
    // as that is reasonable, i.e. couldn't cast i64 -> i32).
    println!("x1 == x2? {}", x2 == x1.into());

    // The same as using as.
    let x2: f64 = x1.into();
    println!("x1 == x2? {}", x2 == x1.into());
}
