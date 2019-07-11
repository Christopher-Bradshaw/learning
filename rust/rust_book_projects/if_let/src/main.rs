fn main() {
    let x = Some(2);
    println!("x is {:?}", x);

    let val = 1;
    println!("initially val is {} but it gets shadowed in the match", val);

    match x {
        // Note - we couldn't define some variable, set it equal to 1
        // and replace the 1 in the first arm with it, and have things
        // still work. Any variable in the match is freshly assigned (see val)
        Some(1) => println!("A some with value of 1"),
        Some(val) => println!("A some with value {}", val),
        None => println!("A none"),
    }

    // Doesn't match (because the some doesn't have value 1)
    if let Some(1) = x {
        println!("x is a some with value 1");
    }
    // Does match (because the some has value 2)
    if let Some(2) = x {
        println!("x is a some with value 2");
    }
    // Does match (and would match all somes)
    if let Some(val) = x {
        println!("x is a some with value {}", val);
    }
}
