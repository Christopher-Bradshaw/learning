fn main() {
    variables();
    constants();
    shadowing();
}

fn variables() {
    println!("\nVARIABLES");
    let mut x = 5;
    println!("x is {}", x);
    // By default variables are immutable.
    // If we hadn't defined x as `mut` then this would be a compiler error
    x = 6;
    println!("x is {}", x);
}

fn constants() {
    println!("\nCONSTANTS");
    // Note that immutable variables are different to constants
    // I'm not 100% sure how though
    const SPEED_OF_LIGHT: f32 = 300_000.; // km/s
    println!("c is {}", SPEED_OF_LIGHT);
}

fn shadowing() {
    println!("\nSHADOWING");
    // Shadowing is allowed
    let x = 5; // immutable
    println!("x is {}", x);

    let x = 6;
    println!("x is {} (redefined)", x);

    let x = "cat";
    println!("x is {} (new type)", x);
}
