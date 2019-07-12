#[no_mangle]
pub extern fn doubler(x: f64) -> f64 {
    println!("Called with x={}", x);
    x * 2.
}
