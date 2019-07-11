// This trait gives us rudimentary printing
#[derive(Debug)]
struct Rectangle {
    length: u32,
    width: u32,
}

fn main() {
    let r = create_rectange(20, 10);
    println!("r has an area of {}", rectangle_area(&r));

    // This is all well and good, but it isn't very oopy. Our data and functions are
    // in separate places. Can we get ome constructors/methods etc? Try impl.
    println!("r has an area of {}", r.area());
    let r2 = create_rectange(2, 1);
    println!("r can hold r2 inside it? {}", r.can_hold_inside(&r2));
    println!("r2 can hold r inside it? {}", r2.can_hold_inside(&r));

    let r3 = Rectangle::square(10);
    println!("r3 is a square, {:#?}", r3);
}

impl Rectangle {
    // Functions whose first argument is not `&self` are "associatd functions".
    // These are often useful as constructors
    fn square(s: u32) -> Rectangle {
        Rectangle {
            length: s,
            width: s,
        }
    }
    // Functions whose first argument is `&self` work like methods.
    fn area(&self) -> u32 {
        self.length * self.width
    }

    fn can_hold_inside(&self, r: &Rectangle) -> bool {
        (r.length < self.length) & (r.width < self.width)
    }
}

// Not that oopy
fn create_rectange(length: u32, width: u32) -> Rectangle {
    // We can do this shorthand because the variables names are the same as the field names
    Rectangle { length, width }
}

fn rectangle_area(r: &Rectangle) -> u32 {
    r.length * r.width
}
