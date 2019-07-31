use std::thread;
use std::time::Duration;

mod mutexes;

fn main() {
    basic_threads();
    threads_with_ownership();

    mutexes::basic_mutexes();
    mutexes::shared_mutex();
}

fn basic_threads() {
    let handle = thread::spawn(|| {
        for i in 1..10 {
            println!("hi number {} from the spawned thread!", i);
            thread::sleep(Duration::from_millis(1));
        }
    });

    for i in 1..5 {
        println!("hi number {} from the main thread!", i);
        thread::sleep(Duration::from_millis(1));
    }

    handle.join().unwrap(); // Wait for this thread to end
}

fn threads_with_ownership() {
    let v = vec![1, 2, 3];

    // You can't borrow v because this thread might outlive v!
    // You need to take ownership of v
    let handle = thread::spawn(move || {
        println!("This is a vector {:?}", v);
    });

    // Now we can't use v down here because, the other thread has taken ownership of it.

    handle.join().unwrap();

    // And down here, we can't use v because it will have been cleaned up after going out of scope
    // when that other thread finished
}
