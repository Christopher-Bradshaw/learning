use std::sync::{Arc, Mutex};
use std::thread;

pub fn basic_mutexes() {
    // Let's say we have some data;
    let data = 5;

    // And we want to mutate this data from multiple threads. To do this we lock the data in a
    // mutex.
    let m = Mutex::new(data);
    println!("Mutex: {:?}", m);

    // Now, to access this data we need to first lock it. This .lock blocks until we get the lock.
    let lr = m.lock();
    println!("Lock taken: {:?}", lr);

    {
        // Assuming things went fine, we can get the value. But note that this is not just the raw
        // value. It is a MutexGuard<T>. The important property of this is that when it falls out
        // of scope, the lock is released! So we don't need to release the lock, we just need the
        // value to fall out of scope. If does this by implementing Drop.
        let mut val = lr.unwrap();
        println!("Underlying val: {:?}", val);

        *val = 2; // MutexGuard implements deref!
    }

    // See how we can get the lock again here, without an explicit unlock.
    {
        let val = m.lock().unwrap();
        println!("Underlying val is now {:?}", val);
    }
    // Or we could have just printed it.
    println!("Mutex: {:?}", m);
}

// Simple goal - have a couple of threads increment a counter
pub fn shared_mutex() {
    // This counter needs to be mutated by multiple threads. It therefore needs to be owned by all
    // of those threads. We need to support multiple ownership.
    // Usually we would do this with Rc. However, we need the reference count to be atomic (else
    // another thread might read an intermediate state). Arc (atomic Rc) does this.
    let counter = Arc::new(Mutex::new(0));

    // Create a clone (which remember is actually just a reference to the original data) for the
    // first thread to own.
    let counter_clone = Arc::clone(&counter);
    let thread1 = thread::spawn(move || {
        let mut val = counter_clone.lock().unwrap();
        *val += 1;
    });

    let counter_clone = Arc::clone(&counter);
    let thread2 = thread::spawn(move || {
        let mut val = counter_clone.lock().unwrap();
        *val += 1;
    });

    thread1.join().unwrap();
    thread2.join().unwrap();
    // All of the references to counter (except the initial one) will have gone out of scope at the end of the thread (as
    // the thread took ownership). Thus the number of counts will be 1.
    println!(
        "Counter is {:?}. Only 1 reference to it (right? {}).",
        counter,
        Arc::strong_count(&counter)
    );
}
