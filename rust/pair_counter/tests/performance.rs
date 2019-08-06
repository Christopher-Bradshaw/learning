extern crate rand;

use pair_counter;
use rand::Rng;
use std::ops::Sub;
use std::time::Instant;

fn gen_data(n_items: usize, box_size: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut d: Vec<f32> = vec![];
    for i in 0..n_items {
        d.push(rng.gen());
        d[i] *= box_size;
    }
    return d;
}

// npts: time (ms)
// 1e3: 50
// 2e3: 113
// 4e3: 315
// 8e3: 990
// 16e3: 3500
// 32e3: 13000
#[test]
fn simple_grid_perf() {
    // These choices are taken to roughly match https://gist.github.com/manodeep/cffd9a5d77510e43ccf0
    let box_size = 420.;
    let n_items: usize = 8000;
    let mut bins = vec![
        0.001, 0.0015, 0.0023, 0.0036, 0.0056, 0.0087, 0.013, 0.020, 0.032, 0.05,
    ];
    for i in 0..bins.len() {
        bins[i] = bins[i] * box_size;
    }

    let x1 = gen_data(n_items, box_size);
    let y1 = gen_data(n_items, box_size);

    let before = Instant::now();
    let res_grid = pair_counter::simple_grid::simple_grid(&x1, &y1, None, None, &bins, box_size);
    let after = Instant::now();
    println!("{:?}", after.sub(before).as_micros());
    println!("{:?}", res_grid);
}
