use crate::{checks, common};
use std::convert::TryFrom;

pub fn simple_grid(
    x1: &[f32],
    y1: &[f32],
    x2: Option<&[f32]>,
    y2: Option<&[f32]>,
    bins: &[f32],
    box_size: f32,
) -> Vec<u32> {
    checks::all_checks(x1, y1, x2, y2, bins, box_size);
    checks::grid_checks(bins, box_size);

    // Construct grid
    let n_gridbox = (box_size / bins[bins.len() - 1]) as u32;
    let bin_size = box_size / (n_gridbox as f32);

    let mut grid: Vec<Vec<GridBox>> = vec![];

    for i in 0..n_gridbox {
        grid.push(vec![]);
        for j in 0..n_gridbox {
            grid[i as usize].push(GridBox::new_with_items(
                i, j, n_gridbox, bin_size, x1, y1, x2, y2,
            ));
        }
    }

    // Do counts
    // Zeroth bin is for anything inside the inside bin. nth bin is anything outside the outside.
    // We throw both away before returning.
    let mut counts: Vec<u32> = vec![0; bins.len() + 1];
    let is_autocorr = common::is_autocorr(x2);

    for i in 0..n_gridbox {
        for j in 0..n_gridbox {
            counts = sum_vec(counts, do_counts(&grid, i, j, bins, box_size, is_autocorr));
        }
    }
    counts[1..bins.len()].to_vec()
}

fn do_counts(
    grid: &Vec<Vec<GridBox>>,
    x_index: u32,
    y_index: u32,
    bins: &[f32],
    box_size: f32,
    is_autocorr: bool,
) -> Vec<u32> {
    let mut counts: Vec<u32> = vec![0; bins.len() + 1];
    let grid_item = &grid[x_index as usize][y_index as usize];

    for [inner_x_index, inner_y_index] in grid_item.owned_neighbors(is_autocorr).iter() {
        let pair_box = &grid[*inner_x_index as usize][*inner_y_index as usize];
        let needs_pbc = (*inner_x_index as i64 - x_index as i64).abs() > 1
            || (*inner_y_index as i64 - y_index as i64).abs() > 1;
        let sub_counts = match is_autocorr {
            // If crosscorr -> always do crosscorr
            false => common::count_pairs_crosscorr(
                &grid_item.x1,
                &grid_item.y1,
                &pair_box.x2,
                &pair_box.y2,
                bins,
                box_size,
                needs_pbc,
            ),
            // If autocorr, do autocorr if between the same mini-bin, else do cross corr
            true => match *inner_x_index == x_index as u32 && *inner_y_index == y_index as u32 {
                true => common::count_pairs_autocorr(&grid_item.x1, &grid_item.y1, bins),
                false => common::count_pairs_crosscorr(
                    &grid_item.x1,
                    &grid_item.y1,
                    &pair_box.x1,
                    &pair_box.y1,
                    bins,
                    box_size,
                    needs_pbc,
                ),
            },
        };
        counts = sum_vec(counts, sub_counts);
    }
    counts
}

fn sum_vec(mut a: Vec<u32>, b: Vec<u32>) -> Vec<u32> {
    for i in 0..b.len() {
        a[i] += b[i];
    }
    a
}

struct GridBox {
    x_index: u32,
    y_index: u32,
    n_gridbox: u32,
    x1: Vec<f32>,
    y1: Vec<f32>,
    x2: Vec<f32>,
    y2: Vec<f32>,
}

impl GridBox {
    pub fn new(x_index: u32, y_index: u32, n_gridbox: u32) -> GridBox {
        GridBox {
            x_index,
            y_index,
            n_gridbox,
            x1: vec![],
            y1: vec![],
            x2: vec![],
            y2: vec![],
        }
    }

    pub fn new_with_items(
        i: u32,
        j: u32,
        n_gridbox: u32,
        bins_size: f32,
        x1: &[f32],
        y1: &[f32],
        x2: Option<&[f32]>,
        y2: Option<&[f32]>,
    ) -> GridBox {
        let mut b = GridBox::new(i, j, n_gridbox);
        let (x_min, x_max) = (bins_size * i as f32, bins_size * ((i + 1) as f32));
        let (y_min, y_max) = (bins_size * j as f32, bins_size * ((j + 1) as f32));

        for i in 0..x1.len() {
            if (x1[i] >= x_min) && (x1[i] < x_max) && (y1[i] >= y_min) && (y1[i] < y_max) {
                b.x1.push(x1[i]);
                b.y1.push(y1[i]);
            }
        }
        match x2 {
            Some(x2) => {
                let y2 = y2.unwrap();
                for i in 0..x2.len() {
                    if (x2[i] >= x_min) && (x2[i] < x_max) && (y2[i] >= y_min) && (y2[i] < y_max) {
                        b.x2.push(x2[i]);
                        b.y2.push(y2[i]);
                    }
                }
            }
            None => (),
        }

        return b;
    }

    // This will soon be in stable
    fn rem_euclid(&self, x: i32, div: u32) -> u32 {
        let mut t: i32 = x % div as i32;
        if t < 0 {
            t += div as i32;
        }
        t as u32
    }

    pub fn owned_neighbors(&self, is_autocorr: bool) -> Vec<[u32; 2]> {
        if is_autocorr {
            return self.autocorr_owned_neighbors().to_vec();
        }
        return self.crosscorr_owned_neighbors().to_vec();
    }

    fn autocorr_owned_neighbors(&self) -> [[u32; 2]; 5] {
        let delta: [[i32; 2]; 5] = [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1]];
        let mut res: [[u32; 2]; 5] = [[0, 0]; 5];
        for i in 0..delta.len() {
            res[i][0] = self.rem_euclid(
                i32::try_from(self.x_index).unwrap() + delta[i][0],
                self.n_gridbox,
            );
            res[i][1] = self.rem_euclid(
                i32::try_from(self.y_index).unwrap() + delta[i][1],
                self.n_gridbox,
            );
        }
        return res;
    }

    fn crosscorr_owned_neighbors(&self) -> [[u32; 2]; 9] {
        let delta: [[i32; 2]; 9] = [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [-1, 1],
            [-1, 0],
            [-1, -1],
            [0, -1],
            [1, -1],
        ];
        let mut res: [[u32; 2]; 9] = [[0, 0]; 9];
        for i in 0..delta.len() {
            res[i][0] = self.rem_euclid(
                i32::try_from(self.x_index).unwrap() + delta[i][0],
                self.n_gridbox,
            );
            res[i][1] = self.rem_euclid(
                i32::try_from(self.y_index).unwrap() + delta[i][1],
                self.n_gridbox,
            );
        }
        return res;
    }
}

#[cfg(test)]
mod test {
    use crate::naive::naive;
    use crate::simple_grid::{simple_grid, GridBox};
    extern crate rand;

    use rand::Rng;

    #[test]
    fn basic_autocorr() {
        let x = vec![0.5, 1.0, 2.0, 3.1, 5.0];
        let y = vec![0.0; x.len()];
        let bins = vec![0.0, 1.01, 1.6];

        let res;

        res = simple_grid(&x, &y, None, None, &bins, 10.);
        assert_eq!(res, vec![2, 2]);
        let res = simple_grid(&y, &x, None, None, &bins, 10.);
        assert_eq!(res, vec![2, 2]);
    }

    #[test]
    fn basic_crosscorr() {
        let x1 = vec![0.5, 1.0, 2.0, 3.1, 5.0];
        let y1 = vec![0.0; x1.len()];
        let x2 = vec![0.75];
        let y2 = vec![0.0];
        let bins = vec![0.0, 1.01, 1.6];

        let res;
        res = simple_grid(&x1, &y1, Some(&x2), Some(&y2), &bins, 7.9);
        assert_eq!(res, vec![2, 1]);
    }

    fn gen_data(n_items: usize, box_size: f32) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let mut d: Vec<f32> = vec![];
        for i in 0..n_items {
            d.push(rng.gen());
            d[i] *= box_size;
        }
        return d;
    }

    #[test]
    fn large_autocorr() {
        let box_size = 100.;
        let n_items: usize = 300;

        let x1 = gen_data(n_items, box_size);
        let y1 = gen_data(n_items, box_size);

        let bins = vec![0.0, 1., 2., 3., 4., 5.];
        let res_grid = simple_grid(&x1, &y1, None, None, &bins, box_size);
        let res_naive = naive(&x1, &y1, None, None, &bins, box_size);
        assert_eq!(res_grid, res_naive);
    }

    #[test]
    fn large_crosscorr() {
        let box_size = 100.;
        let n_items: usize = 300;

        let x1 = gen_data(n_items, box_size);
        let y1 = gen_data(n_items, box_size);
        let x2 = gen_data(n_items * 2, box_size);
        let y2 = gen_data(n_items * 2, box_size);

        let bins = vec![0.0, 1., 2., 3., 4., 5.];
        let res_naive = naive(&x1, &y1, Some(&x2), Some(&y2), &bins, box_size);
        let res_grid_1 = simple_grid(&x1, &y1, Some(&x2), Some(&y2), &bins, box_size);
        assert_eq!(res_grid_1, res_naive);
        let res_grid_2 = simple_grid(&x2, &y2, Some(&x1), Some(&y1), &bins, box_size);
        assert_eq!(res_grid_2, res_naive);
    }

    #[test]
    fn autocorr_owned_neighbors() {
        assert_eq!(
            GridBox::new(4, 4, 10).autocorr_owned_neighbors(),
            [[4, 4], [5, 4], [5, 5], [4, 5], [3, 5]]
        );
        assert_eq!(
            GridBox::new(0, 0, 10).autocorr_owned_neighbors(),
            [[0, 0], [1, 0], [1, 1], [0, 1], [9, 1]]
        );
        assert_eq!(
            GridBox::new(0, 9, 10).autocorr_owned_neighbors(),
            [[0, 9], [1, 9], [1, 0], [0, 0], [9, 0]]
        );
        assert_eq!(
            GridBox::new(9, 9, 10).autocorr_owned_neighbors(),
            [[9, 9], [0, 9], [0, 0], [9, 0], [8, 0]]
        );
    }
}
