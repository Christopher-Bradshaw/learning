use crate::{checks, common};

struct GridBox {
    x_index: i64,
    y_index: i64,
    n_gridbox: i64,
    x1: Vec<f32>,
    y1: Vec<f32>,
    x2: Vec<f32>,
    y2: Vec<f32>,
}

pub fn simple_grid(
    x1: &[f32],
    y1: &[f32],
    x2: Option<&[f32]>,
    y2: Option<&[f32]>,
    bins: &[f32],
    box_size: f32,
) -> Vec<i32> {
    checks::all_checks(x1, y1, x2, y2, bins, box_size);
    checks::grid_checks(bins, box_size);

    let is_autocorr = common::is_autocorr(x2);

    // Construct grid
    let (n_gridbox, bin_size) = grid_size(bins, box_size);

    let mut grid: Vec<Vec<GridBox>> = vec![];

    for i in 0..n_gridbox {
        grid.push(vec![]);
        for j in 0..n_gridbox {
            grid[i as usize].push(GridBox::new_with_items(
                i,
                j,
                n_gridbox,
                bin_size,
                is_autocorr,
                x1,
                y1,
                x2,
                y2,
            ));
        }
    }

    // Do counts
    let mut counts: Vec<i32> = vec![0; bins.len() - 1];

    for i in 0..n_gridbox {
        for j in 0..n_gridbox {
            counts = sum_vec(counts, do_counts(&grid, i, j, bins, box_size, is_autocorr));
        }
    }
    counts
}

fn do_counts(
    grid: &Vec<Vec<GridBox>>,
    x_index: i64,
    y_index: i64,
    bins: &[f32],
    box_size: f32,
    is_autocorr: bool,
) -> Vec<i32> {
    let mut counts: Vec<i32> = vec![0; bins.len() - 1];
    let grid_item = &grid[x_index as usize][y_index as usize];

    for [inner_x_index, inner_y_index] in grid_item.owned_neighbors().iter() {
        let pair_box = &grid[*inner_x_index as usize][*inner_y_index as usize];
        let sub_counts = match is_autocorr {
            // If crosscorr -> always do crosscorr
            false => common::count_pairs_crosscorr(
                &grid_item.x1,
                &grid_item.y1,
                &pair_box.x2,
                &pair_box.y2,
                bins,
                box_size,
            ),
            // If autocorr, do autocorr if between the same mini-bin, else do cross corr
            true => match *inner_x_index == x_index && *inner_y_index == y_index {
                true => common::count_pairs_autocorr(&grid_item.x1, &grid_item.y1, bins, box_size),
                false => common::count_pairs_crosscorr(
                    &grid_item.x1,
                    &grid_item.y1,
                    &pair_box.x1,
                    &pair_box.y1,
                    bins,
                    box_size,
                ),
            },
        };
        counts = sum_vec(counts, sub_counts);
    }
    counts
}

fn sum_vec(mut a: Vec<i32>, b: Vec<i32>) -> Vec<i32> {
    for i in 0..b.len() {
        a[i] += b[i];
    }
    a
}

fn grid_size(bins: &[f32], box_size: f32) -> (i64, f32) {
    let n_gridbox = (box_size / bins[bins.len() - 1]) as i64;
    let bin_size = box_size / (n_gridbox as f32);
    return (n_gridbox, bin_size);
}

impl GridBox {
    pub fn new(x_index: i64, y_index: i64, n_gridbox: i64) -> GridBox {
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
        i: i64,
        j: i64,
        n_gridbox: i64,
        bins_size: f32,
        is_autocorr: bool,
        x1: &[f32],
        y1: &[f32],
        x2_opt: Option<&[f32]>,
        y2_opt: Option<&[f32]>,
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
        if !is_autocorr {
            let x2 = x2_opt.unwrap();
            let y2 = y2_opt.unwrap();
            for i in 0..x2.len() {
                if (x2[i] >= x_min) && (x2[i] < x_max) && (y2[i] >= y_min) && (y2[i] < y_max) {
                    b.x2.push(x2[i]);
                    b.y2.push(y2[i]);
                }
            }
        }
        return b;
    }

    // GridBoxs own themselves + the 4 neighbors below and to the right. This means that
    // they are responsible for checking those bins for pairs.
    pub fn owned_neighbors(&self) -> [[i64; 2]; 5] {
        [
            [
                (self.x_index) % self.n_gridbox,
                (self.y_index) % self.n_gridbox,
            ], // Self
            [
                (self.x_index + 1) % self.n_gridbox,
                (self.y_index) % self.n_gridbox,
            ], // Right
            [
                (self.x_index + 1) % self.n_gridbox,
                (self.y_index + 1) % self.n_gridbox,
            ], // Right and down
            [
                (self.x_index) % self.n_gridbox,
                (self.y_index + 1) % self.n_gridbox,
            ], // Down
            [
                if self.x_index == 0 {
                    self.n_gridbox - 1
                } else {
                    (self.x_index - 1) % self.n_gridbox
                },
                (self.y_index + 1) % self.n_gridbox,
            ], // Left and Down
        ]
    }
}

#[cfg(test)]
mod test {
    // use crate::naive::naive;
    use crate::simple_grid::{simple_grid, GridBox};

    #[test]
    fn basic_autocorr() {
        let x = vec![0.5, 1.0, 2.0, 3.1, 5.0];
        let y = vec![0.0; x.len()];
        let bins = vec![0.0, 1.01, 1.6];

        let res;

        // Test switching x and y
        res = simple_grid(&x, &y, None, None, &bins, 10.);
        assert_eq!(res, vec![2, 2]);

        let res = simple_grid(&y, &x, None, None, &bins, 10.);
        assert_eq!(res, vec![2, 2]);
    }

    #[test]
    fn owned_neighbors() {
        assert_eq!(
            GridBox::new(4, 4, 10).owned_neighbors(),
            [[4, 4], [5, 4], [5, 5], [4, 5], [3, 5]]
        );
        assert_eq!(
            GridBox::new(0, 0, 10).owned_neighbors(),
            [[0, 0], [1, 0], [1, 1], [0, 1], [9, 1]]
        );
        assert_eq!(
            GridBox::new(0, 9, 10).owned_neighbors(),
            [[0, 9], [1, 9], [1, 0], [0, 0], [9, 0]]
        );
        assert_eq!(
            GridBox::new(9, 9, 10).owned_neighbors(),
            [[9, 9], [0, 9], [0, 0], [9, 0], [8, 0]]
        );
    }
}
