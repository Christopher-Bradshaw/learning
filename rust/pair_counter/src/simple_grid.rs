use crate::{checks, common};

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
    let (n_gridbox, bin_size) = grid_size(bins, box_size);

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
    let is_autocorr = common::is_autocorr(x2);
    let mut counts: Vec<u32> = vec![0; bins.len() - 1];

    for i in 0..n_gridbox {
        for j in 0..n_gridbox {
            counts = sum_vec(counts, do_counts(&grid, i, j, bins, box_size, is_autocorr));
        }
    }
    counts
}

fn do_counts(
    grid: &Vec<Vec<GridBox>>,
    x_index: i32,
    y_index: i32,
    bins: &[f32],
    box_size: f32,
    is_autocorr: bool,
) -> Vec<u32> {
    println!("{}", is_autocorr);
    let mut counts: Vec<u32> = vec![0; bins.len() - 1];
    let grid_item = &grid[x_index as usize][y_index as usize];

    for [inner_x_index, inner_y_index] in grid_item.owned_neighbors(is_autocorr).iter() {
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
            true => match *inner_x_index == x_index as u32 && *inner_y_index == y_index as u32 {
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
        println!("{:?}, {:?}", grid_item.x1, grid_item.y1);
        println!("{:?}, {:?}", pair_box.x2, pair_box.y2);
        println!("{:?}", sub_counts);
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

fn grid_size(bins: &[f32], box_size: f32) -> (i32, f32) {
    let n_gridbox = (box_size / bins[bins.len() - 1]) as i32;
    let bin_size = box_size / (n_gridbox as f32);
    return (n_gridbox, bin_size);
}

struct GridBox {
    x_index: i32,
    y_index: i32,
    n_gridbox: i32,
    x1: Vec<f32>,
    y1: Vec<f32>,
    x2: Vec<f32>,
    y2: Vec<f32>,
}

impl GridBox {
    pub fn new(x_index: i32, y_index: i32, n_gridbox: i32) -> GridBox {
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
        i: i32,
        j: i32,
        n_gridbox: i32,
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
    fn rem_euclid(&self, x: i32, div: i32) -> u32 {
        let mut t = x % div;
        if t < 0 {
            t += div;
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
            res[i][0] = self.rem_euclid(delta[i][0] + self.x_index, self.n_gridbox);
            res[i][1] = self.rem_euclid(delta[i][1] + self.y_index, self.n_gridbox);
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
            res[i][0] = self.rem_euclid(delta[i][0] + self.x_index, self.n_gridbox);
            res[i][1] = self.rem_euclid(delta[i][1] + self.y_index, self.n_gridbox);
        }
        return res;
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
