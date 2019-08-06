extern crate ord_subset;
use ord_subset::OrdSubsetSliceExt;

pub fn is_autocorr(x2: Option<&[f32]>) -> bool {
    match x2 {
        None => true,
        Some(_) => false,
    }
}

// This is really slow. Something like https://arxiv.org/pdf/1506.08620.pdf could send it to ~0
fn get_bin_idx(dist: f32, bins: &[f32]) -> usize {
    return match bins.ord_subset_binary_search(&dist) {
        Err(idx) => idx,
        Ok(idx) => idx,
    };
}

pub fn count_pairs_crosscorr(
    x1: &[f32],
    y1: &[f32],
    x2: &[f32],
    y2: &[f32],
    bins: &[f32],
    box_size: f32,
    needs_pbc: bool,
) -> Vec<u32> {
    let mut counts: Vec<u32> = vec![0; bins.len() + 1];
    let mut dist;

    for i in 0..x1.len() {
        for j in 0..x2.len() {
            if needs_pbc {
                dist = compute_distance_with_pbc(x1[i], y1[i], x2[j], y2[j], box_size);
            } else {
                dist = compute_distance(x1[i], y1[i], x2[j], y2[j]);
            }
            counts[get_bin_idx(dist, bins)] += 1;
        }
    }
    counts
}

pub fn count_pairs_autocorr(x1: &[f32], y1: &[f32], bins: &[f32]) -> Vec<u32> {
    let mut counts: Vec<u32> = vec![0; bins.len() + 1];
    let mut dist;

    for i in 0..x1.len() {
        for j in (i + 1)..x1.len() {
            dist = compute_distance(x1[i], y1[i], x1[j], y1[j]);
            counts[get_bin_idx(dist, bins)] += 1;
        }
    }
    counts
}

// If we know we don't need pbc, this is much faster
fn compute_distance(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    ((x1 - x2).powf(2.0) + (y1 - y2).powf(2.0)).sqrt()
}

fn compute_distance_with_pbc(x1: f32, y1: f32, x2: f32, y2: f32, box_size: f32) -> f32 {
    let dx = [
        (x1 - x2).abs(),
        (x1 + box_size - x2).abs(),
        (x1 - box_size - x2).abs(),
    ];
    let dy = [
        (y1 - y2).abs(),
        (y1 + box_size - y2).abs(),
        (y1 - box_size - y2).abs(),
    ];
    fn min(vals: &[f32]) -> f32 {
        let mut min = vals[0];
        for v in vals[1..].iter() {
            min = min.min(*v);
        }
        min
    }
    (min(&dx).powf(2.0) + min(&dy).powf(2.0)).sqrt()
}

#[cfg(test)]
mod test {
    use crate::common::get_bin_idx;

    #[test]
    fn get_bin_idx_test() {
        let bins = [1., 2., 3., 4., 5.];
        assert_eq!(get_bin_idx(2.5, &bins), 2);
    }
}
