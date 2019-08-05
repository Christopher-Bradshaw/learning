pub fn is_autocorr(x2: Option<&[f32]>) -> bool {
    match x2 {
        None => true,
        Some(_) => false,
    }
}

fn get_bin_idx(dist: f32, bins: &[f32]) -> Option<usize> {
    if dist < bins[0] {
        return None;
    }
    for i in 1..bins.len() {
        if dist < bins[i] {
            return Some(i - 1);
        }
    }
    return None;
}

pub fn count_pairs_crosscorr(
    x1: &[f32],
    y1: &[f32],
    x2: &[f32],
    y2: &[f32],
    bins: &[f32],
    box_size: f32,
) -> Vec<i32> {
    let mut counts: Vec<i32> = vec![0; bins.len() - 1];
    let mut dist;

    for i in 0..x1.len() {
        for j in 0..x2.len() {
            dist = compute_distance_with_pbc(x1[i], y1[i], x2[j], y2[j], box_size);
            match get_bin_idx(dist, bins) {
                Some(idx) => counts[idx] += 1,
                None => (),
            }
        }
    }
    counts
}

pub fn count_pairs_autocorr(x1: &[f32], y1: &[f32], bins: &[f32], box_size: f32) -> Vec<i32> {
    let mut counts: Vec<i32> = vec![0; bins.len() - 1];
    let mut dist;

    for i in 0..x1.len() {
        for j in (i + 1)..x1.len() {
            dist = compute_distance_with_pbc(x1[i], y1[i], x1[j], y1[j], box_size);
            match get_bin_idx(dist, bins) {
                Some(idx) => counts[idx] += 1,
                None => (),
            }
        }
    }
    counts
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
