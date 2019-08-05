use crate::{checks, common};

pub fn naive(
    x1: &[f32],
    y1: &[f32],
    x2: Option<&[f32]>,
    y2: Option<&[f32]>,
    bins: &[f32],
    box_size: f32,
) -> Vec<u32> {
    checks::all_checks(x1, y1, x2, y2, bins, box_size);

    return match common::is_autocorr(x2) {
        true => count_pairs_autocorr(x1, y1, bins, box_size),
        false => count_pairs_crosscorr(x1, y1, x2.unwrap(), y2.unwrap(), bins, box_size),
    };
}
fn count_pairs_crosscorr(
    x1: &[f32],
    y1: &[f32],
    x2: &[f32],
    y2: &[f32],
    bins: &[f32],
    box_size: f32,
) -> Vec<u32> {
    let mut counts: Vec<u32> = vec![0; bins.len() - 1];
    let mut dist;

    for i in 0..x1.len() {
        for j in 0..x2.len() {
            dist = compute_distance_with_pbc(x1[i], y1[i], x2[j], y2[j], box_size);
            if dist < bins[0] {
                continue;
            }
            for k in 1..bins.len() {
                if dist < bins[k] {
                    counts[k - 1] += 1;
                    break;
                }
            }
        }
    }
    counts
}

fn count_pairs_autocorr(x1: &[f32], y1: &[f32], bins: &[f32], box_size: f32) -> Vec<u32> {
    let mut counts: Vec<u32> = vec![0; bins.len() - 1];
    let mut dist;

    for i in 0..x1.len() {
        for j in (i + 1)..x1.len() {
            dist = compute_distance_with_pbc(x1[i], y1[i], x1[j], y1[j], box_size);
            if dist < bins[0] {
                continue;
            }
            for k in 1..bins.len() {
                if dist < bins[k] {
                    counts[k - 1] += 1;
                    break;
                }
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

#[cfg(test)]
mod test {
    use crate::naive::naive;

    #[test]
    fn basic_autocorr() {
        let x = vec![0.5, 1.0, 2.0, 3.1, 5.0];
        let y = vec![0.0; x.len()];
        let bins = vec![0.0, 1.01, 1.6];

        let res;

        // Test switching x and y
        res = naive(&x, &y, None, None, &bins, 1000.);
        assert_eq!(res, vec![2, 2]);

        let res = naive(&y, &x, None, None, &bins, 1000.);
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
        // Test switching x and y
        res = naive(&x1, &y1, Some(&x2), Some(&y2), &bins, 1000.);
        assert_eq!(res, vec![2, 1]);
    }

    #[test]
    fn distances_across_boundaries() {
        let x = vec![1., 999.];
        let y = vec![999., 1.];
        let bins = vec![0., 3.];

        let res = naive(&x, &y, None, None, &bins, 1000.);
        assert_eq!(res, vec![1]);
    }
}
