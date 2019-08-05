pub fn naive(
    x1: &[f32],
    y1: &[f32],
    x2: &[f32],
    y2: &[f32],
    bins: &[f32],
    box_size: f32,
) -> std::vec::Vec<i32> {
    checks(x1, y1, x2, y2, bins, box_size);

    let mut counts: Vec<i32> = vec![0; bins.len() - 1];

    let mut dist: f32;
    for i in 0..x1.len() {
        for j in i + 1..x2.len() {
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
    return counts;
}

fn checks(x1: &[f32], y1: &[f32], x2: &[f32], y2: &[f32], bins: &[f32], box_size: f32) {
    assert!(bins.len() > 1, "Need to provide at least 1 bin");
    assert_eq!(x1.len(), y1.len(), "X and Y should be the same length");
    assert_eq!(x2.len(), y2.len(), "X and Y should be the same length");
    for d in [x1, y1, x2, y2].iter() {
        for i in 0..d.len() {
            assert!(d[i] < box_size);
        }
    }
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
    fn basic_funcionality() {
        let x = vec![0.5, 1.0, 2.0, 3.1, 5.0];
        let y = vec![0.0; x.len()];
        let bins = vec![0.0, 1.01, 1.6];

        let res;

        // Test switching x and y
        res = naive(&x, &y, &x, &y, &bins, 1000.);
        assert_eq!(res, vec![2, 2]);

        let res = naive(&y, &x, &y, &x, &bins, 1000.);
        assert_eq!(res, vec![2, 2]);
    }

    #[test]
    fn distances_across_boundaries() {
        let x = vec![1., 999.];
        let y = vec![999., 1.];
        let bins = vec![0., 3.];

        let res = naive(&x, &y, &x, &y, &bins, 1000.);
        assert_eq!(res, vec![1]);
    }
}
