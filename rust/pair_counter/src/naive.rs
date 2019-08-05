// Currently we only support autocorrelations
pub fn naive(x: &[f32], y: &[f32], bins: &[f32]) -> std::vec::Vec<i32> {
    checks(x, y, bins);

    let mut counts: Vec<i32> = vec![0; bins.len() - 1];

    let mut dist: f32;
    for i in 0..x.len() {
        for j in i + 1..x.len() {
            dist = compute_distance(x[i], y[i], x[j], y[j]);
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

fn checks(x: &[f32], y: &[f32], bins: &[f32]) {
    assert!(bins.len() > 1, "Need to provide at least 1 bin");
    assert_eq!(x.len(), y.len(), "X and Y should be the same length");
}

fn compute_distance(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    ((x1 - x2).powf(2.0) + (y1 - y2).powf(2.0)).sqrt()
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
        res = naive(&x, &y, &bins);
        assert_eq!(res, vec![2, 2]);

        let res = naive(&y, &x, &bins);
        assert_eq!(res, vec![2, 2]);
    }
}
