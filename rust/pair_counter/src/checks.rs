pub fn all_checks(
    x1: &[f32],
    y1: &[f32],
    x2_opt: Option<&[f32]>,
    y2_opt: Option<&[f32]>,
    bins: &[f32],
    box_size: f32,
) {
    assert!(bins.len() > 1, "Need to provide at least 1 bin");
    assert_eq!(x1.len(), y1.len(), "X and Y should be the same length");

    let mut x2: &[f32] = &[];
    let mut y2: &[f32] = &[];

    match x2_opt {
        None => {
            assert!(y2_opt == None, "Can't have x2 be None and y2 to Some");
        }
        Some(v) => {
            x2 = v;
            y2 = y2_opt.unwrap();
        }
    }
    assert_eq!(x2.len(), y2.len(), "X and Y should be the same length");
    for d in [x1, y1, x2, y2].iter() {
        for i in 0..d.len() {
            assert!(d[i] < box_size);
        }
    }
    for i in 1..bins.len() {
        assert!(
            bins[i] > bins[i - 1],
            "Bins need to be monotonically increasing"
        );
    }
}

pub fn grid_checks(bins: &[f32], box_size: f32) {
    assert!(box_size / bins[bins.len() - 1] > 3.);
}
