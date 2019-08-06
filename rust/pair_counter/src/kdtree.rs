use crate::kdtree_impl::KDNode;
use crate::{checks, common};

pub fn kdtree(
    x1: &[f32],
    y1: &[f32],
    x2: Option<&[f32]>,
    y2: Option<&[f32]>,
    bins: &[f32],
    box_size: f32,
) -> Vec<u32> {
    checks::all_checks(x1, y1, x2, y2, bins, box_size);

    // Construct tree
    let tree = KDNode::new_tree(x1, y1);

    // Do counts
    let mut counts: Vec<u32> = vec![0; bins.len() + 1];
    counts[1..bins.len()].to_vec()
}
