use statistical;

type More = Option<Box<KDNode>>;
type Vals = Option<Vec<f32>>;

#[derive(Debug, Copy, Clone)]
enum CutDim {
    X,
    Y,
}

#[derive(Debug)]
pub struct KDNode {
    // The values it splits on
    x: f32,
    y: f32,
    // Subtree
    right: More,
    left: More,
    // Vals
    x_vals: Vals,
    y_vals: Vals,
    // Cut dim
    cut_dim: CutDim,
}

impl KDNode {
    pub fn new_tree(x: &[f32], y: &[f32]) -> KDNode {
        return KDNode::new_subtree(x, y, CutDim::X);
    }

    fn new_node(x: f32, y: f32, cut_dim: CutDim) -> KDNode {
        KDNode {
            x,
            y,
            right: None,
            left: None,
            x_vals: None,
            y_vals: None,
            cut_dim,
        }
    }
    fn new_leaf_node(x_vals: &[f32], y_vals: &[f32]) -> KDNode {
        KDNode {
            x: 0.,
            y: 0.,
            right: None,
            left: None,
            x_vals: Some(x_vals.to_vec()),
            y_vals: Some(y_vals.to_vec()),
            cut_dim: CutDim::X,
        }
    }

    fn new_subtree(x: &[f32], y: &[f32], parent_cut_dim: CutDim) -> KDNode {
        // Leafnode
        if x.len() <= 2 {
            return KDNode::new_leaf_node(x, y);
        }

        // Treenode
        let med_x = statistical::median(x);
        let med_y = statistical::median(y);
        // This node is going to be cut on the opposite dim as its parent
        let cut_dim = match parent_cut_dim {
            CutDim::X => CutDim::Y,
            CutDim::Y => CutDim::X,
        };
        let mut node = KDNode::new_node(med_x, med_y, cut_dim);
        let (lx, ly, rx, ry) = KDNode::divide_data(x, y, med_x);
        // left subtree
        node.left = Some(Box::new(KDNode::new_subtree(&lx, &ly, cut_dim)));
        // right subtree
        node.right = Some(Box::new(KDNode::new_subtree(&rx, &ry, cut_dim)));
        return node;
    }

    fn divide_data(
        cut_on: &[f32],
        cut_with: &[f32],
        cut_val: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut ret_on_l = vec![];
        let mut ret_with_l = vec![];
        let mut ret_on_r = vec![];
        let mut ret_with_r = vec![];
        for i in 0..cut_on.len() {
            // Need arg to decided whether lt or gt
            if cut_on[i] < cut_val {
                ret_on_l.push(cut_on[i]);
                ret_with_l.push(cut_with[i]);
            } else {
                ret_on_r.push(cut_on[i]);
                ret_with_r.push(cut_with[i]);
            }
        }
        (ret_on_l, ret_with_l, ret_on_r, ret_with_r)
    }
}

#[cfg(test)]
mod test {
    use crate::kdtree_impl::KDNode;

    #[test]
    fn construct_tree() {
        let x = vec![0., 1., 2.];
        let y = x.clone();
        let tree = KDNode::new_tree(x.as_slice(), y.as_slice());

        println!("{:?}", tree.right);
        println!("{:?}", tree.left);
    }
}
