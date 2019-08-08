use statistical;

type More = Option<Box<KDTree>>;
type Vals = Option<Vec<f32>>;

#[derive(Debug, Copy, Clone)]
enum CutDim {
    X,
    Y,
}

enum CutSide {
    Left,
    Right,
}

#[derive(Debug, Copy, Clone)]
struct Bounds {
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
}

#[derive(Debug, Copy, Clone)]
struct Best {
    x: f32,
    y: f32,
    dist: f32,
}

#[derive(Debug)]
pub struct KDTree {
    // The values it splits on
    cut_val: f32,
    cut_dim: CutDim,
    // Subtree
    right: More,
    left: More,
    // Vals
    x_vals: Vals,
    y_vals: Vals,
    // Cut dim
    min_leafsize: u32,
}

impl KDTree {
    pub fn new_tree(x: &[f32], y: &[f32], min_leafsize: u32) -> KDTree {
        return KDTree::new_subtree(x, y, min_leafsize, CutDim::X);
    }

    // Tree construction
    fn new_node(cut_val: f32, cut_dim: CutDim, min_leafsize: u32) -> KDTree {
        KDTree {
            cut_val,
            cut_dim,
            right: None,
            left: None,
            x_vals: None,
            y_vals: None,
            min_leafsize,
        }
    }
    fn new_leaf_node(x_vals: &[f32], y_vals: &[f32]) -> KDTree {
        KDTree {
            cut_val: 0.,
            cut_dim: CutDim::X,
            right: None,
            left: None,
            x_vals: Some(x_vals.to_vec()),
            y_vals: Some(y_vals.to_vec()),
            min_leafsize: 0,
        }
    }

    fn new_subtree(x: &[f32], y: &[f32], min_leafsize: u32, parent_cut_dim: CutDim) -> KDTree {
        // Leafnode
        if x.len() <= min_leafsize as usize {
            return KDTree::new_leaf_node(x, y);
        }

        // Treenode
        // This node is going to be cut on the opposite dim as its parent
        let (cut_dim, cut_val) = match parent_cut_dim {
            CutDim::X => (CutDim::Y, statistical::median(y)),
            CutDim::Y => (CutDim::X, statistical::median(x)),
        };
        let mut node = KDTree::new_node(cut_val, cut_dim, min_leafsize);
        let (lx, ly, rx, ry) = match cut_dim {
            CutDim::X => KDTree::divide_data(x, y, cut_val),
            CutDim::Y => KDTree::divide_data(y, x, cut_val),
        };
        // left subtree
        node.left = Some(Box::new(KDTree::new_subtree(
            &lx,
            &ly,
            min_leafsize,
            cut_dim,
        )));
        // right subtree
        node.right = Some(Box::new(KDTree::new_subtree(
            &rx,
            &ry,
            min_leafsize,
            cut_dim,
        )));
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

    // Tree methods
    pub fn nearest_neighbor(&self, x: f32, y: f32) -> (f32, f32) {
        let init_bounds = Bounds {
            x_min: -1e10,
            x_max: 1e10,
            y_min: -1e10,
            y_max: 1e10,
        };
        let init_best = Best {
            x: 0.,
            y: 0.,
            dist: 1e10,
        };
        let best = self._nearest_neighbor(x, y, init_best, init_bounds);
        (best.x, best.y)
    }
    fn _nearest_neighbor(&self, x: f32, y: f32, mut best: Best, mut bounds: Bounds) -> Best {
        // Check if leaf node
        if self.left.is_none() {
            for i in 0..self.x_vals.unwrap().len() {
                let dist =
                    (self.x_vals.unwrap()[i] - x).powf(2.) + (self.y_vals.unwrap()[i] - y).powf(2.);
                if dist < best.dist {
                    best = Best {
                        x: self.x_vals.unwrap()[i],
                        y: self.y_vals.unwrap()[i],
                        dist,
                    };
                }
            }
            return best;
        }
        // Decide which subtree to search first
        match self.find_closer_cut(x, y, bounds) {
            CutSide::Left => {
                match self.cut_dim {
                    CutDim::X => {
                        bounds.x_max = self.cut_val;
                        best = self.left.unwrap()._nearest_neighbor(x, y, best, bounds);
                        best = self.right.unwrap()._nearest_neighbor(x, y, best, bounds);
                    }
                    CutDim::Y => {
                        bounds.y_max = self.cut_val;
                        best = self.left.unwrap()._nearest_neighbor(x, y, best, bounds);
                        best = self.right.unwrap()._nearest_neighbor(x, y, best, bounds);
                    }
                };
            }
            CutSide::Right => {
                match self.cut_dim {
                    CutDim::X => {
                        bounds.x_min = self.cut_val;
                        best = self.left.unwrap()._nearest_neighbor(x, y, best, bounds);
                        best = self.right.unwrap()._nearest_neighbor(x, y, best, bounds);
                    }
                    CutDim::Y => {
                        bounds.y_min = self.cut_val;
                        best = self.left.unwrap()._nearest_neighbor(x, y, best, bounds);
                        best = self.right.unwrap()._nearest_neighbor(x, y, best, bounds);
                    }
                };
            }
        };
        return best;
    }

    fn find_closer_cut(&self, x: f32, y: f32, bounds: Bounds) -> CutSide {
        // I find this syntax far more easy to understand...
        fn max(a: f32, b: f32) -> f32 {
            return a.max(b);
        }
        // This is the idea
        // let dx = max(max(bounds.x_min - x, 0.), max(x - bounds.x_max, 0.));

        match self.cut_dim {
            CutDim::X => {
                let dx_left = max(max(bounds.x_min - x, 0.), max(x - self.cut_val, 0.));
                let dx_right = max(max(self.cut_val - x, 0.), max(x - bounds.x_max, 0.));
                if dx_left < dx_right {
                    return CutSide::Left;
                } else {
                    return CutSide::Right;
                }
            }
            CutDim::Y => {
                let dy_left = max(max(bounds.y_min - y, 0.), max(y - self.cut_val, 0.));
                let dy_right = max(max(self.cut_val - y, 0.), max(y - bounds.y_max, 0.));
                if dy_left < dy_right {
                    return CutSide::Left;
                } else {
                    return CutSide::Right;
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::kdtree_impl::KDTree;
    use std::iter::Iterator;

    #[test]
    fn construct_tree() {
        let x = [0., 1., 2.];
        let y = x.clone();
        let tree = KDTree::new_tree(&x, &y, 2);

        println!("{:?}\n", tree);
        println!("{:?}", tree.right);
        println!("{:?}", tree.left);
    }

    #[test]
    fn nearest_neighbor() {
        let x: Vec<f32> = (0u16..10).map(f32::from).collect();
        let y: Vec<f32> = (0u16..10).map(f32::from).collect();
        let tree = KDTree::new_tree(x.as_slice(), y.as_slice(), 2);

        tree.nearest_neighbor(1.1, 1.1);
    }
}
