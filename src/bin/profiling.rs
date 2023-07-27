use std::time::Instant;

use clustering::{bottom_up::NodeHierarchy, sparsify};
use ndarray::Array2;
use ndarray_rand::rand_distr::{Distribution, UnitDisc};
use rand::{self, Rng};

fn get_random_points(num_nodes: usize, num_clusters: usize) -> Array2<f64> {
    let mut points = Array2::zeros((num_nodes, 2));
    let cluster_size = num_nodes / num_clusters;
    let mut index = 0;
    for _ in 0..num_clusters {
        let x = rand::thread_rng().gen_range(10.0..100.0);
        let y = rand::thread_rng().gen_range(10.0..100.0);
        let scale_x = rand::thread_rng().gen_range(1.0..4.0);
        let scale_y = rand::thread_rng().gen_range(1.0..4.0);

        for _ in 0..cluster_size {
            let v: [f64; 2] = UnitDisc.sample(&mut rand::thread_rng());
            points[[index, 0]] = x + v[0] * scale_x;
            points[[index, 1]] = y + v[1] * scale_y;
            index += 1;
        }
    }
    points
}

fn get_distance_matrix(data: &Array2<f64>) -> Array2<i32> {
    let mut dist = Array2::zeros((data.shape()[0], data.shape()[0]));
    for i in 0..data.shape()[0] {
        for j in 0..data.shape()[0] {
            if i != j {
                dist[[i, j]] = (((data[[i, 0]] - data[[j, 0]]).powi(2)
                    + (data[[i, 1]] - data[[j, 1]]).powi(2))
                    * 1000.0) as i32;
            }
        }
    }
    dist
}

fn main() {
    let num_points = 1000;
    let cluster_size = 8;
    let points = get_random_points(num_points, num_points / cluster_size);
    let mut matrix = get_distance_matrix(&points);
    sparsify::interpolate_sparse_entries_with_mean(&mut matrix, 0.1);

    let now = Instant::now();
    let _node_hierarchy = NodeHierarchy::new(
        &matrix,
        num_points / cluster_size,
        cluster_size,
        cluster_size,
        100,
    );
    let elapsed = now.elapsed();
    println!("duration: {elapsed:?}");
}
