use ndarray::Array2;
use rand::seq::SliceRandom;

pub fn fill_sparse_entries_with_mean(matrix: &mut Array2<i32>, percent_of_missing_values: f32) {
    let mut sum: u128 = 0;
    let (n, m) = (matrix.shape()[0], matrix.shape()[1]);
    let total = n * m;
    let mut matrix_indices = Vec::with_capacity(total);
    for i in 0..n {
        for j in 0..m {
            sum += matrix[[i, j]] as u128;
            matrix_indices.push((i, j));
        }
    }
    let mean = (sum / total as u128) as i32;
    matrix_indices.shuffle(&mut rand::thread_rng());
    let num_values = (matrix_indices.len() as f32 * percent_of_missing_values) as usize;

    for (x, y) in matrix_indices.into_iter().take(num_values) {
        matrix[[x, y]] = mean;
    }
}
