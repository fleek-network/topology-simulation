use ndarray::{s, Array2};
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

pub fn interpolate_sparse_entries_with_mean(
    matrix: &mut Array2<i32>,
    percent_of_missing_values: f32,
) {
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
    matrix_indices.shuffle(&mut rand::thread_rng());
    let num_values = (matrix_indices.len() as f32 * percent_of_missing_values) as usize;

    for (x, y) in matrix_indices.iter().take(num_values) {
        // set missing values to zero
        matrix[[*x, *y]] = 0;
    }
    let mean = (sum / total as u128) as i32;

    for (x, y) in matrix_indices.into_iter().take(num_values) {
        let row = matrix.slice(s![.., x]);
        let col = matrix.slice(s![y, ..]);

        let q = 0.3;
        let mut row_values: Vec<i32> = row.iter().copied().collect();
        // TODO: use quick select
        row_values.sort();
        let index = (q * row_values.len() as f64).floor() as usize;
        let quantile = row_values[index];

        let mut dot_prod = 0;
        let mut sum = 0;
        for i in 0..row.shape()[0] {
            if row[i] <= quantile {
                dot_prod += row[i] as u128 * col[i] as u128;
                sum += row[i] as u128;
            }
        }
        if sum != 0 {
            matrix[[x, y]] = (dot_prod / sum) as i32;
        } else {
            matrix[[x, y]] = mean;
        }
    }
}
