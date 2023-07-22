use std::collections::{BTreeMap, BTreeSet};

use ndarray::Array2;

pub fn greedy_pairs(dissim_matrix: &Array2<i32>, a: &[usize], b: &[usize]) -> Vec<(usize, usize)> {
    let (a, b) = if a.len() > b.len() { (a, b) } else { (b, a) };
    let b_set = BTreeSet::from_iter(b.clone());

    // TODO: improve the initial greedy indeces
    // First, sort a by the sum of distances to b for each item in a. For example:
    //
    // a (sorted): [ 0 1 2 3 4 5 6 7 ]
    // len: 8
    // b (unsorted): [ 8 9 ]
    // len: 2
    // indeces to skip for each chunk item: 4
    //
    // Now to form each chunk, iterate over the sorted items, to form chunks of a like so:
    //
    // [ 1 5 ] [ 2 6 ] [ 3 7 ] [ 4 8 ]
    //
    // Hypothetically speaking, this should be a good hueristic because as each chunks first items,
    // which have the lowest sum latency to b, will have the most number of options and will be
    // able to select an optimal pair. We can then iterate over chunks and find best pairs as
    // implemented

    let mut pairs = BTreeMap::new();

    for a_chunk in a.chunks(b.len()) {
        // store a clone of the b set to remove entries from for this chunk
        let mut b_set = b_set.clone();

        // find best options from the b set for each item in the chunk
        for i in a_chunk.iter() {
            // find best option from the b set
            let mut iter = b_set.iter();
            let mut best = *iter.next().unwrap();
            let mut best_dissim = dissim_matrix[(*i, *best)];
            for &j in iter {
                let dissim = dissim_matrix[(*i, *j)];
                if dissim < best_dissim {
                    best = j;
                    best_dissim = dissim;
                }
            }

            // remove the best index from b set for the next iteration
            b_set.remove(best);
            // insert the pairing
            pairs.insert(*i, *best);
        }
    }

    pairs.into_iter().collect()
}

#[test]
fn test_greedy_pairing() {
    use ndarray_rand::rand_distr::UnitDisc;

    fn sample_cluster(
        x: f32,
        y: f32,
        x_scale: f32,
        y_scale: f32,
        num_points: usize,
    ) -> Array2<i32> {
        let mut cluster = Array2::zeros((num_points, 2));

        for i in 0..num_points {
            let v: [f32; 2] =
                rand::prelude::Distribution::sample(&UnitDisc, &mut rand::thread_rng());
            cluster[[i, 0]] = (x + v[0] * x_scale) as i32;
            cluster[[i, 1]] = (y + v[1] * y_scale) as i32;
        }
        cluster
    }

    fn get_distance_matrix(data: &Array2<i32>) -> Array2<i32> {
        let mut dist = Array2::zeros((data.shape()[0], data.shape()[0]));
        for i in 0..data.shape()[0] {
            for j in 0..data.shape()[0] {
                if i != j {
                    dist[[i, j]] =
                        (data[[i, 0]] - data[[j, 0]]).pow(2) + (data[[i, 1]] - data[[j, 1]]).pow(2);
                }
            }
        }
        dist
    }

    let cluster1 = sample_cluster(1., 1., 1000., 1000., 10);
    let cluster2 = sample_cluster(100., 100., 1000., 1000., 10);
    let data =
        ndarray::concatenate(ndarray::Axis(0), &[(&cluster1).into(), (&cluster2).into()]).unwrap();

    let dis_matrix = get_distance_matrix(&data);

    let indeces: Vec<_> = (0..dis_matrix.nrows()).collect();

    let pairs = greedy_pairs(&dis_matrix, &indeces[0..10], &indeces[10..]);
    println!("{pairs:?}");
}

/* TODO: Finish moving to its own function
pub fn hungarian_pairs(
    dissim_matrix: &Array2<i32>,
    a: &[usize],
    b: &[usize],
) -> Vec<(usize, usize)> {
    // build weight matrix
    // The smaller cluster should be placed on the left
    let (a, b) = if a.len() > b.len() { (a, b) } else { (b, a) };

    let (left_len, right_len) = (a.len(), b.len());
    let mut weights = Vec::new();
    for i in a {
        for j in b {
            let dissim = dissim_matrix[(*i, *j)];
            weights.push(dissim);
        }
    }

    // Compute pairs with the munkres (hungarian) algorithm
    let (_cost, pairs) = pathfinding::kuhn_munkres::kuhn_munkres_min(
        &Matrix::from_vec(left_len, right_len, weights).unwrap(),
    );

    let mut final_pairs = BTreeMap::new();

    let mut remaining = BTreeSet::from_iter(b);
    for (i, &j) in pairs.iter().enumerate() {
        // add connection to i and j
        let (i, j) = (a[i], b[j]);
        final_pairs.insert(i, j)
        add_connection(&mut indeces, current_path.depth(), i, j);
        remaining.remove(&j);
    }

    // Find best pair for missing nodes
    for &missing in remaining {
        let mut best = clusters[0][0];
        let mut best_dist = dissim_matrix[(best, missing)];
        for possible in &clusters[0][1..] {
            let dist = dissim_matrix[(*possible, missing)];
            if dist < best_dist {
                best = *possible;
                best_dist = dist;
            }
        }
        add_connection(&mut indeces, current_path.depth(), missing, best);
    }
}*/
