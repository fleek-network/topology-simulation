use std::collections::BTreeSet;

use ndarray::Array2;

/// Hueristically sort a's indeces for greedily pairing.
///
/// First, sort a by the sum of distances to b for each item in a.
///
/// ```text
/// a (sorted): [ 0 1 2 3 4 5 6 7 ]
/// len: 8
/// b (unsorted): [ 8 9 ]
/// len: 2
/// ```
///
/// Now iterate through the sorted items, to form chunks where each next index skips n items.
///
/// ```text
/// n: ceil(a len / b len) = 4
/// [ 0 4 ] [ 1 5 ] [ 2 6 ] [ 3 7 ]
/// ```
///
/// Hypothetically speaking, this should be a good hueristic because each chunks first items,
/// which have the lowest sum latency to b, will have the most number of options and will be
/// able to select the optimal pair. In the network, we want to ensure as many low latency
/// connections between clusters, and dont care if some pairings are sub-optimal.
pub fn hueristic_sort(dissim_matrix: &Array2<i32>, a: &mut [usize], b: &[usize]) {
    // 1. compute and sort each node by their sums of dissim to b
    let mut sorted = a.to_vec();
    sorted.sort_by_cached_key(|&i| b.iter().map(|&j| dissim_matrix[(i, j)]).sum::<i32>());

    // 2. reassign indeces
    let len = a.len();
    let n = (len + b.len() - 1) / b.len(); // ceiling 
    let mut iter = a.iter_mut();
    for c in 0..n {
        let mut j = c;
        while j < len {
            *iter.next().unwrap() = sorted[j];
            j += n;
        }
    }
}

/// Greedily pair nodes together by finding their closest match, after a heuristic sort. If a
/// cluster is smaller than the other, it may have more than one connection per node.
pub fn greedy_pairs(dissim_matrix: &Array2<i32>, a: &[usize], b: &[usize]) -> Vec<(usize, usize)> {
    let (a, b) = if a.len() > b.len() { (a, b) } else { (b, a) };
    let mut a = a.to_vec();

    hueristic_sort(dissim_matrix, &mut a, b);

    let b_set = BTreeSet::from_iter(b);
    let mut pairs = Vec::new();
    for chunk in a.chunks(b.len()) {
        // store a clone of the b set to remove entries from for this chunk
        let mut b_set_cloned = b_set.clone();
        // find best options from the b set for each item in the chunk
        for i in chunk.iter() {
            // find best option from the b set and remove it for the next node in a set
            let best = *b_set_cloned
                .iter()
                .min_by_key(|&j| dissim_matrix[(*i, **j)])
                .unwrap();
            b_set_cloned.remove(best);
            pairs.push((*i, *best));
        }
    }

    pairs
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
