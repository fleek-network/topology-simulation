use ndarray::Array2;
use std::collections::{BTreeMap, BinaryHeap, HashSet};

#[derive(Debug, Clone)]
struct Pair {
    a: usize,
    b: usize,
    dissimilarity: f64,
}

impl std::cmp::PartialEq for Pair {
    fn eq(&self, other: &Self) -> bool {
        (self.dissimilarity - other.dissimilarity).abs() < f64::EPSILON
    }
}

impl std::cmp::Eq for Pair {}

impl std::cmp::PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.dissimilarity < other.dissimilarity {
            Some(std::cmp::Ordering::Greater)
        } else if self.dissimilarity > other.dissimilarity {
            Some(std::cmp::Ordering::Less)
        } else {
            Some(std::cmp::Ordering::Equal)
        }
    }
}

impl std::cmp::Ord for Pair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.dissimilarity < other.dissimilarity {
            std::cmp::Ordering::Greater
        } else if self.dissimilarity > other.dissimilarity {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Equal
        }
    }
}

pub fn cluster(dis_matrix: &Array2<f64>) -> (Vec<usize>, Vec<usize>) {
    let all_nodes: HashSet<usize> = (0..dis_matrix.shape()[0]).collect();
    let mut used_nodes = HashSet::new();
    let mut heap = BinaryHeap::new();

    for i in 0..dis_matrix.shape()[0] {
        for j in (i + 1)..dis_matrix.shape()[1] {
            heap.push(Pair {
                a: i,
                b: j,
                dissimilarity: dis_matrix[[i, j]],
            });
        }
    }

    let mut assignments = BTreeMap::new();
    let mut medoids = BTreeMap::new();
    let mut cluster_index = 0;
    while !heap.is_empty() {
        let pair = heap.pop().unwrap();
        if used_nodes.contains(&pair.a) || used_nodes.contains(&pair.b) {
            continue;
        }
        used_nodes.insert(pair.a);
        used_nodes.insert(pair.b);

        assignments
            .entry(cluster_index)
            .or_insert(Vec::new())
            .push(pair.a);
        assignments
            .entry(cluster_index)
            .or_insert(Vec::new())
            .push(pair.b);
        // Use the first node in the cluster as the medoid.
        medoids.insert(pair.a, cluster_index);
        cluster_index += 1;
    }

    let mut unassigned_nodes = all_nodes.difference(&used_nodes);
    if let Some(node_index) = unassigned_nodes.next() {
        if assignments.len() % 2 == 0 {
            // if number of clusters are even, find the closest medoid and add the missing node to
            // this cluster
            let closest_mediod =
                medoids
                    .keys()
                    .fold(*medoids.keys().next().unwrap(), |min_index, index| {
                        if dis_matrix[[*index, *node_index]] < dis_matrix[[min_index, *node_index]]
                        {
                            *index
                        } else {
                            min_index
                        }
                    });

            let medoid_index = medoids.get(&closest_mediod).unwrap();
            assignments
                .entry(*medoid_index)
                .or_insert(Vec::new())
                .push(*node_index);
        } else {
            // if number of clusters are odd, create a new cluster
            assignments.insert(cluster_index, vec![*node_index]);
            medoids.insert(*node_index, cluster_index);
        }
    }

    let mut assignment = vec![0; dis_matrix.shape()[0]];
    for (cluster_index, nodes) in assignments {
        for node in nodes {
            assignment[node] = cluster_index;
        }
    }
    (assignment, medoids.keys().copied().collect())
}

#[cfg(test)]
mod tests {

    use ndarray::Array;

    use super::cluster;

    #[test]
    fn test_xx() {
        let mut matrix = Array::zeros((5, 5));
        matrix[[0, 0]] = 0.0;
        matrix[[0, 1]] = 3.0;
        matrix[[0, 2]] = 2.0;
        matrix[[0, 3]] = 4.0;
        matrix[[0, 4]] = 5.0;

        matrix[[1, 0]] = 3.0;
        matrix[[1, 1]] = 0.0;
        matrix[[1, 2]] = 4.0;
        matrix[[1, 3]] = 2.0;
        matrix[[1, 4]] = 5.0;

        matrix[[2, 0]] = 2.0;
        matrix[[2, 1]] = 4.0;
        matrix[[2, 2]] = 0.0;
        matrix[[2, 3]] = 3.0;
        matrix[[2, 4]] = 5.0;

        matrix[[3, 0]] = 4.0;
        matrix[[3, 1]] = 2.0;
        matrix[[3, 2]] = 3.0;
        matrix[[3, 3]] = 0.0;
        matrix[[3, 4]] = 5.0;

        matrix[[4, 0]] = 4.0;
        matrix[[4, 1]] = 2.0;
        matrix[[4, 2]] = 3.0;
        matrix[[4, 3]] = 0.0;
        matrix[[4, 4]] = 0.0;

        cluster(&matrix);
    }
}
