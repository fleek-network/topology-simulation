use std::collections::{BTreeMap, HashMap};

use ndarray::{iter::IndexedIterMut, Array, Array2};
use serde::{Deserialize, Serialize};

use crate::constrained_fasterpam;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum Cluster {
    Cluster {
        index: usize,
        children: Vec<Cluster>,
    },
    LeafCluster {
        index: usize,
        nodes: Vec<Node>,
    },
}

impl std::fmt::Display for Cluster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.get_string_rep(0))
    }
}

impl Cluster {
    fn get_node_indices(&self) -> Vec<usize> {
        let mut indices = Vec::new();
        Cluster::_get_node_indices(self, &mut indices);
        indices
    }

    fn _get_node_indices(cluster: &Cluster, indices: &mut Vec<usize>) {
        match cluster {
            Cluster::Cluster { index: _, children } => {
                children
                    .iter()
                    .for_each(|child| Cluster::_get_node_indices(child, indices));
            },
            Cluster::LeafCluster { index: _, nodes } => {
                nodes.iter().for_each(|node| indices.push(node.index));
            },
        }
    }

    fn get_string_rep(&self, depth: usize) -> String {
        let mut s = Vec::new();
        let whitespace = vec!["  "; depth].join("");
        match self {
            Cluster::Cluster { index, children } => {
                s.push(format!("{}Cluster ( index: {} )", whitespace, index));
                for cluster in children.iter() {
                    s.push(cluster.get_string_rep(depth + 1));
                }
            },
            Cluster::LeafCluster { index, nodes } => {
                let node_indices: Vec<usize> = nodes.iter().map(|node| node.index).collect();
                s.push(format!(
                    "{}LeafCluster ( index: {}, nodes: {:?} )",
                    whitespace, index, node_indices
                ));
            },
        }
        s.join("\n")
    }

    fn get_index(&self) -> usize {
        match self {
            Cluster::Cluster { index, children: _ } => *index,
            Cluster::LeafCluster { index, nodes: _ } => *index,
        }
    }
}

impl std::fmt::Display for NodeHierarchy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = Vec::new();
        for (_, cluster) in self.clusters.iter() {
            s.push(format!("{}", cluster));
        }
        write!(f, "{}", s.join("\n"))
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct NodeHierarchy {
    clusters: BTreeMap<usize, Cluster>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Node {
    index: usize,
    connection_indices: Vec<Vec<usize>>,
}

impl Node {
    pub fn new(index: usize) -> Self {
        Self {
            index,
            connection_indices: Vec::new(),
        }
    }

    pub fn add_connection(&mut self, node_index: usize, level: usize) {
        if self.connection_indices.len() < level + 1 {
            let missing_levels = 1 + level - self.connection_indices.len();
            (0..missing_levels).for_each(|_| self.connection_indices.push(vec![]));
        }
        // TODO: this check might not be necessary. Otherwise use a set.
        if !self.connection_indices[level].contains(&node_index) {
            self.connection_indices[level].push(node_index);
        }
    }
}

impl NodeHierarchy {
    pub fn new(
        dis_matrix: &Array2<f64>,
        init_num_clusters: usize,
        init_min_size: usize,
        init_max_size: usize,
        max_iter: usize,
    ) -> Self {
        let mut meds = kmedoids::random_initialization(
            dis_matrix.shape()[0],
            init_num_clusters,
            &mut rand::thread_rng(),
        );
        let (_, assignment, _, _): (f64, _, _, _) = constrained_fasterpam::fasterpam(
            dis_matrix,
            &mut meds,
            max_iter,
            init_min_size,
            init_max_size,
        );

        let mut medoids = meds;
        let mut depth = 0;
        let mut assignment = assignment;

        let mut level = NodeHierarchy::new_leaf_hierarchy(&assignment);

        loop {
            let new_dis_matrix = NodeHierarchy::build_matrix(dis_matrix, &medoids);
            //let new_dis_matrix = NodeHierarchy::build_matrix_v2(dis_matrix, &medoids, &assignment);

            let mut new_medoids = kmedoids::random_initialization(
                new_dis_matrix.shape()[0],
                new_dis_matrix.shape()[0] / 2,
                &mut rand::thread_rng(),
            );

            let (_, new_assignment, _, _): (f64, _, _, _) =
                constrained_fasterpam::fasterpam(&new_dis_matrix, &mut new_medoids, max_iter, 2, 2);

            let new_level = NodeHierarchy::new_hierarchy(&mut level, &new_assignment);

            medoids = new_medoids;
            level = new_level;
            depth += 1;
            assignment = new_assignment;

            if medoids.len() <= 3 {
                break;
            }
        }
        level
    }

    pub fn get_assignments(&self) -> BTreeMap<usize, Vec<usize>> {
        let mut assignments = BTreeMap::new();
        for (depth, clusters) in self.get_assignments_map().iter() {
            let mut map = BTreeMap::new();
            for (cluster_index, cluster) in clusters.iter() {
                for node_index in cluster.iter() {
                    map.insert(*node_index, *cluster_index);
                }
            }
            let mut assignment = vec![0; map.len()];
            for (&node_index, &cluster_index) in map.iter() {
                assignment[node_index] = cluster_index;
            }
            assignments.insert(*depth, assignment);
        }
        assignments
    }

    pub fn get_assignments_map(&self) -> BTreeMap<usize, BTreeMap<usize, Vec<usize>>> {
        let mut levels = BTreeMap::new();
        self.clusters
            .values()
            .for_each(|cluster| NodeHierarchy::_get_assignments(cluster, 0, &mut levels));
        levels
    }

    fn _get_assignments(
        cluster: &Cluster,
        depth: usize,
        levels: &mut BTreeMap<usize, BTreeMap<usize, Vec<usize>>>,
    ) {
        let node_indices = cluster.get_node_indices();
        levels
            .entry(depth)
            .or_insert(BTreeMap::new())
            .insert(cluster.get_index(), node_indices);

        if let Cluster::Cluster { index: _, children } = cluster {
            children
                .iter()
                .for_each(|child| NodeHierarchy::_get_assignments(child, depth + 1, levels));
        }
    }

    fn new_hierarchy(below_level: &mut NodeHierarchy, assignment: &[usize]) -> Self {
        // TODO: merge clusters that aren't non-leaf clusters using hungarian algo
        let mut clusters_map = BTreeMap::new();
        for (below_cluster_index, cluster_index) in assignment.iter().enumerate() {
            let cluster = below_level
                .clusters
                .remove(&below_cluster_index)
                .expect("Cluster missing from hierarchy below.");
            clusters_map
                .entry(*cluster_index)
                .or_insert(Vec::new())
                .push(cluster);
        }
        let mut clusters = BTreeMap::new();
        for (cluster_index, children) in clusters_map.into_iter() {
            clusters.insert(
                cluster_index,
                Cluster::Cluster {
                    index: cluster_index,
                    children,
                },
            );
        }
        NodeHierarchy { clusters }
    }

    fn new_leaf_hierarchy(assignment: &[usize]) -> Self {
        let mut clusters_map = BTreeMap::new();
        for (node_index, cluster_index) in assignment.iter().enumerate() {
            clusters_map
                .entry(cluster_index)
                .or_insert(Vec::new())
                .push(Node::new(node_index));
        }
        let mut clusters = BTreeMap::new();
        for (cluster_index, nodes) in clusters_map.into_iter() {
            clusters.insert(
                *cluster_index,
                Cluster::LeafCluster {
                    index: *cluster_index,
                    nodes,
                },
            );
        }
        Self { clusters }
    }

    fn build_matrix(dis_matrix: &Array2<f64>, medoids: &[usize]) -> Array2<f64> {
        let mut medoids = medoids.to_vec();
        medoids.sort();
        let mut new_dis_matrix = Array::zeros((medoids.len(), medoids.len()));
        for (i_new, &i) in medoids.iter().enumerate() {
            for (j_new, &j) in medoids.iter().enumerate() {
                if i_new == j_new {
                    continue;
                }
                new_dis_matrix[[i_new, j_new]] = dis_matrix[[i, j]];
            }
        }
        new_dis_matrix
    }

    fn build_matrix_v2(
        dis_matrix: &Array2<f64>,
        medoids: &[usize],
        assignment: &[usize],
    ) -> Array2<f64> {
        let mut medoids = medoids.to_vec();
        medoids.sort();
        let mut new_dis_matrix = Array::zeros((medoids.len(), medoids.len()));
        let mut clusters = BTreeMap::new();
        for (node_index, cluster_index) in assignment.iter().enumerate() {
            clusters
                .entry(cluster_index)
                .or_insert(Vec::new())
                .push(node_index);
        }

        for (&i, node_indices_i) in clusters.iter() {
            for (&j, node_indices_j) in clusters.iter() {
                if i != j {
                    let mut sum = 0.0;
                    let mut count = 0;
                    for &node_i in node_indices_i.iter() {
                        for &node_j in node_indices_j.iter() {
                            sum += dis_matrix[[node_i, node_j]];
                            count += 1;
                        }
                    }
                    if count > 0 {
                        new_dis_matrix[[*i, *j]] = sum / count as f64;
                    }
                }
            }
        }
        new_dis_matrix
    }

    fn merge_clusters() {}
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use ndarray_rand::rand_distr::{Distribution, UnitDisc};
    use rand::{self, Rng};

    use super::Cluster;
    use crate::{bottom_up::NodeHierarchy, constrained_fasterpam};

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

    fn get_distance_matrix(data: &Array2<f64>) -> Array2<f64> {
        let mut dist = Array2::zeros((data.shape()[0], data.shape()[0]));
        for i in 0..data.shape()[0] {
            for j in 0..data.shape()[0] {
                if i != j {
                    dist[[i, j]] = (data[[i, 0]] - data[[j, 0]]).powi(2)
                        + (data[[i, 1]] - data[[j, 1]]).powi(2);
                }
            }
        }
        dist
    }

    #[test]
    fn test_foo() {
        let points = get_random_points(10, 5);
        let matrix = get_distance_matrix(&points);

        let mut meds =
            kmedoids::random_initialization(matrix.shape()[0], 4, &mut rand::thread_rng());
        println!("meds: {:?}", meds);
        println!("nodes: {:?}", matrix.shape()[0]);
        let (_, labels, _, _): (f64, _, _, _) =
            constrained_fasterpam::fasterpam(&matrix, &mut meds, 100, 2, 2);
        println!("labels: {:?}", labels);
    }

    #[test]
    fn test_new() {
        let points = get_random_points(100, 10);
        let matrix = get_distance_matrix(&points);
        let node_hierarchy = NodeHierarchy::new(&matrix, 10, 8, 12, 100);
        println!("{node_hierarchy}");
        println!();
        println!("{:?}", node_hierarchy.get_assignments());
    }

    #[test]
    fn test_new_leaf_cluster() {
        let cluster_0 = [0, 3, 6];
        let cluster_1 = [1, 4, 5];
        let cluster_2 = [2, 7, 8];
        let target_clusters = vec![cluster_0, cluster_1, cluster_2];

        let assignment = vec![0, 1, 2, 0, 1, 1, 0, 2, 2];
        let node_hierarchy = NodeHierarchy::new_leaf_hierarchy(&assignment);
        assert_eq!(node_hierarchy.clusters.len(), 3);
        for (&i, node_hierarchy) in node_hierarchy.clusters.iter() {
            if let Cluster::LeafCluster {
                index: _index,
                nodes,
            } = node_hierarchy
            {
                let node_indices: Vec<usize> = nodes.iter().map(|node| node.index).collect();
                assert_eq!(node_indices, target_clusters[i]);
            } else {
                panic!("Expected a cluster.");
            }
        }
    }

    #[test]
    fn test_new_hierarchy() {
        let leaf_cluster_0 = [0, 3, 6];
        let leaf_cluster_1 = [1, 4, 5];
        let leaf_cluster_2 = [2, 7, 8];
        let leaf_cluster_3 = [9, 10, 11];
        let target_cluster1 = vec![leaf_cluster_0, leaf_cluster_2];
        let target_cluster2 = vec![leaf_cluster_1, leaf_cluster_3];
        let target_cluster = vec![target_cluster1, target_cluster2];

        let leaf_assignment = vec![0, 1, 2, 0, 1, 1, 0, 2, 2, 3, 3, 3];
        let mut leaf_hierarchy = NodeHierarchy::new_leaf_hierarchy(&leaf_assignment);

        // cluster_0 : [cluster_0, cluster_2]
        // cluster_1 : [cluster_1, cluster_3]
        let assignment = vec![0, 1, 0, 1];
        let node_hierarchy = NodeHierarchy::new_hierarchy(&mut leaf_hierarchy, &assignment);
        assert_eq!(node_hierarchy.clusters.len(), 2);

        for (_, cluster) in node_hierarchy.clusters.iter() {
            if let Cluster::Cluster {
                index: cluster_index,
                children,
            } = cluster
            {
                for (leaf_index, child) in children.iter().enumerate() {
                    if let Cluster::LeafCluster { index: _, nodes } = child {
                        let node_indices: Vec<usize> =
                            nodes.iter().map(|node| node.index).collect();
                        assert_eq!(node_indices, target_cluster[*cluster_index][leaf_index]);
                    } else {
                        panic!("Expected leaf cluster.")
                    }
                }
            } else {
                panic!("Expected non-leaf cluster.")
            }
        }
    }

    #[test]
    fn test_build_matrix() {
        let points = get_random_points(100, 10);
        let matrix = get_distance_matrix(&points);
        let medoids = vec![0, 4, 10, 23, 1, 7, 83, 44, 57, 66];
        let new_matrix = NodeHierarchy::build_matrix(&matrix, &medoids);
        assert_eq!(new_matrix.shape()[0], medoids.len());
        assert_eq!(new_matrix.shape()[1], medoids.len());
        assert_eq!(new_matrix[[1, 3]], matrix[[4, 23]]);
        assert_eq!(new_matrix[[0, 6]], matrix[[0, 83]]);
        assert_eq!(new_matrix[[5, 8]], matrix[[7, 57]]);
    }
}
