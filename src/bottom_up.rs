use std::collections::{BTreeMap, BTreeSet, HashMap};

use ndarray::{Array, Array2};
use pathfinding::prelude::{kuhn_munkres_min, Matrix};
use serde::{Deserialize, Serialize};

use crate::{
    cluster_into_pairs, constrained_fasterpam, pairing,
    types::{SerializedLayer, SerializedNode},
};

impl From<NodeHierarchy> for SerializedLayer {
    fn from(value: NodeHierarchy) -> Self {
        let mut children: Vec<SerializedLayer> = Vec::new();
        let mut total = 0;
        let level_path = vec![];
        for (_index, child) in value.clusters.into_iter() {
            total += child.get_total();
            children.push(child.to_serialized_layer(level_path.clone()));
        }
        SerializedLayer::Group {
            id: "root".to_string(),
            total,
            children,
        }
    }
}

impl From<Cluster> for SerializedLayer {
    fn from(value: Cluster) -> Self {
        let level_path = Vec::new();
        value.to_serialized_layer(level_path)
    }
}

impl From<Node> for SerializedNode {
    fn from(value: Node) -> Self {
        let connections: Vec<Vec<usize>> = value
            .connections
            .into_iter()
            .rev()
            .map(|(_level, indices)| indices)
            .collect();
        Self {
            id: value.index,
            connections,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum Cluster {
    Cluster {
        index: usize,
        total: usize,
        node_indices: Vec<usize>,
        children: Vec<Cluster>,
    },
    LeafCluster {
        index: usize,
        node_indices: Vec<usize>,
        nodes: Vec<Node>,
    },
}

impl std::fmt::Display for Cluster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.get_string_rep(0))
    }
}

impl Cluster {
    fn get_connections(&self, connections: &mut Vec<Vec<usize>>) {
        match self {
            Cluster::Cluster {
                index: _,
                total: _,
                node_indices: _,
                children,
            } => {
                children
                    .iter()
                    .for_each(|child| child.get_connections(connections));
            },
            Cluster::LeafCluster {
                index: _,
                node_indices: _,
                nodes,
            } => {
                nodes.iter().for_each(|node| {
                    connections[node.index] = node.connections.values().flatten().copied().collect()
                });
            },
        }
    }

    fn to_serialized_layer(&self, mut level_path: Vec<String>) -> SerializedLayer {
        match self {
            Cluster::Cluster {
                index,
                total,
                node_indices: _,
                children,
            } => {
                level_path.push(index.to_string());
                let id = level_path.join(".");
                let serialized_children: Vec<SerializedLayer> = children
                    .iter()
                    .map(|cluster| cluster.to_serialized_layer(level_path.clone()))
                    .collect();
                SerializedLayer::Group {
                    id,
                    total: *total,
                    children: serialized_children,
                }
            },
            Cluster::LeafCluster {
                index,
                node_indices: _,
                nodes,
            } => {
                let serialized_nodes: Vec<SerializedNode> =
                    nodes.iter().map(|node| node.clone().into()).collect();
                level_path.push(index.to_string());
                SerializedLayer::Cluster {
                    id: level_path.join("."),
                    nodes: serialized_nodes,
                }
            },
        }
    }

    fn add_connections(
        &mut self,
        connection_map: &mut HashMap<usize, BTreeMap<usize, Vec<usize>>>,
    ) {
        match self {
            Cluster::Cluster {
                index: _,
                total: _,
                node_indices: _,
                children,
            } => {
                children
                    .iter_mut()
                    .for_each(|child| child.add_connections(connection_map));
            },
            Cluster::LeafCluster {
                index: _,
                node_indices: _,
                nodes,
            } => {
                for node in nodes.iter_mut() {
                    if let Some(connections) = connection_map.remove(&node.index) {
                        node.add_connection(connections);
                    }
                }
            },
        }
    }

    fn connect_nodes_in_leaf_cluster(
        &self,
        node_connections: &mut HashMap<usize, BTreeMap<usize, Vec<usize>>>,
    ) {
        if let Cluster::LeafCluster {
            index: _,
            node_indices,
            nodes: _,
        } = self
        {
            for (i, &node_lhs) in node_indices.iter().enumerate() {
                for &node_rhs in node_indices[i + 1..].iter() {
                    node_connections
                        .entry(node_lhs)
                        .or_insert(BTreeMap::new())
                        .entry(0)
                        .or_insert(Vec::new())
                        .push(node_rhs);

                    node_connections
                        .entry(node_rhs)
                        .or_insert(BTreeMap::new())
                        .entry(0)
                        .or_insert(Vec::new())
                        .push(node_lhs);
                }
            }
        }
    }

    fn get_node_indices(&self) -> &[usize] {
        match self {
            Cluster::Cluster {
                index: _,
                total: _,
                node_indices,
                children: _,
            } => node_indices,
            Cluster::LeafCluster {
                index: _,
                node_indices,
                nodes: _,
            } => node_indices,
        }
    }

    fn get_string_rep(&self, depth: usize) -> String {
        let mut s = Vec::new();
        let whitespace = vec!["  "; depth].join("");
        match self {
            Cluster::Cluster {
                index,
                total: _,
                node_indices: _,
                children,
            } => {
                s.push(format!("{}Cluster ( index: {} )", whitespace, index));
                for cluster in children.iter() {
                    s.push(cluster.get_string_rep(depth + 1));
                }
            },
            Cluster::LeafCluster {
                index,
                node_indices,
                nodes: _,
            } => {
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
            Cluster::Cluster {
                index,
                total: _,
                node_indices: _,
                children: _,
            } => *index,
            Cluster::LeafCluster {
                index,
                node_indices: _,
                nodes: _,
            } => *index,
        }
    }

    fn get_total(&self) -> usize {
        match self {
            Cluster::Cluster {
                index: _,
                total,
                node_indices: _,
                children: _,
            } => *total,
            Cluster::LeafCluster {
                index: _,
                node_indices: _,
                nodes,
            } => nodes.len(),
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
    clusters: HashMap<usize, Cluster>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Node {
    index: usize,
    connections: BTreeMap<usize, Vec<usize>>,
}

impl Node {
    pub fn new(index: usize) -> Self {
        Self {
            index,
            connections: BTreeMap::new(),
        }
    }

    pub fn add_connection(&mut self, connections: BTreeMap<usize, Vec<usize>>) {
        self.connections = connections;
    }
}

impl NodeHierarchy {
    pub fn new(
        dis_matrix: &Array2<i32>,
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
        let (_, assignment, _, _): (i32, _, _, _) = constrained_fasterpam::fasterpam(
            dis_matrix,
            &mut meds,
            max_iter,
            init_min_size,
            init_max_size,
        );
        let meds_indices: BTreeSet<usize> = assignment.clone().into_iter().collect();
        let mut medoids: Vec<usize> = meds_indices.iter().map(|&index| meds[index]).collect();
        let mut level = 0;

        let mut hierarchy = NodeHierarchy::new_leaf_hierarchy(&assignment);
        let mut node_connections = HashMap::new();
        hierarchy.connect_nodes_in_leaf_clusters(&mut node_connections);

        loop {
            level += 1;
            let new_dis_matrix = NodeHierarchy::build_matrix(dis_matrix, &medoids);
            //let new_dis_matrix = Array::zeros((medoids.len(), medoids.len()));
            //let new_dis_matrix = NodeHierarchy::build_matrix_v2(dis_matrix, &medoids,
            // &assignment);

            //let mut new_medoids = kmedoids::random_initialization(
            //    new_dis_matrix.shape()[0],
            //    new_dis_matrix.shape()[0] / 2,
            //    &mut rand::thread_rng(),
            //);

            //let (_, new_assignment, _, _): (f64, _, _, _) =
            //    constrained_fasterpam::fasterpam(&new_dis_matrix, &mut new_medoids, max_iter, 2,
            // 2);
            let (new_assignment, _) = cluster_into_pairs::cluster(&new_dis_matrix);
            let meds_indices: BTreeSet<usize> = new_assignment.clone().into_iter().collect();
            let new_medoids: Vec<usize> = meds_indices.iter().map(|&index| meds[index]).collect();

            let new_hierarchy = NodeHierarchy::new_hierarchy(
                &mut hierarchy,
                &mut node_connections,
                dis_matrix,
                &new_assignment,
                level,
            );

            medoids = new_medoids;
            hierarchy = new_hierarchy;

            if medoids.len() <= 2 {
                break;
            }
        }
        hierarchy.connect_nodes_in_root_clusters(dis_matrix, level + 1, &mut node_connections);
        hierarchy.add_node_connections(&mut node_connections);
        hierarchy
    }

    pub fn get_connections(&self) -> Vec<Vec<usize>> {
        let mut connections = vec![vec![]; self.get_node_indices().len()];

        self.clusters
            .values()
            .for_each(|cluster| cluster.get_connections(&mut connections));

        connections
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
            .for_each(|cluster| NodeHierarchy::_get_assignments_map(cluster, 0, &mut levels));
        levels
    }

    fn _get_assignments_map(
        cluster: &Cluster,
        depth: usize,
        levels: &mut BTreeMap<usize, BTreeMap<usize, Vec<usize>>>,
    ) {
        let node_indices = cluster.get_node_indices().to_vec();
        levels
            .entry(depth)
            .or_insert(BTreeMap::new())
            .insert(cluster.get_index(), node_indices);

        if let Cluster::Cluster {
            index: _,
            total: _,
            node_indices: _,
            children,
        } = cluster
        {
            children
                .iter()
                .for_each(|child| NodeHierarchy::_get_assignments_map(child, depth + 1, levels));
        }
    }

    fn get_node_indices(&self) -> Vec<usize> {
        let mut nodes = Vec::new();
        for (_, cluster) in self.clusters.iter() {
            nodes.extend(cluster.get_node_indices());
        }
        nodes
    }

    fn new_hierarchy(
        below_level: &mut NodeHierarchy,
        node_connections: &mut HashMap<usize, BTreeMap<usize, Vec<usize>>>,
        dis_matrix: &Array2<i32>,
        assignment: &[usize],
        level: usize,
    ) -> Self {
        let mut clusters_map = HashMap::new();
        let mut node_indices_map = HashMap::new();

        let mut assignment_clone = assignment.to_vec();
        assignment_clone.sort();
        let mut cluster_indices: Vec<usize> = below_level.clusters.keys().copied().collect();
        cluster_indices.sort();

        for (below_cluster_index, cluster_index) in assignment.iter().enumerate() {
            if !below_level.clusters.contains_key(&below_cluster_index) {
                println!("below_cluster_index: {below_cluster_index}");
            }
            let cluster = below_level
                .clusters
                .remove(&below_cluster_index)
                .expect("Cluster missing from hierarchy below.");
            node_indices_map
                .entry(*cluster_index)
                .or_insert(Vec::new())
                .extend_from_slice(cluster.get_node_indices());
            clusters_map
                .entry(*cluster_index)
                .or_insert(Vec::new())
                .push(cluster);
        }

        // Connect nodes that were merged into a cluster
        for (_, clusters) in clusters_map.iter() {
            for (i, cluster_lhs) in clusters.iter().enumerate() {
                for cluster_rhs in clusters[i + 1..].iter() {
                    let node_indices_lhs = cluster_lhs.get_node_indices();
                    let node_indices_rhs = cluster_rhs.get_node_indices();
                    let connection_map = NodeHierarchy::connect_clusters_greedy(
                        node_indices_lhs,
                        node_indices_rhs,
                        dis_matrix,
                    );
                    connection_map.into_iter().for_each(|(node_index, nodes)| {
                        node_connections
                            .entry(node_index)
                            .or_insert(BTreeMap::new())
                            .entry(level)
                            .or_insert(Vec::new())
                            .extend_from_slice(&nodes);
                    });
                }
            }
        }

        let mut clusters = HashMap::new();
        for (cluster_index, children) in clusters_map.into_iter() {
            let node_indices = node_indices_map.remove(&cluster_index).unwrap();
            clusters.insert(
                cluster_index,
                Cluster::Cluster {
                    index: cluster_index,
                    total: node_indices.len(),
                    node_indices,
                    children,
                },
            );
        }
        NodeHierarchy { clusters }
    }

    fn add_node_connections(
        &mut self,
        node_connections: &mut HashMap<usize, BTreeMap<usize, Vec<usize>>>,
    ) {
        self.clusters
            .iter_mut()
            .for_each(|(_, cluster)| cluster.add_connections(node_connections));
    }

    fn connect_clusters_greedy(
        nodes_lhs: &[usize],
        nodes_rhs: &[usize],
        dis_matrix: &Array2<i32>,
    ) -> HashMap<usize, Vec<usize>> {
        let assignment = pairing::greedy_pairs(dis_matrix, nodes_lhs, nodes_rhs);
        let mut connection_map = HashMap::with_capacity(assignment.len());
        for (a, b) in assignment {
            connection_map.entry(a).or_insert(Vec::new()).push(b);
            connection_map.entry(b).or_insert(Vec::new()).push(a);
        }
        connection_map
    }

    fn _connect_clusters(
        nodes_lhs: &[usize],
        nodes_rhs: &[usize],
        dis_matrix: &Array2<i32>,
    ) -> HashMap<usize, Vec<usize>> {
        // Make sure that nodes_lhs is always equal or smaller than nodes_rhs.
        if nodes_lhs.len() > nodes_rhs.len() {
            nodes_rhs
        } else {
            nodes_lhs
        };

        let mut connection_map = HashMap::new();
        loop {
            // Some nodes in nodes_rhs might not have been assigned if node_rhs.len() >
            // node_lhs.len()
            let unassigned_rhs: Vec<usize> = nodes_rhs
                .iter()
                .filter(|index| !connection_map.contains_key(*index))
                .copied()
                .collect();
            if unassigned_rhs.is_empty() {
                break;
            }

            // Build weight matrix for Hungarian algo.
            let nodes_lhs_ = if nodes_lhs.len() > unassigned_rhs.len() {
                &nodes_lhs[..unassigned_rhs.len()]
            } else {
                nodes_lhs
            };
            let mut weights = Matrix::new(nodes_lhs_.len(), unassigned_rhs.len(), i32::MAX);
            for (i, node_lhs) in nodes_lhs_.iter().enumerate() {
                for (j, node_rhs) in unassigned_rhs.iter().enumerate() {
                    weights[(i, j)] = dis_matrix[[*node_lhs, *node_rhs]];
                }
            }

            let (_, assignment) = kuhn_munkres_min(&weights);
            assignment.iter().enumerate().for_each(|(i, &j)| {
                connection_map
                    .entry(nodes_lhs_[i])
                    .or_insert(Vec::new())
                    .push(unassigned_rhs[j]);

                connection_map
                    .entry(unassigned_rhs[j])
                    .or_insert(Vec::new())
                    .push(nodes_lhs_[i]);
            });
        }
        connection_map
    }

    fn new_leaf_hierarchy(assignment: &[usize]) -> Self {
        let mut clusters_map = HashMap::new();
        for (node_index, cluster_index) in assignment.iter().enumerate() {
            clusters_map
                .entry(cluster_index)
                .or_insert(Vec::new())
                .push(Node::new(node_index));
        }
        let mut clusters = HashMap::new();
        for (cluster_index, nodes) in clusters_map.into_iter() {
            clusters.insert(
                *cluster_index,
                Cluster::LeafCluster {
                    index: *cluster_index,
                    node_indices: nodes.iter().map(|node| node.index).collect(),
                    nodes,
                },
            );
        }
        Self { clusters }
    }

    fn build_matrix(dis_matrix: &Array2<i32>, medoids: &[usize]) -> Array2<i32> {
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

    fn connect_nodes_in_leaf_clusters(
        &self,
        node_connections: &mut HashMap<usize, BTreeMap<usize, Vec<usize>>>,
    ) {
        self.clusters
            .iter()
            .for_each(|(_, cluster)| cluster.connect_nodes_in_leaf_cluster(node_connections));
    }

    fn connect_nodes_in_root_clusters(
        &self,
        dis_matrix: &Array2<i32>,
        level: usize,
        node_connections: &mut HashMap<usize, BTreeMap<usize, Vec<usize>>>,
    ) {
        let cluster_indices: Vec<usize> = self.clusters.keys().copied().collect();
        for i in cluster_indices.iter() {
            for j in cluster_indices[i + 1..].iter() {
                let node_indices_lhs = self.clusters.get(i).unwrap().get_node_indices();
                let node_indices_rhs = self.clusters.get(j).unwrap().get_node_indices();
                let connection_map = NodeHierarchy::connect_clusters_greedy(
                    node_indices_lhs,
                    node_indices_rhs,
                    dis_matrix,
                );
                connection_map.into_iter().for_each(|(node_index, nodes)| {
                    node_connections
                        .entry(node_index)
                        .or_insert(BTreeMap::new())
                        .entry(level)
                        .or_insert(Vec::new())
                        .extend_from_slice(&nodes);
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, time::Instant};

    use ndarray::Array2;
    use ndarray_rand::rand_distr::{Distribution, UnitDisc};
    use rand::{self, Rng};

    use super::Cluster;
    use crate::bottom_up::NodeHierarchy;

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

    #[test]
    fn test_new_bar() {
        //let points = get_random_points(100, 10);
        let num_points = 1000;
        let cluster_size = 10;
        let points = get_random_points(num_points, num_points / cluster_size);
        let matrix = get_distance_matrix(&points);
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
                node_indices: _,
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
    fn test_get_assignments_map() {
        let leaf_cluster_0 = [0, 3, 6];
        let leaf_cluster_1 = [1, 4, 5];
        let leaf_cluster_2 = [2, 7, 8];
        let leaf_cluster_3 = [9, 10, 11];
        let target_cluster1 = vec![leaf_cluster_0, leaf_cluster_2];
        let target_cluster2 = vec![leaf_cluster_1, leaf_cluster_3];
        let _target_cluster = vec![target_cluster1, target_cluster2];

        let leaf_assignment = vec![0, 1, 2, 0, 1, 1, 0, 2, 2, 3, 3, 3];
        let mut leaf_hierarchy = NodeHierarchy::new_leaf_hierarchy(&leaf_assignment);

        // cluster_0 : [cluster_0, cluster_2]
        // cluster_1 : [cluster_1, cluster_3]
        let assignment = vec![0, 1, 0, 1];
        let matrix = Array2::zeros((12, 12)); // dummy matrix
        let mut node_connections = HashMap::new();
        let node_hierarchy = NodeHierarchy::new_hierarchy(
            &mut leaf_hierarchy,
            &mut node_connections,
            &matrix,
            &assignment,
            0,
        );
        let assignment_map = node_hierarchy.get_assignments_map();
        assert_eq!(
            assignment_map.get(&0).unwrap().get(&0).unwrap(),
            &[0, 3, 6, 2, 7, 8]
        );
        assert_eq!(
            assignment_map.get(&0).unwrap().get(&1).unwrap(),
            &[1, 4, 5, 9, 10, 11]
        );
        assert_eq!(assignment_map.get(&1).unwrap().get(&0).unwrap(), &[0, 3, 6]);
        assert_eq!(assignment_map.get(&1).unwrap().get(&1).unwrap(), &[1, 4, 5]);
        assert_eq!(assignment_map.get(&1).unwrap().get(&2).unwrap(), &[2, 7, 8]);
        assert_eq!(
            assignment_map.get(&1).unwrap().get(&3).unwrap(),
            &[9, 10, 11]
        );
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
        let matrix = Array2::zeros((12, 12)); // dummy matrix
        let mut node_connections = HashMap::new();
        let node_hierarchy = NodeHierarchy::new_hierarchy(
            &mut leaf_hierarchy,
            &mut node_connections,
            &matrix,
            &assignment,
            0,
        );
        assert_eq!(node_hierarchy.clusters.len(), 2);

        for (_, cluster) in node_hierarchy.clusters.iter() {
            if let Cluster::Cluster {
                index: cluster_index,
                total: _,
                node_indices: _,
                children,
            } = cluster
            {
                for (leaf_index, child) in children.iter().enumerate() {
                    if let Cluster::LeafCluster {
                        index: _,
                        node_indices: _,
                        nodes,
                    } = child
                    {
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
        let medoids = vec![0, 1, 4, 7, 10, 23, 44, 57, 66, 83];
        let new_matrix = NodeHierarchy::build_matrix(&matrix, &medoids);
        assert_eq!(new_matrix.shape()[0], medoids.len());
        assert_eq!(new_matrix.shape()[1], medoids.len());
        assert_eq!(new_matrix[[1, 3]], matrix[[1, 7]]);
        assert_eq!(new_matrix[[0, 6]], matrix[[0, 44]]);
        assert_eq!(new_matrix[[5, 8]], matrix[[23, 66]]);
    }
}
