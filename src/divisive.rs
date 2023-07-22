use std::{collections::BTreeMap, fmt::Display};

use ndarray::Array2;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    constrained_fasterpam,
    pairing::greedy_pairs,
    types::{SerializedLayer, SerializedNode},
};

/// A divisive hierarchy strategy where we split the nodes into clusters of two, until the size
/// is less than the target n
#[derive(Debug, Clone)]
pub enum DivisiveHierarchy {
    Group {
        id: String,
        total: usize,
        children: Vec<DivisiveHierarchy>,
        nodes: Vec<Node>,
    },
    Cluster {
        id: String,
        nodes: Vec<Node>,
    },
}

impl From<&DivisiveHierarchy> for SerializedLayer {
    fn from(value: &DivisiveHierarchy) -> Self {
        match value {
            DivisiveHierarchy::Group {
                id,
                total,
                children,
                ..
            } => SerializedLayer::Group {
                id: id.clone(),
                total: *total,
                children: children.iter().map(|c| c.into()).collect(),
            },
            DivisiveHierarchy::Cluster { id, nodes } => SerializedLayer::Cluster {
                id: id.clone(),
                nodes: nodes.iter().map(|n| n.into()).collect(),
            },
        }
    }
}

impl Serialize for DivisiveHierarchy {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        SerializedLayer::from(self).serialize(serializer)
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    id: usize,
    connections: BTreeMap<usize, Vec<usize>>,
}

impl From<&Node> for SerializedNode {
    fn from(val: &Node) -> Self {
        SerializedNode {
            id: val.id,
            connections: val.connections.values().cloned().collect(),
        }
    }
}

impl Serialize for Node {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        SerializedNode::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Node {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let temp = SerializedNode::deserialize(deserializer)?;
        Ok(Self {
            id: temp.id,
            connections: BTreeMap::from_iter(temp.connections.iter().cloned().enumerate()),
        })
    }
}

#[derive(Debug, Clone)]
pub struct HierarchyPath(Vec<u8>);

impl HierarchyPath {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self(bytes.to_vec())
    }
    pub fn root() -> Self {
        Self(vec![])
    }
    pub fn depth(&self) -> usize {
        self.0.len()
    }
}

impl Display for HierarchyPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let strings: Vec<_> = self.0.iter().map(|v| v.to_string()).collect();
        if strings.is_empty() {
            write!(f, "root")
        } else {
            write!(f, "{}", strings.join("."))
        }
    }
}

fn add_connection(indeces: &mut [Node], depth: usize, i: usize, j: usize) {
    let jid = indeces[j].id;
    let left = &mut indeces[i];
    left.connections.entry(depth).or_default().push(jid);
    let iid = left.id;
    let right = &mut indeces[j];
    right.connections.entry(depth).or_default().push(iid);
}

impl DivisiveHierarchy {
    /// Create a new divisive hierarchy using constrained fasterpam and selecting first
    pub fn new(dissim_matrix: &Array2<i32>, target_n: usize) -> Self {
        let nodes: Vec<_> = (0..dissim_matrix.nrows())
            .map(|i| Node {
                id: i,
                connections: BTreeMap::new(),
            })
            .collect();

        Self::new_inner(
            &mut rand::thread_rng(),
            dissim_matrix,
            nodes,
            target_n,
            &HierarchyPath::root(),
            false,
        )
    }

    fn new_inner<R: Rng>(
        rng: &mut R,
        dissim_matrix: &Array2<i32>,
        mut indeces: Vec<Node>,
        target_n: usize,
        current_path: &HierarchyPath,
        last: bool,
    ) -> Self {
        if last {
            // add connections to every other node in the cluster for each node in the cluster
            let depth = current_path.depth();
            let ids: Vec<_> = indeces.iter().map(|n| n.id).collect();
            for node in indeces.iter_mut() {
                let conns = node.connections.entry(depth).or_default();
                for id in &ids {
                    if id != &node.id {
                        conns.push(*id);
                    }
                }
            }

            // current list of nodes are within the target size
            Self::Cluster {
                id: current_path.to_string(),
                // collect the top level indeces for the nodes
                nodes: indeces.iter().map(|n| (*n).clone()).collect(),
            }
        } else {
            // Split the current indeces in half using constrained fasterpam with random init
            let mut medoids = rand::seq::index::sample(rng, dissim_matrix.nrows(), 2).into_vec();

            let half = indeces.len() / 2;
            let min = half - 1;
            let max = half + 1;
            let (_, assignments, _, _) = constrained_fasterpam::fasterpam::<_, i32>(
                dissim_matrix,
                &mut medoids,
                100,
                min,
                max,
            );

            let mut clusters = vec![vec![], vec![]];
            for (node, &assignment) in assignments.iter().enumerate() {
                clusters[assignment].push(node);
            }

            // greedily pair nodes together
            let pairs = greedy_pairs(dissim_matrix, &clusters[0], &clusters[1]);
            let depth = current_path.depth();
            for (i, j) in pairs {
                add_connection(&mut indeces, depth, i, j);
            }

            // Recurse new children for each cluster
            // Stopping criteria is if either cluster is less than the target n size
            let last = clusters[0].len() < target_n || clusters[1].len() < target_n;
            let mut children = Vec::with_capacity(2);
            for (path_index, new_indeces) in clusters.iter().enumerate() {
                // build new matrix from medoids
                let mut child_matrix = Array2::zeros((new_indeces.len(), new_indeces.len()));

                for (i, &iidx) in new_indeces.iter().enumerate() {
                    for (mut j, &jidx) in new_indeces[i + 1..].iter().enumerate() {
                        j += i + 1;
                        let dissim = dissim_matrix[(iidx, jidx)];
                        child_matrix[(i, j)] = dissim;
                        child_matrix[(j, i)] = dissim;
                    }
                }

                // create a new child with the new matrix and indeces
                let mut path = current_path.clone();
                path.0.push(path_index as u8);

                let nodes: Vec<_> = new_indeces.iter().map(|&i| indeces[i].clone()).collect();
                let child = Self::new_inner(rng, &child_matrix, nodes, target_n, &path, last);
                children.push(child);
            }

            Self::Group {
                id: current_path.to_string(),
                total: indeces.len(),
                nodes: indeces,
                children,
            }
        }
    }

    /// Get the total number of nodes in the hierarchy
    pub fn n_nodes(&self) -> usize {
        match self {
            Self::Group { total, .. } => *total,
            Self::Cluster { nodes, .. } => nodes.len(),
        }
    }

    /// Collect assignments for each node at each depth of the hierarchy. The last vec of
    /// assignments is the final tree depth.
    pub fn assignments(&self) -> Vec<Vec<usize>> {
        fn inner(
            item: &DivisiveHierarchy,
            data: &mut BTreeMap<usize, (usize, Vec<usize>)>,
            depth: usize,
            total: usize,
        ) {
            let (counter, assignments) = data.entry(depth).or_insert((0, vec![0; total]));
            let current = *counter;
            *counter += 1;
            match item {
                DivisiveHierarchy::Group {
                    children, nodes, ..
                } => {
                    // set assignments
                    for node in nodes {
                        assignments[node.id] = current;
                    }
                    // recurse for each child item
                    for child in children {
                        inner(child, data, depth + 1, total);
                    }
                },
                DivisiveHierarchy::Cluster { nodes, .. } => {
                    for node in nodes {
                        assignments[node.id] = current;
                    }
                },
            }
        }

        let mut data = BTreeMap::new();
        let total = self.n_nodes();
        inner(self, &mut data, 0, total);
        data.into_values().map(|v| v.1).collect()
    }
}
