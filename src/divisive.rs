use std::{collections::BTreeMap, fmt::Display};

use ndarray::Array2;
use pathfinding::prelude::Matrix;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    constrained_fasterpam,
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
    current_id: usize,
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
            current_id: temp.id,
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

impl DivisiveHierarchy {
    /// Create a new divisive hierarchy using constrained fasterpam and selecting first
    pub fn new(dissim_matrix: &Array2<f64>, target_n: usize) -> Self {
        let indeces: Vec<_> = (0..dissim_matrix.nrows()).map(|i| (i, i)).collect();
        Self::new_inner(
            &mut rand::thread_rng(),
            dissim_matrix,
            &indeces,
            target_n,
            &HierarchyPath::root(),
            false,
        )
    }

    fn new_inner<R: Rng>(
        rng: &mut R,
        dissim_matrix: &Array2<f64>,
        indeces: &[(usize, usize)],
        target_n: usize,
        current_path: &HierarchyPath,
        last: bool,
    ) -> Self {
        if last {
            // current list of nodes are within the target size
            Self::Cluster {
                id: current_path.to_string(),
                // collect the top level indeces for the nodes
                nodes: indeces.iter().map(|i| i.0).collect(),
            }
        } else {
            // Split the current indeces in half using constrained fasterpam with random init
            let mut medoids = rand::seq::index::sample(rng, dissim_matrix.nrows(), 2).into_vec();

            let half = indeces.len() / 2;
            let min = half - 1;
            let max = half + 1;
            let (_, assignments, _, _) = constrained_fasterpam::fasterpam::<_, f64>(
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

            // TODO: compute pairings between the clusters using a hungarian problem
            //
            // X: Cluster A nodes
            // Y: Cluster B nodes
            //
            // The smaller cluster should be placed on the left

            // Recurse new children for each cluster
            let last = clusters[0].len() < target_n || clusters[1].len() < target_n;
            let mut children = Vec::with_capacity(2);
            for (path_index, new_indeces) in clusters.iter().enumerate() {
                let nodes: Vec<_> = new_indeces
                    .iter()
                    .map(|&i| {
                        let mut node = indeces[i];
                        node.1 = i;
                        node
                    })
                    .collect();

                // build new matrix from medoids
                let mut child_matrix = Array2::zeros((nodes.len(), nodes.len()));
                for (i, &iidx) in new_indeces.iter().enumerate() {
                    for (j, &jidx) in new_indeces[i + 1..].iter().enumerate() {
                        let dissim = dissim_matrix[(iidx, jidx)];
                        child_matrix[(i, j)] = dissim;
                        child_matrix[(j, i)] = dissim;
                    }
                }
                // pass new matrix and indeces
                let mut path = current_path.clone();
                path.0.push(path_index as u8);
                let child = Self::new_inner(rng, &child_matrix, &nodes, target_n, &path, last);
                children.push(child);
            }

            Self::Group {
                id: current_path.to_string(),
                total: indeces.len(),
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

    /// Collect assignments for each node in the hierarchy, returning cluster indexes
    pub fn assignments(&self) -> Vec<usize> {
        fn inner(counter: &mut usize, assignments: &mut [usize], item: &DivisiveHierarchy) {
            match item {
                DivisiveHierarchy::Group { children, .. } => {
                    for child in children {
                        inner(counter, assignments, child);
                    }
                },
                DivisiveHierarchy::Cluster { nodes, .. } => {
                    *counter += 1;
                    for &node in nodes {
                        assignments[node] = *counter;
                    }
                },
            }
        }

        let total = self.n_nodes();
        let mut assignments = vec![999; total];
        let mut counter = 0;

        inner(&mut counter, &mut assignments, self);

        assignments
    }
}
