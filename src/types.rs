use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct SerializedNode {
    pub id: usize,
    pub connections: Vec<Vec<usize>>,
}

#[derive(Serialize, Deserialize)]
pub enum SerializedLayer {
    Group {
        id: String,
        total: usize,
        children: Vec<SerializedLayer>,
    },
    Cluster {
        id: String,
        nodes: Vec<SerializedNode>,
    },
}
