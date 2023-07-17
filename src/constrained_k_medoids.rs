use std::collections::{BTreeMap, BTreeSet};

use mcmf::{Capacity, Cost, GraphBuilder, Vertex};
use ndarray::Array2;

pub struct ConstrainedKMedoids {
    matrix: Array2<f64>,
    medoids: BTreeSet<usize>,
    k: usize,
    n_nodes: usize,
    total: usize,
    min: usize,
    max: usize,
}

impl ConstrainedKMedoids {
    /// Create a new instance of constrained k medoids. The initial k medoids specifies the
    /// number of clusters that should be created, and the dissimilarity matrix specifies the
    /// number of nodes. Min and max are constraints on the cluster sizes
    pub fn new(
        dissimilarity_matrix: Array2<f64>,
        initial_k_medoids: &[usize],
        min: usize,
        max: usize,
    ) -> Self {
        let total = dissimilarity_matrix.nrows();

        let mut nodes = BTreeSet::from_iter(0..total);
        for i in initial_k_medoids {
            nodes.remove(i);
        }
        let medoids = BTreeSet::from_iter(initial_k_medoids.iter().cloned());
        let k = medoids.len();
        let n_nodes = total - k;

        Self {
            min,
            max,
            k,
            n_nodes,
            total,
            medoids,
            matrix: dissimilarity_matrix,
        }
    }

    /// Create a new instance of constrained k medoids, with random selection for the initial k medoids.
    pub fn with_rand_medoids(
        dissimilarity_matrix: Array2<f64>,
        k: usize,
        min: usize,
        max: usize,
    ) -> Self {
        let medoids = vec![0; k];
        // TODO: random non repeating sequence between 0..matrix.len()

        Self::new(dissimilarity_matrix, &medoids, min, max)
    }

    // Run constrained partitioning around medoids.
    pub fn cpam(&mut self) -> Vec<usize> {
        let meds: Vec<usize> = self.medoids.iter().cloned().collect();
        self.build_solve_graph(&meds)
    }

    /// Build and solve a min cost max flow graph for the given medoids
    ///
    /// - Non-medoids are supply nodes
    /// - medoid indeces do not have a role (hop used for max constraint)
    /// - medoid' indeces are demand nodes
    /// - one artificial demand node to ensure total demand = total supply
    fn build_solve_graph(&self, medoids: &[usize]) -> Vec<usize> {
        // - Edges
        //   - source -> [supply nodes]
        //     - capacity: 1
        //     - cost: 0
        //   - [supply nodes] -> [medoid]
        //     - capacity: 1
        //     - cost: lookup via DS matrix
        //   - [medoid -> medoid'] (all pairs)
        //     - capacity: size_max
        //     - cost: 0
        //   - [medoid'] -> artificial node
        //     - capacity: n_supply
        //     - cost: 0
        //   - [medoid'] -> sink
        //     - capacity: size_min
        //     - cost: 0
        //   - artificial node -> sink
        //     - capacity:  n_supply - n_medoids * size_min
        //     - cost: 0

        const ARTIFICIAL_IDX: usize = usize::MAX;

        let mut graph = GraphBuilder::new();

        for i in 0..self.total {
            if !medoids.contains(&i) {
                // source -> supply node
                graph.add_edge(Vertex::Source, i, Capacity(1), Cost(0));

                for &j in medoids {
                    // supply node -> medoid
                    let cost = (self.matrix[(i, j)] * 1000.) as i32;
                    graph.add_edge(i, j, Capacity(1), Cost(cost));
                }
            }
        }

        for (prime_offset, &idx) in medoids.iter().enumerate() {
            let prime_idx = usize::MAX - prime_offset - 1;

            // medoid -> medoid'
            graph.add_edge(idx, prime_idx, Capacity(self.max as i32), Cost(0));

            // medoid' -> artificial
            graph.add_edge(
                prime_idx,
                ARTIFICIAL_IDX,
                Capacity(self.n_nodes as i32),
                Cost(0),
            );

            // medoid' -> sink
            graph.add_edge(prime_idx, Vertex::Sink, Capacity(self.min as i32), Cost(0));
        }

        // artificial node -> sink
        graph.add_edge(
            ARTIFICIAL_IDX,
            Vertex::Sink,
            Capacity((self.n_nodes - self.k * self.min) as i32),
            Cost(0),
        );

        // solve graph
        let (total_cost, paths) = graph.mcmf();
        println!("{total_cost}");
        let mut mappings: BTreeMap<usize, Vec<usize>> = BTreeMap::new();

        for path in paths {
            let verts = path.vertices();

            let node = verts[1].as_option().unwrap();
            let medoid = verts[2].as_option().unwrap();

            let ids = mappings.entry(medoid).or_insert(vec![medoid]);
            ids.push(node);
        }

        let mut labels = vec![0; self.total];
        for (cluster, nodes) in mappings.values().enumerate() {
            for node in nodes {
                labels[*node] = cluster;
            }
        }
        labels
    }
}
