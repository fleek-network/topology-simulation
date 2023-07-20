use std::{collections::BTreeMap, time::Instant};

use mcmf::{Capacity, Cost, GraphBuilder, Vertex};
use ndarray::Array2;
use rand::Rng;

/// Choose the best medoid within a partition
/// Used by ther alternating algorithm, or when a single cluster is requested.

pub struct ConstrainedKMedoids<'a> {
    matrix: &'a Array2<f64>,
    medoids: Vec<usize>,
    k: usize,
    n_nodes: usize,
    total: usize,
    min: usize,
    max: usize,
}

impl<'a> ConstrainedKMedoids<'a> {
    /// Create a new instance of constrained k medoids. The initial k medoids specifies the
    /// number of clusters that should be created. Min and max are constraints on the cluster sizes
    pub fn new(
        dissimilarity_matrix: &'a Array2<f64>,
        initial_k_medoids: &[usize],
        min: usize,
        max: usize,
    ) -> Self {
        let total = dissimilarity_matrix.nrows();
        let k = initial_k_medoids.len();
        let n_nodes = total - k;

        Self {
            min,
            max,
            k,
            n_nodes,
            total,
            medoids: initial_k_medoids.to_vec(),
            matrix: dissimilarity_matrix,
        }
    }

    /// Create a new instance of constrained k medoids, with random selection for the initial k
    /// medoids.
    pub fn with_rand_medoids<R: Rng>(
        dissimilarity_matrix: &'a Array2<f64>,
        k: usize,
        min: usize,
        max: usize,
        rng: &mut R,
    ) -> Self {
        let medoids = rand::seq::index::sample(rng, dissimilarity_matrix.nrows(), k).into_vec();
        Self::new(dissimilarity_matrix, &medoids, min, max)
    }

    /// Run an immediate min cost max flow graph problem on the data, without iterating on new
    /// medoids.
    pub fn mcmf(&self) -> Vec<usize> {
        self.build_solve_graph(&self.medoids).0
    }

    pub fn _mcmf_rand_iterations(&self, iterations: usize) -> Vec<usize> {
        let _cost = self.build_solve_graph(&self.medoids).1;

        let mut rng = rand::thread_rng();
        let _inner = Self::with_rand_medoids(
            self.matrix,
            self.medoids.len(),
            self.min,
            self.max,
            &mut rng,
        );

        for _ in 0..iterations {}

        vec![]
    }

    pub fn pam(&mut self, max_iterations: usize) -> Vec<usize> {
        let mut iters = 0;
        let mut changed = true;

        let (mut graph, mut cost) = self.build_solve_graph(&self.medoids);

        let mut now = Instant::now();
        while changed && iters < max_iterations {
            println!("iter: {iters}, last: {:?}", now.elapsed());
            now = Instant::now();
            iters += 1;

            let mut tmp = self.medoids.clone();

            for i in 0..self.total {
                // TODO: solution for finding best swap for this node
                for (m, &current_medoid) in self.medoids.iter().enumerate() {
                    // try swapping current medoid for each node and see if cost decreases
                    if i != current_medoid {
                        tmp[m] = i;
                        let (new_graph, new_cost) = self.build_solve_graph(&tmp);
                        if new_cost < cost {
                            changed |= true;
                            (graph, cost) = (new_graph, new_cost);
                        } else {
                            tmp[m] = current_medoid;
                        }
                    }
                }
            }
            if !changed {
                break;
            } else {
                self.medoids = tmp;
            }
        }

        graph
    }

    /// Run constrained alternating k-medoids. Returns a vector of node assignments, and the number
    /// of iterations taken.
    pub fn alternating(&mut self) -> (Vec<usize>, usize) {
        let mut iters = 0;
        let mut changed = true;

        while changed {
            let labels = self.build_solve_graph(&self.medoids).0;
            changed = self.choose_medoid_within_partitions(&labels);
            iters += 1;
        }

        (self.build_solve_graph(&self.medoids).0, iters)
    }

    /// used by alternating algorithm
    fn choose_medoid_within_partitions(&mut self, labels: &[usize]) -> bool {
        let mut changed = false;
        for (current_assignment, current_medoid) in self.medoids.iter_mut().enumerate() {
            let mut best = *current_medoid;
            let mut sum_b = 0.;
            for (i, &assignment) in labels.iter().enumerate() {
                if *current_medoid != i && assignment == current_assignment {
                    sum_b += self.matrix[(*current_medoid, i)];
                }
            }
            for (j, &assignment_j) in labels.iter().enumerate() {
                if j != *current_medoid && assignment_j == current_assignment {
                    let mut sum_j = 0.;
                    for (i, &assignment_i) in labels.iter().enumerate() {
                        if i != j && assignment_i == current_assignment {
                            sum_j += self.matrix[(j, i)];
                        }
                    }
                    if sum_j < sum_b {
                        best = j;
                        sum_b = sum_j;
                    }
                }
            }

            if *current_medoid != best {
                changed = true;
                *current_medoid = best;
            }
        }
        changed
    }

    /// Build and solve a min cost max flow graph for the given medoids
    ///
    /// - Non-medoids are supply nodes
    /// - medoid indeces do not have a role (hop used for max constraint)
    /// - medoid' indeces are demand nodes
    /// - one artificial demand node to ensure total demand = total supply
    fn build_solve_graph(&self, medoids: &[usize]) -> (Vec<usize>, usize) {
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
        //     - capacity:  n_supply - k * size_min
        //     - cost: 0

        const ARTIFICIAL_IDX: usize = usize::MAX;

        let mut graph = GraphBuilder::new();

        // non-medoids
        for i in 0..self.total {
            if !medoids.contains(&i) {
                // source -> supply node
                graph.add_edge(Vertex::Source, i, Capacity(1), Cost(0));

                for j in medoids {
                    // supply node -> medoid
                    let cost = (self.matrix[(i, *j)] * 1000.) as i32;
                    graph.add_edge(i, *j, Capacity(1), Cost(cost));
                }
            }
        }

        // medoids
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
            graph.add_edge(
                prime_idx,
                Vertex::Sink,
                Capacity((self.min) as i32),
                Cost(0),
            );
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

        // build clusters
        let mut clusters: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for path in paths {
            let verts = path.vertices();

            let node = verts[1].as_option().unwrap();
            let medoid = verts[2].as_option().unwrap();

            clusters.entry(medoid).or_insert(vec![medoid]).push(node);
        }

        // build label assignments (and rename cluster ids to sequential numbers)
        // default is 999 for debugging to find missing units
        let mut labels = vec![999; self.total];
        for (cluster, nodes) in clusters.values().enumerate() {
            for node in nodes {
                labels[*node] = cluster;
            }
        }

        // TEMPORARY: Find any missing nodes, and assign them to their closest cluster.
        // TODO: Figure out why some nodes aren't getting a path
        for (i, assignment) in labels.iter_mut().enumerate() {
            if assignment == &999 {
                let mut best = 999;
                let mut diff = f64::MAX;
                for (j, &medoid) in medoids.iter().enumerate() {
                    let diff2 = self.matrix[(i, medoid)];
                    if diff2 < diff {
                        best = j;
                        diff = diff2;
                    }
                }
                *assignment = best;
            }
        }
        (labels, total_cost as usize)
    }
}
