use clustering::{bottom_up::NodeHierarchy, divisive::DivisiveHierarchy};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use ndarray_rand::rand_distr::{Distribution, UnitDisc};
use rand::{self, Rng};

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

fn run_divisive_constrained_fasterpam(dis_matrix: &Array2<i32>) -> DivisiveHierarchy {
    DivisiveHierarchy::new(&mut rand::thread_rng(), dis_matrix, 8)
}

fn run_bottom_up_clustering(dis_matrix: &Array2<i32>) -> NodeHierarchy {
    let target_size = 8;
    let num_clusters = dis_matrix.shape()[0] / target_size;
    NodeHierarchy::new(
        dis_matrix,
        num_clusters,
        target_size - 1,
        target_size + 1,
        100,
    )
}

fn criterion_benchmark(c: &mut Criterion) {
    let sizes = vec![1000, 2000, 5000, 10000, 20000];
    let mut c = c.benchmark_group("Hierarchy");
    c.sample_size(20);

    for size in sizes {
        c.bench_with_input(
            BenchmarkId::new("Divisive Constrained FasterPAM", size),
            &size,
            |b, size| {
                let clusters = (size + 7) / 8;
                let points = get_random_points(*size, clusters);
                let matrix = get_distance_matrix(&points);
                b.iter(|| run_divisive_constrained_fasterpam(black_box(&matrix)))
            },
        );

        c.bench_with_input(
            BenchmarkId::new("Bottom Up Clustering", size),
            &size,
            |b, size| {
                let clusters = (size + 7) / 8;
                let points = get_random_points(*size, clusters);
                let matrix = get_distance_matrix(&points);
                b.iter(|| run_bottom_up_clustering(black_box(&matrix)))
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
