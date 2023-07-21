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

fn get_distance_matrix(data: &Array2<f64>) -> Array2<f64> {
    let mut dist = Array2::zeros((data.shape()[0], data.shape()[0]));
    for i in 0..data.shape()[0] {
        for j in 0..data.shape()[0] {
            if i != j {
                dist[[i, j]] =
                    (data[[i, 0]] - data[[j, 0]]).powi(2) + (data[[i, 1]] - data[[j, 1]]).powi(2);
            }
        }
    }
    dist
}

fn run_fasterpam(dis_matrix: &Array2<f64>, num_clusters: usize) -> (f64, Vec<usize>, usize, usize) {
    let mut meds = kmedoids::random_initialization(
        dis_matrix.shape()[0],
        num_clusters,
        &mut rand::thread_rng(),
    );
    kmedoids::fasterpam(dis_matrix, &mut meds, 100)
}

fn run_constrained_fasterpam(
    dis_matrix: &Array2<f64>,
    num_clusters: usize,
) -> (f64, Vec<usize>, usize, usize) {
    let mut meds = kmedoids::random_initialization(
        dis_matrix.shape()[0],
        num_clusters,
        &mut rand::thread_rng(),
    );
    kmedoids::fasterpam(dis_matrix, &mut meds, 100)
}

fn criterion_benchmark(c: &mut Criterion) {
    let sizes = vec![1000, 2000, 5000, 10000, 20000];

    for size in sizes {
        c.bench_with_input(BenchmarkId::new("FasterPAM", size), &size, |b, size| {
            let clusters = (size + 7) / 8;
            let points = get_random_points(*size, clusters);
            let matrix = get_distance_matrix(&points);
            b.iter(|| run_fasterpam(black_box(&matrix), clusters))
        });

        c.bench_with_input(
            BenchmarkId::new("Constrained FasterPAM", size),
            &size,
            |b, size| {
                let clusters = (size + 7) / 8;
                let points = get_random_points(*size, clusters);
                let matrix = get_distance_matrix(&points);
                b.iter(|| run_constrained_fasterpam(black_box(&matrix), clusters))
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
