use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Array, Dim};
use ndarray_rand::rand_distr::{Distribution, UnitDisc};
use rand::{self, Rng};

fn get_random_points(num_nodes: usize, num_clusters: usize) -> Array<f64, Dim<[usize; 2]>> {
    let mut points = Array::zeros((num_nodes, 2));
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

fn get_distance_matrix(data: &Array<f64, Dim<[usize; 2]>>) -> Array<f64, Dim<[usize; 2]>> {
    let mut dist = Array::zeros((data.shape()[0], data.shape()[0]));
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

fn run_kmedoids(num_nodes: usize, num_clusters: usize) -> (f64, Vec<usize>, usize, usize) {
    let points = get_random_points(num_nodes, num_clusters);
    let dis_matrix = get_distance_matrix(&points);
    let mut meds = kmedoids::random_initialization(
        dis_matrix.shape()[0],
        num_clusters,
        &mut rand::thread_rng(),
    );
    kmedoids::fasterpam(&dis_matrix, &mut meds, 100)
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("kmedoids 10_000", |b| {
        b.iter(|| run_kmedoids(black_box(10_000), black_box(10_000 / 8)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
