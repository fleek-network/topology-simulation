use std::collections::{BTreeMap, HashMap};

use ndarray::{Array, Dim};
use ndarray_rand::rand_distr::{Distribution, UnitDisc};
use serde::{Deserialize, Serialize};

use csv::ReaderBuilder;
use std::error::Error;

use plotters::prelude::*;

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ServerData {
    id: u16,
    name: String,
    title: String,
    location: String,
    state: String,
    country: String,
    state_abbv: String,
    continent: String,
    latitude: f32,
    longitude: f32,
}

fn scatter_plot(
    data: &Array<f64, Dim<[usize; 2]>>,
    assignment: &[usize],
    output_path: &str,
    title: &str,
) {
    let mut series = HashMap::new();
    let mut x_min = f64::MAX;
    let mut x_max = f64::MIN;
    let mut y_min = f64::MAX;
    let mut y_max = f64::MIN;
    for (i, cluster_index) in assignment.iter().enumerate() {
        let x = data[[i, 0]];
        let y = data[[i, 1]];

        x_min = x_min.min(x);
        x_max = x_max.max(x);
        y_min = y_min.min(y);
        y_max = y_max.max(y);
        series
            .entry(cluster_index)
            .or_insert(Vec::new())
            .push((x, y));
    }

    let root_area = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    // make the plot wider
    x_min -= 5.;
    x_max += 5.;
    y_min -= 40.;
    y_max += 20.;

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption(title, ("sans-serif", 40))
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    let colors = vec![BLUE, CYAN, RED, MAGENTA, YELLOW, GREEN, RED];

    series.iter().for_each(|(&cluster_index, points)| {
        let color = if series.len() > colors.len() {
            let h = *cluster_index as f64 / series.len() as f64;
            ViridisRGB::get_color(h)
        } else {
            colors[*cluster_index]
        };
        ctx.draw_series(points.iter().map(|point| Circle::new(*point, 5, color)))
            .unwrap();
    });
}

fn sample_cluster(
    x: f64,
    y: f64,
    x_scale: f64,
    y_scale: f64,
    num_points: usize,
) -> Array<f64, Dim<[usize; 2]>> {
    let mut cluster = Array::zeros((num_points, 2));

    for i in 0..num_points {
        let v: [f64; 2] = UnitDisc.sample(&mut rand::thread_rng());
        cluster[[i, 0]] = x + v[0] * x_scale;
        cluster[[i, 1]] = y + v[1] * y_scale;
    }
    cluster
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

fn run_kmedoids(
    dis_matrix: &Array<f64, Dim<[usize; 2]>>,
    num_clusters: usize,
) -> (f64, Vec<usize>, usize, usize) {
    let mut meds = kmedoids::random_initialization(
        dis_matrix.shape()[0],
        num_clusters,
        &mut rand::thread_rng(),
    );
    kmedoids::fasterpam(dis_matrix, &mut meds, 100)
}

fn read_latency_matrix(path: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let mut buf = Vec::new();

    let mut rdr = ReaderBuilder::new().from_path(path)?;
    for result in rdr.deserialize() {
        let record: Vec<f32> = result?;
        buf.push(record);
    }
    Ok(buf)
}

fn read_metadata(path: &str) -> Result<BTreeMap<u16, ServerData>, Box<dyn Error>> {
    let mut metadata = BTreeMap::new();
    let mut rdr = ReaderBuilder::new().from_path(path)?;
    for result in rdr.deserialize() {
        let record: ServerData = result?;
        metadata.insert(record.id, record);
    }
    Ok(metadata)
}

fn toy_example() {
    let cluster1 = sample_cluster(1., 1., 1., 1., 10);
    let cluster2 = sample_cluster(2., 2., 1., 1., 10);
    let cluster3 = sample_cluster(3., 3., 2., 2., 20);
    let data = ndarray::concatenate(
        ndarray::Axis(0),
        &[(&cluster1).into(), (&cluster2).into(), (&cluster3).into()],
    )
    .unwrap();

    scatter_plot(
        &data,
        &vec![0; data.shape()[0]],
        "before.png",
        "Before Clustering",
    );

    let dis_matrix = get_distance_matrix(&data);
    let (_, assignment, _, _) = run_kmedoids(&dis_matrix, 3);

    scatter_plot(&data, &assignment, "after.png", "After Clustering");
}

fn main() {
    let matrix = read_latency_matrix("matrix.csv").unwrap();
    let metadata = read_metadata("metadata.csv").unwrap();

    let mut data_points = Array::zeros((metadata.len(), 2));
    metadata
        .iter()
        .enumerate()
        .for_each(|(i, (_, server_data))| {
            data_points[[i, 1]] = server_data.latitude as f64;
            data_points[[i, 0]] = server_data.longitude as f64;
        });

    scatter_plot(
        &data_points,
        &vec![0; data_points.shape()[0]],
        "before.png",
        "Before Clustering",
    );

    let mut dissim_matrix = Array::zeros((matrix.len(), matrix.len()));
    for i in 0..matrix.len() {
        for j in 0..matrix.len() {
            dissim_matrix[[i, j]] = matrix[i][j] as f64;
        }
    }
    let num_servers = dissim_matrix.shape()[0];
    let optimal_cluster_size = 8;
    let num_clusters = num_servers / optimal_cluster_size;
    let (_, assignment, num_iterstions, _) = run_kmedoids(&dissim_matrix, num_clusters);
    scatter_plot(&data_points, &assignment, "after.png", "After Clustering");
    println!("num_iterstions: {num_iterstions}");
}
