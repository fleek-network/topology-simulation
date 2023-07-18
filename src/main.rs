mod constrained_fasterpam;
mod constrained_k_medoids;

use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use base64::Engine;
use constrained_k_medoids::ConstrainedKMedoids;
use ndarray::{Array, Dim};
use ndarray_rand::rand_distr::{Distribution, UnitDisc};
use rand::Rng;
use serde::{Deserialize, Serialize};

use csv::ReaderBuilder;
use std::error::Error;

use plotters::prelude::*;

const FONT: &str = "IBM Plex Mono, monospace";

fn main() {
    toy_example();
    run();
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ServerData {
    id: u16,
    title: String,
    country: String,
    latitude: f32,
    longitude: f32,
}

fn histogram(values: &[f64], min_val: f64, max_val: f64, title: &str) -> String {
    let root = BitMapBackend::new("/tmp/histogram.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let data: Vec<u32> = values.iter().map(|v| v.round() as u32).collect();

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(40)
        .margin(5)
        .caption(title, ("monospace", 40.0))
        .build_cartesian_2d(
            (min_val as u32..max_val as u32).into_segmented(),
            0u32..10u32,
        )
        .unwrap();

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(WHITE.mix(0.3))
        .y_desc("Count")
        .x_desc("Bucket")
        .axis_desc_style(("monospace", 15))
        .draw()
        .unwrap();

    chart
        .draw_series(
            Histogram::vertical(&chart)
                .style(BLUE.mix(0.5).filled())
                .data(data.iter().map(|x: &u32| (*x, 1))),
        )
        .unwrap();

    root.present()
        .expect("Unable to write temp histogram to file");

    let file = std::fs::read("/tmp/histogram.png").expect("failed to read temp file");
    base64::engine::general_purpose::STANDARD.encode(file)
}

fn scatter_plot(
    buffer: &mut String,
    data: &Array<f64, Dim<[usize; 2]>>,
    assignment: &[usize],
    title: &str,
) {
    let mut series = BTreeMap::new();
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

    let root_area = SVGBackend::with_string(buffer, (1200, 800)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    // make the plot wider
    x_min -= 5.;
    x_max += 5.;
    y_min -= 40.;
    y_max += 20.;

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption(title, (FONT, 40))
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    let color_map = DerivedColorMap::new(&[
        RGBColor(230, 25, 75),
        RGBColor(60, 180, 75),
        RGBColor(255, 225, 25),
        RGBColor(0, 130, 200),
        RGBColor(245, 130, 48),
        RGBColor(145, 30, 180),
        RGBColor(70, 240, 240),
        RGBColor(240, 50, 230),
        RGBColor(210, 245, 60),
        RGBColor(250, 190, 212),
        RGBColor(0, 128, 128),
        RGBColor(220, 190, 255),
        RGBColor(170, 110, 40),
        RGBColor(255, 250, 200),
        RGBColor(128, 0, 0),
        RGBColor(170, 255, 195),
        RGBColor(128, 128, 0),
        RGBColor(255, 215, 180),
        RGBColor(0, 0, 128),
        RGBColor(128, 128, 128),
        RGBColor(0, 0, 0),
    ]);

    series.iter().for_each(|(&cluster_index, points)| {
        let color = color_map.get_color(*cluster_index as f64 / series.len() as f64);

        ctx.draw_series(points.iter().map(|point| Circle::new(*point, 5, color)))
            .unwrap()
            .label(format!("Cluster {cluster_index} ({})", points.len()))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    });

    ctx.configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .label_font(FONT)
        .margin(15)
        .position(SeriesLabelPosition::LowerLeft)
        .draw()
        .unwrap();
}

fn run_fasterpam(
    dis_matrix: &Array<f64, Dim<[usize; 2]>>,
    num_clusters: usize,
) -> (Vec<usize>, usize, Duration) {
    let mut meds = kmedoids::random_initialization(
        dis_matrix.shape()[0],
        num_clusters,
        &mut rand::thread_rng(),
    );
    let instant = Instant::now();
    let (_, labels, iterations, _): (f64, _, _, _) =
        kmedoids::fasterpam(dis_matrix, &mut meds, 100);
    (labels, iterations, instant.elapsed())
}

fn run_constrained_fasterpam(
    dis_matrix: &Array<f64, Dim<[usize; 2]>>,
    num_clusters: usize,
) -> (Vec<usize>, usize, Duration) {
    let mut meds = kmedoids::random_initialization(
        dis_matrix.shape()[0],
        num_clusters,
        &mut rand::thread_rng(),
    );
    let instant = Instant::now();
    let (_, labels, iterations, _): (f64, _, _, _) =
        constrained_fasterpam::fasterpam(dis_matrix, &mut meds, 100);
    (labels, iterations, instant.elapsed())
}

fn run_constrained_alternating(
    dis_matrix: &Array<f64, Dim<[usize; 2]>>,
    num_clusters: usize,
) -> (Vec<usize>, usize, Duration) {
    let instant = Instant::now();
    let mut alg = ConstrainedKMedoids::with_rand_medoids(
        dis_matrix,
        num_clusters,
        4,
        12,
        &mut rand::thread_rng(),
    );
    let (labels, iterations) = alg.alternating();
    (labels, iterations, instant.elapsed())
}

fn get_random_assignment(num_clusters: usize, num_nodes: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();

    let mut assignment = vec![0; num_nodes];
    for node_idx in 0..num_nodes {
        let cluster_index = rng.gen_range(0..num_clusters);
        assignment[node_idx] = cluster_index;
    }
    assignment
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

fn calculate_cluster_metrics(
    assignment: &[usize],
    latency_matrix: &[Vec<f32>],
) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let mut clusters = BTreeMap::new();
    for (i, cluster_index) in assignment.iter().enumerate() {
        clusters.entry(cluster_index).or_insert(Vec::new()).push(i);
    }
    let mut mean_inner_cluster_latencies = Vec::new();
    let mut cluster_node_count = Vec::new();
    let mut inner_cluster_latency_sums = 0.0;
    let mut inner_cluster_latency_mean = 0.0;
    for node_indices in clusters.values() {
        let mut latency_sum = 0.0;
        let mut count = 0;

        if node_indices.is_empty() {
            continue;
        }

        for (i, &src) in node_indices.iter().enumerate() {
            for &dst in node_indices[i + 1..].iter() {
                let sum = latency_matrix[src][dst] as f64;
                count += 1;
                latency_sum += sum;
                inner_cluster_latency_sums += sum;
            }
        }
        mean_inner_cluster_latencies.push(latency_sum / count as f64);
        inner_cluster_latency_mean += latency_sum / count as f64;
        cluster_node_count.push(node_indices.len() as f64);
    }
    (
        mean_inner_cluster_latencies,
        cluster_node_count,
        inner_cluster_latency_sums,
        inner_cluster_latency_mean / clusters.len() as f64,
    )
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

fn toy_example() {
    let cluster1 = sample_cluster(1., 1., 1., 1., 10);
    let cluster2 = sample_cluster(4., 3., 1., 1., 10);
    let cluster3 = sample_cluster(5., 5., 1., 1., 20);
    let data = ndarray::concatenate(
        ndarray::Axis(0),
        &[(&cluster1).into(), (&cluster2).into(), (&cluster3).into()],
    )
    .unwrap();

    let mut plot_buffer = String::new();

    scatter_plot(
        &mut plot_buffer,
        &data,
        &vec![0; data.shape()[0]],
        "Before Clustering",
    );

    let dis_matrix = get_distance_matrix(&data);

    let (assignment, _, _) = run_fasterpam(&dis_matrix, 3);
    for (i, &a) in assignment.iter().enumerate() {
        if a == 999 {
            println!("missing: {i}");
        }
    }
    scatter_plot(&mut plot_buffer, &data, &assignment, "FasterPAM");

    let (assignment, _, _) = run_constrained_fasterpam(&dis_matrix, 3);
    scatter_plot(
        &mut plot_buffer,
        &data,
        &assignment,
        "WIP Constrained FasterPAM",
    );

    let (assignment, _, _) = run_constrained_alternating(&dis_matrix, 3);
    scatter_plot(
        &mut plot_buffer,
        &data,
        &assignment,
        "WIP Constrained Alternating",
    );

    let html = format!(
        r#"
<!DOCTYPE html>
<html>
<head>
    <title>Topology Simulation Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
        body {{ font-family: '{FONT}', monospace }}
        tr:nth-child(even) {{
            background-color: rgba(150, 212, 212, 0.4);
        }}
    </style>
</head>
<body>
    <h1>Toy Example Report</h1>
    <p>The clustering </p>
    <hr>
    {plot_buffer}
    <hr>
<body>
</html>
          "#
    );
    std::fs::write("toy_report.html", html).unwrap();
}

fn run() {
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

    let mut plot_buffer = String::new();

    scatter_plot(
        &mut plot_buffer,
        &data_points,
        &vec![0; data_points.shape()[0]],
        "Before Clustering",
    );

    let mut dissim_matrix = Array::zeros((matrix.len(), matrix.len()));
    for i in 0..matrix.len() {
        for j in 0..matrix.len() {
            dissim_matrix[[i, j]] = matrix[i][j] as f64;
        }
    }
    let num_servers = dissim_matrix.shape()[0];
    let optimal_cluster_size = 10;
    let num_clusters = num_servers / optimal_cluster_size;
    let mut min_latency_val = f64::MAX;
    let mut max_latency_val = f64::MIN;
    let mut min_size_val = f64::MAX;
    let mut max_size_val = f64::MIN;

    /* BASELINE: RANDOM ASSIGNMENT */

    // establish baseline by random assignment
    let random_assignment = get_random_assignment(num_clusters, num_servers);
    scatter_plot(
        &mut plot_buffer,
        &data_points,
        &random_assignment,
        "Random Assignment",
    );
    let (
        avg_cluster_latencies_baseline,
        cluster_counts_baseline,
        cluster_latency_sum_baseline,
        cluster_latency_mean_baseline,
    ) = calculate_cluster_metrics(&random_assignment, &matrix);

    let mut table_rows_baseline = vec![];
    for i in 0..avg_cluster_latencies_baseline.len() {
        table_rows_baseline.push(format!(
            "<tr><td>{i}</td><td>{}</td><td>{}</td></tr>",
            cluster_counts_baseline[i], avg_cluster_latencies_baseline[i]
        ));
    }

    avg_cluster_latencies_baseline.iter().for_each(|v| {
        min_latency_val = min_latency_val.min(*v);
        max_latency_val = max_latency_val.max(*v);
    });
    cluster_counts_baseline.iter().for_each(|v| {
        min_size_val = min_size_val.min(*v);
        max_size_val = max_size_val.max(*v);
    });

    /* FASTERPAM */

    let (assignment, fasterpam_num_iterations, fasterpam_duration) =
        run_fasterpam(&dissim_matrix, num_clusters);
    scatter_plot(&mut plot_buffer, &data_points, &assignment, "FasterPAM");

    let (
        fasterpam_avg_cluster_latencies,
        fasterpam_cluster_counts,
        fasterpam_cluster_latency_sum,
        fasterpam_cluster_latency_mean,
    ) = calculate_cluster_metrics(&assignment, &matrix);

    let mut fasterpam_table_rows = vec![];
    for i in 0..fasterpam_avg_cluster_latencies.len() {
        fasterpam_table_rows.push(format!(
            "<tr><td>{i}</td><td>{}</td><td>{}</td></tr>",
            fasterpam_cluster_counts[i], fasterpam_avg_cluster_latencies[i]
        ));
    }
    fasterpam_avg_cluster_latencies.iter().for_each(|v| {
        min_latency_val = min_latency_val.min(*v);
        max_latency_val = max_latency_val.max(*v);
    });
    fasterpam_cluster_counts.iter().for_each(|v| {
        min_size_val = min_size_val.min(*v);
        max_size_val = max_size_val.max(*v);
    });

    /* WIP CONSTRAINED FASTERPAM */

    let (assignment, c_fasterpam_num_iterations, c_fasterpam_duration) =
        run_constrained_fasterpam(&dissim_matrix, num_clusters);
    scatter_plot(
        &mut plot_buffer,
        &data_points,
        &assignment,
        "Constrained FasterPAM",
    );
    for (i, &a) in assignment.iter().enumerate() {
        if a == 999 {
            println!("missing: {i}");
        }
    }

    let (
        c_fasterpam_avg_cluster_latencies,
        c_fasterpam_cluster_counts,
        c_fasterpam_cluster_latency_sum,
        c_fasterpam_cluster_latency_mean,
    ) = calculate_cluster_metrics(&assignment, &matrix);

    let mut c_fasterpam_table_rows = vec![];
    for i in 0..c_fasterpam_avg_cluster_latencies.len() {
        c_fasterpam_table_rows.push(format!(
            "<tr><td>{i}</td><td>{}</td><td>{}</td></tr>",
            c_fasterpam_cluster_counts[i], c_fasterpam_avg_cluster_latencies[i]
        ));
    }
    c_fasterpam_avg_cluster_latencies.iter().for_each(|v| {
        min_latency_val = min_latency_val.min(*v);
        max_latency_val = max_latency_val.max(*v);
    });
    c_fasterpam_cluster_counts.iter().for_each(|v| {
        min_size_val = min_size_val.min(*v);
        max_size_val = max_size_val.max(*v);
    });

    // build histograms
    let random_assignment_latency_histogram = histogram(
        &avg_cluster_latencies_baseline,
        min_latency_val,
        max_latency_val,
        "Random Assignment Cluster Latency",
    );
    let random_assignment_sizes_histogram = histogram(
        &cluster_counts_baseline,
        min_size_val,
        max_size_val,
        "Random Assignment Cluster Sizes",
    );
    let fasterpam_latency_histogram = histogram(
        &fasterpam_avg_cluster_latencies,
        min_latency_val,
        max_latency_val,
        "FasterPAM Cluster Latency",
    );
    let fasterpam_sizes_histogram = histogram(
        &fasterpam_cluster_counts,
        min_size_val,
        max_size_val,
        "FasterPAM Cluster Sizes",
    );
    let c_fasterpam_latency_histogram = histogram(
        &c_fasterpam_avg_cluster_latencies,
        min_latency_val,
        max_latency_val,
        "Constrained FasterPAM Cluster Latency",
    );
    let c_fasterpam_sizes_histogram = histogram(
        &c_fasterpam_cluster_counts,
        min_size_val,
        max_size_val,
        "Constrained FasterPAM Cluster Sizes",
    );

    // create html report
    let html = format!(
        r#"
<!DOCTYPE html>
<html>
<head>
    <title>Topology Simulation Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
        body {{ font-family: '{FONT}', monospace }}
        tr:nth-child(even) {{
            background-color: rgba(150, 212, 212, 0.4);
        }}
    </style>
</head>
<body>
    <h1>Topology Simulation Report</h1>
    <p>The clustering </p>
    <hr>
    {plot_buffer}
    <hr>
    <div style="display: flex;">
        <div style="width:100%; margin:1%;">
            <h2>Random Assignment</h2>
            <p>
                Num. Clusters: {num_clusters}</br>
                Latency Sum: {cluster_latency_sum_baseline:.0001}</br>
                Avg. Latency of All Clusters: {cluster_latency_mean_baseline:.0001}</br>
            </p>
            <table>
                <tr>
                    <th>Cluster</th>
                    <th>Count</th>
                    <th>Avg. Latency</th>
                </tr>
                {}
            </table>
        </div>
        <div style="width:100%; margin:1%;">
            <h2>FasterPAM</h2>
            <p>
                Duration: {fasterpam_duration:?}</br>
                Num. Clusters: {num_clusters}</br>
                Num. Iterations: {fasterpam_num_iterations}</br>
                Latency Sum: {fasterpam_cluster_latency_sum:.0001}</br>
                Avg. Latency of All Clusters: {fasterpam_cluster_latency_mean:.0001}</br>
            </p>
            <table>
                <tr>
                    <th>Cluster</th>
                    <th>Count</th>
                    <th>Avg. Latency</th>
                </tr>
                {}
            </table>
        </div>
        <div style="width:100%; margin:1%;">
            <h2>WIP Constrained FasterPAM</h2>
            <p>
                Duration: {c_fasterpam_duration:?}</br>
                Num. Clusters: {num_clusters}</br>
                Num. Iterations: {c_fasterpam_num_iterations}</br>
                Latency Sum: {c_fasterpam_cluster_latency_sum:.0001}</br>
                Avg. Latency of All Clusters: {c_fasterpam_cluster_latency_mean:.0001}</br>
            </p>
            <table>
                <tr>
                    <th>Cluster</th>
                    <th>Count</th>
                    <th>Avg. Latency</th>
                </tr>
                {}
            </table>
        </div>
    </div>

    <h1>Cluster Latency Histograms</h1>
    <div style="display: flex;">
        <img src="data:image/png;base64,{random_assignment_latency_histogram}" width="500" />
        <img src="data:image/png;base64,{fasterpam_latency_histogram}" width="500" />
        <img src="data:image/png;base64,{c_fasterpam_latency_histogram}" width="500" />
    </div>

    <h1>Cluster Sizes Histograms</h1>
    <div style="display: flex;">
        <img src="data:image/png;base64,{random_assignment_sizes_histogram}" width="500" />
        <img src="data:image/png;base64,{fasterpam_sizes_histogram}" width="500" />
        <img src="data:image/png;base64,{c_fasterpam_sizes_histogram}" width="500" />
    </div>
<body>
</html>
          "#,
        table_rows_baseline.join("\n"),
        fasterpam_table_rows.join("\n"),
        c_fasterpam_table_rows.join("\n"),
    );

    std::fs::write("report.html", html).unwrap();
}
