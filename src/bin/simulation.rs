use std::{cell::RefCell, iter::repeat, rc::Rc, time::Duration};

use clustering::divisive::DivisiveHierarchy;
use fxhash::{FxHashMap, FxHashSet};
use ndarray::Array2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use simulon::{api, simulation::SimulationBuilder};

#[derive(Serialize, Deserialize, Debug)]
enum Message {
    Advr(usize),
    Want(usize),
    Payload(usize, Vec<u8>),
}

#[derive(Default)]
struct NodeState {
    conns: FxHashMap<api::RemoteAddr, BroadcastConnection>,
    messages: FxHashMap<usize, Vec<u8>>,
}

struct BroadcastConnection {
    writer: api::OwnedWriter,
    seen: FxHashSet<usize>,
}

impl NodeState {
    fn handle_message_internal(&mut self, id: usize, payload: Vec<u8>) {
        if self.messages.insert(id, payload.clone()).is_none() {
            api::emit(String::from_utf8(payload).unwrap());
        }

        for (_addr, conn) in self.conns.iter_mut() {
            if conn.seen.contains(&id) {
                continue;
            }

            conn.writer.write(&Message::Advr(id));
        }
    }

    pub fn handle_message_from_client(&mut self, id: usize, payload: Vec<u8>) {
        self.handle_message_internal(id, payload);
    }

    pub fn handle_message(&mut self, sender: api::RemoteAddr, id: usize, payload: Vec<u8>) {
        let conn = self.conns.get_mut(&sender).unwrap();
        conn.seen.insert(id);

        self.handle_message_internal(id, payload);
    }

    pub fn handle_advr(&mut self, sender: api::RemoteAddr, id: usize) {
        let conn = self.conns.get_mut(&sender).unwrap();

        conn.seen.insert(id);

        // If we already have the message move on.
        if self.messages.contains_key(&id) {
            return;
        }

        conn.writer.write(&Message::Want(id));
    }

    pub fn handle_want(&mut self, sender: api::RemoteAddr, id: usize) {
        if let Some(payload) = self.messages.get(&id) {
            let conn = self.conns.get_mut(&sender).unwrap();
            conn.seen.insert(id);
            conn.writer.write(&Message::Payload(id, payload.clone()));
        }
    }
}

impl From<api::OwnedWriter> for BroadcastConnection {
    fn from(writer: api::OwnedWriter) -> Self {
        Self {
            writer,
            seen: FxHashSet::default(),
        }
    }
}

type NodeStateRef = Rc<RefCell<NodeState>>;

/// Start a node.
async fn run_node(n: usize) {
    let state = Rc::new(RefCell::new(NodeState::default()));

    // The event loop for accepting connections from the peers.
    api::spawn(listen_for_connections(n, state.clone()));

    // Make the connections.
    api::spawn(make_connections(n, state));
}

async fn listen_for_connections(n: usize, state: NodeStateRef) {
    let mut listener = api::listen(80);
    while let Some(conn) = listener.accept().await {
        if n == *conn.remote() {
            api::spawn(handle_client_connection(state.clone(), conn));
        } else {
            api::spawn(handle_connection(state.clone(), conn));
        }
    }
}

async fn make_connections(n: usize, state: NodeStateRef) {
    api::with_state(|assignments: &Vec<Vec<usize>>| {
        let i = *api::RemoteAddr::whoami();
        assignments[i]
            .iter()
            .cloned()
            .filter(|&j| i < j) // only connect from one side of a pair
            .map(api::RemoteAddr::from_global_index)
            .zip(repeat(state))
            .map(|(addr, state)| async move {
                debug_assert!(*addr < n && i != *addr);
                handle_connection(
                    state,
                    api::connect(addr, 80).await.expect("Could not connect."),
                )
                .await;
            })
            .for_each(api::spawn);
    })
}

/// Handle the connection that is made from a client.
async fn handle_client_connection(state: NodeStateRef, conn: api::Connection) {
    let (mut reader, _writer) = conn.split();
    let msg = reader.recv::<Message>().await.unwrap();
    if let Message::Payload(id, payload) = msg {
        state.borrow_mut().handle_message_from_client(id, payload);
    } else {
        panic!("unexpected.");
    }
}

/// Handle the connection that is made from another node.
async fn handle_connection(state: NodeStateRef, conn: api::Connection) {
    let remote = conn.remote();
    let (mut reader, writer) = conn.split();

    // Insert the writer half of this connection into the state.
    let b_conn: BroadcastConnection = writer.into();
    state.borrow_mut().conns.insert(remote, b_conn);

    while let Some(msg) = reader.recv::<Message>().await {
        match msg {
            Message::Payload(id, payload) => {
                state.borrow_mut().handle_message(remote, id, payload);
            },
            Message::Advr(id) => {
                state.borrow_mut().handle_advr(remote, id);
            },
            Message::Want(id) => {
                state.borrow_mut().handle_want(remote, id);
            },
        }
    }
}

/// Start a client loop which picks a random node and sends a message to it every
/// few seconds.
async fn run_client(n: usize) {
    let mut rng = ChaCha8Rng::from_seed([0; 32]);

    for i in 0..1 {
        let index = rng.gen_range(0..n);
        let addr = api::RemoteAddr::from_global_index(index);

        let mut conn = api::connect(addr, 80).await.expect("Connection failed.");

        let msg = format!("message {i}");
        conn.write(&Message::Payload(i, msg.into()));

        api::sleep(Duration::from_secs(5)).await;
    }
}

fn exec(n: usize) {
    if n == *api::RemoteAddr::whoami() {
        api::spawn(run_client(n));
    } else {
        api::spawn(run_node(n));
    }
}

use simulon::latency::{
    ping::{PingStat, RegionToRegionDistribution},
    LatencyProvider, PingDataLatencyProvider,
};

fn get_matrix(n: usize) -> Array2<i32> {
    struct MeanExtractor(u32);
    impl From<PingStat> for MeanExtractor {
        fn from(value: PingStat) -> Self {
            Self(value.avg)
        }
    }
    impl RegionToRegionDistribution for MeanExtractor {
        fn next<R: rand::Rng>(&mut self, _rng: &mut R) -> u32 {
            self.0
        }
    }

    let mut provider = PingDataLatencyProvider::<MeanExtractor>::default();
    provider.init(n);

    let mut matrix = Array2::zeros((n, n));
    for i in 0..n {
        for j in i + 1..n {
            matrix[(i, j)] = provider.get(i, j).as_micros() as i32;
        }
    }
    matrix
}

pub fn main() {
    const N: usize = 1500;

    let matrix = get_matrix(N);
    let hierarchy = DivisiveHierarchy::new(&mut rand::thread_rng(), &matrix, 8);
    let assignments = hierarchy.connections();

    let report = SimulationBuilder::new(|| exec(N))
        .with_nodes(N + 1)
        .with_state(assignments)
        .set_node_metrics_rate(Duration::ZERO)
        .enable_progress_bar()
        .run(Duration::from_secs(120));

    // write out json report for the simulation
    let file =
        std::fs::File::create("simulation_report.json").expect("failed to open json report file");
    serde_json::to_writer(file, &report).expect("failed to write json report");
}
