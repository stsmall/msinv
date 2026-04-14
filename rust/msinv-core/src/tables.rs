/// TableBuilder: accumulates node/edge rows during simulation.
///
/// At finalization the raw arrays are handed to the Python side for
/// conversion into a tskit TableCollection → TreeSequence. This
/// avoids any Rust dependency on the tskit C library.

pub const NODE_IS_SAMPLE: u32 = 1; // tskit.NODE_IS_SAMPLE

pub struct TableBuilder {
    pub sequence_length: f64,
    pub num_populations: u32,
    // Node table columns
    pub node_flags: Vec<u32>,
    pub node_time: Vec<f64>,
    pub node_population: Vec<i32>,
    // Edge table columns
    pub edge_left: Vec<f64>,
    pub edge_right: Vec<f64>,
    pub edge_parent: Vec<i32>,
    pub edge_child: Vec<i32>,
}

impl TableBuilder {
    pub fn new(sequence_length: f64, num_populations: u32) -> Self {
        Self {
            sequence_length,
            num_populations,
            node_flags: Vec::with_capacity(256),
            node_time: Vec::with_capacity(256),
            node_population: Vec::with_capacity(256),
            edge_left: Vec::with_capacity(1024),
            edge_right: Vec::with_capacity(1024),
            edge_parent: Vec::with_capacity(1024),
            edge_child: Vec::with_capacity(1024),
        }
    }

    /// Add a sample node. Returns the node id.
    pub fn add_sample(&mut self, time: f64, population: i32) -> i32 {
        let id = self.node_flags.len() as i32;
        self.node_flags.push(NODE_IS_SAMPLE);
        self.node_time.push(time);
        self.node_population.push(population);
        id
    }

    /// Add an internal (non-sample) node. Returns the node id.
    pub fn add_internal(&mut self, time: f64, population: i32) -> i32 {
        let id = self.node_flags.len() as i32;
        self.node_flags.push(0);
        self.node_time.push(time);
        self.node_population.push(population);
        id
    }

    /// Add an edge.
    #[inline]
    pub fn add_edge(&mut self, left: f64, right: f64, parent: i32, child: i32) {
        self.edge_left.push(left);
        self.edge_right.push(right);
        self.edge_parent.push(parent);
        self.edge_child.push(child);
    }

    pub fn num_nodes(&self) -> usize {
        self.node_flags.len()
    }

    pub fn num_edges(&self) -> usize {
        self.edge_left.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_and_internal_ids() {
        let mut t = TableBuilder::new(100.0, 1);
        let s0 = t.add_sample(0.0, 0);
        let s1 = t.add_sample(0.0, 0);
        let n2 = t.add_internal(5.0, 0);
        assert_eq!(s0, 0);
        assert_eq!(s1, 1);
        assert_eq!(n2, 2);
        assert_eq!(t.node_flags[0], NODE_IS_SAMPLE);
        assert_eq!(t.node_flags[2], 0);
    }

    #[test]
    fn edge_recording() {
        let mut t = TableBuilder::new(100.0, 1);
        t.add_edge(0.0, 50.0, 2, 0);
        t.add_edge(0.0, 50.0, 2, 1);
        assert_eq!(t.num_edges(), 2);
        assert_eq!(t.edge_parent[0], 2);
    }
}
