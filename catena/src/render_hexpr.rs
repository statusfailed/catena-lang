//! Render open hypergraphs as hexprs
//!
//! In general, every open hypergraph can be rendered as a hexpr like the one below:
//!
//! {[s0 s1 ... sm.]
//!     [.s0 s1 ... sa] edge_0 [t0 t1 ... tn.]
//!     [.s0 s1 ... sa] edge_1 [t0 t1 ... tn.]
//!     ...
//!     [.s0 s1 ... sa] edge_k [t0 t1 ... tn.]
//! [. t0 t1 ... tn]}
use hexpr::{Hexpr, Operation, Variable};
use open_hypergraphs::lax::{NodeId, OpenHypergraph};
use std::str::FromStr;

/// Render an open hypergraph into a hexpr using one tensor factor per boundary
/// spider or generating edge.
pub fn open_hypergraph_to_hexpr(mut f: OpenHypergraph<(), Operation>) -> Hexpr {
    let _ = f.quotient();

    let mut factors = Vec::new();

    // Sources: emit the left boundary spider, exposing each open input node as a
    // named variable in the surrounding tensor.
    if !f.sources.is_empty() {
        factors.push(spider(
            f.sources.iter().map(|node| node_var(*node)).collect(),
            vec![],
        ));
    }

    // Body: emit one factor per hyperedge. Each edge is rendered as its input
    // spider, followed by the edge operation itself, followed by its output
    // spider, matching the sketch in the module comment.
    for (edge, adjacency) in f
        .hypergraph
        .edges
        .iter()
        .cloned()
        .zip(f.hypergraph.adjacency.iter())
    {
        let mut parts = Vec::new();
        if !adjacency.sources.is_empty() {
            parts.push(spider(
                vec![],
                adjacency.sources.iter().map(|node| node_var(*node)).collect(),
            ));
        }
        parts.push(Hexpr::Operation(edge));
        if !adjacency.targets.is_empty() {
            parts.push(spider(
                adjacency.targets.iter().map(|node| node_var(*node)).collect(),
                vec![],
            ));
        }
        factors.push(match parts.len() {
            0 => Hexpr::Tensor(vec![]),
            1 => parts.remove(0),
            _ => Hexpr::Composition(parts),
        });
    }

    // Targets: emit the right boundary spider, collecting the designated open
    // output nodes of the hypergraph.
    if !f.targets.is_empty() {
        factors.push(spider(
            vec![],
            f.targets.iter().map(|node| node_var(*node)).collect(),
        ));
    }

    match factors.len() {
        0 => Hexpr::Tensor(vec![]),
        1 => factors.remove(0),
        _ => Hexpr::Tensor(factors),
    }
}

fn spider(sources: Vec<Variable>, targets: Vec<Variable>) -> Hexpr {
    Hexpr::Frobenius { sources, targets }
}

fn node_var(node: NodeId) -> Variable {
    Variable::from_str(&format!("n{}", node.0)).expect("generated variable should parse")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_single_edge_with_boundary_spiders() {
        let graph = OpenHypergraph::singleton("f".parse().unwrap(), vec![()], vec![(), ()]);
        let rendered = open_hypergraph_to_hexpr(graph);
        assert_eq!(
            rendered.to_string(),
            "{[n0 . ] ([ . n0] f [n1 n2 . ]) [ . n1 n2]}"
        );
    }

    #[test]
    fn renders_identity_as_boundary_only() {
        let graph = OpenHypergraph::<(), Operation>::identity(vec![(), ()]);
        let rendered = open_hypergraph_to_hexpr(graph);
        assert_eq!(rendered.to_string(), "{[n0 n1 . ] [ . n0 n1]}");
    }
}
