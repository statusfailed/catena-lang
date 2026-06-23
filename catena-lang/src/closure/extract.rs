use std::collections::HashMap;

use open_hypergraphs::lax::NodeId;
use thiserror::Error;

use crate::{check::AnnotatedTerm, closure::region::ClosureRegion};

#[derive(Debug, Error)]
pub enum ExtractRegionError {
    #[error("region node n{node} is out of bounds")]
    NodeOutOfBounds { node: usize },
    #[error("region edge e{edge} is out of bounds")]
    EdgeOutOfBounds { edge: usize },
    #[error("region edge e{edge} references node n{node}, which is not in the region")]
    IncidentNodeOutsideRegion { edge: usize, node: usize },
    #[error("region closure wire n{wire} is not in the region")]
    ClosureWireOutsideRegion { wire: usize },
    #[error("region defer input n{wire} is not in the region")]
    DeferInputOutsideRegion { wire: usize },
}

/// Copy the identified closure region into a standalone annotated term.
///
/// The extracted term contains only the region's nodes and edges. Its source
/// interface is the region's recorded `defer` inputs, in that order, and its
/// target interface is the region's closure root.
pub fn extract_region(
    definition: &AnnotatedTerm,
    region: &ClosureRegion,
) -> Result<AnnotatedTerm, ExtractRegionError> {
    validate_region(definition, region)?;

    let mut extracted = AnnotatedTerm::empty();
    let mut node_map = HashMap::<NodeId, NodeId>::new();

    for &node in &region.nodes {
        let label = definition.hypergraph.nodes[node.0].clone();
        let copied = extracted.new_node(label);
        node_map.insert(node, copied);
    }

    for &edge in &region.edges {
        let hyperedge = &definition.hypergraph.adjacency[edge.0];
        let sources = remap_nodes(&node_map, edge.0, &hyperedge.sources)?;
        let targets = remap_nodes(&node_map, edge.0, &hyperedge.targets)?;
        extracted.new_edge(
            definition.hypergraph.edges[edge.0].clone(),
            (sources, targets),
        );
    }

    extracted.sources = remap_interface(&node_map, &region.defer_inputs, |wire| {
        ExtractRegionError::DeferInputOutsideRegion { wire }
    })?;
    extracted.targets = remap_interface(&node_map, &[region.closure_wire], |wire| {
        ExtractRegionError::ClosureWireOutsideRegion { wire }
    })?;

    Ok(extracted)
}

fn validate_region(
    definition: &AnnotatedTerm,
    region: &ClosureRegion,
) -> Result<(), ExtractRegionError> {
    for &node in &region.nodes {
        if node.0 >= definition.hypergraph.nodes.len() {
            return Err(ExtractRegionError::NodeOutOfBounds { node: node.0 });
        }
    }

    for &edge in &region.edges {
        if edge.0 >= definition.hypergraph.edges.len() {
            return Err(ExtractRegionError::EdgeOutOfBounds { edge: edge.0 });
        }
    }

    Ok(())
}

fn remap_nodes(
    node_map: &HashMap<NodeId, NodeId>,
    edge: usize,
    nodes: &[NodeId],
) -> Result<Vec<NodeId>, ExtractRegionError> {
    nodes
        .iter()
        .map(|node| {
            node_map
                .get(node)
                .copied()
                .ok_or(ExtractRegionError::IncidentNodeOutsideRegion { edge, node: node.0 })
        })
        .collect()
}

fn remap_interface(
    node_map: &HashMap<NodeId, NodeId>,
    nodes: &[NodeId],
    error: impl Fn(usize) -> ExtractRegionError,
) -> Result<Vec<NodeId>, ExtractRegionError> {
    nodes
        .iter()
        .map(|node| node_map.get(node).copied().ok_or_else(|| error(node.0)))
        .collect()
}

#[cfg(test)]
mod tests {
    use hexpr::Operation;
    use metacat::{
        theory::{RawTheorySet, Theory, TheoryId, TheorySet},
        tree::Tree,
    };
    use open_hypergraphs::lax::NodeId;

    use super::*;
    use crate::{
        check::{DefinitionTypes, check},
        closure::region::{Obj, closure_region},
        elaborate::elaborate,
        stdlib,
    };

    #[test]
    fn extracted_region_uses_defer_inputs_as_sources() {
        let bool_value = obj("val", vec![obj("bool", vec![])]);
        let closure_type = obj("=>", vec![obj("1", vec![]), bool_value.clone()]);

        let mut definition = AnnotatedTerm::empty();
        let captured = definition.new_node(bool_value.clone());
        let closure = definition.new_node(closure_type.clone());
        let defer = definition.new_edge(op("defer"), (vec![captured], vec![closure]));

        let region = ClosureRegion {
            closure_wire: closure,
            closure_type,
            defer_inputs: vec![captured],
            nodes: vec![captured, closure],
            edges: vec![defer],
        };

        let extracted =
            extract_region(&definition, &region).expect("region extraction should succeed");

        assert_eq!(extracted.hypergraph.edges, vec![op("defer")]);
        assert_eq!(extracted.sources, vec![NodeId(0)]);
        assert_eq!(extracted.targets, vec![NodeId(1)]);
        assert_eq!(extracted.hypergraph.adjacency[0].sources, vec![NodeId(0)]);
        assert_eq!(extracted.hypergraph.adjacency[0].targets, vec![NodeId(1)]);
    }

    #[test]
    fn identity_wire_has_no_extractable_closure_regions() {
        let definition = annotated_program_definition(
            r#"
            (def program id : [a] -> [a] = [x])
            "#,
            "id",
        );
        let closure_wires = closure_wires(&definition);
        let regions =
            closure_region(&definition, &closure_wires).expect("region discovery should succeed");

        assert_eq!(regions.len(), 0);
    }

    #[test]
    fn extracted_composed_closure_region_has_defer_input_type() {
        let definition = annotated_program_definition(
            r#"
            (def program run-bool-id : (bool val) -> ({1 (bool val)} =>) = (
              {[x] bool.t}
              bool.and
              bool.not
              {defer (name.bool.id lift)}
              compose
            ))
            "#,
            "run-bool-id",
        );
        let [region] = closure_region(&definition, &[definition.targets[0]])
            .expect("region discovery should succeed")
            .try_into()
            .expect("expected one closure region");
        let extracted =
            extract_region(&definition, &region).expect("region extraction should succeed");

        assert_eq!(extracted.hypergraph.edges.len(), 4);
        assert_eq!(
            interface_types(&extracted, &extracted.sources),
            interface_types(&definition, &region.defer_inputs)
        );
        assert_eq!(
            interface_types(&extracted, &extracted.targets),
            vec![region.closure_type]
        );
    }

    fn theories_with(source: &'static str) -> (TheorySet, DefinitionTypes) {
        let raw_theories = RawTheorySet::from_texts(stdlib::sources().chain([source]))
            .expect("test theories should parse");
        let elaborated = elaborate(raw_theories).expect("test theories should elaborate");
        let theory_set = TheorySet::from_raw(elaborated).expect("test theories should load");
        let definition_types = check(&theory_set).expect("test theories should typecheck");
        (theory_set, definition_types)
    }

    fn annotated_program_definition(source: &'static str, definition: &str) -> AnnotatedTerm {
        let (theory_set, definition_types) = theories_with(source);
        let program = TheoryId("program".parse().expect("program theory id should parse"));
        let definition: Operation = definition
            .parse()
            .expect("program definition name should parse");
        let theory = theory_set
            .theories
            .get(&program)
            .expect("program theory should exist");
        let Theory::Theory { arrows, .. } = theory else {
            panic!("program should be a theory");
        };
        let arrow = arrows
            .get(&definition)
            .expect("program definition should exist");
        let mut body = arrow
            .definition
            .clone()
            .expect("program arrow should be a definition");
        body.quotient().ok();
        let labels = definition_types
            .get(&program)
            .and_then(|definitions| definitions.get(&definition))
            .cloned()
            .expect("program definition should have checked node types");
        body.with_nodes(|_| labels)
            .expect("checked node labels should match definition graph")
    }

    fn closure_wires(definition: &AnnotatedTerm) -> Vec<NodeId> {
        definition
            .hypergraph
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, object)| is_closure_type(object).then_some(NodeId(index)))
            .collect()
    }

    fn is_closure_type(object: &Obj) -> bool {
        let Tree::Node(operation, _, children) = object else {
            return false;
        };
        operation.as_str() == "=>" && children.len() == 2
    }

    fn interface_types(term: &AnnotatedTerm, interface: &[NodeId]) -> Vec<Obj> {
        interface
            .iter()
            .map(|node| term.hypergraph.nodes[node.0].clone())
            .collect()
    }

    fn obj(name: &str, children: Vec<Obj>) -> Obj {
        Tree::Node(op(name), 0, children)
    }

    fn op(name: &str) -> Operation {
        name.parse().expect("test operation should parse")
    }
}
