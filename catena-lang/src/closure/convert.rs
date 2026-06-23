use hexpr::Operation;
use metacat::tree::Tree;
use open_hypergraphs::lax::NodeId;
use thiserror::Error;

use crate::{
    check::AnnotatedTerm,
    closure::{
        body::{ClosureBodyError, closure_body},
        extract::{ExtractRegionError, extract_region},
        region::{ClosureRegion, ClosureRegionError, closure_region},
        rewrite::{RewriteRegionError, rewrite_region},
    },
};

const CLOSURE_TYPE: &str = "=>";

type Obj = Tree<(), Operation>;

pub type ConvertedClosures = Vec<(NodeId, AnnotatedTerm)>;

#[derive(Debug, Error)]
pub enum ConvertError {
    #[error(transparent)]
    Region(#[from] ClosureRegionError),
    #[error(transparent)]
    Extract(#[from] ExtractRegionError),
    #[error(transparent)]
    Body(#[from] ClosureBodyError),
    #[error(transparent)]
    Rewrite(#[from] RewriteRegionError),
}

/// Convert closure-typed output regions of an annotated term.
///
/// Returns the rewritten term plus the newly generated closure body terms. Each
/// generated body is paired with the original closure output node id that caused
/// it to be created.
pub fn convert(
    definition: &AnnotatedTerm,
) -> Result<(AnnotatedTerm, ConvertedClosures), ConvertError> {
    let closure_wires = closure_output_wires(definition);
    let regions = closure_region(definition, &closure_wires)?;

    let mut closures = Vec::new();
    let mut rewrites = Vec::new();
    for region in regions {
        let extracted = extract_region(definition, &region)?;
        let body = closure_body(&extracted)?;
        let replacement = replacement_region(definition, &region);
        closures.push((region.closure_wire, body));
        rewrites.push((region, replacement));
    }

    let mut rewritten = definition.clone();
    rewrites.sort_by_key(|(region, _)| {
        (
            region
                .nodes
                .iter()
                .map(|node| node.0)
                .max()
                .unwrap_or_default(),
            region
                .edges
                .iter()
                .map(|edge| edge.0)
                .max()
                .unwrap_or_default(),
        )
    });
    for (region, replacement) in rewrites.into_iter().rev() {
        rewritten = rewrite_region(&rewritten, &region, &replacement)?;
    }

    Ok((rewritten, closures))
}

fn closure_output_wires(definition: &AnnotatedTerm) -> Vec<NodeId> {
    definition
        .targets
        .iter()
        .copied()
        .filter(|wire| {
            definition
                .hypergraph
                .nodes
                .get(wire.0)
                .is_some_and(is_closure_type)
        })
        .collect()
}

fn replacement_region(definition: &AnnotatedTerm, region: &ClosureRegion) -> AnnotatedTerm {
    let mut replacement = AnnotatedTerm::empty();
    let sources = region
        .defer_inputs
        .iter()
        .map(|wire| replacement.new_node(definition.hypergraph.nodes[wire.0].clone()))
        .collect::<Vec<_>>();
    let target = replacement.new_node(region.closure_type.clone());
    replacement.new_edge(
        closure_operation(region.closure_wire),
        (sources.clone(), vec![target]),
    );
    replacement.sources = sources;
    replacement.targets = vec![target];
    replacement
}

fn closure_operation(closure_wire: NodeId) -> Operation {
    format!("closure.{}", closure_wire.0)
        .parse()
        .expect("generated closure operation should parse")
}

fn is_closure_type(object: &Obj) -> bool {
    let Tree::Node(operation, _, children) = object else {
        return false;
    };
    operation.as_str() == CLOSURE_TYPE && children.len() == 2
}

#[cfg(test)]
mod tests {
    use metacat::{
        theory::{RawTheorySet, Theory, TheoryId, TheorySet},
        tree::Tree,
    };

    use super::*;
    use crate::{
        check::{DefinitionTypes, check},
        elaborate::elaborate,
        stdlib,
    };

    #[test]
    fn convert_closure_output_splices_placeholder_and_returns_body() {
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
        let original_target = definition.targets[0];

        let (rewritten, closures) = convert(&definition).expect("conversion should succeed");

        assert_eq!(closures.len(), 1);
        assert_eq!(closures[0].0, original_target);
        assert_eq!(closures[0].1.hypergraph.edges.len(), 7);
        assert_eq!(rewritten.hypergraph.edges.len(), 4);
        assert!(
            rewritten
                .hypergraph
                .edges
                .iter()
                .any(|operation| operation.as_str() == format!("closure.{}", original_target.0))
        );
        assert_eq!(
            interface_types(&closures[0].1, &closures[0].1.sources),
            vec![obj("val", vec![obj("bool", vec![])]), obj("1", vec![])]
        );
        assert_eq!(
            interface_types(&closures[0].1, &closures[0].1.targets),
            vec![obj("val", vec![obj("bool", vec![])])]
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

    fn interface_types(term: &AnnotatedTerm, interface: &[NodeId]) -> Vec<Obj> {
        interface
            .iter()
            .map(|node| term.hypergraph.nodes[node.0].clone())
            .collect()
    }

    fn obj(name: &str, children: Vec<Obj>) -> Obj {
        Tree::Node(
            name.parse().expect("test operation should parse"),
            0,
            children,
        )
    }
}
