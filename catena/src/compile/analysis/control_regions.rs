use std::collections::HashMap;

use open_hypergraphs::{
    lax::{NodeId, OpenHypergraph as LaxOpenHypergraph, functor::Functor},
    strict::vec::FiniteFunction,
};

use crate::{
    compile::{
        CompileGraph, CompileTheory,
        analysis::{
            layering::{
                BoundaryRelation, NestedGraph, coproduct_over_parent, quotient_over_parent,
            },
            partition::{OperationId, OperationRegion, RegionKind},
            wires::is_interleaved_control_operation,
        },
        graph_ops::{Graph, operation_inputs, operation_name, operation_outputs},
    },
    pass::inline::Inline,
};

#[derive(Debug, Clone)]
pub struct ControlRegionGraph {
    pub region_index: usize,
    pub region_operations: Vec<OperationId>,
    pub graph: Graph,
    pub morphism: ControlRegionMorphism,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlRegionMorphism {
    /// Result wire id -> parent region wire id, when the resolved wire is visible through the original interleaved control region.
    pub wires: Vec<Option<NodeId>>,
    /// Result operation id -> parent interleaved control operation id.
    pub operations: Vec<OperationId>,
}

pub(super) fn process_control_regions(
    parent: &CompileGraph,
    regions: &[OperationRegion],
) -> Vec<ControlRegionGraph> {
    regions
        .iter()
        .enumerate()
        .filter(|(_, region)| matches!(region.kind, RegionKind::InterleavedControl))
        .map(|(region_index, region)| expand_control_region(parent, region_index, region))
        .collect()
}

// Expand one interleaved control region from a data graph into a native control graph, preserving a non-injective morphism back to the original region.
fn expand_control_region(
    parent: &CompileGraph,
    region_index: usize,
    region: &OperationRegion,
) -> ControlRegionGraph {
    let nested_graph = coproduct_over_parent(
        region
            .operations
            .iter()
            .copied()
            .map(|operation_id| expand_interleaved_control_call(parent, operation_id))
            .collect::<Vec<_>>(),
    );
    let resolved = quotient_over_parent(&parent.graph, nested_graph);

    ControlRegionGraph {
        region_index,
        region_operations: region.operations.clone(),
        morphism: ControlRegionMorphism {
            wires: resolved
                .boundary_relation
                .parent_wires_by_child_wire(resolved.graph.h.w.0.len()),
            operations: resolved.parent_operations.clone(),
        },
        graph: resolved.graph,
    }
}

// Expand a `control.foo` operation from a data graph. The native control child
// may have many boundary wires, but the interleaved call site can expose them
// through packed unary wires. Only child boundary projections are remapped to
// the call site; internal wires stay unmapped.
fn expand_interleaved_control_call(
    parent: &CompileGraph,
    operation_id: OperationId,
) -> NestedGraph {
    debug_assert!(is_interleaved_control_operation(
        &parent.graph,
        operation_id
    ));
    let operation = operation_name(&parent.graph, operation_id);
    let native_control_child =
        control_definition_for_operation(parent, operation).unwrap_or_else(|| {
            panic!(
                "interleaved control operation `{operation}` must resolve to a native control graph"
            )
        });

    let expanded_control_graph = inline_control_definitions(native_control_child);

    let expanded_source_wires = boundary_table(&expanded_control_graph.s);
    let expanded_target_wires = boundary_table(&expanded_control_graph.t);
    let call_inputs = operation_inputs(&parent.graph, operation_id).collect::<Vec<_>>();
    let call_outputs = operation_outputs(&parent.graph, operation_id).collect::<Vec<_>>();
    let boundary_relation = BoundaryRelation::from_boundaries(
        operation_id,
        (&expanded_source_wires, &call_inputs),
        (&expanded_target_wires, &call_outputs),
    );

    NestedGraph::under_parent_operation(operation_id, expanded_control_graph, boundary_relation)
}

// Expand a native control graph by inlining non-primitive control operations
// inside it. Mapping the expanded boundary to an interleaved call site is handled
// by `expand_interleaved_control_call`.
fn inline_control_definitions(graph: &CompileGraph) -> Graph {
    debug_assert!(matches!(graph.theory, CompileTheory::Control));
    let definitions = graph
        .children
        .iter()
        .filter(|child| matches!(child.graph.theory, CompileTheory::Control))
        .map(|child| {
            (
                child
                    .operation
                    .parse()
                    .expect("control child operation name must be valid"),
                LaxOpenHypergraph::from_strict(inline_control_definitions(&child.graph)),
            )
        })
        .collect::<HashMap<_, _>>();
    if definitions.is_empty() {
        return graph.graph.clone();
    }

    let mut expanded = LaxOpenHypergraph::from_strict(graph.graph.clone());
    for _ in 0..64 {
        let inlinable = expanded
            .hypergraph
            .edges
            .iter()
            .any(|operation| definitions.contains_key(operation));
        if !inlinable {
            return expanded.to_strict();
        }
        expanded = Inline {
            definitions: definitions.clone(),
        }
        .map_arrow(&expanded);
    }

    panic!(
        "too many control-definition inline iterations while expanding `{}`",
        graph.definition_name
    )
}

// Look up the child graph that gives the native control definition for an operation. Operations without such a child are treated as primitives.
fn control_definition_for_operation<'a>(
    graph: &'a CompileGraph,
    operation: &str,
) -> Option<&'a CompileGraph> {
    graph
        .children
        .iter()
        .find(|child| child.operation == operation)
        .map(|child| &child.graph)
        .filter(|child| matches!(child.theory, CompileTheory::Control))
}

fn boundary_table(boundary: &FiniteFunction) -> Vec<NodeId> {
    boundary.table.0.iter().copied().map(NodeId).collect()
}
