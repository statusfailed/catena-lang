use std::collections::HashMap;

use open_hypergraphs::{
    lax::NodeId,
    strict::vec::{
        FiniteFunction, Hypergraph, IndexedCoproduct, OpenHypergraph, SemifiniteFunction, VecArray,
    },
};

use crate::{
    compile::{
        analysis::partition::OperationId,
        graph_ops::{Graph, operation_inputs, operation_outputs},
    },
    lang::Obj,
};

// A span from a child graph boundary to its parent graph boundary. The vector
// index is the apex element; `child_wires` and `parent_wires` are the two legs.
#[derive(Debug, Clone)]
pub(super) struct BoundaryRelation {
    pub(super) child_wires: Vec<NodeId>,
    pub(super) parent_wires: Vec<NodeId>,
}

impl BoundaryRelation {
    pub(super) fn from_boundaries(
        source: (&[NodeId], &[NodeId]),
        target: (&[NodeId], &[NodeId]),
    ) -> Self {
        let links = boundary_links(source.0, source.1)
            .chain(boundary_links(target.0, target.1))
            .fold(Vec::<(NodeId, NodeId)>::new(), |mut links, link| {
                if !links.iter().any(|(child_wire, _)| *child_wire == link.0) {
                    links.push(link);
                }
                links
            });
        Self::from_links(links)
    }

    fn empty() -> Self {
        Self {
            child_wires: Vec::new(),
            parent_wires: Vec::new(),
        }
    }

    fn from_links(links: Vec<(NodeId, NodeId)>) -> Self {
        let (child_wires, parent_wires) = links.into_iter().unzip();
        Self {
            child_wires,
            parent_wires,
        }
    }

    fn shifted_child_wires(&self, offset: usize) -> impl Iterator<Item = (NodeId, NodeId)> + '_ {
        self.child_wires
            .iter()
            .copied()
            .zip(self.parent_wires.iter().copied())
            .map(move |(child_wire, parent_wire)| (NodeId(child_wire.0 + offset), parent_wire))
    }

    pub(super) fn fiber_points_by_wire(
        &self,
        graph: &Graph,
        parent_operations: &[OperationId],
        parent: &Graph,
        wire_count: usize,
    ) -> Vec<Option<BoundaryFiberPoint>> {
        debug_assert_eq!(self.child_wires.len(), self.parent_wires.len());
        let mut fiber_positions = HashMap::<BoundaryFiber, usize>::new();
        let mut fiber_points = vec![None; wire_count];
        for (child_wire, parent_wire) in self
            .child_wires
            .iter()
            .copied()
            .zip(self.parent_wires.iter().copied())
        {
            let side = boundary_side(graph, child_wire)
                .expect("boundary relation child wire must be on the child graph boundary");
            let parent_operation = parent_operation_for_boundary_link(
                graph,
                parent_operations,
                parent,
                child_wire,
                parent_wire,
                side,
            )
            .expect("boundary relation must land on the mapped parent operation boundary");
            let position = fiber_positions
                .entry(BoundaryFiber {
                    parent_operation,
                    parent_wire,
                    side,
                })
                .or_default();
            fiber_points[child_wire.0] = Some(BoundaryFiberPoint {
                parent_wire,
                fiber_position: *position,
            });
            *position += 1;
        }
        fiber_points
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BoundarySide {
    Source,
    Target,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BoundaryFiber {
    parent_operation: OperationId,
    parent_wire: NodeId,
    side: BoundarySide,
}

#[derive(Debug, Clone)]
pub(super) struct NestedGraph {
    pub(super) graph: Graph,
    pub(super) parent_operations: Vec<OperationId>,
    pub(super) boundary_relation: BoundaryRelation,
}

impl NestedGraph {
    pub(super) fn under_parent_operation(
        parent_operation: OperationId,
        graph: Graph,
        boundary_relation: BoundaryRelation,
    ) -> Self {
        let operation_count = graph.h.x.0.len();
        Self {
            graph,
            parent_operations: vec![parent_operation; operation_count],
            boundary_relation,
        }
    }

    pub(super) fn validate_against_parent(&self, parent: &Graph) {
        assert_eq!(
            self.parent_operations.len(),
            self.graph.h.x.0.len(),
            "nested graph must map every child operation to a parent operation"
        );
        assert_eq!(
            self.boundary_relation.child_wires.len(),
            self.boundary_relation.parent_wires.len(),
            "boundary relation legs must have the same apex size"
        );
        for parent_operation in &self.parent_operations {
            assert!(
                *parent_operation < parent.h.x.0.len(),
                "nested graph maps to missing parent operation {parent_operation}"
            );
        }
        for child_wire in &self.boundary_relation.child_wires {
            assert!(
                child_wire.0 < self.graph.h.w.0.len(),
                "boundary relation references missing child wire {:?}",
                child_wire
            );
        }
        for parent_wire in &self.boundary_relation.parent_wires {
            assert!(
                parent_wire.0 < parent.h.w.0.len(),
                "boundary relation references missing parent wire {:?}",
                parent_wire
            );
        }

        for child_operation in 0..self.graph.h.x.0.len() {
            let parent_operation = self.parent_operations[child_operation];
            let parent_boundary = operation_boundary(parent, parent_operation);
            let child_boundary = operation_boundary(&self.graph, child_operation);
            for (child_wire, parent_wire) in self
                .boundary_relation
                .child_wires
                .iter()
                .zip(&self.boundary_relation.parent_wires)
            {
                if child_boundary.contains(child_wire) {
                    assert!(
                        parent_boundary.contains(parent_wire),
                        "child operation {child_operation} maps to parent operation {parent_operation}, but boundary relation maps child wire {:?} to unrelated parent wire {:?}",
                        child_wire,
                        parent_wire
                    );
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct BoundaryFiberPoint {
    pub(super) parent_wire: NodeId,
    pub(super) fiber_position: usize,
}

pub(super) fn tensor_nested_graphs(nested_graphs: Vec<NestedGraph>) -> NestedGraph {
    let mut wires = Vec::<Obj>::new();
    let mut operations = Vec::new();
    let mut source_lengths = Vec::new();
    let mut target_lengths = Vec::new();
    let mut source_values = Vec::new();
    let mut target_values = Vec::new();
    let mut source_boundary = Vec::new();
    let mut target_boundary = Vec::new();
    let mut parent_operations = Vec::new();
    let mut boundary_links = Vec::<(NodeId, NodeId)>::new();

    for nested_graph in nested_graphs {
        let wire_base = wires.len();
        wires.extend(nested_graph.graph.h.w.0.0.iter().cloned());
        source_boundary.extend(
            boundary_table(&nested_graph.graph.s)
                .into_iter()
                .map(|wire| wire.0 + wire_base),
        );
        target_boundary.extend(
            boundary_table(&nested_graph.graph.t)
                .into_iter()
                .map(|wire| wire.0 + wire_base),
        );
        boundary_links.extend(
            nested_graph
                .boundary_relation
                .shifted_child_wires(wire_base),
        );

        for operation_id in 0..nested_graph.graph.h.x.0.len() {
            operations.push(nested_graph.graph.h.x.0.0[operation_id].clone());
            let sources = operation_sources(&nested_graph.graph, operation_id)
                .into_iter()
                .map(|wire| wire.0 + wire_base)
                .collect::<Vec<_>>();
            let targets = operation_targets(&nested_graph.graph, operation_id)
                .into_iter()
                .map(|wire| wire.0 + wire_base)
                .collect::<Vec<_>>();
            source_lengths.push(sources.len());
            target_lengths.push(targets.len());
            source_values.extend(sources);
            target_values.extend(targets);
        }
        parent_operations.extend(nested_graph.parent_operations);
    }

    let wire_count = wires.len();
    let graph = OpenHypergraph {
        s: finite_function(source_boundary, wire_count),
        t: finite_function(target_boundary, wire_count),
        h: Hypergraph {
            s: indexed_coproduct(source_lengths, source_values, wire_count),
            t: indexed_coproduct(target_lengths, target_values, wire_count),
            w: SemifiniteFunction::new(VecArray(wires)),
            x: SemifiniteFunction::new(VecArray(operations)),
        },
    }
    .validate()
    .expect("tensor of nested graphs must be valid");

    NestedGraph {
        graph,
        parent_operations,
        boundary_relation: if boundary_links.is_empty() {
            BoundaryRelation::empty()
        } else {
            BoundaryRelation::from_links(boundary_links)
        },
    }
}

fn boundary_links<'a>(
    child_boundary: &'a [NodeId],
    parent_boundary: &'a [NodeId],
) -> impl Iterator<Item = (NodeId, NodeId)> + 'a {
    child_boundary
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, child_wire)| {
            related_parent_wire(index, parent_boundary).map(|parent_wire| (child_wire, parent_wire))
        })
}

fn related_parent_wire(boundary_index: usize, parent_boundary: &[NodeId]) -> Option<NodeId> {
    if parent_boundary.len() == 1 {
        parent_boundary.first().copied()
    } else {
        parent_boundary.get(boundary_index).copied()
    }
}

fn boundary_table(boundary: &FiniteFunction) -> Vec<NodeId> {
    boundary.table.0.iter().copied().map(NodeId).collect()
}

fn operation_sources(graph: &Graph, operation_id: OperationId) -> Vec<NodeId> {
    operation_inputs(graph, operation_id).collect()
}

fn operation_targets(graph: &Graph, operation_id: OperationId) -> Vec<NodeId> {
    operation_outputs(graph, operation_id).collect()
}

fn operation_boundary(graph: &Graph, operation_id: OperationId) -> Vec<NodeId> {
    operation_inputs(graph, operation_id)
        .chain(operation_outputs(graph, operation_id))
        .collect()
}

fn boundary_side(graph: &Graph, wire: NodeId) -> Option<BoundarySide> {
    if boundary_table(&graph.s).contains(&wire) {
        Some(BoundarySide::Source)
    } else if boundary_table(&graph.t).contains(&wire) {
        Some(BoundarySide::Target)
    } else {
        None
    }
}

fn parent_operation_for_boundary_link(
    graph: &Graph,
    parent_operations: &[OperationId],
    parent: &Graph,
    child_wire: NodeId,
    parent_wire: NodeId,
    side: BoundarySide,
) -> Option<OperationId> {
    (0..graph.h.x.0.len()).find_map(|child_operation| {
        let child_boundary = operation_side_boundary(graph, child_operation, side);
        if !child_boundary.contains(&child_wire) {
            return None;
        }
        let parent_operation = parent_operations[child_operation];
        let parent_boundary = operation_side_boundary(parent, parent_operation, side);
        parent_boundary
            .contains(&parent_wire)
            .then_some(parent_operation)
    })
}

fn operation_side_boundary(
    graph: &Graph,
    operation_id: OperationId,
    side: BoundarySide,
) -> Vec<NodeId> {
    match side {
        BoundarySide::Source => operation_inputs(graph, operation_id).collect(),
        BoundarySide::Target => operation_outputs(graph, operation_id).collect(),
    }
}

fn indexed_coproduct(
    segment_lengths: Vec<usize>,
    values: Vec<usize>,
    target: usize,
) -> IndexedCoproduct<FiniteFunction> {
    let total = segment_lengths.iter().sum::<usize>();
    debug_assert_eq!(total, values.len());
    let sources = FiniteFunction::new(VecArray(segment_lengths), total + 1)
        .expect("segment lengths must form a valid indexed coproduct");
    let values = finite_function(values, target);
    IndexedCoproduct::new(sources, values).expect("incidence must be valid")
}

fn finite_function(table: Vec<usize>, target: usize) -> FiniteFunction {
    FiniteFunction::new(VecArray(table), target).expect("finite function table must be valid")
}
