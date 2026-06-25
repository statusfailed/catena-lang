use hexpr::Operation;
use metacat::tree::Tree;
use thiserror::Error;

use crate::check::AnnotatedTerm;

const CLOSURE_TYPE: &str = "=>";
const PRODUCT_TYPE: &str = "*";
const UNIT_TYPE: &str = "1";
const DEFER: &str = "defer";
const COMPOSE: &str = "compose";
const RUN: &str = "run";
const PRODUCT_ELIM: &str = "*.elim";

type Obj = Tree<(), Operation>;

#[derive(Debug, Error)]
pub enum ClosureBodyError {
    #[error("extracted closure region must have exactly one target, found {actual}")]
    TargetArity { actual: usize },
    #[error("extracted closure region target node n{wire} is out of bounds")]
    TargetOutOfBounds { wire: usize },
    #[error("extracted closure region target node n{wire} is not closure-typed")]
    TargetNotClosureTyped { wire: usize },
}

/// Build the function body for an extracted closure region.
///
/// Given an extracted term `t : X -> (A => B)`, this produces a term
/// `X, A -> B` by appending the inlined evaluation sequence:
/// `defer ; compose ; run`.
pub fn closure_body(extracted: &AnnotatedTerm) -> Result<AnnotatedTerm, ClosureBodyError> {
    let [closure_wire] = extracted.targets.as_slice() else {
        return Err(ClosureBodyError::TargetArity {
            actual: extracted.targets.len(),
        });
    };
    let closure_type = extracted.hypergraph.nodes.get(closure_wire.0).ok_or(
        ClosureBodyError::TargetOutOfBounds {
            wire: closure_wire.0,
        },
    )?;
    let (domain, codomain) =
        closure_parts(closure_type).ok_or(ClosureBodyError::TargetNotClosureTyped {
            wire: closure_wire.0,
        })?;

    let mut body = extracted.clone();
    let unit = unit_type();
    let environment = packed_environment_source(&mut body);

    let argument = body.new_node(domain.clone());
    let deferred_argument = body.new_node(closure_type_of(unit.clone(), domain.clone()));
    let composed = body.new_node(closure_type_of(unit, codomain.clone()));
    let output = body.new_node(codomain.clone());

    body.new_edge(op(DEFER), (vec![argument], vec![deferred_argument]));
    body.new_edge(
        op(COMPOSE),
        (vec![deferred_argument, *closure_wire], vec![composed]),
    );
    body.new_edge(op(RUN), (vec![composed], vec![output]));

    body.sources = vec![environment, argument];
    body.targets = vec![output];

    Ok(body)
}

fn packed_environment_source(body: &mut AnnotatedTerm) -> open_hypergraphs::lax::NodeId {
    let environment = body.sources.clone();
    let environment_types = interface_types(body, &environment);
    match environment.as_slice() {
        [] => body.new_node(unit_type()),
        [only] => *only,
        _ => {
            let packed = body.new_node(pack_object(&environment_types));
            unpack_environment(body, packed, &environment, &environment_types);
            packed
        }
    }
}

fn unpack_environment(
    body: &mut AnnotatedTerm,
    packed: open_hypergraphs::lax::NodeId,
    components: &[open_hypergraphs::lax::NodeId],
    component_types: &[Obj],
) {
    match components {
        [] | [_] => {}
        [left, right] => {
            body.new_edge(op(PRODUCT_ELIM), (vec![packed], vec![*left, *right]));
        }
        [left, rest @ ..] => {
            let tail_type = pack_object(&component_types[1..]);
            let tail = body.new_node(tail_type);
            body.new_edge(op(PRODUCT_ELIM), (vec![packed], vec![*left, tail]));
            unpack_environment(body, tail, rest, &component_types[1..]);
        }
    }
}

fn closure_parts(object: &Obj) -> Option<(&Obj, &Obj)> {
    let Tree::Node(operation, _, children) = object else {
        return None;
    };
    if operation.as_str() != CLOSURE_TYPE {
        return None;
    }
    let [domain, codomain] = children.as_slice() else {
        return None;
    };
    Some((domain, codomain))
}

fn closure_type_of(domain: Obj, codomain: Obj) -> Obj {
    Tree::Node(op(CLOSURE_TYPE), 0, vec![domain, codomain])
}

fn unit_type() -> Obj {
    Tree::Node(op(UNIT_TYPE), 0, vec![])
}

fn pack_object(objects: &[Obj]) -> Obj {
    match objects {
        [] => unit_type(),
        [only] => only.clone(),
        [head, tail @ ..] => Tree::Node(op(PRODUCT_TYPE), 0, vec![head.clone(), pack_object(tail)]),
    }
}

fn interface_types(term: &AnnotatedTerm, interface: &[open_hypergraphs::lax::NodeId]) -> Vec<Obj> {
    interface
        .iter()
        .map(|node| term.hypergraph.nodes[node.0].clone())
        .collect()
}

fn op(name: &str) -> Operation {
    name.parse().expect("generated operation should parse")
}
