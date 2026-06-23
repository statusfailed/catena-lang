use hexpr::Operation;
use metacat::tree::Tree;
use thiserror::Error;

use crate::check::AnnotatedTerm;

const CLOSURE_TYPE: &str = "=>";
const UNIT_TYPE: &str = "1";
const DEFER: &str = "defer";
const COMPOSE: &str = "compose";
const RUN: &str = "run";

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

    body.sources.push(argument);
    body.targets = vec![output];

    Ok(body)
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

fn op(name: &str) -> Operation {
    name.parse().expect("generated operation should parse")
}
