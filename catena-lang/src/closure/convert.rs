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

const FN_TYPE: &str = "->";
const NAME_PREFIX: &str = "name.";
const PRODUCT_TYPE: &str = "*";
const UNIT_TYPE: &str = "1";
const VALUE_TYPE: &str = "val";

type Obj = Tree<(), Operation>;

#[derive(Debug, Clone)]
pub struct Converted {
    pub definition: AnnotatedTerm,
    pub closures: Vec<ConvertedClosure>,
}

#[derive(Debug, Clone)]
pub struct ConvertedClosure {
    pub node: NodeId,
    pub term: AnnotatedTerm,
    pub type_info: TypeInfo,
}

impl ConvertedClosure {
    pub fn name(&self, definition_name: &Operation) -> Operation {
        closure_operation(definition_name, self.node)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeInfo {
    pub environment: Vec<Obj>, // X (context, possibly multiple wires)
    pub domain: Obj,           // A (always packed)
    pub codomain: Obj,         // B (always packed)
}

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
/// Returns the rewritten term plus the generated closure body terms. Each
/// generated closure records the original closure node id and the type
/// information needed by the caller to elaborate its `name.closure.*` operation.
pub fn convert(
    definition_name: &Operation,
    definition: &AnnotatedTerm,
    closure_wires: &[NodeId],
) -> Result<Converted, ConvertError> {
    let regions = closure_region(definition, closure_wires)?;

    let mut closures = Vec::new();
    let mut rewrites = Vec::new();
    for region in regions {
        let extracted = extract_region(definition, &region)?;
        let body = closure_body(&extracted)?;
        let type_info = type_info(definition, &region)?;
        let replacement = replacement_region(definition_name, definition, &region, &type_info);
        closures.push(ConvertedClosure {
            node: region.closure_wire,
            term: body,
            type_info,
        });
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

    Ok(Converted {
        definition: rewritten,
        closures,
    })
}

fn replacement_region(
    definition_name: &Operation,
    definition: &AnnotatedTerm,
    region: &ClosureRegion,
    type_info: &TypeInfo,
) -> AnnotatedTerm {
    let mut replacement = AnnotatedTerm::empty();
    let sources = region
        .defer_inputs
        .iter()
        .map(|wire| replacement.new_node(definition.hypergraph.nodes[wire.0].clone()))
        .collect::<Vec<_>>();
    let function_pointer = replacement.new_node(function_pointer_type(
        [
            type_info.environment.clone(),
            vec![type_info.domain.clone()],
        ]
        .concat(),
        vec![type_info.codomain.clone()],
    ));
    replacement.new_edge(
        name_operation(definition_name, region.closure_wire),
        (vec![], vec![function_pointer]),
    );
    replacement.sources = sources;
    replacement.targets = [replacement.sources.clone(), vec![function_pointer]].concat();
    replacement
}

fn type_info(definition: &AnnotatedTerm, region: &ClosureRegion) -> Result<TypeInfo, ConvertError> {
    let environment = region
        .defer_inputs
        .iter()
        .map(|wire| definition.hypergraph.nodes[wire.0].clone())
        .collect::<Vec<_>>();
    let (domain, codomain) = closure_parts(&region.closure_type)
        .expect("closure region type should be a binary closure type");
    Ok(TypeInfo {
        environment,
        domain: domain.clone(),
        codomain: codomain.clone(),
    })
}

fn closure_operation(definition_name: &Operation, closure_wire: NodeId) -> Operation {
    format!("closure.{}.{}", definition_name, closure_wire.0)
        .parse()
        .expect("generated closure operation should parse")
}

fn name_operation(definition_name: &Operation, closure_wire: NodeId) -> Operation {
    format!(
        "{NAME_PREFIX}{}",
        closure_operation(definition_name, closure_wire)
    )
    .parse()
    .expect("generated name operation should parse")
}

fn closure_parts(object: &Obj) -> Option<(&Obj, &Obj)> {
    let Tree::Node(operation, _, children) = object else {
        return None;
    };
    if operation.as_str() != "=>" {
        return None;
    }
    let [domain, codomain] = children.as_slice() else {
        return None;
    };
    Some((domain, codomain))
}

fn function_pointer_type(sources: Vec<Obj>, targets: Vec<Obj>) -> Obj {
    value_type(function_type(pack_object(sources), pack_object(targets)))
}

fn function_type(domain: Obj, codomain: Obj) -> Obj {
    Tree::Node(op(FN_TYPE), 0, vec![domain, codomain])
}

fn value_type(inner: Obj) -> Obj {
    Tree::Node(op(VALUE_TYPE), 0, vec![inner])
}

fn pack_object(objects: Vec<Obj>) -> Obj {
    match objects.as_slice() {
        [] => Tree::Node(op(UNIT_TYPE), 0, vec![]),
        [only] => only.clone(),
        _ => Tree::Node(op(PRODUCT_TYPE), 0, objects),
    }
}

fn op(name: &str) -> Operation {
    name.parse().expect("generated operation should parse")
}
