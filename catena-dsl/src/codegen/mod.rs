//! HIP codegen.
//!
//! This module lowers closure-converted, typed hypergraphs into a small dataflow
//! GPU artifact. Report generation should render this artifact, not make codegen
//! decisions itself.

pub mod fn_ptrs;
pub mod gpu;
pub mod lower_types;
mod prelude;

use std::collections::BTreeMap;

use hexpr::Operation;
use metacat::{
    ssa::{SSAError, ssa},
    theory::TheoryId,
};
use open_hypergraphs::lax::NodeId;
use thiserror::Error;

use crate::{
    codegen::{
        fn_ptrs::{FnPtrSymbol, FnPtrSymbolError, direct_fn_ptr_symbols},
        lower_types::{CType, LowerTypeError, LoweredType, lower_type},
    },
    report::{AnnotatedTerm, TheoryTermMap},
};

pub type GpuModuleMap = BTreeMap<Operation, GpuModule>;

const PROGRAM_THEORY: &str = "program";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuModule {
    pub name: String,
    pub entry: GpuFunction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuFunction {
    pub name: String,
    pub sources: Vec<GpuVar>,
    pub targets: Vec<GpuVar>,
    pub assignments: Vec<GpuAssign>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuAssign {
    pub op: Operation,
    pub inputs: Vec<GpuValue>,
    pub outputs: Vec<GpuVar>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuValue {
    Var(GpuVar),
    FnSymbol(FnPtrSymbol),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuVar {
    pub node: NodeId,
    pub name: String,
    pub lowered: LoweredType,
}

/// Codegen for all functions, producing per-definition GPU modules.
pub fn codegen(terms: &TheoryTermMap) -> Result<GpuModuleMap, CodegenError> {
    let mut modules = BTreeMap::new();

    let theory_id = TheoryId(
        PROGRAM_THEORY
            .parse()
            .expect("program theory id should parse"),
    );
    let Some(definitions) = terms.get(&theory_id) else {
        return Ok(modules);
    };

    for (definition_name, term) in definitions {
        modules.insert(
            definition_name.clone(),
            codegen_definition(&format!("{theory_id}.{definition_name}"), term)?,
        );
    }

    Ok(modules)
}

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error(transparent)]
    Ssa(#[from] SSAError),
    #[error("failed to quotient transformed term before codegen: {0:?}")]
    Quotient(open_hypergraphs::strict::vec::FiniteFunction),
    #[error(transparent)]
    LowerType(#[from] LowerTypeError),
    #[error(transparent)]
    FnPtrSymbol(#[from] FnPtrSymbolError),
}

/// Lower one closure-converted, type-annotated definition into the dataflow GPU artifact.
///
/// Direct `name.*` producers are recorded as symbolic function values instead of runtime
/// assignments. All other edges become SSA-style `GpuAssign`s over lowered runtime wires.
fn codegen_definition(
    qualified_name: &str,
    term: &AnnotatedTerm,
) -> Result<GpuModule, CodegenError> {
    let fn_symbols = direct_fn_ptr_symbols(term)?;

    let mut term = term.clone();
    term.quotient().map_err(CodegenError::Quotient)?;

    let mut sources = Vec::new();
    for source in &term.sources {
        let var = var(*source, &term)?;
        if matches!(var.lowered, LoweredType::Runtime(_)) {
            sources.push(var);
        }
    }

    let mut targets = Vec::new();
    for target in &term.targets {
        let var = var(*target, &term)?;
        if matches!(var.lowered, LoweredType::Runtime(_)) {
            targets.push(var);
        }
    }

    let mut assignments = Vec::new();
    for assignment in ssa(term.clone().to_strict())? {
        if assignment.op.as_str().starts_with("name.") {
            continue;
        }

        let inputs = assignment
            .sources
            .iter()
            .map(|(node, _)| {
                if let Some(symbol) = fn_symbols.get(node) {
                    Ok(GpuValue::FnSymbol(symbol.clone()))
                } else {
                    Ok(GpuValue::Var(var(*node, &term)?))
                }
            })
            .collect::<Result<Vec<_>, CodegenError>>()?;
        let outputs = assignment
            .targets
            .iter()
            .map(|(node, _)| var(*node, &term))
            .collect::<Result<Vec<_>, CodegenError>>()?;

        assignments.push(GpuAssign {
            op: assignment.op,
            inputs,
            outputs,
        });
    }

    let name = sanitize_ident(qualified_name);
    Ok(GpuModule {
        name: name.clone(),
        entry: GpuFunction {
            name,
            sources,
            targets,
            assignments,
        },
    })
}

fn var(node: NodeId, term: &AnnotatedTerm) -> Result<GpuVar, CodegenError> {
    Ok(GpuVar {
        node,
        name: node_var(node),
        lowered: lower_type(&term.hypergraph.nodes[node.0])?,
    })
}

fn node_var(node: NodeId) -> String {
    format!("x{}", node.0)
}

fn sanitize_ident(name: &str) -> String {
    let mut ident = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    if ident.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        ident.insert(0, '_');
    }
    ident
}

pub fn runtime_type(var: &GpuVar) -> Option<&CType> {
    match &var.lowered {
        LoweredType::Runtime(ty) => Some(ty),
        LoweredType::Erased => None,
    }
}
