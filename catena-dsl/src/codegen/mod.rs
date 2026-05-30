//! HIP codegen.
//!
//! This module lowers closure-converted, typed hypergraphs into a small dataflow
//! GPU artifact. Report generation should render this artifact, not make codegen
//! decisions itself.

pub mod fn_ptrs;
pub mod gpu;
pub mod lower_types;
mod prelude;

use std::collections::{BTreeMap, VecDeque};

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
    pub call_symbol: Option<String>,
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SpecializationKey {
    sources: Vec<CType>,
    targets: Vec<CType>,
    static_inputs: Vec<FnPtrSymbol>,
}

struct PendingInstance {
    op: Operation,
    name: String,
    overrides: BTreeMap<usize, LoweredType>,
}

struct CodegenState<'a> {
    definitions: &'a BTreeMap<Operation, AnnotatedTerm>,
    modules: GpuModuleMap,
    instances: BTreeMap<(Operation, SpecializationKey), String>,
    queue: VecDeque<PendingInstance>,
    next_specialization_id: usize,
}

/// Codegen for all functions, producing per-definition GPU modules.
pub fn codegen(terms: &TheoryTermMap) -> Result<GpuModuleMap, CodegenError> {
    let theory_id = TheoryId(
        PROGRAM_THEORY
            .parse()
            .expect("program theory id should parse"),
    );
    let Some(definitions) = terms.get(&theory_id) else {
        return Ok(BTreeMap::new());
    };

    let mut state = CodegenState {
        definitions,
        modules: BTreeMap::new(),
        instances: BTreeMap::new(),
        queue: VecDeque::new(),
        next_specialization_id: 0,
    };

    for (definition_name, term) in definitions {
        let Some(key) = entrypoint_key(term)? else {
            continue;
        };
        let name = sanitize_ident(&format!("{theory_id}.{definition_name}"));
        state
            .instances
            .insert((definition_name.clone(), key.clone()), name.clone());
        state.queue.push_back(PendingInstance {
            op: definition_name.clone(),
            name,
            overrides: BTreeMap::new(),
        });
    }

    while let Some(instance) = state.queue.pop_front() {
        let module_key: Operation = instance
            .name
            .parse()
            .expect("generated function name should parse as operation");
        if state.modules.contains_key(&module_key) {
            continue;
        }
        let term = state
            .definitions
            .get(&instance.op)
            .expect("queued specialization should have a definition");
        let module = state.codegen_definition(term, &instance)?;
        state.modules.insert(module_key, module);
    }

    Ok(state.modules)
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
    #[error("definition `{0}` is used with non-monomorphic runtime interface")]
    NonMonomorphicUse(Operation),
}

impl CodegenState<'_> {
    /// Lower one closure-converted, type-annotated definition into the dataflow GPU artifact.
    ///
    /// Direct `name.*` producers are recorded as symbolic function values instead of runtime
    /// assignments. Calls to other `program` definitions are resolved to generated specialization
    /// symbols and enqueue those specializations as needed.
    fn codegen_definition(
        &mut self,
        term: &AnnotatedTerm,
        instance: &PendingInstance,
    ) -> Result<GpuModule, CodegenError> {
        let fn_symbols = direct_fn_ptr_symbols(term)?;

        let mut term = term.clone();
        term.quotient().map_err(CodegenError::Quotient)?;

        let mut sources = Vec::new();
        for source in &term.sources {
            let var = var(*source, &term, &instance.overrides)?;
            if matches!(var.lowered, LoweredType::Runtime(_)) {
                sources.push(var);
            }
        }

        let mut targets = Vec::new();
        for target in &term.targets {
            let var = var(*target, &term, &instance.overrides)?;
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
                        Ok(GpuValue::Var(var(*node, &term, &instance.overrides)?))
                    }
                })
                .collect::<Result<Vec<_>, CodegenError>>()?;
            let outputs = assignment
                .targets
                .iter()
                .map(|(node, _)| var(*node, &term, &instance.overrides))
                .collect::<Result<Vec<_>, CodegenError>>()?;

            let call_symbol = if self.definitions.contains_key(&assignment.op) {
                Some(self.ensure_specialization(&assignment.op, &inputs, &outputs)?)
            } else {
                None
            };

            assignments.push(GpuAssign {
                op: assignment.op,
                call_symbol,
                inputs,
                outputs,
            });
        }

        Ok(GpuModule {
            name: instance.name.clone(),
            entry: GpuFunction {
                name: instance.name.clone(),
                sources,
                targets,
                assignments,
            },
        })
    }

    fn ensure_specialization(
        &mut self,
        op: &Operation,
        inputs: &[GpuValue],
        outputs: &[GpuVar],
    ) -> Result<String, CodegenError> {
        let key = specialization_key(inputs, outputs)
            .ok_or_else(|| CodegenError::NonMonomorphicUse(op.clone()))?;
        if let Some(name) = self.instances.get(&(op.clone(), key.clone())) {
            return Ok(name.clone());
        }

        let name = sanitize_ident(&format!(
            "{PROGRAM_THEORY}.{op}__{}",
            self.next_specialization_id
        ));
        self.next_specialization_id += 1;
        let overrides = specialization_overrides(
            self.definitions
                .get(op)
                .expect("specialized operation should have a definition"),
            inputs,
            outputs,
        );
        self.instances
            .insert((op.clone(), key.clone()), name.clone());
        self.queue.push_back(PendingInstance {
            op: op.clone(),
            name: name.clone(),
            overrides,
        });
        Ok(name)
    }
}

fn var(
    node: NodeId,
    term: &AnnotatedTerm,
    overrides: &BTreeMap<usize, LoweredType>,
) -> Result<GpuVar, CodegenError> {
    Ok(GpuVar {
        node,
        name: node_var(node),
        lowered: overrides
            .get(&node.0)
            .cloned()
            .unwrap_or(lower_type(&term.hypergraph.nodes[node.0])?),
    })
}

fn entrypoint_key(term: &AnnotatedTerm) -> Result<Option<SpecializationKey>, CodegenError> {
    let mut sources = Vec::new();
    let mut targets = Vec::new();
    for source in &term.sources {
        if let LoweredType::Runtime(ty) = lower_type(&term.hypergraph.nodes[source.0])? {
            sources.push(ty);
        }
    }
    for target in &term.targets {
        if let LoweredType::Runtime(ty) = lower_type(&term.hypergraph.nodes[target.0])? {
            targets.push(ty);
        }
    }
    if sources.is_empty() && targets.is_empty() {
        return Ok(None);
    }
    Ok(Some(SpecializationKey {
        sources,
        targets,
        static_inputs: Vec::new(),
    }))
}

fn specialization_key(inputs: &[GpuValue], outputs: &[GpuVar]) -> Option<SpecializationKey> {
    let mut sources = Vec::new();
    let mut static_inputs = Vec::new();
    for input in inputs {
        match input {
            GpuValue::Var(var) => {
                if let LoweredType::Runtime(ty) = &var.lowered {
                    sources.push(ty.clone());
                }
            }
            GpuValue::FnSymbol(symbol) => static_inputs.push(symbol.clone()),
        }
    }
    let mut targets = Vec::new();
    for output in outputs {
        if let LoweredType::Runtime(ty) = &output.lowered {
            targets.push(ty.clone());
        }
    }
    if sources.is_empty() && targets.is_empty() && static_inputs.is_empty() {
        return None;
    }
    Some(SpecializationKey {
        sources,
        targets,
        static_inputs,
    })
}

fn specialization_overrides(
    term: &AnnotatedTerm,
    inputs: &[GpuValue],
    outputs: &[GpuVar],
) -> BTreeMap<usize, LoweredType> {
    let mut overrides = BTreeMap::new();
    for (node, input) in term.sources.iter().zip(inputs.iter()) {
        if let GpuValue::Var(var) = input {
            overrides.insert(node.0, var.lowered.clone());
        }
    }
    for (node, output) in term.targets.iter().zip(outputs.iter()) {
        overrides.insert(node.0, output.lowered.clone());
    }
    overrides
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

#[cfg(test)]
mod tests {
    use super::*;

    use metacat::tree::Tree;
    use open_hypergraphs::lax::OpenHypergraph;

    fn op(name: &str) -> Operation {
        name.parse().unwrap()
    }

    fn node(name: &str, children: Vec<Tree<(), Operation>>) -> Tree<(), Operation> {
        Tree::Node(op(name), 0, children)
    }

    fn var(node: usize, name: &str, lowered: LoweredType) -> GpuVar {
        GpuVar {
            node: NodeId(node),
            name: name.to_string(),
            lowered,
        }
    }

    #[test]
    fn function_symbols_are_part_of_specialization_key() {
        let output = var(2, "x2", LoweredType::Runtime(CType::Bool));
        let foo_key = specialization_key(
            &[GpuValue::FnSymbol(FnPtrSymbol { target: op("foo") })],
            std::slice::from_ref(&output),
        )
        .unwrap();
        let bar_key = specialization_key(
            &[GpuValue::FnSymbol(FnPtrSymbol { target: op("bar") })],
            &[output],
        )
        .unwrap();

        assert_ne!(foo_key, bar_key);
    }

    #[test]
    fn erased_only_generic_definition_is_not_an_entrypoint() {
        let term = OpenHypergraph::identity(vec![Tree::Leaf(0, ())]);

        assert!(entrypoint_key(&term).unwrap().is_none());
    }

    #[test]
    fn runtime_interface_definition_is_an_entrypoint() {
        let term = OpenHypergraph::identity(vec![node("val", vec![node("bool", vec![])])]);

        assert_eq!(
            entrypoint_key(&term).unwrap().unwrap(),
            SpecializationKey {
                sources: vec![CType::Bool],
                targets: vec![CType::Bool],
                static_inputs: Vec::new(),
            }
        );
    }
}
