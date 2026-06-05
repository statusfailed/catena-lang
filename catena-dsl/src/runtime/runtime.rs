use thiserror::Error;

use std::{collections::HashMap, path::PathBuf};

use super::artifact::{Artifact, ArtifactError};
use super::executor::{CallFrame, ExecutorError};
use super::value::{Value, ValueKind};
use crate::{
    codegen::{lower_types::CType, GpuModuleMap},
    compile::CompileFailure,
};

/// Run catena programs with the C backend
#[derive(Debug)]
pub struct Runtime {
    artifact: Artifact,
    signatures: HashMap<String, FunctionSignature>,
}

#[derive(Debug, Clone)]
pub(crate) struct FunctionSignature {
    pub(crate) symbol: String,
    pub(crate) inputs: Vec<ValueKind>,
    pub(crate) outputs: Vec<ValueKind>,
}

#[derive(Debug, Error)]
pub enum InitError {
    #[error("Failed to parse program: {0}")]
    Parse(#[from] metacat::theory::ast::ParseRawError),
    #[error(transparent)]
    Compile(#[from] CompileFailure),
    #[error("compile report did not contain GPU modules")]
    MissingGpuModules,
    #[error("failed to write generated report files: {0}")]
    DumpReport(#[from] std::io::Error),
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
}

#[derive(Debug, Error)]
pub enum ExecError {
    #[error("Unknown function '{0}'")]
    UnknownFunction(String),
    #[error("Argument {index} expected {expected:?}, got {actual:?}")]
    TypeMismatch {
        index: usize,
        expected: ValueKind,
        actual: ValueKind,
    },
    #[error("Function '{name}' expected {expected} inputs, got {actual}")]
    InputArityMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("Function '{name}' expected {expected} outputs, got {actual}")]
    OutputArityMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("Executor error: {0}")]
    Executor(#[from] ExecutorError),
}

impl Runtime {
    pub fn new<I>(paths: I) -> Result<Runtime, InitError>
    where
        I: IntoIterator<Item = PathBuf>,
    {
        let raw_theories = metacat::theory::RawTheorySet::from_files(paths)?;
        let report = crate::compile::compile(raw_theories)?;
        let modules = report
            .gpu_modules
            .as_ref()
            .ok_or(InitError::MissingGpuModules)?;
        let signatures = signatures(modules)?;

        let report_dir = tempfile::Builder::new().prefix("catena-report-").tempdir()?;
        report.dump_to_dir(report_dir.path())?;
        let cpp_path = report_dir.path().join("gpu/program.cpp");
        let artifact = super::artifact::compile(&cpp_path)?;

        Ok(Self {
            artifact,
            signatures,
        })
    }

    /// Run 'fn_name', which must have M arguments, and return its N arguments.
    pub fn exec<const M: usize, const N: usize>(
        &self,
        fn_name: &str,
        args: [Value; M],
    ) -> Result<[Value; N], ExecError> {
        let signature = self
            .signatures
            .get(fn_name)
            .ok_or_else(|| ExecError::UnknownFunction(fn_name.to_string()))?;

        if signature.inputs.len() != M {
            return Err(ExecError::InputArityMismatch {
                name: fn_name.to_string(),
                expected: signature.inputs.len(),
                actual: M,
            });
        }
        if signature.outputs.len() != N {
            return Err(ExecError::OutputArityMismatch {
                name: fn_name.to_string(),
                expected: signature.outputs.len(),
                actual: N,
            });
        }

        let input_values = collect_inputs(&args, signature)?;
        let mut output_values: Vec<Value> = signature
            .outputs
            .iter()
            .copied()
            .map(Value::zeroed)
            .collect();

        let mut frame_args = Vec::with_capacity(M + N);
        for value in &input_values {
            frame_args.push(value.as_input_arg());
        }
        for value in &mut output_values {
            frame_args.push(value.as_output_arg());
        }

        super::executor::exec(
            self.artifact.path(),
            &signature.symbol,
            CallFrame {
                args: &mut frame_args,
            },
        )?;

        Ok(output_values
            .try_into()
            .expect("output arity already validated"))
    }
}

fn signatures(modules: &GpuModuleMap) -> Result<HashMap<String, FunctionSignature>, InitError> {
    let mut signatures = HashMap::new();
    for module in modules.values() {
        let Some(inputs) = module
            .entry
            .sources
            .iter()
            .map(|var| {
                let ty = crate::codegen::runtime_type(var)
                    .expect("GpuFunction sources should be runtime-lowered");
                value_kind(ty)
            })
            .collect::<Option<Vec<_>>>() else {
                continue;
            };
        let Some(outputs) = module
            .entry
            .targets
            .iter()
            .map(|var| {
                let ty = crate::codegen::runtime_type(var)
                    .expect("GpuFunction targets should be runtime-lowered");
                value_kind(ty)
            })
            .collect::<Option<Vec<_>>>() else {
                continue;
            };

        signatures.insert(
            module.entry.name.clone(),
            FunctionSignature {
                symbol: module.entry.name.clone(),
                inputs,
                outputs,
            },
        );
    }
    Ok(signatures)
}

fn value_kind(ty: &CType) -> Option<ValueKind> {
    match ty {
        CType::Bool => Some(ValueKind::Bool),
        CType::U64 => Some(ValueKind::U64),
        _ => None,
    }
}

fn collect_inputs<const M: usize>(
    args: &[Value; M],
    signature: &FunctionSignature,
) -> Result<Vec<Value>, ExecError> {
    args.iter()
        .zip(signature.inputs.iter().copied())
        .enumerate()
        .map(|(index, (value, expected))| validate_input(index, value, expected))
        .collect()
}

// Verify that an input value has the expected kind, but does *not* check deeply (e.g., that an
// index belongs to its declared extent.
fn validate_input(index: usize, value: &Value, expected: ValueKind) -> Result<Value, ExecError> {
    match (value, expected) {
        (Value::Bool(value), ValueKind::Bool) => Ok(Value::Bool(*value)),
        (Value::U64(value), ValueKind::U64) => Ok(Value::U64(*value)),
        _ => Err(ExecError::TypeMismatch {
            index,
            expected,
            actual: value.kind(),
        }),
    }
}
