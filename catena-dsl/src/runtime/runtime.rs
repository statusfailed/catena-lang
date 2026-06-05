use thiserror::Error;

use std::{collections::HashMap, path::PathBuf};

use super::artifact::{Artifact, ArtifactError};
use super::executor::{CallFrame, ExecutorError};
use super::{
    signature::{FunctionSignature, signatures},
    value::{Value, ValueKind},
};
use crate::compile::CompileFailure;

/// Run catena programs with the C backend
#[derive(Debug)]
pub struct Runtime {
    artifact: Artifact,
    signatures: HashMap<String, FunctionSignature>,
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
    /// Construct a new runtime from a list of paths, interpreted as catena programs (&stdlib)
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
        let signatures = signatures(modules);

        let report_dir = tempfile::Builder::new()
            .prefix("catena-report-")
            .tempdir()?;
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

        // Check arity/coarity lines up with what's in the function signature.
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

        let mut output_values: Vec<Value> = signature
            .outputs
            .iter()
            .copied()
            .map(Value::zeroed)
            .collect();

        // Construct the `ArgValue`s with which to call the function
        let mut frame_args = Vec::with_capacity(M + N);
        for (index, (value, expected)) in args
            .iter()
            .zip(signature.inputs.iter().copied())
            .enumerate()
        {
            if value.kind() != expected {
                return Err(ExecError::TypeMismatch {
                    index,
                    expected,
                    actual: value.kind(),
                });
            }
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
