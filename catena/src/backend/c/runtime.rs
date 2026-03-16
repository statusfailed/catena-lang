use thiserror::Error;

use super::compile::{CompileError, SharedObject};
use super::executor::{ArgValue, CallFrame, ExecutorError};
use super::value::{Value, ValueKind};

/// Run catena programs with the C backend
#[derive(Debug)]
pub struct Runtime {
    artifact: SharedObject,
}

pub type InitError = CompileError;

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
    pub fn new(source: &str) -> Result<Runtime, InitError> {
        let artifact = super::compile::compile(source)?;
        Ok(Self { artifact })
    }

    /// Run 'fn_name', which must have M arguments, and return its N arguments.
    pub fn exec<const M: usize, const N: usize>(
        &self,
        fn_name: &str,
        args: [Value; M],
    ) -> Result<[Value; N], ExecError> {
        let signature = self
            .artifact
            .signature(fn_name)
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

        let abi_inputs = collect_inputs(&args, signature)?;
        let mut abi_outputs: Vec<AbiValue> = signature
            .outputs
            .iter()
            .copied()
            .map(AbiValue::zeroed)
            .collect();

        let mut frame_args = Vec::with_capacity(M + N);
        for value in &abi_inputs {
            frame_args.push(value.as_input_arg());
        }
        for value in &mut abi_outputs {
            frame_args.push(value.as_output_arg());
        }

        super::executor::exec(
            self.artifact.path(),
            &signature.symbol,
            CallFrame {
                args: &mut frame_args,
            },
        )?;

        let output_values: Vec<Value> = abi_outputs
            .into_iter()
            .zip(signature.outputs.iter().copied())
            .map(|(value, kind)| value.into_runtime_value(kind))
            .collect();

        Ok(output_values
            .try_into()
            .expect("output arity already validated"))
    }
}

fn collect_inputs<const M: usize>(
    args: &[Value; M],
    signature: &super::compile::FunctionSignature,
) -> Result<Vec<AbiValue>, ExecError> {
    args.iter()
        .zip(signature.inputs.iter().copied())
        .enumerate()
        .map(|(index, (value, expected))| validate_input(index, value, expected))
        .collect()
}

// Verify that an input value has the expected kind, but does *not* check deeply (e.g., that an
// index belongs to its declared extent.
fn validate_input(
    index: usize,
    value: &Value,
    expected: ValueKind,
) -> Result<AbiValue, ExecError> {
    match (value, expected) {
        (Value::Extent(value), ValueKind::Extent) => Ok(AbiValue::U64(*value as u64)),
        (Value::Index(value), ValueKind::Index) => Ok(AbiValue::U64(*value as u64)),
        (Value::F32(value), ValueKind::F32) => Ok(AbiValue::F32(*value)),
        _ => Err(ExecError::TypeMismatch {
            index,
            expected,
            actual: value.kind(),
        }),
    }
}

#[derive(Debug)]
enum AbiValue {
    U64(u64),
    F32(f32),
}

impl AbiValue {
    fn zeroed(kind: ValueKind) -> Self {
        match kind {
            ValueKind::Extent | ValueKind::Index => AbiValue::U64(0),
            ValueKind::F32 => AbiValue::F32(0.0),
        }
    }

    fn as_input_arg(&self) -> ArgValue<'_> {
        match self {
            AbiValue::U64(value) => ArgValue::U64(value),
            AbiValue::F32(value) => ArgValue::F32(value),
        }
    }

    fn as_output_arg(&mut self) -> ArgValue<'_> {
        match self {
            AbiValue::U64(value) => ArgValue::OutU64(value),
            AbiValue::F32(value) => ArgValue::OutF32(value),
        }
    }

    fn into_runtime_value(self, kind: ValueKind) -> Value {
        match (self, kind) {
            (AbiValue::U64(value), ValueKind::Extent) => Value::Extent(value as usize),
            (AbiValue::U64(value), ValueKind::Index) => Value::Index(value as usize),
            (AbiValue::F32(value), ValueKind::F32) => Value::F32(value),
            _ => unreachable!("compile/runtime kind mismatch"),
        }
    }
}
