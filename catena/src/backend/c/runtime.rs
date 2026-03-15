use std::sync::Mutex;
use thiserror::Error;

pub use super::compile::CompileError;

/// Run catena programs with the C backend
#[derive(Debug)]
pub struct Runtime {
    artifact: Mutex<Option<super::compile::SharedObject>>,
}

/// Public interface for marshalling values into/out of the runtime
#[derive(Debug)]
pub enum Value {
    Extent(usize),
}

// An opaque pointer to a value reference.
// NOTE: these *cannot* be safely copied; we rely on them being 'consumable'.
#[derive(Debug)]
pub struct ValueRef;

// Error types
#[derive(Debug, Error)]
pub enum ExecError {}

impl Runtime {
    pub fn new() -> Runtime {
        Self {
            artifact: Mutex::new(None),
        }
    }

    // Compile the standard library and all its functions.
    // Later, we'll need to allow multiple modules + auto-load the stdlib.
    pub fn compile(&self, source: &str) -> Result<(), CompileError> {
        let artifact = super::compile::compile(source)?;
        let _ = artifact.path();
        *self.artifact.lock().unwrap() = Some(artifact);
        Ok(())
    }

    // Move a value into the runtime
    pub fn value(&self, _value: Value) -> ValueRef {
        todo!()
    }

    /// Run 'fn_name', which must have M arguments, and return its N arguments.
    pub fn exec<const M: usize, const N: usize>(
        &self,
        fn_name: &str,
        args: [ValueRef; M],
    ) -> Result<[ValueRef; N], ExecError> {
        let artifact = self.artifact.lock().unwrap();
        let symbol = artifact
            .as_ref()
            .and_then(|artifact| artifact.symbol(fn_name))
            .unwrap_or(fn_name);
        println!("{fn_name} [{symbol}] ({args:?})");
        todo!()
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}
