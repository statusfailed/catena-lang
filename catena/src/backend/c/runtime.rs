use thiserror::Error;

pub use super::compile::CompileError;

#[derive(Debug)]
pub struct Runtime {}

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
        Self {}
    }

    // Compile the standard library and all its functions.
    // Later, we'll need to allow multiple modules + auto-load the stdlib.
    pub fn compile(&self, source: &str) -> Result<(), CompileError> {
        println!("todo: read {} bytes", source.len());
        Ok(())
    }

    // Move a value into the runtime
    pub fn value(&self, _value: Value) -> ValueRef {
        ValueRef
    }

    /// Run 'fn_name', which must have M arguments, and return its N arguments.
    pub fn exec<const M: usize, const N: usize>(
        &self,
        fn_name: &str,
        args: [ValueRef; M],
    ) -> Result<[ValueRef; N], ExecError> {
        println!("{fn_name}({args:?})");
        todo!()
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}
