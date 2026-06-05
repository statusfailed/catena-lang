//! Execute compiled C backend functions through a small ABI-oriented interface.

use std::ffi::c_void;
use std::path::Path;

use libffi::middle::{Arg, Cif, CodePtr, Type};
use libloading::Library;
use thiserror::Error;

/// The set of possible types in the image of lowering a valid catena boundary type
#[derive(Debug)]
pub enum AbiValue<'a> {
    U8(&'a u8),
    U64(&'a u64),
}

/// Role of an [`AbiValue`] as either input or output
#[derive(Debug)]
pub enum ArgValue<'a> {
    Val(AbiValue<'a>),
    Out(AbiValue<'a>),
}

/// List of arguments & return ptrs passed on function invocation
#[derive(Debug)]
pub struct CallFrame<'a> {
    pub args: &'a mut [ArgValue<'a>],
}

#[derive(Debug, Error)]
pub enum ExecutorError {
    #[error("Failed to load shared object: {0}")]
    LoadLibrary(#[source] libloading::Error),
    #[error("Failed to resolve symbol '{symbol}': {source}")]
    LoadSymbol {
        symbol: String,
        #[source]
        source: libloading::Error,
    },
}

/// Invoke a compiled symbol using the generated C ABI.
///
/// The executor only knows about ABI-level scalar slots and output pointers.
/// Catena-specific type mapping belongs in `runtime`.
///
/// TODO: this introduces a bunch of overhead which should be cached:
///     - looking up symbols by name
///     - computing Vec<Type> from Vec<ArgValue>
pub(crate) fn exec(
    so_path: &Path,
    symbol: &str,
    frame: CallFrame<'_>,
) -> Result<(), ExecutorError> {
    // Load shared object
    let library = unsafe { Library::new(so_path) }.map_err(ExecutorError::LoadLibrary)?;

    // Get function symbol by name
    let symbol_name = format!("{symbol}\0");
    let function =
        unsafe { library.get::<*mut c_void>(symbol_name.as_bytes()) }.map_err(|source| {
            ExecutorError::LoadSymbol {
                symbol: symbol.to_string(),
                source,
            }
        })?;

    // Compute list of ABI types
    let types = frame
        .args
        .iter()
        .map(ArgValue::ffi_type)
        .collect::<Vec<_>>();

    // Outputs have to be treated separately: we need to give cif a *pointer to the pointer*,
    // but we actually only have a pointer! So we'll need to create the double-pointers briefly
    // while we cif.call.
    let mut pointer_args = Vec::new();
    for arg in frame.args.iter_mut() {
        if let ArgValue::Out(value) = arg {
            pointer_args.push(value.as_pointer_arg());
        }
    }

    // Create a Cif to call function symbol
    let cif = Cif::new(types, Type::void());
    let mut pointer_index = 0usize;
    let args: Vec<Arg<'_>> = frame
        .args
        .iter()
        .map(|arg| match arg {
            ArgValue::Val(value) => value.as_arg(),
            ArgValue::Out(_) => {
                let ptr = &pointer_args[pointer_index];
                pointer_index += 1;
                Arg::new(ptr)
            }
        })
        .collect();

    unsafe {
        cif.call::<()>(CodePtr(*function), &args);
    }
    Ok(())
}

impl AbiValue<'_> {
    fn ffi_type(&self) -> Type {
        match self {
            AbiValue::U8(_) => Type::u8(),
            AbiValue::U64(_) => Type::u64(),
        }
    }

    fn as_arg(&self) -> Arg<'_> {
        match self {
            AbiValue::U8(value) => Arg::new(*value),
            AbiValue::U64(value) => Arg::new(*value),
        }
    }

    fn as_pointer_arg(&self) -> *const c_void {
        match self {
            AbiValue::U8(slot) => (*slot as *const u8).cast::<c_void>(),
            AbiValue::U64(slot) => (*slot as *const u64).cast::<c_void>(),
        }
    }
}

impl ArgValue<'_> {
    fn ffi_type(&self) -> Type {
        match self {
            ArgValue::Val(value) => value.ffi_type(),
            ArgValue::Out(_) => Type::pointer(),
        }
    }
}
