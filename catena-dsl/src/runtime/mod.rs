//! Catena Hip Runtime

/// Public API for creating values to pass into generated catena code
pub mod value;

/// Compile generated HIP/C++ to a shared object.
pub mod artifact;

/// Marshal catena values into the C ABI and invoke compiled symbols
pub mod executor;

/// manage and run compiled catena programs
pub mod runtime;

//#[cfg(test)]
//mod tests;

pub use runtime::InitError;
pub use runtime::Runtime;
pub use value::Value;
pub use value::ValueKind;
