//! Catena C backend

/// Generate C code for a catena program
pub mod codegen;

/// Compile a catena program to a .so file using the C backend
pub mod compile;

/// manage and run compiled catena programs
pub mod runtime;

pub use runtime::Runtime;
pub use runtime::Value;
