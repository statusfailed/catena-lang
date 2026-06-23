//! Closure conversion
//!
//! Given an AnnotatedTerm f containing a node n with closure type A => B,
//! identify its "smuggled environment" X, and produce:
//!
//! 1. A new term `closure.f.n : X ● A -> B`
//! 2. The *name* `name.closure.f.n : X ● A ● B -> (X ● A -> B)` (as in name elaboration)
//! 3. A modified `f` whose closure is replaced by `name.closure.f.n`
//!
//! This module exports two main things:
//!
//! - The `convert` function is the low level, per-definition implementation of this procedure.
//! - The `theory` module closure-converts a whole theory by lowering `if` calls to `ifc`.

// NOTE: this is really doing rewriting internally.
// In future, we should integrate using 'proper' rewrite machinery, rather than reimplementing it
// ad-hoc here.

// Identifies a region of an AnnotatedTerm corresponding to a closure.
pub(crate) mod region;

// Create an AnnotatedTerm `t` corresponding to a subregion identified by the `region` module
pub(crate) mod extract;

// From the extracted AnnotatedTerm create `closure.f.n` as `(f × defer) ; compose ; run`
// (see also `cmc.hex`'s "evaluate" function)
pub(crate) mod body;

// Replace an identified closure region with a caller-provided lowered body.
pub(crate) mod rewrite;

// Integrate region identification, extraction, body construction, and splicing.
pub mod convert;

// Convert a whole theory: for each arrow, identify the closure nodes appearing as (closure) inputs
// to a primitive (`if`), and closure-convert them.
pub mod theory;

#[cfg(test)]
mod tests;
