use hexpr::Operation;
use metacat::theory::TheorySet;

use crate::report::CompileReport;

// TODO: Write a function `compile` which:
//
// - Elaborates input to include function names (finitary CMC)
// - Typechecks
// - Generates a `StructuredProgram` for each definition
// - Renders as a single CUDA file
// - Produces a CompileReport which contains all intermediate data, including graphs rendered with
//   open-hypergraphs-dot for each definition + the result of each pass.
//
// NOTE: *definitions* will never be inlined.
//
// At each stage, write debug output to an (optionally supplied) directory.
// Choose meaningful names for each file; render SVGs of terms where possible.
// Provide a top-level HTML file
//
// This should

/// Compile all definitions from 'theory' into a list of [`catena::structured::StructuredProgram`].
fn compile(theory_name: &Operation, theories: &TheorySet) -> CompileReport {
    let _ = (theory_name, theories);

    // Elaborate theories

    todo!()
}
