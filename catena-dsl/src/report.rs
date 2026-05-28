mod gpu;
mod svg;

use std::{fs, io, path::Path};

use catena::structured::StructuredProgram;
use hexpr::Operation;
use metacat::{
    theory::{RawTheorySet, TheoryId, TheorySet},
    tree::Tree,
};
use open_hypergraphs::lax::OpenHypergraph;
use std::collections::BTreeMap;

use crate::check::PartialDefinitionTypes;

/// A definition graph whose nodes are annotated with their computed object types.
pub type AnnotatedTerm = OpenHypergraph<Tree<(), Operation>, Operation>;
/// Generic storage for per-theory, per-definition graph results produced by compiler passes.
pub type TheoryTermMap = BTreeMap<TheoryId, BTreeMap<Operation, AnnotatedTerm>>;
/// Generic storage for per-theory, per-definition structured codegen results.
pub type StructuredProgramMap = BTreeMap<TheoryId, BTreeMap<Operation, StructuredProgram>>;

#[derive(Debug)]
pub struct CompileReport {
    pub raw_theories: RawTheorySet,
    pub elaborated: Option<RawTheorySet>,
    pub theory_set: Option<TheorySet>,
    pub definition_types: Option<BTreeMap<TheoryId, BTreeMap<Operation, Vec<Tree<(), Operation>>>>>,
    pub partial_definition_types: Option<PartialDefinitionTypes>,
    pub forgotten_closures: Option<TheoryTermMap>,
    pub structured_programs: Option<StructuredProgramMap>,
}

impl CompileReport {
    pub fn new(raw_theories: RawTheorySet) -> Self {
        Self {
            raw_theories,
            elaborated: None,
            theory_set: None,
            definition_types: None,
            partial_definition_types: None,
            forgotten_closures: None,
            structured_programs: None,
        }
    }
}

impl CompileReport {
    pub fn dump_to_dir(&self, dir: impl AsRef<Path>) -> io::Result<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        fs::write(
            dir.join("raw_theories.hex"),
            self.raw_theories.to_hexpr_text(),
        )?;
        if let Some(elaborated) = &self.elaborated {
            fs::write(dir.join("elaborated.hex"), elaborated.to_hexpr_text())?;
        }
        svg::dump_svgs(self, &dir.join("svgs"))?;
        gpu::dump_gpu(self, &dir.join("gpu"))?;
        Ok(())
    }
}
