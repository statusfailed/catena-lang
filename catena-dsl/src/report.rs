use std::{fs, io, path::Path};

use hexpr::Operation;
use metacat::{theory::{RawTheorySet, TheoryId, TheorySet}, tree::Tree};
use std::collections::BTreeMap;

pub struct CompileReport {
    pub raw_theories: RawTheorySet,
    pub elaborated: RawTheorySet,
    pub theory_set: TheorySet,
    pub definition_types: BTreeMap<TheoryId, BTreeMap<Operation, Vec<Tree<(), Operation>>>>,
}

impl CompileReport {
    pub fn dump_to_dir(&self, dir: impl AsRef<Path>) -> io::Result<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        fs::write(dir.join("raw_theories.hex"), self.raw_theories.to_hexpr_text())?;
        fs::write(dir.join("elaborated.hex"), self.elaborated.to_hexpr_text())?;
        Ok(())
    }
}
