use std::{fs, io, path::Path};

use metacat::theory::RawTheorySet;

pub struct CompileReport {
    pub raw_theories: RawTheorySet,
    pub elaborated: RawTheorySet,
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
