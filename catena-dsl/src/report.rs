use metacat::theory::{RawTheorySet, TheorySet};

pub struct CompileReport {
    pub raw_theories: RawTheorySet,
    pub elaborated: TheorySet,
}
