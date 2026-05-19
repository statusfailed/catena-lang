mod name_symbols;

use metacat::theory::{GraphError, RawTheorySet, ast::ExtensionsError};
use thiserror::Error;

const NAT_THEORY: &str = "nat";

#[derive(Debug, Error)]
pub enum ElaborateError {
    #[error(transparent)]
    Extensions(#[from] ExtensionsError),
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Load(#[from] metacat::theory::LoadError),
}

pub fn elaborate(mut raw: RawTheorySet) -> Result<RawTheorySet, ElaborateError> {
    raw = raw.with_extensions()?;

    let theory_names: Vec<_> = raw
        .theories
        .iter()
        .filter(|(_, theory)| theory.syntax_category.as_str() != NAT_THEORY)
        .map(|(name, _)| name.clone())
        .collect();

    for theory_name in theory_names {
        name_symbols::elaborate_theory(&mut raw, &theory_name)?;
    }

    Ok(raw)
}
