use std::{fs, io, path::Path};

use hexpr::Operation;
use metacat::theory::{Theory, TheoryId};
use open_hypergraphs_dot::{Options, svg::to_svg_with};

use crate::report::CompileReport;

pub fn dump_svgs(report: &CompileReport, dir: &Path) -> io::Result<()> {
    fs::create_dir_all(dir)?;

    for (theory_id, definition_types) in &report.definition_types {
        let theory = report
            .theory_set
            .theories
            .get(theory_id)
            .ok_or_else(|| invalid_data(format!("missing theory `{theory_id}` in TheorySet")))?;
        let Theory::Theory { syntax, arrows } = theory else {
            continue;
        };
        let syntax_theory = report
            .theory_set
            .theories
            .get(syntax)
            .ok_or_else(|| invalid_data(format!("missing syntax theory `{syntax}`")))?;

        for (definition_name, node_types) in definition_types {
            let arrow = arrows.get(definition_name).ok_or_else(|| {
                invalid_data(format!(
                    "missing definition `{definition_name}` in theory `{theory_id}`"
                ))
            })?;
            let mut term = arrow.definition.clone().ok_or_else(|| {
                invalid_data(format!(
                    "definition `{definition_name}` in theory `{theory_id}` has no body"
                ))
            })?;
            term.quotient()
                .map_err(|error| invalid_data(format!("failed to quotient `{definition_name}`: {error:?}`")))?;

            let labels: Vec<String> = node_types
                .iter()
                .map(|ty| {
                    ty.try_pretty(Some(&|op: &Operation| {
                        syntax_theory.coarity_of(op).ok_or_else(|| {
                            invalid_data(format!("coarity lookup failed for operation `{op}`"))
                        })
                    }))
                })
                .collect::<Result<_, _>>()?;

            let labelled = term
                .with_nodes(|_| labels)
                .ok_or_else(|| invalid_data("labels length mismatch".to_string()))?;
            let svg = to_svg_with(&labelled, &Options::default().display().lr())?;

            let definition_dir = dir.join(qualified_definition_dir(theory_id, definition_name));
            fs::create_dir_all(&definition_dir)?;
            fs::write(definition_dir.join("checked.svg"), svg)?;
        }
    }

    Ok(())
}

fn qualified_definition_dir(theory_id: &TheoryId, definition_name: &Operation) -> String {
    format!("{theory_id}.{definition_name}")
}

fn invalid_data(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
