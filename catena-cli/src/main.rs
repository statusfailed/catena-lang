use catena::lower::{Pass, lower};

use colored::*;
use clap::{Parser, Subcommand, ValueEnum};
use open_hypergraphs::lax::OpenHypergraph;
use std::path::PathBuf;

use catena::backend::c::codegen::codegen;
use hexpr::try_interpret;
use metacat::{check::check, syntax::TheoryBundle, theory::OperationKey};

#[derive(Parser)]
#[command(name = "catena", version=env!("CARGO_PKG_VERSION"))]
#[command(about = "catena compiler")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Check all definitions in a source file
    Check {
        #[arg()]
        path: PathBuf,
    },

    /// Run codegen for a given pass
    Codegen {
        #[arg()]
        path: PathBuf,
        #[arg()]
        definition: String,
    },

    /// Run compiler passes up to the given pass and output SVG
    Lower {
        #[arg()]
        pass: PassArg,
        #[arg()]
        path: PathBuf,
        #[arg()]
        definition: String,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PassArg {
    Check,
    Erase,
    ForgetBound,
    ExpandEta,
    DiscardNaturality,
}

impl From<PassArg> for Pass {
    fn from(value: PassArg) -> Self {
        match value {
            PassArg::Check => Pass::Check,
            PassArg::Erase => Pass::Erase,
            PassArg::ForgetBound => Pass::ForgetBound,
            PassArg::ExpandEta => Pass::ExpandEta,
            PassArg::DiscardNaturality => Pass::DiscardNaturality,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Check { path } => check_file(path),
        Command::Codegen { path, definition } => {
            let bundle = TheoryBundle::from_file(path)?;
            let lowered = lower(&bundle, Pass::DiscardNaturality, &definition)?;
            println!("{}", codegen(lowered, "out"));
            Ok(())
        }
        Command::Lower {
            path,
            pass,
            definition,
        } => lower_command(TheoryBundle::from_file(path)?, pass.into(), &definition),
    }
}

fn check_file(path: PathBuf) -> anyhow::Result<()> {
    let TheoryBundle {
        object_theory,
        arrow_theory,
        definitions,
        ..
    } = TheoryBundle::from_file(path)?;

    for declaration in definitions.values() {
        let def_hexpr = declaration.definition.as_ref().unwrap();

        let mut term = forget_labels(try_interpret(&arrow_theory, def_hexpr)?);
        let source = forget_labels(try_interpret(&object_theory, &declaration.source_map)?);
        let target = forget_labels(try_interpret(&object_theory, &declaration.target_map)?);

        match check(&arrow_theory, source, target, &mut term) {
            Ok(_) => {
                println!(
                    "{} {} : {} -> {}",
                    "[✓]".green(),
                    declaration.name,
                    declaration.source_map,
                    declaration.target_map
                );
            }
            Err(e) => {
                println!(
                    "{} {} : {} -> {}",
                    "[✗]".red(),
                    declaration.name,
                    declaration.source_map,
                    declaration.target_map
                );
                println!("Checking '{}' failed: {}", declaration.name, e);
            }
        }
    }

    Ok(())
}

fn forget_labels<O, A>(f: OpenHypergraph<O, A>) -> OpenHypergraph<(), A> {
    f.map_nodes(|_| ())
}

fn lower_command(bundle: TheoryBundle, until: Pass, definition: &str) -> anyhow::Result<()> {
    let current = lower(&bundle, until, definition)?;

    // Pretty-print
    let coarity =
        |op: &OperationKey| -> usize { bundle.object_theory.type_maps(op).1.targets.len() };

    let labels: Vec<String> = current
        .hypergraph
        .nodes
        .iter()
        .map(|n| n.pretty(Some(&coarity)))
        .collect();

    use open_hypergraphs_dot::{Options, svg::to_svg_with};
    use std::io::Write;

    let opts = Options::default().display();
    std::io::stdout().write_all(&to_svg_with(
        &current
            .with_nodes(|_| labels)
            .expect("labels length mismatch"),
        &opts,
    )?)?;

    Ok(())
}
