use anyhow::anyhow;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

use hexpr::*;
use metacat::{check::check, syntax::TheoryBundle, theory::OperationKey};
use open_hypergraphs::lax::{OpenHypergraph, functor::Functor};

use catena::lang::{Arr, Obj};
use catena::pass::{erase::Erase, expand_eta::ExpandEta, forget_bound::ForgetBound};

#[derive(Parser)]
#[command(name = "catena", version=env!("CARGO_PKG_VERSION"))]
#[command(about = "catena compiler")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Pass {
    Check,
    Erase,
    ForgetBound,
    ExpandEta,
}

/// Passes after Check, in order. Each entry is (pass name, functor).
const LOWER_PASSES: &[(
    Pass,
    fn(&OpenHypergraph<Obj, Arr>) -> OpenHypergraph<Obj, Arr>,
)] = &[
    (Pass::Erase, |t| Erase.map_arrow(t)),
    (Pass::ForgetBound, |t| ForgetBound.map_arrow(t)),
    (Pass::ExpandEta, |t| ExpandEta.map_arrow(t)),
];

#[derive(Subcommand)]
enum Command {
    /// Run compiler passes up to the given pass and output SVG
    Lower {
        #[arg()]
        path: PathBuf,
        #[arg()]
        pass: Pass,
        #[arg()]
        definition: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Lower {
            path,
            pass,
            definition,
        } => lower(path, pass, &definition),
    }
}

fn lower(path: PathBuf, until: Pass, definition: &str) -> anyhow::Result<()> {
    let TheoryBundle {
        arrow_theory,
        object_theory,
        definitions,
    } = TheoryBundle::from_file(path)?;

    let declaration = definitions
        .get(&definition.parse()?)
        .ok_or_else(|| anyhow!("no such definition: {definition}"))?;

    let definition_hexpr = declaration
        .definition
        .clone()
        .ok_or(anyhow!("not a definition: {definition}"))?;

    let mut term = forget_labels(try_interpret(&arrow_theory, &definition_hexpr)?);
    let source = forget_labels(try_interpret(&object_theory, &declaration.source_map)?);
    let target = forget_labels(try_interpret(&object_theory, &declaration.target_map)?);

    // Check (always runs first)
    let result = check(&arrow_theory, source, target, &mut term)
        .map_err(|e| anyhow!("typechecking failed: {e:?}"))?;

    let mut current = term.with_nodes(|_| result).unwrap();

    // Run subsequent passes in order, stopping after the requested one
    if until != Pass::Check {
        for &(pass, apply) in LOWER_PASSES {
            current = apply(&current);
            current.quotient();
            if pass == until {
                break;
            }
        }
    }

    // Pretty-print
    let coarity = |op: &OperationKey| -> usize { object_theory.type_maps(op).1.targets.len() };

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

fn forget_labels<O, A>(f: OpenHypergraph<O, A>) -> OpenHypergraph<(), A> {
    f.map_nodes(|_| ())
}
