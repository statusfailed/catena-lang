use anyhow::anyhow;
use clap::{Parser, Subcommand, ValueEnum};
use std::collections::HashMap;
use std::path::PathBuf;

use hexpr::*;
use metacat::{check::check, prop::Nat, syntax::TheoryBundle, theory::OperationKey};
use open_hypergraphs::lax::{OpenHypergraph, functor::Functor};

use catena::codegen::c::codegen;
use catena::lang::{Arr, Obj};
use catena::pass::{
    discard_naturality::discard_naturality, erase::Erase, expand_eta::ExpandEta,
    forget_bound::ForgetBound, inline::Inline,
};

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
    Inline,
    Erase,
    ForgetBound,
    ExpandEta,
    DiscardNaturality,
}

#[derive(Subcommand)]
enum Command {
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
        pass: Pass,
        #[arg()]
        path: PathBuf,
        #[arg()]
        definition: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
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
        } => lower_command(TheoryBundle::from_file(path)?, pass, &definition),
    }
}

/// Construct the compiler lowering passes
fn lower_passes(
    arrow_theory: &metacat::theory::Theory<OperationKey>,
    object_theory: &metacat::theory::Theory<Nat>,
    definitions: &HashMap<Operation, metacat::syntax::Declaration>,
) -> anyhow::Result<
    Vec<(
        Pass,
        Box<dyn Fn(&OpenHypergraph<Obj, Arr>) -> OpenHypergraph<Obj, Arr>>,
    )>,
> {
    let inline = {
        let name = "f32.sum";
        let op: Operation = name.parse()?;
        let decl = definitions
            .get(&op)
            .ok_or_else(|| anyhow!("no such definition: {name}"))?;
        let arrow = interpret_definition(arrow_theory, object_theory, decl)?;
        Inline {
            definitions: HashMap::from([(OperationKey(op), arrow)]),
        }
    };

    Ok(vec![
        (Pass::Inline, Box::new(move |t| inline.map_arrow(t))),
        (Pass::Erase, Box::new(|t| Erase.map_arrow(t))),
        (Pass::ForgetBound, Box::new(|t| ForgetBound.map_arrow(t))),
        (Pass::ExpandEta, Box::new(|t| ExpandEta.map_arrow(t))),
        (
            Pass::DiscardNaturality,
            Box::new(|t| discard_naturality(t.clone()).expect("discard_naturality failed")),
        ),
    ])
}

/// Lower a term by applying passes until the specified pass
fn lower(
    bundle: &TheoryBundle,
    until: Pass,
    definition: &str,
) -> anyhow::Result<OpenHypergraph<metacat::tree::Tree<(), OperationKey>, OperationKey>> {
    let TheoryBundle {
        arrow_theory,
        object_theory,
        definitions,
    } = bundle;

    let declaration = definitions
        .get(&definition.parse()?)
        .ok_or_else(|| anyhow!("no such definition: {definition}"))?;

    // Check (always runs first)
    let mut current = interpret_definition(&arrow_theory, &object_theory, declaration)?;

    // Run subsequent passes in order, stopping after the requested one
    if until != Pass::Check {
        for (pass, apply) in lower_passes(&arrow_theory, &object_theory, &definitions)? {
            current = apply(&current);
            current.quotient();
            if pass == until {
                break;
            }
        }
    }

    Ok(current)
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

fn interpret_definition(
    arrow_theory: &metacat::theory::Theory<OperationKey>,
    object_theory: &metacat::theory::Theory<Nat>,
    declaration: &metacat::syntax::Declaration,
) -> anyhow::Result<OpenHypergraph<Obj, Arr>> {
    let definition_hexpr = declaration
        .definition
        .clone()
        .ok_or_else(|| anyhow!("not a definition"))?;
    let mut term = forget_labels(try_interpret(arrow_theory, &definition_hexpr)?);
    let source = forget_labels(try_interpret(object_theory, &declaration.source_map)?);
    let target = forget_labels(try_interpret(object_theory, &declaration.target_map)?);
    let result = check(arrow_theory, source, target, &mut term)
        .map_err(|e| anyhow!("typechecking failed: {e:?}"))?;
    Ok(term.with_nodes(|_| result).unwrap())
}

fn forget_labels<O, A>(f: OpenHypergraph<O, A>) -> OpenHypergraph<(), A> {
    f.map_nodes(|_| ())
}
