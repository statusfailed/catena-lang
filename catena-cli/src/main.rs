use anyhow::anyhow;
use clap::{Parser, Subcommand, ValueEnum};
use std::collections::HashMap;
use std::path::PathBuf;

use hexpr::*;
use metacat::{check::check, syntax::TheoryBundle, theory::OperationKey};
use open_hypergraphs::lax::{functor::Functor, OpenHypergraph};

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
    _bundle: &TheoryBundle,
) -> anyhow::Result<
    Vec<(
        Pass,
        Box<dyn Fn(&OpenHypergraph<Obj, Arr>) -> OpenHypergraph<Obj, Arr>>,
    )>,
> {
    Ok(vec![
        (Pass::Erase, Box::new(|t| Erase.map_arrow(t))),
        (Pass::ForgetBound, Box::new(|t| ForgetBound.map_arrow(t))),
        (Pass::ExpandEta, Box::new(|t| ExpandEta.map_arrow(t))),
        (
            Pass::DiscardNaturality,
            Box::new(|t| discard_naturality(t.clone()).expect("discard_naturality failed")),
        ),
    ])
}

fn inline(
    bundle: &TheoryBundle,
    t: &mut OpenHypergraph<(), Arr>,
) -> anyhow::Result<OpenHypergraph<(), Arr>> {
    let inline = {
        let names = ["f32.sum", "ones-2d", "id-matrix-2d"];
        let mut inline_defs = HashMap::new();
        for name in names {
            let op: Operation = name.parse()?;
            //let hexpr = bundle
            //.definitions
            //.get(&op)
            //.and_then(|v| v.definition.clone())
            //.ok_or_else(|| anyhow!("no such definition: {name}"))?;
            let arrow = declaration_term(bundle, &op)?;
            inline_defs.insert(OperationKey(op), arrow);
        }
        Inline {
            definitions: inline_defs,
        }
    };
    t.quotient().unwrap();
    Ok(inline.map_arrow(t))
}

/// Lower a term by applying passes until the specified pass
/// TODO: add a post-processing hook on `lower` to transform any pass into readable strings - used
/// for lower command -> svg
fn lower(
    bundle: &TheoryBundle,
    until: Pass,
    definition: &str,
) -> anyhow::Result<OpenHypergraph<metacat::tree::Tree<(), OperationKey>, OperationKey>> {
    let key: Operation = definition.parse()?;
    let declaration = bundle
        .definitions
        .get(&key)
        .ok_or_else(|| anyhow!("no such definition: {definition}"))?;

    // Get term from declaration & key
    // NOTE: we *must* inline before typechecking: we need annotated nodes to be specialised to the
    // types applied to each definition.
    let mut current = declaration_term(bundle, &key)?;
    let current = inline(bundle, &mut current)?;

    // Check inlined
    let mut current = compute_types(bundle, declaration, current)?;

    // Run subsequent passes in order, stopping after the requested one
    if until != Pass::Check {
        for (pass, apply) in lower_passes(bundle)? {
            current = apply(&current);
            current.quotient_witness().unwrap();
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

    use open_hypergraphs_dot::{svg::to_svg_with, Options};
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

fn declaration_term(
    bundle: &TheoryBundle,
    key: &Operation,
) -> anyhow::Result<OpenHypergraph<(), Arr>> {
    let hexpr = bundle
        .definitions
        .get(key)
        .and_then(|declaration| declaration.definition.clone())
        .ok_or_else(|| anyhow!("no such definition: {key}"))?;

    Ok(forget_labels(try_interpret(&bundle.arrow_theory, &hexpr)?))
}

fn compute_types(
    bundle: &TheoryBundle,
    declaration: &metacat::syntax::Declaration,
    term: OpenHypergraph<(), Arr>,
) -> anyhow::Result<OpenHypergraph<Obj, Arr>> {
    let mut term = term;
    let source = forget_labels(try_interpret(
        &bundle.object_theory,
        &declaration.source_map,
    )?);
    let target = forget_labels(try_interpret(
        &bundle.object_theory,
        &declaration.target_map,
    )?);
    let result = check(&bundle.arrow_theory, source, target, &mut term)
        .map_err(|e| anyhow!("typechecking failed: {e:?}"))?;
    Ok(term.with_nodes(|_| result).unwrap())
}

fn forget_labels<O, A>(f: OpenHypergraph<O, A>) -> OpenHypergraph<(), A> {
    f.map_nodes(|_| ())
}
