//! Mutually interleave the two "control" and "data" theories by adding "product-packed" data
//! arrows to control, and "coproduct-packed" control arrows to data.
//!
//! As a concrete example, the fragment of `stdlib.hex` would be elaborated with the "arr"
//! declarations commented.
//!
//! ```text
//! (theory syntax nat {
//!   # product/unit, coproduct/counit
//!   (arr * : 2 -> 1)
//!   (arr 1 : 0 -> 1)
//!   (arr + : 2 -> 1)
//!   (arr 0 : 0 -> 1)
//!
//!   # Builtin dtype definitions
//!   (def f32 : 0 -> 1 = (32 bitvec))
//! })
//!
//! # define the "data theory" (in this example, we just have one dataflow map)
//! (theory data syntax {
//!   (arr f32.add : {f32 f32} -> f32)
//! })
//!
//! # define the control theory - we'll assume no arrows added by the user in this minimal example
//! (theory control syntax {
//!   # elaboration adds a generating arrow to control: data.f32.add.
//!   # By convention, the compiler knows to interpret this as the data theory's "f32.add" arrow.
//!   (arr data.f32.add : ({f32 f32} *) -> f32)
//! })
//!
//! # conversely, if control had a map with multiple outputs, elaboration would add a
//! # coproduct-packed version to data.
//! (theory control syntax {
//!   (arr branch : flag -> {f32 f32})
//! })
//!
//! (theory data syntax {
//!   # elaboration adds a generating arrow to data: control.branch.
//!   # By convention, the compiler knows to interpret this as the control theory's "branch" arrow.
//!   (arr control.branch : flag -> ({f32 f32} +))
//! })
//! ```
use hexpr::{Hexpr, Operation};
use metacat::theory::{
    RawTheorySet, Theory,
    ast::{RawTheory, RawTheoryArrow},
};
use open_hypergraphs::{category::Arrow, lax::OpenHypergraph};

use crate::render_hexpr::open_hypergraph_to_hexpr;

/// Interleave "control" maps into "data" and vice-versa.
// Sketch:
// - Interpret the syntax theory (so we can get type map coarities)
// - Compute declarations to add for "control-in-data"
// - Compute declarations to add for "data-in-control"
// - Add both sets to their corresponding theories
// We compute the arrows to add before adding so that *synthesized* arrows are not copied.
pub fn interleave(syntax: &Theory, raw: &mut RawTheorySet) {
    let control_name: Operation = "control".parse().expect("valid operation");
    let data_name: Operation = "data".parse().expect("valid operation");

    let Some(control) = raw.theories.get(&control_name).cloned() else {
        return;
    };
    let Some(data) = raw.theories.get(&data_name).cloned() else {
        return;
    };

    let control_in_data = tensor_pack_embed(
        syntax,
        &control,
        &data,
        "+".parse().expect("valid operation"),
        "0".parse().expect("valid operation"),
    );
    let data_in_control = tensor_pack_embed(
        syntax,
        &data,
        &control,
        "*".parse().expect("valid operation"),
        "1".parse().expect("valid operation"),
    );

    if let Some(data_theory) = raw.theories.get_mut(&data_name) {
        for arrow in control_in_data {
            data_theory.arrows.insert(arrow.name.clone(), arrow);
        }
    }
    if let Some(control_theory) = raw.theories.get_mut(&control_name) {
        for arrow in data_in_control {
            control_theory.arrows.insert(arrow.name.clone(), arrow);
        }
    }
}

/// Let C and D be symmetric monoidal categories over the same syntax category S.
/// This function creates a generating operation in D for each generating operation in C with
/// arity/coarity `1 -> 1`, whose type is the left-biased "tensor-packing" defined in `pack`.
///
/// Procedure: for each arrow (both declared arr and definitions def) in C,
///
///  - Compute a "packed" version of the type, so e.g., a type map like {f32 f32} becomes `({f32 f32} *)`
///  - Create an arr declaration (as syntax) with the packed type maps in dataflow
///
/// Returns a list of `TheoryArrow` declarations (not definitions) to add to D.
fn tensor_pack_embed(
    syntax: &Theory,
    source: &RawTheory,
    target: &RawTheory,
    tensor: Operation,
    unit: Operation,
) -> Vec<RawTheoryArrow> {
    source
        .arrows
        .values()
        .filter_map(|arrow| {
            let lifted_name: Operation = format!("{}.{}", source.name, arrow.name)
                .parse()
                .expect("lifted operation name should parse");
            if target.arrows.contains_key(&lifted_name) {
                return None;
            }

            Some(RawTheoryArrow {
                name: lifted_name,
                type_maps: (
                    pack_type_map(&arrow.type_maps.0, syntax, &tensor, &unit),
                    pack_type_map(&arrow.type_maps.1, syntax, &tensor, &unit),
                ),
                definition: None,
            })
        })
        .collect()
}

/// Let `A = A₀ × A₁ × ... × Am` be an object (list of generating objects) in a chosen syntax category.
/// Then `pack(A, ●, I)` computes a hexpr with type `A₀ × A₁ × ... × Am -> A₀ ● A₁ ● ... ● Am`,
/// where `●` is a chosen binary tensor product, and `I` its unit.
/// concretely, we have cases:
///
/// - `I` when `m = 0`
/// - `A` when `m = 1`
/// - `head(A) ● pack(tail(A))` when `m ≥ 2`
fn pack(object_size: usize, tensor: Operation, unit: Operation) -> Hexpr {
    let mut packed = OpenHypergraph::identity(vec![(); object_size]);
    match object_size {
        0 => {
            let unit_node = packed.new_node(());
            packed.new_edge(unit, (Vec::new(), vec![unit_node]));
            packed.targets = vec![unit_node];
        }
        1 => {}
        _ => {
            let mut inputs = packed.targets.clone();
            while inputs.len() > 1 {
                let left = inputs.remove(0);
                let right = inputs.remove(0);
                let packed_node = packed.new_node(());
                packed.new_edge(tensor.clone(), (vec![left, right], vec![packed_node]));
                inputs.insert(0, packed_node);
            }
            packed.targets = inputs;
        }
    }

    open_hypergraph_to_hexpr(packed)
}

// Have: interpreted and raw type maps for a given arrow
// Want:

fn pack_type_map(map: &Hexpr, syntax: &Theory, tensor: &Operation, unit: &Operation) -> Hexpr {
    let interpreted = hexpr::try_interpret(&syntax.local_signature(), map)
        .expect("type map should interpret in the resolved syntax theory");
    match interpreted.target().len() {
        1 => map.clone(),
        n => Hexpr::Composition(vec![map.clone(), pack(n, tensor.clone(), unit.clone())]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metacat::{
        check::check,
        theory::{Theory, TheoryId, TheorySet},
    };

    #[test]
    fn interleaved_arrows_typecheck_in_both_directions() {
        let source = r#"
        (theory syntax nat {
          (arr flag : 0 -> 1)
          (arr f32 : 0 -> 1)
          (arr * : 2 -> 1)
          (arr 1 : 0 -> 1)
          (arr + : 2 -> 1)
          (arr 0 : 0 -> 1)
        })

        (theory data syntax {
          (arr f32.add : {f32 f32} -> f32)
        })

        (theory control syntax {
          (arr branch : flag -> {f32 f32})
          (def id : f32 -> f32 = [x0])
        })
        "#;

        let scaffold = TheorySet::from_text(source).unwrap();
        let syntax = scaffold
            .theories
            .get(&TheoryId("syntax".parse().unwrap()))
            .unwrap();
        let mut raw = RawTheorySet::from_text(source).unwrap();

        interleave(syntax, &mut raw);

        let control = raw.theories.get_mut(&op("control")).unwrap();
        let data_add = control.arrows.get(&op("data.f32.add")).unwrap().clone();
        control.arrows.insert(
            op("use_data_add"),
            RawTheoryArrow {
                name: op("use_data_add"),
                type_maps: data_add.type_maps.clone(),
                definition: Some(Hexpr::Operation(op("data.f32.add"))),
            },
        );

        let data = raw.theories.get_mut(&op("data")).unwrap();
        let branch = data.arrows.get(&op("control.branch")).unwrap().clone();
        let control_id = data.arrows.get(&op("control.id")).unwrap().clone();
        data.arrows.insert(
            op("use_branch"),
            RawTheoryArrow {
                name: op("use_branch"),
                type_maps: branch.type_maps.clone(),
                definition: Some(Hexpr::Operation(op("control.branch"))),
            },
        );
        data.arrows.insert(
            op("use_control_id"),
            RawTheoryArrow {
                name: op("use_control_id"),
                type_maps: control_id.type_maps.clone(),
                definition: Some(Hexpr::Operation(op("control.id"))),
            },
        );

        let loaded = TheorySet::from_text(&render_raw_theory_set(&raw)).unwrap();
        assert_definition_typechecks(&loaded, "control", "use_data_add");
        assert_definition_typechecks(&loaded, "data", "use_branch");
        assert_definition_typechecks(&loaded, "data", "use_control_id");
    }

    fn op(name: &str) -> Operation {
        name.parse().unwrap()
    }

    fn render_raw_theory_set(raw: &RawTheorySet) -> String {
        raw.theories
            .values()
            .map(render_raw_theory)
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    fn render_raw_theory(theory: &RawTheory) -> String {
        let declarations = theory
            .arrows
            .values()
            .map(render_raw_arrow)
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "(theory {} {} {{\n{}\n}})",
            theory.name, theory.syntax_category, declarations
        )
    }

    fn render_raw_arrow(arrow: &RawTheoryArrow) -> String {
        match &arrow.definition {
            Some(definition) => format!(
                "  (def {} : {} -> {} = {})",
                arrow.name, arrow.type_maps.0, arrow.type_maps.1, definition
            ),
            None => format!(
                "  (arr {} : {} -> {})",
                arrow.name, arrow.type_maps.0, arrow.type_maps.1
            ),
        }
    }

    fn assert_definition_typechecks(theories: &TheorySet, theory_name: &str, arrow_name: &str) {
        let theory_id = TheoryId(op(theory_name));
        let theory = theories.theories.get(&theory_id).unwrap();
        let Theory::Theory { arrows, .. } = theory else {
            panic!("expected user theory");
        };
        let arrow = arrows.get(&op(arrow_name)).unwrap();
        let mut body = arrow.definition.clone().expect("expected definition");
        check(theory, arrow.type_maps.0.clone(), arrow.type_maps.1.clone(), &mut body).unwrap();
    }
}
