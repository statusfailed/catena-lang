use hexpr::Operation;
use metacat::{
    theory::{RawTheorySet, Theory, TheoryId, TheorySet},
    tree::Tree,
};
use open_hypergraphs::lax::NodeId;

use crate::{
    check::{AnnotatedTerm, DefinitionTypes, check},
    closure::{
        body::closure_body,
        convert::convert,
        extract::extract_region,
        region::{Obj, closure_region},
    },
    elaborate::elaborate,
    stdlib,
};

#[test]
fn identity_wire_has_no_closure_work_to_do() {
    let definition = annotated_program_definition(
        r#"
        (def program id : [a] -> [a] = [x])
        "#,
        "id",
    );

    let closure_wires = closure_wires(&definition);
    let regions =
        closure_region(&definition, &closure_wires).expect("region discovery should succeed");
    let converted = convert(&op("id"), &definition).expect("conversion should succeed");

    // No closure regions appear, none converted
    assert_eq!(regions.len(), 0);
    assert_eq!(converted.closures.len(), 0);

    // input definition is (approx.) the same
    assert_same_definition_interface(&converted.definition, &definition);
}

#[test]
fn deferred_bool_id_closure_converts_through_each_stage() {
    let definition = annotated_program_definition(
        r#"
        (def program run-bool-id : (bool val) -> ({1 (bool val)} =>) = (
          {[x] bool.t}
          bool.and
          bool.not
          {defer (name.bool.id lift)}
          compose
        ))
        "#,
        "run-bool-id",
    );
    let original_target = definition.targets[0];

    // Find the (single) closure region at the output node
    let [region] = closure_region(&definition, &[original_target])
        .expect("region discovery should succeed")
        .try_into()
        .expect("expected one closure region");
    assert_eq!(region.edges.len(), 4);

    // Extract the region into its own AnnotatedTerm, check it has expected sources (deferred data)
    // and targets (the closure type)
    let extracted = extract_region(&definition, &region).expect("region extraction should succeed");
    assert_eq!(extracted.hypergraph.edges.len(), 4);
    assert_eq!(
        interface_types(&extracted, &extracted.sources),
        interface_types(&definition, &region.defer_inputs)
    );
    assert_eq!(
        interface_types(&extracted, &extracted.targets),
        vec![region.closure_type.clone()]
    );

    // Transform the closure into an arrow of type (X ● A -> B)
    // where here: X = (val bool), A = 1, B = val bool.
    let body = closure_body(&extracted).expect("closure body construction should succeed");
    assert_eq!(
        body.hypergraph.edges.len(),
        extracted.hypergraph.edges.len() + 3
    );
    assert_eq!(
        interface_types(&body, &body.sources),
        vec![obj("val", vec![obj("bool", vec![])]), obj("1", vec![])]
    );
    assert_eq!(
        interface_types(&body, &body.targets),
        vec![obj("val", vec![obj("bool", vec![])])]
    );

    // Run the 'convert' method
    let definition_name = op("run-bool-id");
    let converted = convert(&definition_name, &definition).expect("conversion should succeed");

    // There should be one converted closure
    assert_eq!(converted.closures.len(), 1);

    // Check the closure type is (A => B)
    let converted_closure = &converted.closures[0];
    assert_eq!(converted_closure.node, original_target);

    // Check name is expected (closure.original-def.node_id)
    assert_eq!(
        converted_closure.name(&definition_name),
        op(&format!("closure.run-bool-id.{}", original_target.0))
    );

    // Check equivalent to body
    assert_eq!(
        converted_closure.term.hypergraph.edges.len(),
        body.hypergraph.edges.len()
    );

    // Check X = val(bool), A = 1, B = val(bool)
    assert_eq!(
        converted_closure.name_info.environment,
        vec![obj("val", vec![obj("bool", vec![])])]
    );
    assert_eq!(converted_closure.name_info.domain, obj("1", vec![]));
    assert_eq!(
        converted_closure.name_info.codomain,
        obj("val", vec![obj("bool", vec![])])
    );

    // Check converted definition has type
    //         X ● (X * A -> B)
    // val(bool) ● (val(bool) * 1 -> val(bool))
    assert_converted_definition(
        &converted.definition,
        4,
        vec![obj("val", vec![obj("bool", vec![])])],
        vec![
            obj("val", vec![obj("bool", vec![])]),
            function_pointer_type(
                vec![obj("val", vec![obj("bool", vec![])]), obj("1", vec![])],
                vec![obj("val", vec![obj("bool", vec![])])],
            ),
        ],
    );

    // Verify that original definition uses the *name* of the closure conversion
    assert!(
        converted
            .definition
            .hypergraph
            .edges
            .iter()
            .any(|operation| operation.as_str()
                == format!("name.closure.run-bool-id.{}", original_target.0))
    );
}

fn theories_with(source: &'static str) -> (TheorySet, DefinitionTypes) {
    let raw_theories = RawTheorySet::from_texts(stdlib::sources().chain([source]))
        .expect("test theories should parse");
    let elaborated = elaborate(raw_theories).expect("test theories should elaborate");
    let theory_set = TheorySet::from_raw(elaborated).expect("test theories should load");
    let definition_types = check(&theory_set).expect("test theories should typecheck");
    (theory_set, definition_types)
}

fn annotated_program_definition(source: &'static str, definition: &str) -> AnnotatedTerm {
    let (theory_set, definition_types) = theories_with(source);
    let program = TheoryId("program".parse().expect("program theory id should parse"));
    let definition: Operation = definition
        .parse()
        .expect("program definition name should parse");
    let theory = theory_set
        .theories
        .get(&program)
        .expect("program theory should exist");
    let Theory::Theory { arrows, .. } = theory else {
        panic!("program should be a theory");
    };
    let arrow = arrows
        .get(&definition)
        .expect("program definition should exist");
    let mut body = arrow
        .definition
        .clone()
        .expect("program arrow should be a definition");
    body.quotient().ok();
    let labels = definition_types
        .get(&program)
        .and_then(|definitions| definitions.get(&definition))
        .cloned()
        .expect("program definition should have checked node types");
    body.with_nodes(|_| labels)
        .expect("checked node labels should match definition graph")
}

fn closure_wires(definition: &AnnotatedTerm) -> Vec<NodeId> {
    definition
        .hypergraph
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, object)| is_closure_type(object).then_some(NodeId(index)))
        .collect()
}

fn is_closure_type(object: &Obj) -> bool {
    let Tree::Node(operation, _, children) = object else {
        return false;
    };
    operation.as_str() == "=>" && children.len() == 2
}

fn interface_types(term: &AnnotatedTerm, interface: &[NodeId]) -> Vec<Obj> {
    interface
        .iter()
        .map(|node| term.hypergraph.nodes[node.0].clone())
        .collect()
}

fn assert_same_definition_interface(actual: &AnnotatedTerm, expected: &AnnotatedTerm) {
    assert_converted_definition(
        actual,
        expected.hypergraph.edges.len(),
        interface_types(expected, &expected.sources),
        interface_types(expected, &expected.targets),
    );
}

fn assert_converted_definition(
    definition: &AnnotatedTerm,
    edges: usize,
    source_types: Vec<Obj>,
    target_types: Vec<Obj>,
) {
    assert_eq!(definition.hypergraph.edges.len(), edges);
    assert_eq!(
        interface_types(definition, &definition.sources),
        source_types
    );
    assert_eq!(
        interface_types(definition, &definition.targets),
        target_types
    );
}

fn obj(name: &str, children: Vec<Obj>) -> Obj {
    Tree::Node(op(name), 0, children)
}

fn function_pointer_type(sources: Vec<Obj>, targets: Vec<Obj>) -> Obj {
    obj(
        "val",
        vec![obj("->", vec![pack_object(sources), pack_object(targets)])],
    )
}

fn pack_object(objects: Vec<Obj>) -> Obj {
    match objects.as_slice() {
        [] => obj("1", vec![]),
        [only] => only.clone(),
        _ => obj("*", objects),
    }
}

fn op(name: &str) -> Operation {
    name.parse().expect("test operation should parse")
}
