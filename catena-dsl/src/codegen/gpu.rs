use std::collections::{BTreeMap, HashSet};

use catena::structured::{Primitive, StructuredProgram, ir::Stmt};
use hexpr::Operation;
use metacat::tree::Tree;
use open_hypergraphs::lax::NodeId;
use thiserror::Error;

use crate::{codegen::types, report::AnnotatedTerm};

#[derive(Debug, Error)]
pub enum GpuRenderError {
    #[error("missing node type for `{0}`")]
    MissingNodeType(String),
    #[error("type `{0:?}` is unsupported in GPU codegen")]
    UnsupportedType(metacat::tree::Tree<(), Operation>),
    #[error("unsupported structured statement in GPU renderer")]
    UnsupportedStmt,
    #[error("gpu.materialize expects one output in StructuredProgram")]
    InvalidMaterializeOutput,
    #[error("gpu.materialize is missing a launch parameter input in StructuredProgram")]
    MissingMaterializeLaunchParams,
    #[error("gpu.materialize is missing a kernel function input in StructuredProgram")]
    MissingMaterializeKernel,
}

pub fn render_program(
    program: &StructuredProgram,
    term: &AnnotatedTerm,
) -> Result<String, GpuRenderError> {
    let mut term = term.clone();
    term.quotient().ok();
    let node_types = node_type_map(&term);
    let source_nodes: HashSet<_> = term.sources.iter().map(|n| n.0).collect();
    let mut declared = source_nodes
        .iter()
        .map(|node| node_var(NodeId(*node)))
        .collect::<HashSet<_>>();

    let mut out = String::new();
    out.push_str(runtime_prelude());
    out.push('\n');
    for (index, stmt) in program.body.iter().enumerate() {
        if let Stmt::Primitive(primitive) = stmt
            && primitive.name == "gpu.materialize"
        {
            render_materialize_kernel(
                &mut out,
                &format!("{}_materialize_{index}", program.entry.name),
                primitive,
                &node_types,
            )?;
            out.push('\n');
        }
    }
    out.push_str(&format!("void {}(", program.entry.name));
    out.push_str(
        &program
            .entry
            .params
            .iter()
            .enumerate()
            .map(|(index, param)| {
                if index < term.sources.len() {
                    types::gpu_param_decl(
                        &term.hypergraph.nodes[term.sources[index].0],
                        &param.name,
                        false,
                    )
                } else {
                    let target_index = index - term.sources.len();
                    types::gpu_param_decl(
                        &term.hypergraph.nodes[term.targets[target_index].0],
                        &param.name,
                        true,
                    )
                }
                .ok_or_else(|| {
                    let ty = if index < term.sources.len() {
                        &term.hypergraph.nodes[term.sources[index].0]
                    } else {
                        let target_index = index - term.sources.len();
                        &term.hypergraph.nodes[term.targets[target_index].0]
                    };
                    GpuRenderError::UnsupportedType(ty.clone())
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .join(", "),
    );
    out.push_str(") {\n");

    for (index, stmt) in program.body.iter().enumerate() {
        render_stmt(
            &mut out,
            stmt,
            &node_types,
            &mut declared,
            &format!("{}_materialize_{index}", program.entry.name),
        )?;
    }

    out.push_str("}\n");
    Ok(out)
}

fn runtime_prelude() -> &'static str {
    r#"#include <hip/hip_runtime.h>
#include <stdint.h>

typedef uint8_t catena_unit_t;
typedef uint8_t catena_gpu_state_t;

typedef struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} catena_dim3_t;

typedef struct {
    uint64_t thread_id;
} catena_gpu_env_t;

typedef struct {
    catena_dim3_t grid_dim;
    catena_dim3_t block_dim;
} catena_launch_params_t;

typedef struct {
    void *data;
    uint64_t len;
} catena_gpu_buf_t;

static inline uint64_t catena_launch_len(catena_launch_params_t params) {
    return (uint64_t)params.grid_dim.x * params.grid_dim.y * params.grid_dim.z
        * params.block_dim.x * params.block_dim.y * params.block_dim.z;
}

static inline void bool_not(uint8_t arg0, uint8_t *out1) {
    *out1 = !arg0;
}

static inline void bool_or(uint8_t arg0, uint8_t arg1, uint8_t *out2) {
    *out2 = arg0 || arg1;
}

static inline void bool_and(uint8_t arg0, uint8_t arg1, uint8_t *out2) {
    *out2 = arg0 && arg1;
}

static inline void bool_id(uint8_t arg0, uint8_t *out1) {
    *out1 = arg0;
}

static inline void bool_copy(uint8_t arg0, uint8_t *out1, uint8_t *out2) {
    *out1 = arg0;
    *out2 = arg0;
}

static inline void bool_li(uint8_t arg0, uint8_t *out1) {
    *out1 = arg0;
}
"#
}

fn render_stmt(
    out: &mut String,
    stmt: &Stmt,
    node_types: &BTreeMap<String, metacat::tree::Tree<(), Operation>>,
    declared: &mut HashSet<String>,
    materialize_kernel_name: &str,
) -> Result<(), GpuRenderError> {
    match stmt {
        Stmt::Primitive(primitive) => render_primitive(
            out,
            primitive,
            node_types,
            declared,
            materialize_kernel_name,
        ),
        Stmt::Assign { lhs, rhs } => {
            out.push_str(&format!("    {lhs} = {rhs};\n"));
            Ok(())
        }
        Stmt::Return => {
            out.push_str("    return;\n");
            Ok(())
        }
        _ => Err(GpuRenderError::UnsupportedStmt),
    }
}

fn render_primitive(
    out: &mut String,
    primitive: &Primitive,
    node_types: &BTreeMap<String, metacat::tree::Tree<(), Operation>>,
    declared: &mut HashSet<String>,
    materialize_kernel_name: &str,
) -> Result<(), GpuRenderError> {
    for output in &primitive.outputs {
        if declared.insert(output.clone()) {
            let ty = node_types
                .get(output)
                .ok_or_else(|| GpuRenderError::MissingNodeType(output.clone()))?;
            let decl = types::gpu_local_decl(ty, output)
                .ok_or_else(|| GpuRenderError::UnsupportedType(ty.clone()))?;
            out.push_str(&format!("    {};\n", decl));
        }
    }

    match primitive.name.as_str() {
        "bool.t" => {
            let [output] = primitive.outputs.as_slice() else {
                return Err(GpuRenderError::UnsupportedStmt);
            };
            out.push_str(&format!("    {output} = 1;\n"));
        }
        "bool.f" => {
            let [output] = primitive.outputs.as_slice() else {
                return Err(GpuRenderError::UnsupportedStmt);
            };
            out.push_str(&format!("    {output} = 0;\n"));
        }
        "bool.not" => {
            let [input] = primitive.inputs.as_slice() else {
                return Err(GpuRenderError::UnsupportedStmt);
            };
            let [output] = primitive.outputs.as_slice() else {
                return Err(GpuRenderError::UnsupportedStmt);
            };
            out.push_str(&format!("    {output} = !{input};\n"));
        }
        "bool.and" => {
            let [lhs, rhs] = primitive.inputs.as_slice() else {
                return Err(GpuRenderError::UnsupportedStmt);
            };
            let [output] = primitive.outputs.as_slice() else {
                return Err(GpuRenderError::UnsupportedStmt);
            };
            out.push_str(&format!("    {output} = {lhs} && {rhs};\n"));
        }
        "bool.or" => {
            let [lhs, rhs] = primitive.inputs.as_slice() else {
                return Err(GpuRenderError::UnsupportedStmt);
            };
            let [output] = primitive.outputs.as_slice() else {
                return Err(GpuRenderError::UnsupportedStmt);
            };
            out.push_str(&format!("    {output} = {lhs} || {rhs};\n"));
        }
        "bool.ifc" => {
            let [env_true, fn_true, env_false, fn_false, flag, arg] = primitive.inputs.as_slice()
            else {
                return Err(GpuRenderError::UnsupportedStmt);
            };
            let [output] = primitive.outputs.as_slice() else {
                return Err(GpuRenderError::UnsupportedStmt);
            };
            out.push_str(&format!(
                "    if ({flag}) {{ {fn_true}({env_true}, {arg}, &{output}); }} else {{ {fn_false}({env_false}, {arg}, &{output}); }}\n"
            ));
        }
        "unit.intro" => {
            // The singleton object is erased from the wire-level representation.
        }
        "eval" => {
            let Some((func, args)) = primitive.inputs.split_last() else {
                return Err(GpuRenderError::UnsupportedStmt);
            };
            let mut call_args = args.to_vec();
            let mut output_ptrs = primitive
                .outputs
                .iter()
                .map(|output| format!("&{output}"))
                .collect::<Vec<_>>();
            call_args.append(&mut output_ptrs);
            out.push_str(&format!("    {func}({});\n", call_args.join(", ")));
        }
        "gpu.materialize" => {
            render_materialize_call(out, materialize_kernel_name, primitive, node_types)?;
        }
        _ if primitive.name.starts_with("name.") => {
            let [output] = primitive.outputs.as_slice() else {
                return Err(GpuRenderError::UnsupportedStmt);
            };
            let target = sanitize_ident(primitive.name.trim_start_matches("name."));
            out.push_str(&format!("    {output} = {target};\n"));
        }
        _ => {
            out.push_str(&format!("    /* TODO: lower `{}` */\n", primitive.name));
        }
    }

    Ok(())
}

fn render_materialize_kernel(
    out: &mut String,
    kernel_name: &str,
    primitive: &Primitive,
    node_types: &BTreeMap<String, metacat::tree::Tree<(), Operation>>,
) -> Result<(), GpuRenderError> {
    let [output] = primitive.outputs.as_slice() else {
        return Err(GpuRenderError::InvalidMaterializeOutput);
    };
    let output_type = node_types
        .get(output)
        .ok_or_else(|| GpuRenderError::MissingNodeType(output.clone()))?;
    let element_type = types::gpu_buffer_element_type(output_type)
        .ok_or_else(|| GpuRenderError::UnsupportedType(output_type.clone()))?;
    let (_, kernel_input, arg_inputs) = materialize_inputs(primitive, node_types)?;
    let kernel_type = node_types
        .get(kernel_input)
        .ok_or_else(|| GpuRenderError::MissingNodeType(kernel_input.to_string()))?;
    let kernel_decl = types::gpu_param_decl(kernel_type, "kernel", false)
        .ok_or_else(|| GpuRenderError::UnsupportedType(kernel_type.clone()))?;
    let arg_decls = arg_inputs
        .iter()
        .map(|input| {
            let ty = node_types
                .get(*input)
                .ok_or_else(|| GpuRenderError::MissingNodeType((*input).to_string()))?;
            types::gpu_param_decl(ty, input, false)
                .ok_or_else(|| GpuRenderError::UnsupportedType(ty.clone()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    out.push_str(&format!(
        "__global__ void {kernel_name}({element_type} *out, uint64_t len, {kernel_decl}"
    ));
    for arg_decl in arg_decls {
        out.push_str(", ");
        out.push_str(&arg_decl);
    }
    out.push_str(") {\n");
    out.push_str("    uint64_t thread_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;\n");
    out.push_str("    if (thread_id >= len) { return; }\n");
    out.push_str("    catena_gpu_env_t env = { thread_id };\n");
    out.push_str("    catena_gpu_state_t state = 0;\n");
    out.push_str("    catena_gpu_state_t next_state = 0;\n");
    out.push_str(&format!("    {element_type} value;\n"));
    out.push_str("    kernel(env, state");
    for arg in arg_inputs {
        out.push_str(", ");
        out.push_str(arg);
    }
    out.push_str(", &next_state, &value);\n");
    out.push_str("    out[thread_id] = value;\n");
    out.push_str("}\n");

    Ok(())
}

fn render_materialize_call(
    out: &mut String,
    kernel_name: &str,
    primitive: &Primitive,
    node_types: &BTreeMap<String, metacat::tree::Tree<(), Operation>>,
) -> Result<(), GpuRenderError> {
    let [output] = primitive.outputs.as_slice() else {
        return Err(GpuRenderError::InvalidMaterializeOutput);
    };
    let output_type = node_types
        .get(output)
        .ok_or_else(|| GpuRenderError::MissingNodeType(output.clone()))?;
    let element_type = types::gpu_buffer_element_type(output_type)
        .ok_or_else(|| GpuRenderError::UnsupportedType(output_type.clone()))?;
    let (launch_params, kernel_input, arg_inputs) = materialize_inputs(primitive, node_types)?;
    out.push_str(&format!(
        "    uint64_t {output}_len = catena_launch_len({launch_params});\n"
    ));
    out.push_str(&format!("    {element_type} *{output}_data = nullptr;\n"));
    out.push_str(&format!(
        "    hipMalloc((void **)&{output}_data, {output}_len * sizeof({element_type}));\n"
    ));
    out.push_str(&format!(
        "    {kernel_name}<<<dim3({launch_params}.grid_dim.x, {launch_params}.grid_dim.y, {launch_params}.grid_dim.z), dim3({launch_params}.block_dim.x, {launch_params}.block_dim.y, {launch_params}.block_dim.z)>>>\n"
    ));
    out.push_str(&format!(
        "        ({output}_data, {output}_len, {kernel_input}"
    ));
    for arg in arg_inputs {
        out.push_str(", ");
        out.push_str(arg);
    }
    out.push_str(");\n");
    out.push_str(&format!("    {output}.data = {output}_data;\n"));
    out.push_str(&format!("    {output}.len = {output}_len;\n"));
    Ok(())
}

fn materialize_inputs<'a>(
    primitive: &'a Primitive,
    node_types: &BTreeMap<String, metacat::tree::Tree<(), Operation>>,
) -> Result<(&'a str, &'a str, Vec<&'a str>), GpuRenderError> {
    let launch_params = primitive
        .inputs
        .iter()
        .find(|input| {
            node_types.get(*input).is_some_and(|ty| {
                is_named_type(ty, "gpu.launch_params")
                    || is_val_wrapped_named_type(ty, "gpu.launch_params")
            })
        })
        .map(String::as_str)
        .ok_or(GpuRenderError::MissingMaterializeLaunchParams)?;
    let kernel = primitive
        .inputs
        .iter()
        .find(|input| {
            node_types
                .get(*input)
                .is_some_and(is_value_wrapped_function_type)
        })
        .map(String::as_str)
        .ok_or(GpuRenderError::MissingMaterializeKernel)?;
    let args = primitive
        .inputs
        .iter()
        .filter(|input| input.as_str() != launch_params && input.as_str() != kernel)
        .map(String::as_str)
        .collect();
    Ok((launch_params, kernel, args))
}

fn is_named_type(ty: &Tree<(), Operation>, name: &str) -> bool {
    matches!(ty, Tree::Node(op, 0, children) if op.as_str() == name && children.is_empty())
}

fn is_val_wrapped_named_type(ty: &Tree<(), Operation>, name: &str) -> bool {
    matches!(
        ty,
        Tree::Node(op, 0, children)
            if (op.as_str() == "val" || op.as_str() == "value")
                && matches!(children.as_slice(), [inner] if is_named_type(inner, name))
    )
}

fn is_value_wrapped_function_type(ty: &Tree<(), Operation>) -> bool {
    matches!(
        ty,
        Tree::Node(op, 0, children)
            if (op.as_str() == "val" || op.as_str() == "value")
                && matches!(children.as_slice(), [Tree::Node(fn_op, 0, _)] if fn_op.as_str() == "->")
    )
}

fn node_type_map(term: &AnnotatedTerm) -> BTreeMap<String, metacat::tree::Tree<(), Operation>> {
    term.hypergraph
        .nodes
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, ty)| (node_var(NodeId(index)), ty))
        .collect()
}

fn node_var(node: NodeId) -> String {
    format!("x{}", node.0)
}

fn sanitize_ident(name: &str) -> String {
    let mut ident = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    if ident.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        ident.insert(0, '_');
    }
    ident
}
