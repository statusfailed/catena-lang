use std::collections::{HashMap, HashSet};

use hexpr::Operation;
use thiserror::Error;

use crate::codegen::{
    GpuAssign, GpuFunction, GpuModule, GpuValue, GpuVar, lower_types::CType, prelude::GPU_PRELUDE,
    runtime_type,
};

#[derive(Debug, Error)]
pub enum GpuRenderError {
    #[error("type `{0:?}` is erased and cannot be rendered here")]
    ErasedType(GpuVar),
    #[error("type `{0:?}` is unsupported in GPU renderer")]
    UnsupportedType(CType),
    #[error("unsupported GPU operation `{0}`")]
    UnsupportedOp(Operation),
    #[error("operation `{op}` expected {expected} inputs, found {actual}")]
    InvalidInputCount {
        op: Operation,
        expected: usize,
        actual: usize,
    },
    #[error("operation `{op}` expected {expected} outputs, found {actual}")]
    InvalidOutputCount {
        op: Operation,
        expected: usize,
        actual: usize,
    },
    #[error("gpu.materialize is missing launch params")]
    MissingMaterializeLaunchParams,
    #[error("gpu.materialize is missing function input")]
    MissingMaterializeFunction,
}

/// Render a single GPU dataflow module as standalone HIP/C++ source.
///
/// Codegen has already produced the semantic `GpuModule`; this renderer is responsible only for
/// turning that artifact into text. It still contains primitive-specific lowering for the current
/// small backend surface, but it should not inspect the original Catena graph or report state.
pub fn render_module(module: &GpuModule) -> Result<String, GpuRenderError> {
    let mut out = String::new();
    out.push_str(GPU_PRELUDE);
    out.push('\n');

    // Materialization is represented in the dataflow body as one assignment, but HIP needs an
    // auxiliary `__global__` kernel in addition to the host wrapper function.
    for assignment in &module.entry.assignments {
        if assignment.op.as_str() == "gpu.materialize" {
            render_materialize_kernel(&mut out, &materialize_kernel_name(assignment)?, assignment)?;
            out.push('\n');
        }
    }

    // The entry function is the ordinary host-callable wrapper for this definition.
    render_function(&mut out, &module.entry)?;
    Ok(out)
}

fn render_function(out: &mut String, function: &GpuFunction) -> Result<(), GpuRenderError> {
    // A materialize assignment can use a function pointer whose target element type is carried by
    // the materialized buffer, not by the ordinary lowered function-pointer type.
    let materialize_fn_outputs = materialize_fn_output_types(function)?;
    out.push_str(&format!("void {}(", function.name));
    let mut params = Vec::new();

    // Sources render as ordinary parameters; targets render as output-pointer parameters.
    for source in &function.sources {
        if let Some(element) = materialize_fn_outputs.get(&source.node)
            && matches!(runtime_type(source), Some(CType::FunctionPointer { .. }))
        {
            params.push(materialize_fn_ptr_decl(source, element)?);
        } else {
            params.push(param_decl(source, false)?);
        }
    }
    for target in &function.targets {
        params.push(param_decl(target, true)?);
    }
    out.push_str(&params.join(", "));
    out.push_str(") {\n");

    // Source variables are already declared by the function signature. Assignment outputs become
    // local variables the first time they are produced.
    let mut declared = function
        .sources
        .iter()
        .map(|source| source.name.clone())
        .collect::<HashSet<_>>();

    for assignment in &function.assignments {
        for output in &assignment.outputs {
            if runtime_type(output).is_some() && declared.insert(output.name.clone()) {
                out.push_str(&format!("    {};\n", local_decl(output)?));
            }
        }
        render_assignment(out, assignment)?;
    }

    // Multiple outputs are returned by writing computed target wires into pointer parameters.
    for result in &function.targets {
        out.push_str(&format!("    *out_{} = {};\n", result.name, result.name));
    }
    out.push_str("    return;\n");
    out.push_str("}\n");
    Ok(())
}

fn render_assignment(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    match assignment.op.as_str() {
        "bool.t" => {
            let [] = assignment.inputs.as_slice() else {
                return Err(invalid_inputs(assignment, 0));
            };
            let [output] = assignment.outputs.as_slice() else {
                return Err(invalid_outputs(assignment, 1));
            };
            out.push_str(&format!("    {} = 1;\n", output.name));
        }
        "bool.f" => {
            let [] = assignment.inputs.as_slice() else {
                return Err(invalid_inputs(assignment, 0));
            };
            let [output] = assignment.outputs.as_slice() else {
                return Err(invalid_outputs(assignment, 1));
            };
            out.push_str(&format!("    {} = 0;\n", output.name));
        }
        "bool.not" => {
            let [input] = assignment.inputs.as_slice() else {
                return Err(invalid_inputs(assignment, 1));
            };
            let [output] = assignment.outputs.as_slice() else {
                return Err(invalid_outputs(assignment, 1));
            };
            out.push_str(&format!("    {} = !{};\n", output.name, value_expr(input)));
        }
        "bool.and" => render_binary_bool(out, assignment, "&&")?,
        "bool.or" => render_binary_bool(out, assignment, "||")?,
        "bool.ifc" => render_bool_ifc(out, assignment)?,
        "unit.intro" => {}
        "eval" => render_eval(out, assignment)?,
        "gpu.materialize" => render_materialize_call(out, assignment)?,
        op => {
            return Err(GpuRenderError::UnsupportedOp(
                op.parse().unwrap_or_else(|_| assignment.op.clone()),
            ));
        }
    }
    Ok(())
}

fn render_binary_bool(
    out: &mut String,
    assignment: &GpuAssign,
    operator: &str,
) -> Result<(), GpuRenderError> {
    let [lhs, rhs] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 2));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    {} = {} {operator} {};\n",
        output.name,
        value_expr(lhs),
        value_expr(rhs)
    ));
    Ok(())
}

fn render_bool_ifc(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [env_true, fn_true, env_false, fn_false, flag, arg] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 6));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    out.push_str(&format!(
        "    if ({flag}) {{ {fn_true}({env_true}, {arg}, &{output}); }} else {{ {fn_false}({env_false}, {arg}, &{output}); }}\n",
        flag = value_expr(flag),
        fn_true = callable_expr(fn_true),
        env_true = value_expr(env_true),
        arg = value_expr(arg),
        output = output.name,
        fn_false = callable_expr(fn_false),
        env_false = value_expr(env_false),
    ));
    Ok(())
}

fn render_eval(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let Some((func, args)) = assignment.inputs.split_last() else {
        return Err(invalid_inputs(assignment, 1));
    };
    let mut call_args = args.iter().map(value_expr).collect::<Vec<_>>();
    call_args.extend(
        assignment
            .outputs
            .iter()
            .filter(|output| runtime_type(output).is_some())
            .map(|output| format!("&{}", output.name)),
    );
    out.push_str(&format!(
        "    {}({});\n",
        callable_expr(func),
        call_args.join(", ")
    ));
    Ok(())
}

fn render_materialize_kernel(
    out: &mut String,
    kernel_name: &str,
    assignment: &GpuAssign,
) -> Result<(), GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    let CType::Pointer(element) =
        runtime_type(output).ok_or_else(|| GpuRenderError::ErasedType(output.clone()))?
    else {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(output).unwrap().clone(),
        ));
    };
    let (_, func, args) = materialize_parts(assignment)?;

    out.push_str(&format!(
        "__global__ void {kernel_name}({} *out, uint64_t len",
        c_type(element)
    ));
    if let GpuValue::Var(var) = func {
        out.push_str(", ");
        out.push_str(&materialize_fn_ptr_decl(var, element)?);
    }
    for arg in &args {
        if let GpuValue::Var(var) = arg {
            if runtime_type(var).is_some() {
                out.push_str(", ");
                out.push_str(&param_decl(var, false)?);
            }
        }
    }
    out.push_str(") {\n");
    out.push_str("    uint64_t thread_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;\n");
    out.push_str("    if (thread_id >= len) { return; }\n");
    out.push_str("    catena_gpu_env_t env = { thread_id };\n");
    out.push_str("    catena_gpu_state_t state = 0;\n");
    out.push_str("    catena_gpu_state_t next_state = 0;\n");
    out.push_str(&format!("    {} value;\n", c_type(element)));
    out.push_str("    ");
    out.push_str(&callable_expr(func));
    out.push_str("(env, state");
    for arg in args {
        if let GpuValue::Var(var) = arg
            && runtime_type(var).is_some()
        {
            out.push_str(", ");
            out.push_str(&var.name);
        }
    }
    out.push_str(", &next_state, &value);\n");
    out.push_str("    out[thread_id] = value;\n");
    out.push_str("}\n");
    Ok(())
}

fn render_materialize_call(out: &mut String, assignment: &GpuAssign) -> Result<(), GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    let CType::Pointer(element) =
        runtime_type(output).ok_or_else(|| GpuRenderError::ErasedType(output.clone()))?
    else {
        return Err(GpuRenderError::UnsupportedType(
            runtime_type(output).unwrap().clone(),
        ));
    };
    let (launch, func, args) = materialize_parts(assignment)?;
    let launch = value_expr(launch);
    let kernel_name = materialize_kernel_name(assignment)?;

    out.push_str(&format!(
        "    uint64_t {name}_len = catena_launch_len({launch});\n",
        name = output.name
    ));
    out.push_str(&format!(
        "    {} *{name}_data = nullptr;\n",
        c_type(element),
        name = output.name
    ));
    out.push_str(&format!(
        "    hipMalloc((void **)&{name}_data, {name}_len * sizeof({element}));\n",
        name = output.name,
        element = c_type(element)
    ));
    out.push_str(&format!(
        "    {kernel_name}<<<dim3({launch}.grid_dim.x, {launch}.grid_dim.y, {launch}.grid_dim.z), dim3({launch}.block_dim.x, {launch}.block_dim.y, {launch}.block_dim.z)>>>\n"
    ));
    out.push_str(&format!(
        "        ({name}_data, {name}_len",
        name = output.name
    ));
    if matches!(func, GpuValue::Var(_)) {
        out.push_str(", ");
        out.push_str(&value_expr(func));
    }
    for arg in args {
        if let GpuValue::Var(var) = arg
            && runtime_type(var).is_some()
        {
            out.push_str(", ");
            out.push_str(&var.name);
        }
    }
    out.push_str(");\n");
    out.push_str(&format!("    {} = {}_data;\n", output.name, output.name));
    Ok(())
}

fn materialize_parts(
    assignment: &GpuAssign,
) -> Result<(&GpuValue, &GpuValue, Vec<&GpuValue>), GpuRenderError> {
    let launch = assignment
        .inputs
        .iter()
        .find(|input| {
            matches!(
                input,
                GpuValue::Var(var)
                    if matches!(runtime_type(var), Some(CType::Named(name)) if name == "catena_launch_params_t")
            )
        })
        .ok_or(GpuRenderError::MissingMaterializeLaunchParams)?;
    let func = assignment
        .inputs
        .iter()
        .find(|input| {
            matches!(
                input,
                GpuValue::FnSymbol(_)
                    | GpuValue::Var(GpuVar {
                        lowered: crate::codegen::lower_types::LoweredType::Runtime(
                            CType::FunctionPointer { .. }
                        ),
                        ..
                    })
            )
        })
        .ok_or(GpuRenderError::MissingMaterializeFunction)?;
    let args = assignment
        .inputs
        .iter()
        .filter(|input| !std::ptr::eq(*input, launch) && !std::ptr::eq(*input, func))
        .collect();
    Ok((launch, func, args))
}

fn materialize_fn_output_types(
    function: &GpuFunction,
) -> Result<HashMap<open_hypergraphs::lax::NodeId, CType>, GpuRenderError> {
    let mut outputs = HashMap::new();
    for assignment in &function.assignments {
        if assignment.op.as_str() != "gpu.materialize" {
            continue;
        }
        let [output] = assignment.outputs.as_slice() else {
            return Err(invalid_outputs(assignment, 1));
        };
        let CType::Pointer(element) =
            runtime_type(output).ok_or_else(|| GpuRenderError::ErasedType(output.clone()))?
        else {
            return Err(GpuRenderError::UnsupportedType(
                runtime_type(output).unwrap().clone(),
            ));
        };
        let (_, func, _) = materialize_parts(assignment)?;
        if let GpuValue::Var(var) = func {
            outputs.insert(var.node, (**element).clone());
        }
    }
    Ok(outputs)
}

fn materialize_kernel_name(assignment: &GpuAssign) -> Result<String, GpuRenderError> {
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };
    Ok(format!("materialize_{}", output.name))
}

fn param_decl(var: &GpuVar, by_pointer: bool) -> Result<String, GpuRenderError> {
    let ty = runtime_type(var).ok_or_else(|| GpuRenderError::ErasedType(var.clone()))?;
    if let CType::FunctionPointer { .. } = ty {
        return Ok(fn_ptr_decl(ty, &var.name));
    }
    if by_pointer {
        Ok(format!("{} *out_{}", c_type(ty), var.name))
    } else {
        Ok(format!("{} {}", c_type(ty), var.name))
    }
}

fn materialize_fn_ptr_decl(var: &GpuVar, element: &CType) -> Result<String, GpuRenderError> {
    let Some(CType::FunctionPointer { inputs, outputs }) = runtime_type(var) else {
        return param_decl(var, false);
    };
    let mut outputs = outputs.clone();
    if outputs.last() != Some(element) {
        outputs.push(element.clone());
    }
    Ok(fn_ptr_decl(
        &CType::FunctionPointer {
            inputs: inputs.clone(),
            outputs,
        },
        &var.name,
    ))
}

fn local_decl(var: &GpuVar) -> Result<String, GpuRenderError> {
    let ty = runtime_type(var).ok_or_else(|| GpuRenderError::ErasedType(var.clone()))?;
    if let CType::FunctionPointer { .. } = ty {
        Ok(fn_ptr_decl(ty, &var.name))
    } else {
        Ok(format!("{} {}", c_type(ty), var.name))
    }
}

fn fn_ptr_decl(ty: &CType, name: &str) -> String {
    match ty {
        CType::FunctionPointer { inputs, outputs } => {
            let mut params = inputs
                .iter()
                .enumerate()
                .map(|(index, ty)| format!("{} arg{index}", c_type(ty)))
                .collect::<Vec<_>>();
            let input_count = params.len();
            params.extend(
                outputs
                    .iter()
                    .enumerate()
                    .map(|(index, ty)| format!("{} *out{}", c_type(ty), input_count + index)),
            );
            if params.is_empty() {
                format!("void (*{name})(void)")
            } else {
                format!("void (*{name})({})", params.join(", "))
            }
        }
        other => format!("{} {name}", c_type(other)),
    }
}

fn c_type(ty: &CType) -> String {
    match ty {
        CType::Unit => "catena_unit_t".to_string(),
        CType::Bool => "uint8_t".to_string(),
        CType::U64 => "uint64_t".to_string(),
        CType::F32 => "float".to_string(),
        CType::Pointer(inner) => format!("{} *", c_type(inner)),
        CType::FunctionPointer { .. } => "/* fn */".to_string(),
        CType::Named(name) => name.clone(),
    }
}

fn value_expr(value: &GpuValue) -> String {
    match value {
        GpuValue::Var(var) => var.name.clone(),
        GpuValue::FnSymbol(symbol) => sanitize_ident(symbol.target.as_str()),
    }
}

fn callable_expr(value: &GpuValue) -> String {
    value_expr(value)
}

fn invalid_inputs(assignment: &GpuAssign, expected: usize) -> GpuRenderError {
    GpuRenderError::InvalidInputCount {
        op: assignment.op.clone(),
        expected,
        actual: assignment.inputs.len(),
    }
}

fn invalid_outputs(assignment: &GpuAssign, expected: usize) -> GpuRenderError {
    GpuRenderError::InvalidOutputCount {
        op: assignment.op.clone(),
        expected,
        actual: assignment.outputs.len(),
    }
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
