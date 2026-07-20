use crate::codegen::{
    GpuAssign, GpuDialect,
    components::value_expr,
    gpu::GpuRenderError,
    render_utils::{invalid_inputs, invalid_outputs},
};

pub(in crate::codegen) const OP: &str = "f32.gemm-row-major-rhs-transposed";

pub(in crate::codegen) fn render(
    out: &mut String,
    assignment: &GpuAssign,
    dialect: GpuDialect,
) -> Result<(), GpuRenderError> {
    let [input, weight, rows, columns, reduction, output_len] = assignment.inputs.as_slice() else {
        return Err(invalid_inputs(assignment, 6));
    };
    let [output] = assignment.outputs.as_slice() else {
        return Err(invalid_outputs(assignment, 1));
    };

    out.push_str(&format!(
        "    {output} = nullptr;\n\
         if ({output_len} != 0) {{\n\
             catena_host_gpu_check({alloc}((void **)&{output}, {output_len} * sizeof(float)));\n\
             catena_platform_sgemm(\n\
                 (float *){input}, (float *){weight},\n\
                 {rows}, {columns}, {reduction}, {output});\n\
         }}\n",
        output = output.name,
        output_len = value_expr(output_len),
        alloc = dialect.managed_alloc_fn(),
        input = value_expr(input),
        weight = value_expr(weight),
        rows = value_expr(rows),
        columns = value_expr(columns),
        reduction = value_expr(reduction),
    ));
    Ok(())
}
