use crate::codegen::GpuDialect;

pub fn render_gpu_prelude(dialect: GpuDialect) -> String {
    let profile_helpers = match dialect {
        GpuDialect::Hip => {
            r#"
#ifndef __HIP_DEVICE_COMPILE__
typedef hipEvent_t catena_profile_event_t;

typedef struct {
    catena_profile_event_t start;
    catena_profile_event_t stop;
} catena_profile_span_t;

__host__ static inline bool catena_profile_enabled() {
    const char *value = getenv("CATENA_GPU_PROFILE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

__host__ static inline catena_profile_span_t catena_profile_start() {
    catena_profile_span_t span = {nullptr, nullptr};
    if (!catena_profile_enabled()) {
        return span;
    }
    catena_host_gpu_check(hipEventCreate(&span.start));
    catena_host_gpu_check(hipEventCreate(&span.stop));
    catena_host_gpu_check(hipEventRecord(span.start, nullptr));
    return span;
}

__host__ static inline void catena_profile_finish(
    catena_profile_span_t span,
    const char *name,
    uint64_t work_items) {
    if (span.start == nullptr) {
        return;
    }
    catena_host_gpu_check(hipEventRecord(span.stop, nullptr));
    catena_host_gpu_check(hipEventSynchronize(span.stop));
    float elapsed_ms = 0.0f;
    catena_host_gpu_check(hipEventElapsedTime(&elapsed_ms, span.start, span.stop));
    fprintf(stderr, "CATENA_PROFILE\t%s\t%llu\t%.6f\n",
        name, (unsigned long long)work_items, elapsed_ms);
    catena_host_gpu_check(hipEventDestroy(span.start));
    catena_host_gpu_check(hipEventDestroy(span.stop));
}
#endif
"#
        }
        GpuDialect::Cuda => {
            r#"
#ifndef __CUDA_ARCH__
typedef cudaEvent_t catena_profile_event_t;

typedef struct {
    catena_profile_event_t start;
    catena_profile_event_t stop;
} catena_profile_span_t;

__host__ static inline bool catena_profile_enabled() {
    const char *value = getenv("CATENA_GPU_PROFILE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

__host__ static inline catena_profile_span_t catena_profile_start() {
    catena_profile_span_t span = {nullptr, nullptr};
    if (!catena_profile_enabled()) {
        return span;
    }
    catena_host_gpu_check(cudaEventCreate(&span.start));
    catena_host_gpu_check(cudaEventCreate(&span.stop));
    catena_host_gpu_check(cudaEventRecord(span.start, nullptr));
    return span;
}

__host__ static inline void catena_profile_finish(
    catena_profile_span_t span,
    const char *name,
    uint64_t work_items) {
    if (span.start == nullptr) {
        return;
    }
    catena_host_gpu_check(cudaEventRecord(span.stop, nullptr));
    catena_host_gpu_check(cudaEventSynchronize(span.stop));
    float elapsed_ms = 0.0f;
    catena_host_gpu_check(cudaEventElapsedTime(&elapsed_ms, span.start, span.stop));
    fprintf(stderr, "CATENA_PROFILE\t%s\t%llu\t%.6f\n",
        name, (unsigned long long)work_items, elapsed_ms);
    catena_host_gpu_check(cudaEventDestroy(span.start));
    catena_host_gpu_check(cudaEventDestroy(span.stop));
}
#endif
"#
        }
    };

    let blas_helpers = match dialect {
        GpuDialect::Hip => {
            r#"
#ifndef __HIP_DEVICE_COMPILE__
__host__ static inline void catena_host_blas_check(rocblas_status status) {
    if (status != rocblas_status_success) {
        fprintf(stderr, "catena rocBLAS error: %d\n", (int)status);
        fflush(stderr);
        __builtin_trap();
    }
}

__host__ static inline rocblas_handle catena_blas_handle() {
    static rocblas_handle handle = [] {
        rocblas_handle created = nullptr;
        catena_host_blas_check(rocblas_create_handle(&created));
        return created;
    }();
    return handle;
}

__host__ static inline void catena_platform_sgemm(
    float *input,
    float *weight,
    uint64_t rows,
    uint64_t columns,
    uint64_t reduction,
    float *output) {
    catena_assert(rows <= INT32_MAX && columns <= INT32_MAX && reduction <= INT32_MAX);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    catena_profile_span_t profile = catena_profile_start();
    catena_host_blas_check(rocblas_sgemm(
        catena_blas_handle(),
        rocblas_operation_transpose,
        rocblas_operation_none,
        (rocblas_int)columns,
        (rocblas_int)rows,
        (rocblas_int)reduction,
        &alpha,
        weight,
        (rocblas_int)reduction,
        input,
        (rocblas_int)reduction,
        &beta,
        output,
        (rocblas_int)columns));
    catena_profile_finish(profile, "rocblas_sgemm", rows * columns);
}
#endif
"#
        }
        GpuDialect::Cuda => {
            r#"
#ifndef __CUDA_ARCH__
__host__ static inline void catena_host_blas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "catena cuBLAS error: %d\n", (int)status);
        fflush(stderr);
        __builtin_trap();
    }
}

__host__ static inline cublasHandle_t catena_blas_handle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t created = nullptr;
        catena_host_blas_check(cublasCreate(&created));
        return created;
    }();
    return handle;
}

__host__ static inline void catena_platform_sgemm(
    float *input,
    float *weight,
    uint64_t rows,
    uint64_t columns,
    uint64_t reduction,
    float *output) {
    catena_assert(rows <= INT32_MAX && columns <= INT32_MAX && reduction <= INT32_MAX);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    catena_profile_span_t profile = catena_profile_start();
    catena_host_blas_check(cublasSgemm(
        catena_blas_handle(),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        (int)columns,
        (int)rows,
        (int)reduction,
        &alpha,
        weight,
        (int)reduction,
        input,
        (int)reduction,
        &beta,
        output,
        (int)columns));
    catena_profile_finish(profile, "cublas_sgemm", rows * columns);
}
#endif
"#
        }
    };

    format!(
        r#"#include <{runtime_header}>
#include <{blas_header}>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef uint8_t catena_unit_t;
typedef uint8_t catena_gpu_state_t;

typedef struct {{
    uint32_t x;
    uint32_t y;
    uint32_t z;
}} catena_dim3_t;

typedef struct {{
    uint64_t thread_id;
}} catena_gpu_env_t;

typedef struct {{
    catena_dim3_t grid_dim;
    catena_dim3_t block_dim;
}} catena_launch_params_t;

typedef struct {{
    void *data;
    uint64_t len;
}} catena_mem_t;

typedef struct {{
    void *data;
    uint64_t len;
}} catena_gpu_buf_t;

__host__ __device__ static inline void catena_assert(uint8_t condition) {{
    if (!condition) {{
#ifndef {device_compile_guard}
        fprintf(stderr, "catena assertion failed\n");
        fflush(stderr);
#endif
        __builtin_trap();
    }}
}}

#ifndef {device_compile_guard}
__host__ static inline void catena_host_gpu_check({error_type} err) {{
    if (err != {success_value}) {{
        fprintf(stderr, "catena GPU error: %s\n", {error_string_fn}(err));
        fflush(stderr);
        __builtin_trap();
    }}
}}

#endif

__host__ __device__ static inline uint64_t catena_launch_len(catena_launch_params_t params) {{
    return (uint64_t)params.grid_dim.x * params.grid_dim.y * params.grid_dim.z
        * params.block_dim.x * params.block_dim.y * params.block_dim.z;
}}

__host__ __device__ static inline void bool_not(uint8_t arg0, uint8_t *out1) {{
    *out1 = !arg0;
}}

__host__ __device__ static inline void bool_or(uint8_t arg0, uint8_t arg1, uint8_t *out2) {{
    *out2 = arg0 || arg1;
}}

__host__ __device__ static inline void bool_and(uint8_t arg0, uint8_t arg1, uint8_t *out2) {{
    *out2 = arg0 && arg1;
}}

__host__ __device__ static inline void bool_id(uint8_t arg0, uint8_t *out1) {{
    *out1 = arg0;
}}

__host__ __device__ static inline void bool_copy(uint8_t arg0, uint8_t *out1, uint8_t *out2) {{
    *out1 = arg0;
    *out2 = arg0;
}}

__host__ __device__ static inline void bool_li(uint8_t arg0, uint8_t *out1) {{
    *out1 = arg0;
}}

__host__ __device__ static inline float catena_u32_bitcast_f32(uint32_t bits) {{
    union {{
        uint32_t u;
        float f;
    }} value;
    value.u = bits;
    return value.f;
}}

__host__ __device__ static inline uint32_t catena_f32_bitcast_u32(float value) {{
    union {{
        uint32_t u;
        float f;
    }} bits;
    bits.f = value;
    return bits.u;
}}

{profile_helpers}

{blas_helpers}

"#,
        runtime_header = dialect.runtime_header(),
        blas_header = dialect.blas_header(),
        device_compile_guard = dialect.device_compile_guard(),
        error_type = dialect.error_type(),
        success_value = dialect.success_value(),
        error_string_fn = dialect.error_string_fn(),
        profile_helpers = profile_helpers,
        blas_helpers = blas_helpers,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_gpu_check_is_host_only() {
        let prelude = render_gpu_prelude(GpuDialect::Hip);

        assert!(
            prelude.contains(
                "#ifndef __HIP_DEVICE_COMPILE__\n__host__ static inline void catena_host_gpu_check(hipError_t err)"
            )
        );
        assert!(!prelude.contains("catena_gpu_check"));
        assert!(!prelude.contains("__device__ static inline void catena_host_gpu_check"));
    }
}
