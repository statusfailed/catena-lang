pub const GPU_PRELUDE: &str = r#"#include <hip/hip_runtime.h>
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
"#;
