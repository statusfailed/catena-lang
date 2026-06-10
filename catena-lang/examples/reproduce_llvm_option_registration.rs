use std::{
    path::Path,
    process::{Command, ExitStatus},
};

use libloading::os::unix::{Library, RTLD_LAZY, RTLD_LOCAL};

const HIP_SOURCE: &str = r#"
#include <hip/hip_runtime.h>

__global__ void repro_kernel() {}
"#;

fn main() -> anyhow::Result<()> {
    let reproduce_crash = std::env::args().any(|arg| arg == "--repro");
    let dir = tempfile::Builder::new()
        .prefix("catena-llvm-repro-")
        .tempdir()?;
    let cpp_path = dir.path().join("module.cpp");
    let so_path = dir.path().join("module.so");

    std::fs::write(&cpp_path, HIP_SOURCE)?;
    compile_shared_object(&cpp_path, &so_path)?;

    if reproduce_crash {
        eprintln!("repro: using normal dlopen/dlclose; this should crash on the second load");
    } else {
        eprintln!("repro: using RTLD_NODELETE; this should survive the second load");
        eprintln!("repro: pass --repro to run the crashing path");
    }

    eprintln!("repro: load generated HIP shared object");
    let library = load_library(&so_path, reproduce_crash)?;
    eprintln!("repro: drop generated HIP shared object");
    drop(library);

    eprintln!("repro: load generated HIP shared object again");
    let _library = load_library(&so_path, reproduce_crash)?;

    eprintln!("repro: second load succeeded");
    Ok(())
}

fn load_library(path: &Path, reproduce_crash: bool) -> Result<Library, libloading::Error> {
    let flags = if reproduce_crash {
        RTLD_LAZY | RTLD_LOCAL
    } else {
        RTLD_LAZY | RTLD_LOCAL | libc::RTLD_NODELETE
    };
    unsafe { Library::open(Some(path), flags) }
}

fn compile_shared_object(cpp_path: &Path, so_path: &Path) -> anyhow::Result<()> {
    let output = Command::new("hipcc")
        .arg("-shared")
        .arg("-fPIC")
        .arg("-O2")
        .arg("--std=c++17")
        .arg(cpp_path)
        .arg("-o")
        .arg(so_path)
        .output()
        .map_err(|error| anyhow::anyhow!("hipcc is unavailable: {error}"))?;

    if !output.status.success() {
        anyhow::bail!(
            "HIP/C++ compilation failed with status {}: {}",
            status_text(output.status),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

fn status_text(status: ExitStatus) -> String {
    status
        .code()
        .map(|code| code.to_string())
        .unwrap_or_else(|| "terminated by signal".to_string())
}
