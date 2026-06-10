use std::{
    env,
    path::{Path, PathBuf},
    process::{Command, ExitStatus},
    sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

use catena_lang::{codegen::gpu::render_modules, compile::compile};
use libloading::Library;
use metacat::theory::RawTheorySet;
use tempfile::TempDir;

const STDLIB: &[&str] = &[
    include_str!("../stdlib/cmc.hex"),
    include_str!("../stdlib/value.hex"),
    include_str!("../stdlib/buf.hex"),
    include_str!("../stdlib/index.hex"),
    include_str!("../stdlib/data.hex"),
    include_str!("../stdlib/fn.hex"),
    include_str!("../stdlib/product.hex"),
    include_str!("../stdlib/gpu.hex"),
];

struct Artifact {
    _dir: TempDir,
    path: PathBuf,
}

struct LoadedArtifact {
    _artifact: Artifact,
    _library: Library,
    _hip: Option<Library>,
}

fn main() -> anyhow::Result<()> {
    let artifact_count = env_usize("CATENA_REPRO_ARTIFACTS", 1);
    let load_count = env_usize("CATENA_REPRO_LOADS", 2);
    let threads = env_usize("CATENA_REPRO_THREADS", 1);
    let keep_loaded = env_bool("CATENA_REPRO_KEEP_LOADED", false);
    let parallel = env_bool("CATENA_REPRO_PARALLEL", false);
    let load_hip = env_bool("CATENA_REPRO_LOAD_HIP", false);

    eprintln!(
        "repro: building {artifact_count} generated HIP shared objects named module.so; \
         then performing {load_count} runtime-style load cycles"
    );
    eprintln!(
        "repro: CATENA_REPRO_PARALLEL={parallel}; CATENA_REPRO_THREADS={threads}; \
         CATENA_REPRO_KEEP_LOADED={keep_loaded}; CATENA_REPRO_LOAD_HIP={load_hip}"
    );
    eprintln!(
        "repro: each cycle loads the generated .so, optionally loads libamdhip64.so, \
         and then either retains or drops both handles"
    );
    eprintln!(
        "repro: the default run is intentionally minimal: load one generated .so, \
         drop it, then load it again in the same process"
    );
    eprintln!(
        "repro: tune with CATENA_REPRO_ARTIFACTS, CATENA_REPRO_LOADS, \
         CATENA_REPRO_THREADS, CATENA_REPRO_KEEP_LOADED, CATENA_REPRO_PARALLEL, \
         CATENA_REPRO_LOAD_HIP"
    );
    eprintln!("repro: set CATENA_REPRO_LOADS=0 to keep loading until the process aborts");

    let mut artifacts = Vec::with_capacity(artifact_count);
    for index in 0..artifact_count {
        eprintln!("repro: compile artifact {index}");
        artifacts.push(build_artifact(index)?);
    }

    eprintln!("repro: if the LLVM option-registration bug triggers, this process will abort");
    let loaded = if parallel {
        load_parallel(artifacts, load_count, threads, keep_loaded, load_hip)?
    } else {
        load_sequential(&artifacts, load_count, keep_loaded, load_hip)?
    };

    eprintln!(
        "repro: completed {load_count} shared-object load attempts without triggering the bug; \
         retained {} loaded libraries",
        loaded.len(),
    );
    Ok(())
}

fn load_parallel(
    artifacts: Vec<Artifact>,
    load_count: usize,
    threads: usize,
    keep_loaded: bool,
    load_hip: bool,
) -> anyhow::Result<Vec<LoadedArtifact>> {
    let artifacts = Arc::new(artifacts);
    let next_load = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(threads));
    let mut handles = Vec::with_capacity(threads);
    for thread_index in 0..threads {
        let artifacts = Arc::clone(&artifacts);
        let next_load = Arc::clone(&next_load);
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(
            move || -> anyhow::Result<Vec<LoadedArtifact>> {
                barrier.wait();
                let mut loaded = Vec::new();
                loop {
                    let load_index = next_load.fetch_add(1, Ordering::Relaxed);
                    if load_count != 0 && load_index >= load_count {
                        break;
                    }
                    let artifact_index = (load_index + thread_index) % artifacts.len();
                    eprintln!(
                        "repro: thread {thread_index} before Library::new \
                         load={load_index} artifact={artifact_index}"
                    );
                    let artifact = clone_artifact_for_load(&artifacts[artifact_index])?;
                    let library = unsafe { Library::new(artifact.path.as_path()) }?;
                    eprintln!(
                        "repro: thread {thread_index} after Library::new \
                         load={load_index} artifact={artifact_index}"
                    );
                    let hip = if load_hip {
                        eprintln!(
                            "repro: thread {thread_index} before load libamdhip64.so \
                             load={load_index} artifact={artifact_index}"
                        );
                        let hip = load_hip_library()?;
                        eprintln!(
                            "repro: thread {thread_index} after load libamdhip64.so \
                             load={load_index} artifact={artifact_index}"
                        );
                        Some(hip)
                    } else {
                        None
                    };

                    if keep_loaded {
                        loaded.push(LoadedArtifact {
                            _artifact: artifact,
                            _library: library,
                            _hip: hip,
                        });
                    } else {
                        drop(hip);
                        drop(library);
                        drop(artifact);
                        eprintln!(
                            "repro: thread {thread_index} after drop \
                             load={load_index} artifact={artifact_index}"
                        );
                    }
                }
                Ok(loaded)
            },
        ));
    }

    let mut loaded = Vec::new();
    for handle in handles {
        loaded.extend(handle.join().expect("loader thread panicked")?);
    }
    Ok(loaded)
}

fn load_sequential(
    artifacts: &[Artifact],
    load_count: usize,
    keep_loaded: bool,
    load_hip: bool,
) -> anyhow::Result<Vec<LoadedArtifact>> {
    let mut loaded = Vec::new();
    let mut load_index = 0usize;
    while load_count == 0 || load_index < load_count {
        let artifact_index = load_index % artifacts.len();
        eprintln!("repro: before Library::new load={load_index} artifact={artifact_index}");
        let artifact = clone_artifact_for_load(&artifacts[artifact_index])?;
        let library = unsafe { Library::new(artifact.path.as_path()) }?;
        eprintln!("repro: after Library::new load={load_index} artifact={artifact_index}");
        let hip = if load_hip {
            eprintln!(
                "repro: before load libamdhip64.so load={load_index} artifact={artifact_index}"
            );
            let hip = load_hip_library()?;
            eprintln!(
                "repro: after load libamdhip64.so load={load_index} artifact={artifact_index}"
            );
            Some(hip)
        } else {
            None
        };

        if keep_loaded {
            loaded.push(LoadedArtifact {
                _artifact: artifact,
                _library: library,
                _hip: hip,
            });
        } else {
            drop(hip);
            drop(library);
            drop(artifact);
            eprintln!("repro: after drop load={load_index} artifact={artifact_index}");
        }
        load_index += 1;
    }
    Ok(loaded)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(default)
}

fn build_artifact(index: usize) -> anyhow::Result<Artifact> {
    let source = program_source(index);
    let mut sources = STDLIB.to_vec();
    sources.push(&source);
    let raw = RawTheorySet::from_texts(sources)?;
    let report = compile(raw).map_err(|failure| failure.cause)?;
    let modules = report
        .gpu_modules
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("compile report did not contain GPU modules"))?;
    let cpp = render_modules(modules)?;

    let dir = tempfile::Builder::new()
        .prefix("catena-llvm-repro-")
        .tempdir()?;
    let cpp_path = dir.path().join("module.cpp");
    let so_path = dir.path().join("module.so");
    std::fs::write(&cpp_path, cpp)?;
    compile_shared_object(&cpp_path, &so_path)?;
    Ok(Artifact {
        _dir: dir,
        path: so_path,
    })
}

fn clone_artifact_for_load(artifact: &Artifact) -> anyhow::Result<Artifact> {
    let dir = tempfile::Builder::new()
        .prefix("catena-llvm-repro-load-")
        .tempdir()?;
    let path = dir.path().join("module.so");
    std::fs::copy(&artifact.path, &path)?;
    Ok(Artifact { _dir: dir, path })
}

fn compile_shared_object(cpp_path: &Path, so_path: &Path) -> anyhow::Result<()> {
    let output = Command::new("hipcc")
        .arg("-shared")
        .arg("-fPIC")
        .arg("-O2")
        .arg("--std=c++17")
        .arg("-ffp-contract=off")
        .arg("-fno-fast-math")
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

fn load_hip_library() -> anyhow::Result<Library> {
    if let Ok(library) = unsafe { Library::new("libamdhip64.so") } {
        return Ok(library);
    }
    for env_var in ["ROCM_PATH", "HIP_PATH"] {
        let Ok(root) = env::var(env_var) else {
            continue;
        };
        let path = PathBuf::from(root).join("lib/libamdhip64.so");
        if let Ok(library) = unsafe { Library::new(path) } {
            return Ok(library);
        }
    }
    unsafe { Library::new(Path::new("libamdhip64.so")) }
        .map_err(|error| anyhow::anyhow!("failed to load HIP runtime library: {error}"))
}

fn status_text(status: ExitStatus) -> String {
    status
        .code()
        .map(|code| code.to_string())
        .unwrap_or_else(|| "terminated by signal".to_string())
}

fn program_source(index: usize) -> String {
    match index % 5 {
        0 => format!("(def program repro-{index} : (bool val) -> (bool val) = bool.not)"),
        1 => format!(
            r#"
            (def program repro-{index} : [] -> (u64 val) = (
              ({{u64.one u64.one}} u64.add)
              {{[x . x x]}}
              u64.mul
            ))
            "#
        ),
        2 => {
            format!("(def program repro-{index} : [] -> (u64 val) = const.u64.0xDEADBEEFDEADBEEF)")
        }
        3 => format!("(def program repro-{index} : [] -> (u32 val) = const.u32.0xDEADBEEF)"),
        _ => format!(
            r#"
            (def program repro-{index} : ([n.] (cap.own mem)) -> ([n.] (u64 val)) = (
              mem.cast.u64
              {{
                (u64.assert-nz ix.zero)
                [b]
              }}
              ix
            ))
            "#
        ),
    }
}
