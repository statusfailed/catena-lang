use std::{fs, io, path::Path};

use crate::{codegen::gpu::render_module, report::CompileReport};

pub fn dump_gpu(report: &CompileReport, dir: &Path) -> io::Result<()> {
    let Some(gpu_modules) = &report.gpu_modules else {
        return Ok(());
    };

    fs::create_dir_all(dir)?;

    for (definition_name, module) in gpu_modules {
        let rendered = render_module(module).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to render GPU code for `program.{definition_name}`: {error}"),
            )
        })?;
        fs::write(dir.join(format!("program.{definition_name}.cpp")), rendered)?;
    }

    Ok(())
}
