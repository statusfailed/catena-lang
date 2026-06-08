use std::path::PathBuf;

use catena_dsl::runtime::{Runtime, Value};

fn main() -> anyhow::Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let runtime = Runtime::new([
        root.join("stdlib/cmc.hex"),
        root.join("stdlib/value.hex"),
        root.join("stdlib/buf.hex"),
        root.join("stdlib/index.hex"),
        root.join("stdlib/base/data.hex"),
        root.join("stdlib/base/fn.hex"),
        root.join("stdlib/base/product.hex"),
        root.join("stdlib/gpu.hex"),
        root.join("example.hex"),
    ])?;

    let [result] = runtime.exec("program_not", [false.into()])?;
    println!("program_not(false): {result:?}");

    let values = [0x123456789abcdef0_u64, 7, 11];
    println!(
        "array-head input: [{}]",
        values
            .iter()
            .map(|value| format!("0x{value:x}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let input = runtime.mem_u64(&values)?;
    let [head] = runtime.exec("program_array_head_u64", [input])?;
    let Value::U64(head) = head else {
        anyhow::bail!("program_array_head_u64 returned non-u64 value: {head:?}");
    };

    println!(
        "program_array_head_u64: 0x{head:x} (expected 0x{:x})",
        values[0]
    );
    anyhow::ensure!(
        head == values[0],
        "array head mismatch: got 0x{head:x}, expected 0x{:x}",
        values[0]
    );

    Ok(())
}
