use std::{
    fs::File,
    io::{self, Write},
    path::PathBuf,
};

use catena_lang::{
    codegen::GpuDialect,
    runtime::{Runtime, Value},
    stdlib,
};
use clap::Parser;
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use tokenizers::Tokenizer;

const LAYERS: usize = 24;
const GENERATED_TOKENS: usize = 10;
const HEADS: u64 = 32;
const HEAD_DIM: u64 = 64;
const HIDDEN_SIZE: u64 = HEADS * HEAD_DIM;
const VOCAB_SIZE: u64 = 49_152;
const INTERMEDIATE_SIZE: u64 = 8_192;
const EXTENTS: usize = 13;
const FORWARD_INPUTS: usize = 2 + EXTENTS + LAYERS * 9 + 1;

#[derive(Parser)]
#[command(name = "smollm2")]
struct Args {
    /// Directory containing model.safetensors and tokenizer.json.
    model_dir: PathBuf,

    /// Prompt to continue.
    prompt: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let tokenizer_path = args.model_dir.join("tokenizer.json");
    let weights_path = args.model_dir.join("model.safetensors");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| anyhow::anyhow!("failed to load {}: {error}", tokenizer_path.display()))?;
    let model_prompt = format!(
        "<|im_start|>system\n\
         You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n\
         <|im_start|>user\n{}<|im_end|>\n\
         <|im_start|>assistant\n",
        args.prompt
    );
    let encoding = tokenizer
        .encode(model_prompt, false)
        .map_err(|error| anyhow::anyhow!("failed to tokenize prompt: {error}"))?;
    let mut token_ids = encoding
        .get_ids()
        .iter()
        .copied()
        .map(u64::from)
        .collect::<Vec<_>>();
    anyhow::ensure!(!token_ids.is_empty(), "the prompt produced no tokens");

    let mut decoder = tokenizer.decode_stream(true);
    let stdout = io::stdout();
    let mut output = stdout.lock();
    for &token_id in encoding.get_ids() {
        decoder
            .step(token_id)
            .map_err(|error| anyhow::anyhow!("failed to decode prompt: {error}"))?;
    }
    write!(output, "User: {}\nAssistant: ", args.prompt)?;
    output.flush()?;

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let runtime = Runtime::new(
        stdlib::paths_from(&root).chain([
            root.join("examples/nn.hex"),
            root.join("examples/smollm2.hex"),
        ]),
        configured_gpu_dialect()?,
    )?;

    let file = File::open(&weights_path)?;
    let mapped = unsafe { Mmap::map(&file)? };
    let checkpoint = SafeTensors::deserialize(&mapped)?;
    let mut inputs = model_inputs(&runtime, &checkpoint, &token_ids)?;

    for _ in 0..GENERATED_TOKENS {
        update_sequence_inputs(&runtime, &mut inputs, &token_ids)?;
        let [logits] = runtime.exec_borrowed("smollm2.forward", &inputs)?;
        let Value::Mem(logits) = logits else {
            anyhow::bail!("smollm2.forward returned a non-memory value");
        };
        let logits = logits.to_f32_vec();
        let expected = token_ids.len() * VOCAB_SIZE as usize;
        anyhow::ensure!(
            logits.len() == expected,
            "expected {expected} logits, got {}",
            logits.len()
        );
        let final_token_logits = &logits[expected - VOCAB_SIZE as usize..];
        let next_token = final_token_logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(token, _)| u64::try_from(token))
            .transpose()?
            .ok_or_else(|| anyhow::anyhow!("the final logits row was empty"))?;
        token_ids.push(next_token);

        if let Some(text) = decoder
            .step(u32::try_from(next_token)?)
            .map_err(|error| anyhow::anyhow!("failed to decode generated token: {error}"))?
        {
            write!(output, "{text}")?;
            output.flush()?;
        }
    }

    writeln!(output)?;
    Ok(())
}

fn model_inputs(
    runtime: &Runtime,
    checkpoint: &SafeTensors<'_>,
    token_ids: &[u64],
) -> anyhow::Result<[Value; FORWARD_INPUTS]> {
    let mut inputs = Vec::with_capacity(FORWARD_INPUTS);
    inputs.push(runtime.mem_u64(token_ids)?);
    push_weight(
        runtime,
        checkpoint,
        "model.embed_tokens.weight",
        &mut inputs,
    )?;
    inputs.extend(sequence_extents(token_ids.len())?.map(Value::u64));

    for layer in 0..LAYERS {
        let prefix = format!("model.layers.{layer}");
        for suffix in [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            push_weight(
                runtime,
                checkpoint,
                &format!("{prefix}.{suffix}"),
                &mut inputs,
            )?;
        }
    }
    push_weight(runtime, checkpoint, "model.norm.weight", &mut inputs)?;

    inputs.try_into().map_err(|values: Vec<Value>| {
        anyhow::anyhow!("expected {FORWARD_INPUTS} inputs, got {}", values.len())
    })
}

fn update_sequence_inputs(
    runtime: &Runtime,
    inputs: &mut [Value; FORWARD_INPUTS],
    token_ids: &[u64],
) -> anyhow::Result<()> {
    inputs[0] = runtime.mem_u64(token_ids)?;
    for (slot, extent) in inputs[2..2 + EXTENTS]
        .iter_mut()
        .zip(sequence_extents(token_ids.len())?)
    {
        *slot = Value::u64(extent);
    }
    Ok(())
}

fn sequence_extents(tokens: usize) -> anyhow::Result<[u64; EXTENTS]> {
    let tokens = u64::try_from(tokens)?;
    Ok([
        tokens,
        HEADS,
        HEAD_DIM,
        HIDDEN_SIZE,
        VOCAB_SIZE,
        tokens * HIDDEN_SIZE,
        VOCAB_SIZE * HIDDEN_SIZE,
        tokens * VOCAB_SIZE,
        INTERMEDIATE_SIZE,
        tokens * INTERMEDIATE_SIZE,
        HIDDEN_SIZE * HIDDEN_SIZE,
        INTERMEDIATE_SIZE * HIDDEN_SIZE,
        HIDDEN_SIZE * INTERMEDIATE_SIZE,
    ])
}

fn push_weight(
    runtime: &Runtime,
    checkpoint: &SafeTensors<'_>,
    name: &str,
    inputs: &mut Vec<Value>,
) -> anyhow::Result<()> {
    let tensor = checkpoint.tensor(name)?;
    let values = tensor_to_f32(name, tensor)?;
    inputs.push(runtime.mem_f32(&values)?);
    Ok(())
}

fn tensor_to_f32(name: &str, tensor: TensorView<'_>) -> anyhow::Result<Vec<f32>> {
    let data = tensor.data();
    match tensor.dtype() {
        Dtype::BF16 => Ok(data
            .chunks_exact(2)
            .map(|bytes| {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                f32::from_bits(u32::from(bits) << 16)
            })
            .collect()),
        Dtype::F32 => Ok(data
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
            .collect()),
        dtype => anyhow::bail!("tensor {name} has unsupported dtype {dtype:?}"),
    }
}

fn configured_gpu_dialect() -> anyhow::Result<GpuDialect> {
    match std::env::var("CATENA_GPU_DIALECT") {
        Ok(value) if value == "hip" => Ok(GpuDialect::Hip),
        Ok(value) if value == "cuda" => Ok(GpuDialect::Cuda),
        Ok(value) => anyhow::bail!("invalid CATENA_GPU_DIALECT `{value}`; expected hip or cuda"),
        Err(std::env::VarError::NotPresent) => Ok(GpuDialect::Hip),
        Err(error) => anyhow::bail!("invalid CATENA_GPU_DIALECT: {error}"),
    }
}
