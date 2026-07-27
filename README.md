# Web-RWKV-FFI

Simple FFI for [`web-rwkv`](https://github.com/cryscan/web-rwkv).

## CLI

The crate also ships a small CLI for offline conversions:

```bash
cargo run --bin web-rwkv-ffi-cli -- convert-pth-to-st \
  --input /path/to/model.pth \
  --output /path/to/model.st

cargo run --bin web-rwkv-ffi-cli -- convert-pth-to-prefab \
  --input /path/to/model.pth \
  --output /path/to/model-nf4.prefab \
  --quant-nf4 256
```

For the prefab command, `--quant`, `--quant-nf4`, and `--quant-sf4` follow the same semantics as the FFI APIs: they quantize the first N layers of each type. Values larger than the model layer count simply quantize all layers of that type.

## APIs

The FFI exports the following APIs:

```rust
pub struct Sampler {
    pub temp: f32,
    pub top_p: f32,
    pub top_k: usize,
}

pub struct ModelOutput {
    pub len: usize,
    pub data: *mut f32,
}

pub struct ModelInfoOutput {
    pub version: usize,
    pub num_layer: usize,
    pub num_hidden: usize,
    pub num_emb: usize,
    pub num_vocab: usize,
    pub num_head: usize,
}

pub struct StateRaw {
    pub len: usize,
    pub data: *mut f32,
}

/// Initialize logger and RNG. Call this once before everything.
pub fn init(seed: u64);
/// Set the RNG seed.
pub fn seed(seed: u64);
/// Load a runtime. Returns 0 on success, -1 on error.
pub fn load(model: *const c_char, quant: usize, quant_nf4: usize, quant_sf4: usize, fp16: bool, batch: usize) -> i32;
/// Load a prefab model. Returns 0 on success, -1 on error.
pub fn load_prefab(model: *const c_char, fp16: bool, batch: usize) -> i32;
/// Save the currently loaded model as a prefab. Returns 0 on success, -1 on error.
pub fn save_prefab(output_path: *const c_char) -> i32;
/// Load a model with rescale. Returns 0 on success, -1 on error.
pub fn load_with_rescale(model: *const c_char, quant: usize, quant_nf4: usize, quant_sf4: usize, rescale: usize, fp16: bool, batch: usize) -> i32;
/// Load an extended model (for Othello and other demos). Returns 0 on success, -1 on error.
pub fn load_extended(model: *const c_char, quant: usize, quant_nf4: usize, quant_sf4: usize, fp16: bool, batch: usize) -> i32;
/// Load a runtime from pth. Returns 0 on success, -1 on error.
pub fn load_pth(model: *const c_char, quant: usize, quant_nf4: usize, quant_sf4: usize, fp16: bool, batch: usize, callback: Option<extern "C" fn(f32)>) -> i32;
/// Clear the model state.
pub fn clear_state(batch: usize);
/// Get the model state.
pub fn get_state(batch: usize) -> StateRaw;
/// Set the model state.
pub fn set_state(data: StateRaw, batch: usize);
/// Free the model state.
pub fn free_state(state: StateRaw);
/// Generate the next token prediction given the input tokens and a sampler.
pub fn infer(tokens: *const u32, len: usize, sampler: Sampler) -> u32;
/// Compute the model's raw output (next token prediction only) given the input tokens.
pub fn infer_raw_last(tokens: *const u32, len: usize) -> ModelOutput;
/// Compute the model's raw output (predictions of all tokens) given the input tokens.
pub fn infer_raw_full(tokens: *const u32, len: usize) -> ModelOutput;
/// Delete the model output vector created by the infer functions.
pub fn free_raw(output: ModelOutput);
// Returns the model info.
pub fn get_model_info() -> ModelInfoOutput;
// Release the model.
pub fn release();
/// Convert a pth file to a st file. Returns 0 on success, -1 on error.
pub fn convert_pth_to_st(input_path: *const c_char, output_path: *const c_char) -> i32;
/// Convert a pth file directly to a prefab file. Returns 0 on success, -1 on error.
pub fn convert_pth_to_prefab(input_path: *const c_char, output_path: *const c_char, quant: usize, quant_nf4: usize, quant_sf4: usize, fp16: bool, batch: usize, callback: Option<extern "C" fn(f32)>) -> i32;
```
