typedef unsigned long uintptr_t;
typedef unsigned long long uint64_t;
typedef unsigned short uint16_t;

struct Sampler
{
  float temp;
  float top_p;
  uintptr_t top_k;
};

extern "C" {
/// Initialize logger and RNG. Call this once before everything.
void init(uint64_t seed);

/// Set the RNG seed.
void seed(uint64_t seed);

/// Load a runtime.
///
/// # Safety
///
/// The caller must ensure that `model` is valid.
void load(const char *model, uintptr_t quant, uintptr_t quant_nf4);

void load_with_rescale(const char *model, uintptr_t quant, uintptr_t quant_nf4, uintptr_t rescale);

/// Clear the model state.
void clear_state();

/// Generate the next token prediction given the input tokens and a sampler.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` does not exceed the actual length of `tokens`.
uint16_t infer(const uint16_t *tokens,
               uintptr_t len,
               struct Sampler sampler);

} // extern "C"