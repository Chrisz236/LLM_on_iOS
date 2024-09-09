# llama.cpp prefill/decoding function

- Source: https://github.com/ggerganov/llama.cpp/blob/daa9623ab051a8162ae750b150b9522571b55f21/src/llama.cpp#L16058C1-L16354C2

## Explain 

This code is responsible for decoding a batch of tokens using a transformer model in the **Llama context** (`llama_context`) and performing various steps for evaluating a transformer model's forward pass. It handles batching, key-value (KV) cache management, and extracting outputs such as logits (model predictions) or embeddings.

Here’s a breakdown of what the code does:

### Function Overview:

- **`llama_decode_internal`** takes in a `llama_context` (which contains the model and its parameters) and a batch of tokens to be processed. It returns:
  - `0` for success,
  - a positive value for a warning,
  - a negative value for an error.

### Key Components:

#### 1. **Input Validation**:
```cpp
lctx.is_encoding = false;
const uint32_t n_tokens_all = batch_all.n_tokens;

if (n_tokens_all == 0) {
    LLAMA_LOG_ERROR("%s: n_tokens == 0", __func__);
    return -1;
}
```
- The function starts by setting the context's `is_encoding` flag to `false`, indicating that this is a decoding operation.
- It checks if the batch contains any tokens (`n_tokens_all`). If not, it returns an error (`-1`).

#### 2. **Token Validation**:
```cpp
for (uint32_t i = 0; i < n_tokens_all; ++i) {
    if (batch_all.token[i] < 0 || (uint32_t)batch_all.token[i] >= lctx.model.vocab.n_vocab) {
        LLAMA_LOG_ERROR("%s: invalid token[%d] = %d", __func__, i, batch_all.token[i]);
        return -1;
    }
}
```
- This loop checks if the tokens in the batch are valid by ensuring they are within the valid vocabulary range (`0` to `vocab.n_vocab`).
- If any token is invalid, the function returns an error (`-1`).

#### 3. **Context Setup**:
```cpp
const auto & model = lctx.model;
const auto & hparams = model.hparams;
const auto & cparams = lctx.cparams;
```
- The context (`lctx`) is used to reference the model and its hyperparameters (`hparams`) and context parameters (`cparams`).

#### 4. **Batch Type Assertions**:
```cpp
GGML_ASSERT((!batch_all.token && batch_all.embd) || (batch_all.token && !batch_all.embd)); // NOLINT
GGML_ASSERT(n_tokens_all <= cparams.n_batch);
```
- Ensures that the batch contains either tokens or embeddings, but not both.
- Checks that the number of tokens in the batch does not exceed the batch size defined in the context parameters.

#### 5. **Output Calculation**:
```cpp
if (batch_all.logits && !embd_pooled) {
    for (uint32_t i = 0; i < n_tokens_all; ++i) {
        n_outputs += batch_all.logits[i] != 0;
    }
} else if (lctx.logits_all || embd_pooled) {
    n_outputs = n_tokens_all;
} else {
    n_outputs = 1;
}
```
- The function calculates how many outputs will be generated (`n_outputs`). This depends on whether logits (predictions) or pooled embeddings are requested.
  - If `logits` are requested and embeddings are not pooled, it counts the number of non-zero logits.
  - If logits for all tokens are needed or embeddings are pooled, it sets `n_outputs` to the total number of tokens.
  - Otherwise, it only produces one output (likely the last token).

#### 6. **Batch Processing**:
```cpp
while (lctx.sbatch.n_tokens > 0) {
    llama_ubatch ubatch;
    if (kv_self.recurrent) {
        ubatch = lctx.sbatch.split_equal(n_ubatch);
    } else {
        ubatch = lctx.sbatch.split_simple(n_ubatch);
    }
    const uint32_t n_tokens = ubatch.n_tokens;
```
- This loop processes the batch in **mini-batches** (`ubatch`) to handle multiple tokens at once. The function splits the batch into smaller units (`n_ubatch`) based on the context's ability to handle them.
  - **Recurrent Model**: If the model is recurrent, it splits the batch into equal-length sequences.
  - **Non-Recurrent Model**: Otherwise, it splits the batch into simple sequences.

#### 7. **Key-Value (KV) Cache Management**:
```cpp
if (hparams.causal_attn) {
    llama_kv_cache_update(&lctx);
    if (!llama_kv_cache_find_slot(kv_self, ubatch)) {
        return 1;
    }
}
```
- If the model uses causal attention (meaning that past tokens can influence future tokens but not the other way around), the KV cache is updated.
  - The cache helps speed up the transformer model's attention mechanism by storing previously computed key-value pairs.
  - If the cache is not available or there is an issue finding a slot for the current batch, it returns a warning (`1`).

#### 8. **Graph Building and Scheduling**:
```cpp
ggml_cgraph * gf = llama_build_graph(lctx, ubatch, false);
ggml_backend_sched_alloc_graph(lctx.sched, gf);
llama_set_inputs(lctx, ubatch);
llama_graph_compute(lctx, gf, n_threads, threadpool);
```
- The function builds a computational graph (`llama_build_graph`) that represents the forward pass through the model for the current batch.
- The graph is then scheduled for execution using the Metal backend (`ggml_backend_sched_alloc_graph`).
- The inputs (tokens) are set, and the graph is computed (`llama_graph_compute`), which involves running the forward pass of the model over the batch.

#### 9. **Extracting Logits**:
```cpp
if (res) {
    ggml_backend_tensor_get_async(backend_res, res, logits_out, 0, n_outputs_new*n_vocab*sizeof(float));
}
```
- If logits are needed, the function extracts them from the result tensor (`res`) and stores them in the logits buffer (`lctx.logits`).
- The logits represent the probabilities for the next token in the sequence.

#### 10. **Extracting Embeddings**:
```cpp
if (embd) {
    ggml_backend_tensor_get_async(backend_embd, embd, embd_out, 0, n_outputs_new*n_embd*sizeof(float));
}
```
- If embeddings are requested, the function extracts them and stores them in the `lctx.embd` buffer.
- Embeddings are vector representations of tokens that are useful for tasks like generating contextualized text or performing downstream tasks like classification.

#### 11. **Key-Value Cache and Final Steps**:
```cpp
kv_self.head += n_tokens;
if (kv_self.head >= kv_self.size) {
    kv_self.head = 0;
}
```
- The KV cache is updated by moving the cache head forward by the number of tokens processed.
- The function then decides whether to defragment the KV cache based on a fragmentation threshold.

#### 12. **Return Success**:
```cpp
return 0;
```
- If everything runs successfully, the function returns `0`.

### Summary:

This function decodes a batch of tokens by evaluating a transformer model in the context of a LLaMA-based language model. It splits the batch into smaller units, updates the KV cache (used for attention mechanisms), builds and computes a computational graph, and extracts outputs (logits or embeddings) from the model. The function also handles memory and resource management, ensuring the model’s internal state, such as the KV cache, remains consistent across multiple forward passes.