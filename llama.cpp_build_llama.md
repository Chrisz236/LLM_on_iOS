# llama.cpp build compute graph for llama model

- Source: https://github.com/ggerganov/llama.cpp/blob/daa9623ab051a8162ae750b150b9522571b55f21/src/llama.cpp#L9968C5-L10112C6

## Explain

The function `build_llama()` constructs a compute graph for a transformer-based LLM (such as LLaMA) using GGML. This function defines the operations for each layer of the transformer model, building a graph that represents the forward pass of the model. Here’s a detailed breakdown of what this function does:

### 1. **Initialize the Compute Graph**:
   - The function starts by creating an empty graph using `ggml_new_graph_custom`. The size of the graph is determined based on the model's maximum number of nodes.

   ```cpp
   struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);
   ```

### 2. **Token and Embedding Initialization**:
   - **Token Count**: A mutable variable `n_tokens` keeps track of the number of tokens to process. Initially, this is set from `this->n_tokens`, which refers to the current input token count.
   - **Input Embedding (`inpL`)**: The input tokens are transformed into embeddings using `llm_build_inp_embd()`. These embeddings represent the input tokens in a dense vector space.
   - **Position Embedding (`inp_pos`)**: The position embedding is generated, which is necessary for the model to incorporate the position of each token in the sequence.
   - **Attention Mask (`KQ_mask`)**: A mask is created for the self-attention mechanism, preventing attention from being applied to future tokens in autoregressive models.

   ```cpp
   struct ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
   struct ggml_tensor * inp_pos = build_inp_pos();
   struct ggml_tensor * KQ_mask = build_inp_KQ_mask();
   ```

### 3. **Processing Through Transformer Layers**:
   - The model processes the input through several transformer layers in a loop (`n_layer` layers).
   
   ```cpp
   for (int il = 0; il < n_layer; ++il) {
       struct ggml_tensor * inpSA = inpL;
   ```

#### 3.1 **Layer Normalization**:
   - Each layer starts with a normalization step (e.g., LayerNorm or RMSNorm). This is standard in transformer architectures to stabilize the learning process and improve performance.

   ```cpp
   cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
   ```

#### 3.2 **Self-Attention Mechanism**:
   - The function implements the self-attention mechanism, a core part of transformers, by performing the following:

     - **Query (`Qcur`)**: Queries are computed using a linear transformation (LoRA, in this case). If a bias exists, it is added.
     - **Key (`Kcur`)**: Keys are similarly computed with a LoRA transformation and bias.
     - **Value (`Vcur`)**: Values are also computed using LoRA.

     ```cpp
     struct ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
     struct ggml_tensor * Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
     struct ggml_tensor * Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
     ```

   - **Rotary Position Embedding (RoPE)**: RoPE is applied to the queries and keys to encode positional information. This ensures that the model can differentiate between different positions in the sequence.
   
   ```cpp
   Qcur = ggml_rope_ext(...);  // Applies RoPE to Qcur
   Kcur = ggml_rope_ext(...);  // Applies RoPE to Kcur
   ```

   - **Attention Calculation**: The queries, keys, and values are combined with the attention mask to compute the attention scores. The attention mechanism allows the model to focus on relevant parts of the input.

   ```cpp
   cur = llm_build_kv(ctx0, lctx, kv_self, gf, model.layers[il].wo, model.layers[il].bo, Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, il);
   ```

   - **Key-Value (KV) Cache**: The self-attention mechanism makes use of a **KV cache**. The KV cache stores key-value pairs from previous tokens, allowing the model to efficiently retrieve information without recalculating attention for past tokens. This is a major optimization for autoregressive generation.

#### 3.3 **Feed-Forward Network (FFN)**:
   - After self-attention, the output passes through a feed-forward network (FFN), which consists of linear transformations and an activation function (e.g., SiLU).

   ```cpp
   cur = llm_build_ffn(ctx0, lctx, cur, model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL, model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL, model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
   ```

   - **Mixture of Experts (MoE)**: Some models may implement a Mixture of Experts (MoE) mechanism, where different "experts" (FFNs) are chosen dynamically depending on the input. If `model.layers[il].ffn_gate_inp` is non-null, the function builds an MoE FFN instead.

   ```cpp
   cur = llm_build_moe_ffn(ctx0, lctx, cur, model.layers[il].ffn_gate_inp, model.layers[il].ffn_up_exps, model.layers[il].ffn_gate_exps, model.layers[il].ffn_down_exps, n_expert, n_expert_used, LLM_FFN_SILU, true, false, 0.0, cb, il);
   ```

   - **Residual Connection**: A residual connection is added, which sums the original input (`ffn_inp`) and the FFN output. This is crucial for training stability and efficiency.

   ```cpp
   cur = ggml_add(ctx0, cur, ffn_inp);
   ```

#### 3.4 **End of Layer Processing**:
   - After the last transformer layer, the function skips computing outputs for unused tokens if applicable (e.g., during batched processing or generation).

   ```cpp
   if (il == n_layer - 1) {
       struct ggml_tensor * inp_out_ids = build_inp_out_ids();
       cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
   }
   ```

### 4. **Final Layer and Output Computation**:
   - After all layers are processed, the function normalizes the final output and passes it through the language model's head (a final linear transformation) to map it back to token logits, used for making predictions.

   ```cpp
   cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
   cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
   ```

### 5. **Finalize the Compute Graph**:
   - The last step is to finalize the compute graph. This prepares the graph for execution by the backend (e.g., Metal, OpenCL, etc.).

   ```cpp
   ggml_build_forward_expand(gf, cur);
   return gf;
   ```

### Summary:
1. **Initialize** the compute graph and input embeddings.
2. **Process** the input through multiple transformer layers, performing attention and feed-forward computations, including optimizations like the **KV cache**.
3. **Normalize** and project the final output through the language model head.
4. **Finalize** the graph for execution.

This code builds the core computation graph of an LLM, ensuring efficient use of memory (via the KV cache) and performing key transformer operations (self-attention, FFN, etc.).