# llama.cpp backend compute cgraph

- Source: https://github.com/ggerganov/llama.cpp/blob/daa9623ab051a8162ae750b150b9522571b55f21/ggml/src/ggml-backend.c#L1774C1-L1867C2

## Explain

This code is part of a backend scheduling system that computes splits of a graph (a computational task broken down into subgraphs) across different hardware backends. It handles the transfer of data between backends and orchestrates asynchronous computation for efficient parallel execution.

### Key Concepts:

1. **Graph Splits**: 
   The overall computation graph is divided into smaller subgraphs (splits), each handled by a different backend (e.g., GPU, CPU, or other accelerators). These splits are processed independently but may depend on data from other splits.

2. **Backends**:
   A backend refers to a hardware or processing unit (like a GPU or CPU) that executes part of the graph. Each backend is responsible for processing a subset of the computation.

3. **Events and Synchronization**:
   Events track the progress of computations or data transfers across backends. Synchronization ensures that operations are performed in the correct order, particularly when data is shared between backends.

### Key Parts of the Code:

1. **Copy Input Tensors to Split Backend**:
   Each split may require input tensors that are computed by a different backend. This part of the code ensures that the input tensors are copied to the correct backend before computation starts.
   - If the tensor is flagged as an input (`GGML_TENSOR_FLAG_INPUT`), the copy is performed immediately, preventing overwriting by the user before the copy completes.
   - For other tensors, the code waits for the backend to finish its previous operations before copying.

   There are two types of copy operations:
   - **Synchronous Copy**: If the backend doesn’t support asynchronous copy, the code waits for the backend to finish its operations before copying the tensor.
   - **Asynchronous Copy**: If supported, the tensor is copied asynchronously to improve performance.

2. **Graph Computation**:
   - The graph (or subgraph) corresponding to each split is processed using the backend’s compute function, typically asynchronously (`ggml_backend_graph_compute_async`). This offloads the graph computation to the backend.
   - If a callback function (`sched->callback_eval`) is specified, the code allows the user to evaluate whether certain nodes in the graph are needed before proceeding. This lets users control when certain parts of the computation are done based on their needs.

3. **Recording Events**:
   After the data is copied and the computation is triggered, events are recorded to track when the operations are complete. These events are used for synchronization in later steps.

4. **Managing Multiple Copies**:
   The system supports multiple copies of data to handle scenarios where different backends need to work on the same data at the same time. After processing one split, the scheduler moves on to the next copy (`sched->cur_copy`) to handle the next round of computations.

### Step-by-Step Explanation:

1. **Loop Over Splits**:
   - The scheduler iterates over all splits in the graph. Each split is processed by its corresponding backend (`split_backend_id`).

2. **Copy Inputs**:
   - For each input tensor in the split, the code checks if it needs to copy the tensor from another backend. This ensures that each split has the necessary data to perform its computation.

3. **Graph Computation**:
   - If no user-defined callback is provided (`sched->callback_eval`), the code directly submits the split's graph for computation to the backend. Otherwise, the callback function allows for custom control over which nodes in the graph are evaluated and computed.
   - Asynchronous computation is used to avoid blocking the main program while the backend processes the graph.

4. **Synchronization and Events**:
   - After submitting the graph for computation, the scheduler records events to track the status of the computation. This helps in ensuring proper synchronization between different splits and backends.
   
5. **Handle Multiple Copies**:
   - The `cur_copy` is incremented to handle multiple copies of data in case different splits or backends need to work on the same data in parallel. This mechanism ensures that each backend has access to a separate copy of the data it needs to compute.

6. **Return Success**:
   - Once all splits are processed, the function returns `GGML_STATUS_SUCCESS`, indicating that the computation of the graph splits was successful.

### Optimization and Parallelism:
- This approach allows for efficient parallelism by leveraging different hardware backends, copying data asynchronously where possible, and managing multiple copies of the input data to prevent contention.
- The use of asynchronous execution reduces idle time and improves the performance of large computations on multi-backend systems.