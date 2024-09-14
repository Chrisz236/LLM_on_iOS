# llama.cpp backend assignment of nodes in cgraph

- Source: https://github.com/ggerganov/llama.cpp/blob/daa9623ab051a8162ae750b150b9522571b55f21/ggml/src/ggml-backend.c#L1281C1-L1736C2

## Explain

This function, `ggml_backend_sched_split_graph`, is responsible for assigning backends to various operations (nodes) in a computational graph and splitting the graph into subgraphs that can be computed on the same backend. Here's a detailed explanation of the process:

### Key Steps and Operations:

1. **Reset Scheduler State**:
   The function begins by resetting the state of the scheduler `sched`, clearing the current splits, and resetting the graph input counts.
   - The memory context used for scheduling is re-initialized using `ggml_init()`.
   
   ```cpp
   sched->n_splits = 0;
   sched->n_graph_inputs = 0;
   sched->is_reset = false;
   sched->ctx = ggml_init(params);
   ```

2. **Pass 1: Backend Assignment for Pre-Allocated Inputs**:
   - The function assigns a backend to each input node (`leaf`) and operation node (`node`) in the graph.
   - The backend is determined based on the current state of the scheduler, using the function `ggml_backend_sched_backend_id_from_cur()`.
   - If the backend has already been assigned (i.e., it is not `-1`), it won't be overwritten.

   ```cpp
   for (int i = 0; i < graph->n_leafs; i++) {
       struct ggml_tensor * leaf = graph->leafs[i];
       if (*leaf_backend_id == -1) {
           *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
       }
   }
   ```

3. **Pass 2: Backend Assignment Expansion**:
   - **Expand GPU Backend Downwards**: Assigns the same backend to adjacent nodes for more efficient execution. It prioritizes non-CPU (GPU) backends to expand the graph downwards, ignoring CPU nodes (lowest priority).
   - **Expand GPU Backend Upwards**: Similar to the downwards expansion but in reverse (bottom-up). Again, CPU nodes are skipped.
   - **Expand Other Backends**: If some nodes still haven't been assigned a backend, it expands further to other available backends.
   
   These expansions aim to group adjacent operations that can be performed on the same backend, improving execution efficiency by minimizing data transfers between different backends.

   ```cpp
   for (int i = 0; i < graph->n_nodes; i++) {
       struct ggml_tensor * node = graph->nodes[i];
       if (*node_backend_id != -1) {
           cur_backend_id = *node_backend_id;
       } else if (cur_backend_id != -1) {
           ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
       }
   }
   ```

4. **Pass 3: Backend Upgrades**:
   - This pass upgrades nodes to higher-priority backends when their inputs support the required buffer types. For example, if a tensor is already in the memory space of a higher-priority backend, the function will upgrade the node to use that backend.
   - It verifies that all the sources (inputs) of the node are compatible with the backend before making the upgrade.

   ```cpp
   for (int i = 0; i < graph->n_nodes; i++) {
       struct ggml_tensor * node = graph->nodes[i];
       if (*node_backend_id == -1) {
           // Find backend with the most supported inputs
       } else {
           // Upgrade to higher priority backend if possible
       }
   }
   ```

5. **Pass 4: Assign Backends to Remaining Inputs**:
   - If a node has a source (`view_src`) and the backend is still unassigned, it inherits the backend of the source.
   - Similarly, all unassigned inputs to nodes are given the same backend as the current node.

   ```cpp
   for (int i = 0; i < graph->n_nodes; i++) {
       struct ggml_tensor * node = graph->nodes[i];
       if (node->view_src != NULL && *cur_backend_id == -1) {
           *cur_backend_id = tensor_backend_id(node->view_src);
       }
   }
   ```

6. **Pass 5: Split Graph into Subgraphs**:
   - This pass splits the graph into subgraphs where each subgraph is assigned to a specific backend. It identifies points where the graph needs to be split based on the backend of each node and its inputs.
   - When a split is necessary (for example, if a node has inputs on a different backend), the current split is completed, and a new split is started.

   ```cpp
   for (; i < graph->n_nodes; i++) {
       struct ggml_tensor * node = graph->nodes[i];
       if (node_backend_id != cur_backend_id || need_new_split) {
           split->i_end = i;
           i_split++;
           split = &sched->splits[i_split];
           split->backend_id = node_backend_id;
           split->i_start = i;
       }
   }
   ```

7. **Copying Inputs for Subgraphs**:
   - When subgraphs are created, the inputs to each subgraph might need to be copied to the backend where that subgraph will be computed.
   - This part of the code handles creating these copies, ensuring that the data is correctly allocated and moved between backends if necessary.

   ```cpp
   for (int j = 0; j < split->n_inputs; j++) {
       struct ggml_tensor * input = split->inputs[j];
       struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);
       // Add dependencies to ensure correct data allocation
   }
   ```

### Summary:
- **Backend Assignment**: The function assigns computational backends (like CPU, GPU) to each node in the graph, ensuring efficient execution by grouping nodes that can be computed on the same backend.
- **Graph Splitting**: It splits the graph into subgraphs where each subgraph is assigned to a specific backend. This reduces the overhead of transferring data between backends.
- **Input Copying**: For subgraphs that require inputs from different backends, it ensures that the necessary data is copied between backends.

This function is crucial for optimizing multi-backend computations in systems where some operations are better suited for GPUs, and others may need to be handled by CPUs or specialized hardware.