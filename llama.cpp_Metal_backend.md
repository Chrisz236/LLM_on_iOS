# llama.cpp init Metal compute backend

- Source: https://github.com/ggerganov/llama.cpp/blob/daa9623ab051a8162ae750b150b9522571b55f21/ggml/src/ggml-metal.m#L303C1-L687C2

## Explain

This function, `ggml_metal_init`, initializes the **Metal backend** for the **GGML** library on iOS. It sets up the GPU context, loads the Metal kernels, and configures the necessary components for running operations on the GPU using Apple’s Metal framework.

Here’s a breakdown of what the code does:

### 1. **Device Selection and Setup**:
```objc
id<MTLDevice> device = MTLCreateSystemDefaultDevice();
GGML_METAL_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);
```
- **Selects the Default Metal Device**: This picks the default GPU available on the iOS device using `MTLCreateSystemDefaultDevice()`. The selected device is then stored in the Metal context structure.
- Logs the device name for debugging.

### 2. **Context Initialization**:
```objc
struct ggml_backend_metal_context * ctx = calloc(1, sizeof(struct ggml_backend_metal_context));
ctx->device = device;
ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_BUFFERS);
ctx->queue = [ctx->device newCommandQueue];
ctx->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);
```
- **Context Memory Allocation**: A context (`ctx`) for the Metal backend is allocated.
- **Command Queue Creation**: A Metal command queue is created (`newCommandQueue`) to manage and submit commands to the GPU.
- **Dispatch Queue**: A concurrent dispatch queue is created for parallel processing of tasks.

### 3. **Metal Library Loading**:
The code tries to load the Metal shader library in multiple ways:
- **Precompiled Metal Library (`.metallib`)**:
  ```objc
  NSString * path_lib = [bundle pathForResource:@"default" ofType:@"metallib"];
  if (try_metallib && path_lib != nil) {
      metal_library = [ctx->device newLibraryWithURL:libURL error:&error];
  }
  ```
  - The function first checks if there is a precompiled Metal library (`.metallib` file) in the app bundle. If it exists, it loads it.
  - In read world case, it runs into this way
  
- **Embedded Metal Source**:
  ```objc
  NSString * src = [[NSString alloc] initWithBytes:ggml_metallib_start length:(ggml_metallib_end-ggml_metallib_start) encoding:NSUTF8StringEncoding];
  ```
  - If no precompiled library is found, it tries to load an embedded Metal shader source.

- **Load Metal Source from File**:
  ```objc
  NSString * path_source = [bundle pathForResource:@"ggml-metal" ofType:@"metal"];
  NSString * src = [NSString stringWithContentsOfFile:path_source encoding:NSUTF8StringEncoding error:&error];
  metal_library = [ctx->device newLibraryWithSource:src options:options error:&error];
  ```
  - If neither is found, it loads the Metal source code from a `.metal` file in the bundle or the current working directory and compiles it.

If any of these steps fail, an error is logged and the function returns `NULL`.

### 4. **GPU Family Detection**:
```objc
for (int i = MTLGPUFamilyApple1 + 20; i >= MTLGPUFamilyApple1; --i) {
    if ([ctx->device supportsFamily:i]) {
        GGML_METAL_LOG_INFO("%s: GPU family: MTLGPUFamilyApple%d\n", __func__, i - (int) MTLGPUFamilyApple1 + 1);
        break;
    }
}
```
- **Determine GPU Family**: The function checks what GPU family the device supports (such as `MTLGPUFamilyApple9`), which is crucial for deciding whether certain advanced Metal features are supported (e.g., SIMD group reduction).
  
### 5. **Kernel Compilation**:
```objc
#define GGML_METAL_ADD_KERNEL(e, name, supported) \
if (supported) { \
    struct ggml_metal_kernel * kernel = &ctx->kernels[e]; \
    id<MTLFunction> metal_function = [metal_library newFunctionWithName:@"kernel_"#name]; \
    kernel->pipeline = [ctx->device newComputePipelineStateWithFunction:metal_function error:&error]; \
}
```
- **Load Metal Kernels**: For each Metal function (or "kernel"), the function checks if it is supported on the GPU and then compiles it into a pipeline state object (`newComputePipelineStateWithFunction`). Each kernel is stored in the `ctx->kernels` array for future use.
- If any kernel fails to load or compile, an error is logged.
- Check this part source code!!!. It register ALL predefined Metal kernels with `GGML_METAL_ADD_KERNEL` to `ctx->kernels`.

### 6. **GPU Feature Checks**:
```objc
ctx->support_simdgroup_reduction = [ctx->device supportsFamily:MTLGPUFamilyApple7];
ctx->support_simdgroup_mm = [ctx->device supportsFamily:MTLGPUFamilyApple7];
GGML_METAL_LOG_INFO("%s: simdgroup reduction support = %s\n", __func__, ctx->support_simdgroup_reduction ? "true" : "false");
```
- The function checks whether the GPU supports **SIMD group reduction** (needed for certain types of parallelized calculations) and **SIMD group matrix multiplication**, which are advanced features introduced in newer Apple GPU families.

### 7. **Return the Metal Context**:
```objc
return ctx;
```
- After setting up the context, compiling the kernels, and configuring the GPU features, the function returns the initialized Metal context (`ctx`). This context is ready to be used for GPU-accelerated computations with the Metal framework.

### Key Steps in the Function:
1. **Pick a Metal-capable GPU**: The function selects the default Metal device (GPU) on the iOS device.
2. **Allocate and Initialize Context**: Allocates memory for the Metal context and sets up necessary Metal objects like command queues.
3. **Load Metal Shaders**: Attempts to load a precompiled Metal library (`.metallib`), embedded Metal code, or source from a file. It compiles the Metal kernels and pipelines.
4. **Check GPU Features**: Verifies which GPU features are supported by the current device.
5. **Return Initialized Context**: Returns the Metal context for use in subsequent GPU computations.

### Summary:
This function is responsible for initializing a **Metal backend** for the GGML library on iOS. It selects the GPU, compiles Metal shaders, checks the available GPU features, and configures the context to run computations on the GPU. If any error occurs during initialization (e.g., failure to load Metal shaders), the function logs the error and returns `NULL`.