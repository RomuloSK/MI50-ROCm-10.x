// Minimal native HIP runtime smoke for MI50/gfx906.
//
// This exercises device discovery, allocation, copies, streams, events, a
// wave64-compatible kernel launch, and synchronization. It deliberately does
// not use an ISA override or assume matrix cores, BF16, or FP8.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

__global__ void add_one(const float* input, float* output, std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = input[index] + 1.0f;
  }
}

bool is_gfx906(const hipDeviceProp_t& properties) {
  return std::strstr(properties.gcnArchName, "gfx906") != nullptr;
}

int fail(const char* operation, hipError_t error) {
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(error));
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  const bool graph_only = argc == 2 && std::strcmp(argv[1], "--graph") == 0;
  if (argc > 1 && !graph_only) {
    std::fprintf(stderr, "usage: %s [--graph]\n", argv[0]);
    return 2;
  }
  int device_count = 0;
  hipError_t error = hipGetDeviceCount(&device_count);
  if (error != hipSuccess) {
    return fail("hipGetDeviceCount", error);
  }
  if (device_count < 1) {
    std::fprintf(stderr, "no HIP device is visible\n");
    return 2;
  }

  error = hipSetDevice(0);
  if (error != hipSuccess) {
    return fail("hipSetDevice", error);
  }
  hipDeviceProp_t properties{};
  error = hipGetDeviceProperties(&properties, 0);
  if (error != hipSuccess) {
    return fail("hipGetDeviceProperties", error);
  }
  if (!is_gfx906(properties)) {
    std::fprintf(stderr, "device 0 is %s, expected native gfx906\n", properties.gcnArchName);
    return 3;
  }

  constexpr std::size_t count = 4096;
  constexpr std::size_t bytes = count * sizeof(float);
  float* host_input = static_cast<float*>(std::malloc(bytes));
  float* host_output = static_cast<float*>(std::malloc(bytes));
  float* device_input = nullptr;
  float* device_output = nullptr;
  if (host_input == nullptr || host_output == nullptr) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(host_input);
    std::free(host_output);
    return 4;
  }
  for (std::size_t index = 0; index < count; ++index) {
    host_input[index] = static_cast<float>(index) * 0.25f;
    host_output[index] = 0.0f;
  }

  error = hipMalloc(reinterpret_cast<void**>(&device_input), bytes);
  if (error != hipSuccess) {
    std::free(host_input);
    std::free(host_output);
    return fail("hipMalloc(input)", error);
  }
  error = hipMalloc(reinterpret_cast<void**>(&device_output), bytes);
  if (error != hipSuccess) {
    hipFree(device_input);
    std::free(host_input);
    std::free(host_output);
    return fail("hipMalloc(output)", error);
  }

  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  int result = 0;
  do {
    error = hipStreamCreate(&stream);
    if (error != hipSuccess) {
      result = fail("hipStreamCreate", error);
      break;
    }
    if (graph_only) {
      // Keep copies outside capture so pageable host buffers and graph
      // support are tested independently.  The captured node is the same
      // native gfx906 kernel used by the baseline smoke.
      error = hipMemcpyAsync(device_input, host_input, bytes, hipMemcpyHostToDevice, stream);
      if (error == hipSuccess) error = hipStreamSynchronize(stream);
      if (error == hipSuccess) error = hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
      constexpr unsigned int threads = 256;
      const unsigned int blocks = static_cast<unsigned int>((count + threads - 1) / threads);
      if (error == hipSuccess) {
        hipLaunchKernelGGL(add_one, dim3(blocks), dim3(threads), 0, stream, device_input,
                           device_output, count);
        error = hipGetLastError();
      }
      if (error == hipSuccess) error = hipStreamEndCapture(stream, &graph);
      if (error == hipSuccess)
        error = hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
      if (error == hipSuccess) error = hipGraphLaunch(graph_exec, stream);
      if (error == hipSuccess) error = hipStreamSynchronize(stream);
      if (error == hipSuccess)
        error = hipMemcpyAsync(host_output, device_output, bytes, hipMemcpyDeviceToHost, stream);
      if (error == hipSuccess) error = hipStreamSynchronize(stream);
      if (error != hipSuccess) {
        result = fail("HIP graph execution", error);
        break;
      }
      for (std::size_t index = 0; index < count; ++index) {
        if (std::fabs(host_output[index] - (host_input[index] + 1.0f)) > 1e-5f) {
          std::fprintf(stderr, "graph result mismatch at %zu: got %f expected %f\n", index,
                       host_output[index], host_input[index] + 1.0f);
          result = 5;
          break;
        }
      }
      if (result == 0)
        std::printf("MI50 HIP graph smoke passed: %s\n", properties.gcnArchName);
      break;
    }
    error = hipEventCreate(&start);
    if (error != hipSuccess) {
      result = fail("hipEventCreate(start)", error);
      break;
    }
    error = hipEventCreate(&stop);
    if (error != hipSuccess) {
      result = fail("hipEventCreate(stop)", error);
      break;
    }
    error = hipMemcpyAsync(device_input, host_input, bytes, hipMemcpyHostToDevice, stream);
    if (error != hipSuccess) {
      result = fail("hipMemcpyAsync(HtoD)", error);
      break;
    }
    error = hipEventRecord(start, stream);
    if (error != hipSuccess) {
      result = fail("hipEventRecord(start)", error);
      break;
    }
    constexpr unsigned int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    hipLaunchKernelGGL(add_one, dim3(blocks), dim3(threads), 0, stream, device_input, device_output, count);
    error = hipGetLastError();
    if (error != hipSuccess) {
      result = fail("add_one launch", error);
      break;
    }
    error = hipEventRecord(stop, stream);
    if (error != hipSuccess) {
      result = fail("hipEventRecord(stop)", error);
      break;
    }
    error = hipMemcpyAsync(host_output, device_output, bytes, hipMemcpyDeviceToHost, stream);
    if (error != hipSuccess) {
      result = fail("hipMemcpyAsync(DtoH)", error);
      break;
    }
    error = hipStreamSynchronize(stream);
    if (error != hipSuccess) {
      result = fail("hipStreamSynchronize", error);
      break;
    }
    for (std::size_t index = 0; index < count; ++index) {
      if (std::fabs(host_output[index] - (host_input[index] + 1.0f)) > 1e-5f) {
        std::fprintf(stderr, "result mismatch at %zu: got %f expected %f\n", index,
                     host_output[index], host_input[index] + 1.0f);
        result = 5;
        break;
      }
    }
    if (result == 0) {
      float milliseconds = 0.0f;
      error = hipEventElapsedTime(&milliseconds, start, stop);
      if (error != hipSuccess) {
        result = fail("hipEventElapsedTime", error);
      } else {
        std::printf("MI50 HIP runtime smoke passed: %s, %.3f ms\n", properties.gcnArchName,
                    milliseconds);
      }
    }
  } while (false);

  if (stop != nullptr) {
    hipEventDestroy(stop);
  }
  if (start != nullptr) {
    hipEventDestroy(start);
  }
  // If capture failed between BeginCapture and EndCapture, close the capture
  // before destroying its stream so a runtime error cannot leak a graph state.
  if (graph_only && result != 0 && stream != nullptr && graph == nullptr) {
    hipGraph_t aborted_graph = nullptr;
    if (hipStreamEndCapture(stream, &aborted_graph) == hipSuccess && aborted_graph != nullptr)
      hipGraphDestroy(aborted_graph);
  }
  if (graph_exec != nullptr) {
    hipGraphExecDestroy(graph_exec);
  }
  if (graph != nullptr) {
    hipGraphDestroy(graph);
  }
  if (stream != nullptr) {
    hipStreamDestroy(stream);
  }
  hipFree(device_output);
  hipFree(device_input);
  std::free(host_output);
  std::free(host_input);
  return result;
}
