// Native GCN5.1 packed INT8 dot-product primitive smoke for gfx906.
//
// MI50 exposes the scalar/vector dot4 instruction even when a higher-level
// rocBLAS INT8 GEMM catalog is absent.  This test validates that primitive as
// a possible quantized-inference fallback without claiming a full GEMM path.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

__global__ void dot4_kernel(const char4* input_a, const char4* input_b, int* output,
                            std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = amd_mixed_dot(input_a[index], input_b[index], 0, false);
}

bool is_gfx906_wave64(const hipDeviceProp_t& properties) {
  return std::strstr(properties.gcnArchName, "gfx906") != nullptr && properties.warpSize == 64;
}

int fail(const char* operation, hipError_t error) {
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(error));
  return 1;
}

}  // namespace

int main() {
  int device_count = 0;
  hipError_t error = hipGetDeviceCount(&device_count);
  if (error == hipErrorNoDevice || device_count < 1) {
    std::fprintf(stderr, "INT8 dot4 smoke: GPU-test-pending (no HIP device)\n");
    return 77;
  }
  if (error != hipSuccess) return fail("hipGetDeviceCount", error);
  hipDeviceProp_t properties{};
  error = hipGetDeviceProperties(&properties, 0);
  if (error != hipSuccess) return fail("hipGetDeviceProperties", error);
  if (!is_gfx906_wave64(properties)) {
    std::fprintf(stderr, "device 0 is %s/wave%d, expected gfx906/wave64\n", properties.gcnArchName,
                 properties.warpSize);
    return 2;
  }

  constexpr std::size_t count = 256;
  std::vector<char4> host_a(count, make_char4(1, 2, 3, 4));
  std::vector<char4> host_b(count, make_char4(2, 2, 2, 2));
  std::vector<int> host_output(count, 0);
  char4* device_a = nullptr;
  char4* device_b = nullptr;
  int* device_output = nullptr;
  error = hipMalloc(reinterpret_cast<void**>(&device_a), sizeof(char4) * count);
  if (error == hipSuccess) error = hipMalloc(reinterpret_cast<void**>(&device_b), sizeof(char4) * count);
  if (error == hipSuccess) error = hipMalloc(reinterpret_cast<void**>(&device_output), sizeof(int) * count);
  if (error == hipSuccess) error = hipMemcpy(device_a, host_a.data(), sizeof(char4) * count, hipMemcpyHostToDevice);
  if (error == hipSuccess) error = hipMemcpy(device_b, host_b.data(), sizeof(char4) * count, hipMemcpyHostToDevice);
  if (error != hipSuccess) {
    hipFree(device_output);
    hipFree(device_b);
    hipFree(device_a);
    return fail("INT8 dot4 input setup", error);
  }
  constexpr unsigned int threads = 256;
  hipLaunchKernelGGL(dot4_kernel, dim3(1), dim3(threads), 0, nullptr, device_a, device_b,
                     device_output, count);
  error = hipGetLastError();
  if (error == hipSuccess) error = hipDeviceSynchronize();
  if (error == hipSuccess)
    error = hipMemcpy(host_output.data(), device_output, sizeof(int) * count, hipMemcpyDeviceToHost);
  hipFree(device_output);
  hipFree(device_b);
  hipFree(device_a);
  if (error != hipSuccess) return fail("INT8 dot4 execution", error);
  for (std::size_t index = 0; index < count; ++index) {
    if (host_output[index] != 20) {
      std::fprintf(stderr, "INT8 dot4 mismatch at %zu: got=%d expected=20\n", index,
                   host_output[index]);
      return 3;
    }
  }
  std::printf("MI50 packed INT8 dot4 primitive passed: %s\n", properties.gcnArchName);
  return 0;
}
