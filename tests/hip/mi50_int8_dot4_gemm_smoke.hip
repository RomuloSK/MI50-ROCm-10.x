// Native packed-dot4 INT8 -> INT32 GEMM fallback smoke for gfx906.
//
// This is intentionally a small, readable fallback kernel rather than a
// replacement for rocBLAS.  It provides a real end-to-end quantized GEMM
// building block when a rocBLAS INT8 catalog is unavailable, using the GCN5.1
// packed dot4 primitive and no newer matrix ISA.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

__global__ void int8_dot4_gemm(const std::int8_t* matrix_a, const std::int8_t* matrix_b,
                               std::int32_t* matrix_c, int m, int n, int k) {
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m || col >= n) return;
  std::int32_t accumulator = 0;
  for (int offset = 0; offset + 3 < k; offset += 4) {
    const char4 a = make_char4(matrix_a[row * k + offset], matrix_a[row * k + offset + 1],
                               matrix_a[row * k + offset + 2], matrix_a[row * k + offset + 3]);
    const char4 b = make_char4(matrix_b[offset * n + col], matrix_b[(offset + 1) * n + col],
                               matrix_b[(offset + 2) * n + col], matrix_b[(offset + 3) * n + col]);
    accumulator += amd_mixed_dot(a, b, 0, false);
  }
  for (int offset = k - (k % 4); offset < k; ++offset)
    accumulator += static_cast<std::int32_t>(matrix_a[row * k + offset]) * matrix_b[offset * n + col];
  matrix_c[row * n + col] = accumulator;
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
    std::fprintf(stderr, "INT8 dot4 GEMM smoke: GPU-test-pending (no HIP device)\n");
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

  constexpr int m = 48;
  constexpr int n = 40;
  constexpr int k = 68;  // Deliberately exercises the scalar tail after dot4 groups.
  std::vector<std::int8_t> host_a(static_cast<std::size_t>(m) * k);
  std::vector<std::int8_t> host_b(static_cast<std::size_t>(k) * n);
  std::vector<std::int32_t> host_c(static_cast<std::size_t>(m) * n, 0);
  std::vector<std::int32_t> expected(host_c.size(), 0);
  for (int row = 0; row < m; ++row) {
    for (int offset = 0; offset < k; ++offset) host_a[row * k + offset] = (row + offset) % 7 - 3;
  }
  for (int offset = 0; offset < k; ++offset) {
    for (int col = 0; col < n; ++col) host_b[offset * n + col] = (2 * offset + col) % 5 - 2;
  }
  for (int row = 0; row < m; ++row)
    for (int col = 0; col < n; ++col)
      for (int offset = 0; offset < k; ++offset)
        expected[row * n + col] += static_cast<std::int32_t>(host_a[row * k + offset]) * host_b[offset * n + col];

  std::int8_t* device_a = nullptr;
  std::int8_t* device_b = nullptr;
  std::int32_t* device_c = nullptr;
  error = hipMalloc(reinterpret_cast<void**>(&device_a), host_a.size());
  if (error == hipSuccess) error = hipMalloc(reinterpret_cast<void**>(&device_b), host_b.size());
  if (error == hipSuccess) error = hipMalloc(reinterpret_cast<void**>(&device_c), host_c.size() * sizeof(std::int32_t));
  if (error == hipSuccess) error = hipMemcpy(device_a, host_a.data(), host_a.size(), hipMemcpyHostToDevice);
  if (error == hipSuccess) error = hipMemcpy(device_b, host_b.data(), host_b.size(), hipMemcpyHostToDevice);
  if (error != hipSuccess) {
    hipFree(device_c);
    hipFree(device_b);
    hipFree(device_a);
    return fail("INT8 dot4 GEMM input setup", error);
  }
  const dim3 block(16, 16);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  hipLaunchKernelGGL(int8_dot4_gemm, grid, block, 0, nullptr, device_a, device_b, device_c, m, n, k);
  error = hipGetLastError();
  if (error == hipSuccess) error = hipDeviceSynchronize();
  if (error == hipSuccess)
    error = hipMemcpy(host_c.data(), device_c, host_c.size() * sizeof(std::int32_t), hipMemcpyDeviceToHost);
  hipFree(device_c);
  hipFree(device_b);
  hipFree(device_a);
  if (error != hipSuccess) return fail("INT8 dot4 GEMM execution", error);
  for (std::size_t index = 0; index < host_c.size(); ++index) {
    if (host_c[index] != expected[index]) {
      std::fprintf(stderr, "INT8 dot4 GEMM mismatch at %zu: got=%d expected=%d\n", index,
                   host_c[index], expected[index]);
      return 3;
    }
  }
  std::printf("MI50 packed INT8 dot4 GEMM fallback passed: %s\n", properties.gcnArchName);
  return 0;
}
