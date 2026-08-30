// Native hipBLAS SGEMM correctness smoke for MI50/gfx906.
//
// hipBLAS is expected to route this operation to the mature rocBLAS/Tensile
// backend.  The test intentionally uses only FP32 and does not opt into
// hipBLASLt, matrix cores, BF16 or FP8.

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int fail_hip(const char* operation, hipError_t error) {
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(error));
  return 1;
}

int fail_hipblas(const char* operation, hipblasStatus_t status) {
  std::fprintf(stderr, "%s failed: hipBLAS status=%s\n", operation,
               hipblasStatusToString(status));
  return 1;
}

bool is_gfx906_wave64(const hipDeviceProp_t& properties) {
  return std::strstr(properties.gcnArchName, "gfx906") != nullptr && properties.warpSize == 64;
}

}  // namespace

int main() {
  int device_count = 0;
  hipError_t hip_error = hipGetDeviceCount(&device_count);
  if (hip_error == hipErrorNoDevice || device_count < 1) {
    std::fprintf(stderr, "hipBLAS smoke: GPU-test-pending (no HIP device)\n");
    return 77;
  }
  if (hip_error != hipSuccess) return fail_hip("hipGetDeviceCount", hip_error);
  hipDeviceProp_t properties{};
  hip_error = hipGetDeviceProperties(&properties, 0);
  if (hip_error != hipSuccess) return fail_hip("hipGetDeviceProperties", hip_error);
  if (!is_gfx906_wave64(properties)) {
    std::fprintf(stderr, "device 0 is %s/wave%d, expected gfx906/wave64\n", properties.gcnArchName,
                 properties.warpSize);
    return 2;
  }

  constexpr int m = 32;
  constexpr int n = 32;
  constexpr int k = 32;
  std::vector<float> host_a(static_cast<std::size_t>(m) * k, 1.0f);
  std::vector<float> host_b(static_cast<std::size_t>(k) * n, 2.0f);
  std::vector<float> host_c(static_cast<std::size_t>(m) * n, 0.0f);
  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;
  hip_error = hipMalloc(reinterpret_cast<void**>(&device_a), host_a.size() * sizeof(float));
  if (hip_error == hipSuccess)
    hip_error = hipMalloc(reinterpret_cast<void**>(&device_b), host_b.size() * sizeof(float));
  if (hip_error == hipSuccess)
    hip_error = hipMalloc(reinterpret_cast<void**>(&device_c), host_c.size() * sizeof(float));
  if (hip_error == hipSuccess)
    hip_error = hipMemcpy(device_a, host_a.data(), host_a.size() * sizeof(float), hipMemcpyHostToDevice);
  if (hip_error == hipSuccess)
    hip_error = hipMemcpy(device_b, host_b.data(), host_b.size() * sizeof(float), hipMemcpyHostToDevice);
  if (hip_error != hipSuccess) {
    hipFree(device_c);
    hipFree(device_b);
    hipFree(device_a);
    return fail_hip("hipBLAS input setup", hip_error);
  }

  hipblasHandle_t handle = nullptr;
  hipblasStatus_t status = hipblasCreate(&handle);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  if (status == HIPBLAS_STATUS_SUCCESS)
    status = hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha, device_a, m,
                           device_b, k, &beta, device_c, m);
  if (status == HIPBLAS_STATUS_SUCCESS) hip_error = hipDeviceSynchronize();
  if (status == HIPBLAS_STATUS_SUCCESS && hip_error == hipSuccess)
    hip_error = hipMemcpy(host_c.data(), device_c, host_c.size() * sizeof(float), hipMemcpyDeviceToHost);
  if (handle != nullptr) hipblasDestroy(handle);
  hipFree(device_c);
  hipFree(device_b);
  hipFree(device_a);
  if (status != HIPBLAS_STATUS_SUCCESS) return fail_hipblas("hipblasSgemm", status);
  if (hip_error != hipSuccess) return fail_hip("hipBLAS output", hip_error);
  for (std::size_t index = 0; index < host_c.size(); ++index) {
    if (std::fabs(host_c[index] - 2.0f * k) > 1e-4f) {
      std::fprintf(stderr, "hipBLAS mismatch at %zu: got=%f expected=%f\n", index, host_c[index],
                   2.0f * k);
      return 3;
    }
  }
  std::printf("MI50 hipBLAS SGEMM passed via the native gfx906 path: %s\n", properties.gcnArchName);
  return 0;
}
