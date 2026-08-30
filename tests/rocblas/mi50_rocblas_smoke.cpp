// Native rocBLAS FP32/FP64 correctness smoke for MI50/gfx906.
//
// This covers the supported Tensile path without assuming matrix cores,
// BF16, FP8 or hipBLASLt. It returns 77 when no device is visible so the
// source can be compiled and staged before the cards arrive.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int fail_hip(const char* operation, hipError_t error) {
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(error));
  return 1;
}

int fail_rocblas(const char* operation, rocblas_status status) {
  std::fprintf(stderr, "%s failed: rocBLAS status=%d\n", operation, static_cast<int>(status));
  return 1;
}

bool is_gfx906_wave64(const hipDeviceProp_t& properties) {
  return std::strstr(properties.gcnArchName, "gfx906") != nullptr && properties.warpSize == 64;
}

template <typename Scalar, typename Status, Status (*Gemm)(rocblas_handle, rocblas_operation,
                                                             rocblas_operation, rocblas_int,
                                                             rocblas_int, rocblas_int, const Scalar*,
                                                             const Scalar*, rocblas_int, const Scalar*,
                                                             rocblas_int, const Scalar*, Scalar*,
                                                             rocblas_int)> 
int run_gemm(rocblas_handle handle, const char* label) {
  constexpr rocblas_int m = 32;
  constexpr rocblas_int n = 32;
  constexpr rocblas_int k = 32;
  constexpr rocblas_int lda = m;
  constexpr rocblas_int ldb = k;
  constexpr rocblas_int ldc = m;
  std::vector<Scalar> host_a(static_cast<std::size_t>(m) * k, Scalar(1));
  std::vector<Scalar> host_b(static_cast<std::size_t>(k) * n, Scalar(2));
  std::vector<Scalar> host_c(static_cast<std::size_t>(m) * n, Scalar(0));
  Scalar* device_a = nullptr;
  Scalar* device_b = nullptr;
  Scalar* device_c = nullptr;
  hipError_t hip_error = hipMalloc(reinterpret_cast<void**>(&device_a), host_a.size() * sizeof(Scalar));
  if (hip_error == hipSuccess)
    hip_error = hipMalloc(reinterpret_cast<void**>(&device_b), host_b.size() * sizeof(Scalar));
  if (hip_error == hipSuccess)
    hip_error = hipMalloc(reinterpret_cast<void**>(&device_c), host_c.size() * sizeof(Scalar));
  if (hip_error == hipSuccess)
    hip_error = hipMemcpy(device_a, host_a.data(), host_a.size() * sizeof(Scalar), hipMemcpyHostToDevice);
  if (hip_error == hipSuccess)
    hip_error = hipMemcpy(device_b, host_b.data(), host_b.size() * sizeof(Scalar), hipMemcpyHostToDevice);
  if (hip_error != hipSuccess) {
    hipFree(device_c);
    hipFree(device_b);
    hipFree(device_a);
    return fail_hip("rocBLAS input setup", hip_error);
  }

  const Scalar alpha = Scalar(1);
  const Scalar beta = Scalar(0);
  Status status = Gemm(handle, rocblas_operation_none, rocblas_operation_none, m, n, k, &alpha,
                       device_a, lda, device_b, ldb, &beta, device_c, ldc);
  if (status == rocblas_status_success) {
    hip_error = hipDeviceSynchronize();
    if (hip_error == hipSuccess)
      hip_error = hipMemcpy(host_c.data(), device_c, host_c.size() * sizeof(Scalar), hipMemcpyDeviceToHost);
  }
  hipFree(device_c);
  hipFree(device_b);
  hipFree(device_a);
  if (status != rocblas_status_success) return fail_rocblas(label, status);
  if (hip_error != hipSuccess) return fail_hip("rocBLAS output synchronization", hip_error);

  const Scalar expected = Scalar(2 * k);
  for (std::size_t index = 0; index < host_c.size(); ++index) {
    if (std::fabs(static_cast<double>(host_c[index] - expected)) > 1e-4 * static_cast<double>(expected)) {
      std::fprintf(stderr, "%s mismatch at %zu: got=%g expected=%g\n", label, index,
                   static_cast<double>(host_c[index]), static_cast<double>(expected));
      return 2;
    }
  }
  std::printf("rocBLAS %s passed\n", label);
  return 0;
}

int run_half_gemm(rocblas_handle handle) {
  constexpr rocblas_int m = 32;
  constexpr rocblas_int n = 32;
  constexpr rocblas_int k = 32;
  constexpr rocblas_int lda = m;
  constexpr rocblas_int ldb = k;
  constexpr rocblas_int ldc = m;
  using HostHalf = half;
  std::vector<HostHalf> host_a(static_cast<std::size_t>(m) * k, HostHalf(1.0f));
  std::vector<HostHalf> host_b(static_cast<std::size_t>(k) * n, HostHalf(2.0f));
  std::vector<HostHalf> host_c(static_cast<std::size_t>(m) * n, HostHalf(0.0f));
  HostHalf* device_a = nullptr;
  HostHalf* device_b = nullptr;
  HostHalf* device_c = nullptr;
  hipError_t hip_error = hipMalloc(reinterpret_cast<void**>(&device_a), host_a.size() * sizeof(HostHalf));
  if (hip_error == hipSuccess)
    hip_error = hipMalloc(reinterpret_cast<void**>(&device_b), host_b.size() * sizeof(HostHalf));
  if (hip_error == hipSuccess)
    hip_error = hipMalloc(reinterpret_cast<void**>(&device_c), host_c.size() * sizeof(HostHalf));
  if (hip_error == hipSuccess)
    hip_error = hipMemcpy(device_a, host_a.data(), host_a.size() * sizeof(HostHalf), hipMemcpyHostToDevice);
  if (hip_error == hipSuccess)
    hip_error = hipMemcpy(device_b, host_b.data(), host_b.size() * sizeof(HostHalf), hipMemcpyHostToDevice);
  if (hip_error != hipSuccess) {
    hipFree(device_c);
    hipFree(device_b);
    hipFree(device_a);
    return fail_hip("rocBLAS hgemm input setup", hip_error);
  }
  const HostHalf alpha = HostHalf(1.0f);
  const HostHalf beta = HostHalf(0.0f);
  rocblas_status status = rocblas_hgemm(
      handle, rocblas_operation_none, rocblas_operation_none, m, n, k,
      reinterpret_cast<const rocblas_half*>(&alpha), reinterpret_cast<const rocblas_half*>(device_a), lda,
      reinterpret_cast<const rocblas_half*>(device_b), ldb, reinterpret_cast<const rocblas_half*>(&beta),
      reinterpret_cast<rocblas_half*>(device_c), ldc);
  if (status == rocblas_status_success) hip_error = hipDeviceSynchronize();
  if (status == rocblas_status_success && hip_error == hipSuccess)
    hip_error = hipMemcpy(host_c.data(), device_c, host_c.size() * sizeof(HostHalf), hipMemcpyDeviceToHost);
  hipFree(device_c);
  hipFree(device_b);
  hipFree(device_a);
  if (status != rocblas_status_success) return fail_rocblas("rocblas_hgemm", status);
  if (hip_error != hipSuccess) return fail_hip("rocBLAS hgemm output", hip_error);
  const float expected = static_cast<float>(2 * k);
  for (std::size_t index = 0; index < host_c.size(); ++index) {
    const float observed = static_cast<float>(host_c[index]);
    if (std::fabs(observed - expected) > 0.05f) {
      std::fprintf(stderr, "hgemm mismatch at %zu: got=%f expected=%f\n", index, observed, expected);
      return 2;
    }
  }
  std::printf("rocBLAS hgemm (FP16) passed\n");
  return 0;
}

}  // namespace

int main() {
  int device_count = 0;
  hipError_t hip_error = hipGetDeviceCount(&device_count);
  if (hip_error == hipErrorNoDevice) {
    std::fprintf(stderr, "rocBLAS smoke: GPU-test-pending (no HIP device)\n");
    return 77;
  }
  if (hip_error != hipSuccess) return fail_hip("hipGetDeviceCount", hip_error);
  if (device_count < 1) {
    std::fprintf(stderr, "rocBLAS smoke: GPU-test-pending (no HIP device)\n");
    return 77;
  }
  hipDeviceProp_t properties{};
  hip_error = hipGetDeviceProperties(&properties, 0);
  if (hip_error != hipSuccess) return fail_hip("hipGetDeviceProperties", hip_error);
  if (!is_gfx906_wave64(properties)) {
    std::fprintf(stderr, "device 0 is %s/wave%d, expected gfx906/wave64\n", properties.gcnArchName,
                 properties.warpSize);
    return 3;
  }

  rocblas_handle handle = nullptr;
  rocblas_status status = rocblas_create_handle(&handle);
  if (status != rocblas_status_success) return fail_rocblas("rocblas_create_handle", status);
  int result = run_half_gemm(handle);
  if (result == 0) result = run_gemm<float, rocblas_status, rocblas_sgemm>(handle, "sgemm");
  if (result == 0) result = run_gemm<double, rocblas_status, rocblas_dgemm>(handle, "dgemm");
  rocblas_destroy_handle(handle);
  return result;
}
