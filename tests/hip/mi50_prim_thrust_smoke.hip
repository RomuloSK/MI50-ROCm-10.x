// Native rocPRIM/rocThrust reduction smoke for MI50/gfx906.

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

int fail_hip(const char* operation, hipError_t error) {
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(error));
  return 1;
}

}  // namespace

int main() {
  int device_count = 0;
  hipError_t hip_error = hipGetDeviceCount(&device_count);
  if (hip_error == hipErrorNoDevice || device_count < 1) {
    std::fprintf(stderr, "PRIM/Thrust smoke: GPU-test-pending (no HIP device)\n");
    return 77;
  }
  if (hip_error != hipSuccess) return fail_hip("hipGetDeviceCount", hip_error);
  hipDeviceProp_t properties{};
  hip_error = hipGetDeviceProperties(&properties, 0);
  if (hip_error != hipSuccess) return fail_hip("hipGetDeviceProperties", hip_error);
  if (std::strstr(properties.gcnArchName, "gfx906") == nullptr || properties.warpSize != 64) {
    std::fprintf(stderr, "device is %s/wave%d, expected gfx906/wave64\n", properties.gcnArchName,
                 properties.warpSize);
    return 2;
  }

  constexpr std::size_t count = 1024;
  std::vector<int> host_input(count);
  for (std::size_t index = 0; index < count; ++index) host_input[index] = static_cast<int>(index % 17);
  const int expected = std::accumulate(host_input.begin(), host_input.end(), 0);
  int* input = nullptr;
  int* output = nullptr;
  hip_error = hipMalloc(reinterpret_cast<void**>(&input), count * sizeof(int));
  if (hip_error == hipSuccess) hip_error = hipMalloc(reinterpret_cast<void**>(&output), sizeof(int));
  if (hip_error == hipSuccess)
    hip_error = hipMemcpy(input, host_input.data(), count * sizeof(int), hipMemcpyHostToDevice);
  if (hip_error != hipSuccess) return fail_hip("rocPRIM input setup", hip_error);

  size_t temporary_size = 0;
  hip_error = rocprim::reduce(nullptr, temporary_size, input, output, count, rocprim::plus<int>());
  void* temporary = nullptr;
  if (hip_error == hipSuccess && temporary_size > 0) hip_error = hipMalloc(&temporary, temporary_size);
  if (hip_error == hipSuccess)
    hip_error = rocprim::reduce(temporary, temporary_size, input, output, count, rocprim::plus<int>());
  int prim_result = 0;
  if (hip_error == hipSuccess) hip_error = hipMemcpy(&prim_result, output, sizeof(prim_result), hipMemcpyDeviceToHost);
  if (temporary != nullptr) hipFree(temporary);
  hipFree(output);
  if (hip_error != hipSuccess) {
    hipFree(input);
    return fail_hip("rocPRIM reduce", hip_error);
  }
  if (prim_result != expected) {
    std::fprintf(stderr, "rocPRIM mismatch: got=%d expected=%d\n", prim_result, expected);
    hipFree(input);
    return 3;
  }

  thrust::device_ptr<int> device_begin = thrust::device_pointer_cast(input);
  int thrust_result = thrust::reduce(device_begin, device_begin + count, 0, thrust::plus<int>());
  hip_error = hipDeviceSynchronize();
  hipFree(input);
  if (hip_error != hipSuccess) return fail_hip("rocThrust synchronize", hip_error);
  if (thrust_result != expected) {
    std::fprintf(stderr, "rocThrust mismatch: got=%d expected=%d\n", thrust_result, expected);
    return 4;
  }
  std::printf("MI50 rocPRIM/rocThrust smoke passed: %s\n", properties.gcnArchName);
  return 0;
}
