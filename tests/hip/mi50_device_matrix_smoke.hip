// Native MI50 device/precision/peer-access smoke.
//
// This is a bring-up test, not a performance benchmark.  It deliberately uses
// ordinary VALU-compatible FP32/FP64 kernels and validates that every visible
// device identifies itself as gfx906 with a wave64 execution width.  Peer
// access is reported when available and can be made mandatory with the
// MI50_REQUIRE_PEER=1 environment variable.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

__global__ void add_float(const float* input, float* output, std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = input[index] + 1.25f;
}

__global__ void add_double(const double* input, double* output, std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = input[index] + 2.5;
}

bool is_gfx906(const hipDeviceProp_t& properties) {
  return std::strstr(properties.gcnArchName, "gfx906") != nullptr;
}

int fail(const char* operation, hipError_t error) {
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(error));
  return 1;
}

bool require_peer() {
  const char* value = std::getenv("MI50_REQUIRE_PEER");
  return value != nullptr && std::strcmp(value, "0") != 0 && std::strcmp(value, "OFF") != 0;
}

}  // namespace

int main() {
  int device_count = 0;
  hipError_t error = hipGetDeviceCount(&device_count);
  if (error != hipSuccess) return fail("hipGetDeviceCount", error);
  if (device_count < 1) {
    std::fprintf(stderr, "no HIP device is visible\n");
    return 2;
  }

  std::vector<hipDeviceProp_t> properties(static_cast<std::size_t>(device_count));
  for (int device = 0; device < device_count; ++device) {
    error = hipGetDeviceProperties(&properties[static_cast<std::size_t>(device)], device);
    if (error != hipSuccess) return fail("hipGetDeviceProperties", error);
    const auto& prop = properties[static_cast<std::size_t>(device)];
    if (!is_gfx906(prop)) {
      std::fprintf(stderr, "device %d is %s, expected native gfx906\n", device, prop.gcnArchName);
      return 3;
    }
    if (prop.warpSize != 64) {
      std::fprintf(stderr, "device %d reports warpSize=%d, expected wave64\n", device,
                   prop.warpSize);
      return 4;
    }
  }

  constexpr std::size_t count = 2048;
  constexpr unsigned int threads = 256;
  const unsigned int blocks = static_cast<unsigned int>((count + threads - 1) / threads);
  std::vector<float> host_float(count);
  std::vector<double> host_double(count);
  std::vector<float> result_float(count, 0.0f);
  std::vector<double> result_double(count, 0.0);
  for (std::size_t index = 0; index < count; ++index) {
    host_float[index] = static_cast<float>(index) * 0.125f;
    host_double[index] = static_cast<double>(index) * 0.25;
  }

  error = hipSetDevice(0);
  if (error != hipSuccess) return fail("hipSetDevice(0)", error);
  float* device_float_in = nullptr;
  float* device_float_out = nullptr;
  double* device_double_in = nullptr;
  double* device_double_out = nullptr;
  error = hipMalloc(reinterpret_cast<void**>(&device_float_in), count * sizeof(float));
  if (error != hipSuccess) return fail("hipMalloc(float input)", error);
  error = hipMalloc(reinterpret_cast<void**>(&device_float_out), count * sizeof(float));
  if (error != hipSuccess) return fail("hipMalloc(float output)", error);
  error = hipMalloc(reinterpret_cast<void**>(&device_double_in), count * sizeof(double));
  if (error != hipSuccess) return fail("hipMalloc(double input)", error);
  error = hipMalloc(reinterpret_cast<void**>(&device_double_out), count * sizeof(double));
  if (error != hipSuccess) return fail("hipMalloc(double output)", error);

  error = hipMemcpy(device_float_in, host_float.data(), count * sizeof(float), hipMemcpyHostToDevice);
  if (error == hipSuccess)
    error = hipMemcpy(device_double_in, host_double.data(), count * sizeof(double), hipMemcpyHostToDevice);
  if (error != hipSuccess) return fail("hipMemcpy precision inputs", error);
  hipLaunchKernelGGL(add_float, dim3(blocks), dim3(threads), 0, nullptr, device_float_in,
                     device_float_out, count);
  error = hipGetLastError();
  if (error == hipSuccess)
    hipLaunchKernelGGL(add_double, dim3(blocks), dim3(threads), 0, nullptr, device_double_in,
                       device_double_out, count);
  if (error != hipSuccess) return fail("precision kernel launch", error);
  error = hipDeviceSynchronize();
  if (error != hipSuccess) return fail("hipDeviceSynchronize", error);
  error = hipMemcpy(result_float.data(), device_float_out, count * sizeof(float), hipMemcpyDeviceToHost);
  if (error == hipSuccess)
    error = hipMemcpy(result_double.data(), device_double_out, count * sizeof(double), hipMemcpyDeviceToHost);
  if (error != hipSuccess) return fail("hipMemcpy precision outputs", error);
  for (std::size_t index = 0; index < count; ++index) {
    if (std::fabs(result_float[index] - (host_float[index] + 1.25f)) > 1e-5f ||
        std::fabs(result_double[index] - (host_double[index] + 2.5)) > 1e-12) {
      std::fprintf(stderr, "precision result mismatch at %zu\n", index);
      return 5;
    }
  }
  hipFree(device_double_out);
  hipFree(device_double_in);
  hipFree(device_float_out);
  hipFree(device_float_in);

  bool peer_observed = false;
  for (int source = 0; source < device_count; ++source) {
    for (int destination = 0; destination < device_count; ++destination) {
      if (source == destination) continue;
      int can_access = 0;
      error = hipDeviceCanAccessPeer(&can_access, destination, source);
      if (error != hipSuccess) return fail("hipDeviceCanAccessPeer", error);
      if (!can_access) continue;
      peer_observed = true;
      error = hipSetDevice(destination);
      if (error != hipSuccess) return fail("hipSetDevice(peer destination)", error);
      error = hipDeviceEnablePeerAccess(source, 0);
      if (error != hipSuccess && error != hipErrorPeerAccessAlreadyEnabled)
        return fail("hipDeviceEnablePeerAccess", error);
      std::printf("peer access available: source=%d destination=%d\n", source, destination);
    }
  }
  if (device_count > 1 && require_peer() && !peer_observed) {
    std::fprintf(stderr, "no peer-access path found for a multi-device system\n");
    return 6;
  }
  std::printf("MI50 device matrix smoke passed: devices=%d peer_access=%s\n", device_count,
              peer_observed ? "available" : "not-reported");
  return 0;
}
