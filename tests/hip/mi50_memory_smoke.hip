// Bounded native memory/allocator smoke for MI50/gfx906.
//
// The default allocation is deliberately modest; set MI50_MEMORY_TEST_MIB
// after bring-up to exercise a larger fraction of each card without consuming
// all VRAM needed by the runtime or the model under test.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int fail(const char* operation, hipError_t error) {
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(error));
  return 1;
}

std::size_t requested_bytes() {
  constexpr std::size_t default_mib = 256;
  constexpr std::size_t maximum_mib = 1024;
  const char* value = std::getenv("MI50_MEMORY_TEST_MIB");
  if (value == nullptr || *value == '\0') return default_mib * 1024 * 1024;
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0 || parsed > maximum_mib) {
    std::fprintf(stderr, "MI50_MEMORY_TEST_MIB must be 1..%zu (got %s)\n", maximum_mib, value);
    return 0;
  }
  return static_cast<std::size_t>(parsed) * 1024 * 1024;
}

}  // namespace

int main() {
  const std::size_t requested = requested_bytes();
  if (requested == 0) return 2;
  int device_count = 0;
  hipError_t error = hipGetDeviceCount(&device_count);
  if (error == hipErrorNoDevice || device_count < 1) {
    std::fprintf(stderr, "memory smoke: GPU-test-pending (no HIP device)\n");
    return 77;
  }
  if (error != hipSuccess) return fail("hipGetDeviceCount", error);

  constexpr std::size_t probe_bytes = 4096;
  std::vector<std::uint8_t> host_probe(probe_bytes);
  for (int device = 0; device < device_count; ++device) {
    error = hipSetDevice(device);
    if (error != hipSuccess) return fail("hipSetDevice", error);
    hipDeviceProp_t properties{};
    error = hipGetDeviceProperties(&properties, device);
    if (error != hipSuccess) return fail("hipGetDeviceProperties", error);
    if (std::strstr(properties.gcnArchName, "gfx906") == nullptr || properties.warpSize != 64) {
      std::fprintf(stderr, "device %d is %s/wave%d, expected gfx906/wave64\n", device,
                   properties.gcnArchName, properties.warpSize);
      return 3;
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    error = hipMemGetInfo(&free_bytes, &total_bytes);
    if (error != hipSuccess) return fail("hipMemGetInfo", error);
    // Leave at least 75% of currently free VRAM available to the inference
    // process and cap the test to the user-selected upper bound.
    const std::size_t safe_bytes = std::min(requested, free_bytes / 4);
    if (safe_bytes < probe_bytes) {
      std::fprintf(stderr, "device %d has too little free VRAM for the probe (%zu bytes)\n", device,
                   free_bytes);
      return 4;
    }
    std::uint8_t* allocation = nullptr;
    error = hipMalloc(reinterpret_cast<void**>(&allocation), safe_bytes);
    if (error != hipSuccess) return fail("hipMalloc", error);
    error = hipMemset(allocation, 0x5a, safe_bytes);
    if (error == hipSuccess) error = hipMemcpy(host_probe.data(), allocation, probe_bytes, hipMemcpyDeviceToHost);
    if (error != hipSuccess) {
      hipFree(allocation);
      return fail("memory probe read", error);
    }
    if (!std::all_of(host_probe.begin(), host_probe.end(), [](std::uint8_t value) { return value == 0x5a; })) {
      hipFree(allocation);
      std::fprintf(stderr, "device %d returned an incorrect memset pattern\n", device);
      return 5;
    }
    std::fill(host_probe.begin(), host_probe.end(), 0xa5);
    error = hipMemcpy(allocation + safe_bytes - probe_bytes, host_probe.data(), probe_bytes,
                      hipMemcpyHostToDevice);
    if (error == hipSuccess) error = hipDeviceSynchronize();
    hipError_t free_error = hipFree(allocation);
    if (error != hipSuccess) return fail("memory probe write", error);
    if (free_error != hipSuccess) return fail("hipFree", free_error);
    std::printf("MI50 memory smoke passed: device=%d arch=%s tested=%zuMiB free=%zuMiB total=%zuMiB\n",
                device, properties.gcnArchName, safe_bytes / (1024 * 1024),
                free_bytes / (1024 * 1024), total_bytes / (1024 * 1024));
  }
  return 0;
}
