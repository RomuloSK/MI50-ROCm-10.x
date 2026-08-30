// Native RCCL all-reduce smoke for a homogeneous MI50/gfx906 system.
//
// The test intentionally requires two visible devices: a single-card result
// cannot prove the peer path needed for dual-card inference. It returns 77
// when the hardware tier is not available, so pre-hardware CI stays explicit.

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int fail_hip(const char* operation, hipError_t error) {
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(error));
  return 1;
}

int fail_nccl(const char* operation, ncclResult_t error) {
  std::fprintf(stderr, "%s failed: %s\n", operation, ncclGetErrorString(error));
  return 1;
}

bool is_gfx906_wave64(const hipDeviceProp_t& properties) {
  return std::strstr(properties.gcnArchName, "gfx906") != nullptr && properties.warpSize == 64;
}

}  // namespace

int main() {
  int device_count = 0;
  hipError_t hip_error = hipGetDeviceCount(&device_count);
  if (hip_error != hipSuccess) return fail_hip("hipGetDeviceCount", hip_error);
  if (device_count < 2) {
    std::fprintf(stderr, "RCCL smoke: two native gfx906 devices are required (found %d)\n",
                 device_count);
    return 77;
  }

  std::vector<hipDeviceProp_t> properties(static_cast<std::size_t>(device_count));
  for (int device = 0; device < device_count; ++device) {
    hip_error = hipGetDeviceProperties(&properties[static_cast<std::size_t>(device)], device);
    if (hip_error != hipSuccess) return fail_hip("hipGetDeviceProperties", hip_error);
    if (!is_gfx906_wave64(properties[static_cast<std::size_t>(device)])) {
      std::fprintf(stderr, "device %d is %s/wave%d, expected gfx906/wave64\n", device,
                   properties[static_cast<std::size_t>(device)].gcnArchName,
                   properties[static_cast<std::size_t>(device)].warpSize);
      return 2;
    }
  }

  ncclUniqueId unique_id{};
  ncclResult_t nccl_error = ncclGetUniqueId(&unique_id);
  if (nccl_error != ncclSuccess) return fail_nccl("ncclGetUniqueId", nccl_error);

  std::vector<ncclComm_t> communicators(static_cast<std::size_t>(device_count));
  nccl_error = ncclGroupStart();
  if (nccl_error != ncclSuccess) return fail_nccl("ncclGroupStart(init)", nccl_error);
  for (int rank = 0; rank < device_count; ++rank) {
    nccl_error = ncclCommInitRank(&communicators[static_cast<std::size_t>(rank)], device_count,
                                  unique_id, rank);
    if (nccl_error != ncclSuccess) {
      ncclGroupEnd();
      return fail_nccl("ncclCommInitRank", nccl_error);
    }
  }
  nccl_error = ncclGroupEnd();
  if (nccl_error != ncclSuccess) return fail_nccl("ncclGroupEnd(init)", nccl_error);

  constexpr std::size_t count = 4096;
  std::vector<float> host_input(count);
  std::vector<float> host_output(count, 0.0f);
  std::vector<float*> device_input(static_cast<std::size_t>(device_count), nullptr);
  std::vector<float*> device_output(static_cast<std::size_t>(device_count), nullptr);
  std::vector<hipStream_t> streams(static_cast<std::size_t>(device_count), nullptr);
  for (std::size_t index = 0; index < count; ++index) host_input[index] = 1.0f + index * 0.001f;

  for (int rank = 0; rank < device_count; ++rank) {
    hip_error = hipSetDevice(rank);
    if (hip_error == hipSuccess)
      hip_error = hipStreamCreate(&streams[static_cast<std::size_t>(rank)]);
    if (hip_error == hipSuccess)
      hip_error = hipMalloc(reinterpret_cast<void**>(&device_input[static_cast<std::size_t>(rank)]),
                            count * sizeof(float));
    if (hip_error == hipSuccess)
      hip_error = hipMalloc(reinterpret_cast<void**>(&device_output[static_cast<std::size_t>(rank)]),
                            count * sizeof(float));
    if (hip_error == hipSuccess)
      hip_error = hipMemcpyAsync(device_input[static_cast<std::size_t>(rank)], host_input.data(),
                                 count * sizeof(float), hipMemcpyHostToDevice,
                                 streams[static_cast<std::size_t>(rank)]);
    if (hip_error != hipSuccess) return fail_hip("RCCL input setup", hip_error);
  }

  nccl_error = ncclGroupStart();
  if (nccl_error != ncclSuccess) return fail_nccl("ncclGroupStart(allreduce)", nccl_error);
  for (int rank = 0; rank < device_count; ++rank) {
    nccl_error = ncclAllReduce(
        device_input[static_cast<std::size_t>(rank)], device_output[static_cast<std::size_t>(rank)],
        count, ncclFloat32, ncclSum, communicators[static_cast<std::size_t>(rank)],
        streams[static_cast<std::size_t>(rank)]);
    if (nccl_error != ncclSuccess) {
      ncclGroupEnd();
      return fail_nccl("ncclAllReduce", nccl_error);
    }
  }
  nccl_error = ncclGroupEnd();
  if (nccl_error != ncclSuccess) return fail_nccl("ncclGroupEnd(allreduce)", nccl_error);

  const float expected_scale = static_cast<float>(device_count);
  for (int rank = 0; rank < device_count; ++rank) {
    hip_error = hipSetDevice(rank);
    if (hip_error == hipSuccess)
      hip_error = hipStreamSynchronize(streams[static_cast<std::size_t>(rank)]);
    if (hip_error == hipSuccess)
      hip_error = hipMemcpy(host_output.data(), device_output[static_cast<std::size_t>(rank)],
                            count * sizeof(float), hipMemcpyDeviceToHost);
    if (hip_error != hipSuccess) return fail_hip("RCCL output synchronization", hip_error);
    for (std::size_t index = 0; index < count; ++index) {
      const float expected = host_input[index] * expected_scale;
      if (std::fabs(host_output[index] - expected) > 1e-4f * expected_scale) {
        std::fprintf(stderr, "all-reduce mismatch rank=%d index=%zu got=%f expected=%f\n", rank,
                     index, host_output[index], expected);
        return 3;
      }
    }
  }

  for (int rank = 0; rank < device_count; ++rank) {
    hipSetDevice(rank);
    hipFree(device_output[static_cast<std::size_t>(rank)]);
    hipFree(device_input[static_cast<std::size_t>(rank)]);
    hipStreamDestroy(streams[static_cast<std::size_t>(rank)]);
    ncclCommDestroy(communicators[static_cast<std::size_t>(rank)]);
  }
  std::printf("MI50 RCCL all-reduce smoke passed: devices=%d\n", device_count);
  return 0;
}
