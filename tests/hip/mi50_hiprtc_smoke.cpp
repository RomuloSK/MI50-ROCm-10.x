// Native HIPRTC compilation and module-launch smoke for MI50/gfx906.
//
// The program compiles a small kernel to a real gfx906 code object before it
// checks for /dev/kfd.  This makes the compiler/COMGR part testable on a
// pre-hardware host while keeping module execution explicitly pending.

#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr const char* kSource = R"HIP(
extern "C" __global__ void mi50_rtc_add(const float* input, float* output,
                                        unsigned long long count) {
  unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = input[index] + 3.0f;
}
)HIP";

int fail_hip(const char* operation, hipError_t error) {
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(error));
  return 1;
}

int fail_rtc(const char* operation, hiprtcResult result, hiprtcProgram program = nullptr) {
  std::fprintf(stderr, "%s failed: %s\n", operation, hiprtcGetErrorString(result));
  if (program != nullptr) {
    size_t log_size = 0;
    if (hiprtcGetProgramLogSize(program, &log_size) == HIPRTC_SUCCESS && log_size > 1) {
      std::vector<char> log(log_size, '\0');
      if (hiprtcGetProgramLog(program, log.data()) == HIPRTC_SUCCESS)
        std::fprintf(stderr, "hipRTC log:\n%s\n", log.data());
    }
  }
  return 1;
}

bool is_gfx906(const hipDeviceProp_t& properties) {
  return std::strstr(properties.gcnArchName, "gfx906") != nullptr && properties.warpSize == 64;
}

}  // namespace

int main() {
  hiprtcProgram program = nullptr;
  hiprtcResult rtc_result = hiprtcCreateProgram(&program, kSource, "mi50_rtc_add.hip", 0,
                                                nullptr, nullptr);
  if (rtc_result != HIPRTC_SUCCESS) return fail_rtc("hiprtcCreateProgram", rtc_result);
  const char* options[] = {"--gpu-architecture=gfx906"};
  rtc_result = hiprtcCompileProgram(program, 1, options);
  if (rtc_result != HIPRTC_SUCCESS) {
    const int result = fail_rtc("hiprtcCompileProgram(gfx906)", rtc_result, program);
    hiprtcDestroyProgram(&program);
    return result;
  }
  size_t code_size = 0;
  rtc_result = hiprtcGetCodeSize(program, &code_size);
  if (rtc_result != HIPRTC_SUCCESS || code_size == 0) {
    const int result = rtc_result == HIPRTC_SUCCESS
                           ? (std::fprintf(stderr, "hipRTC returned an empty gfx906 code object\n"), 1)
                           : fail_rtc("hiprtcGetCodeSize", rtc_result, program);
    hiprtcDestroyProgram(&program);
    return result;
  }
  std::vector<char> code(code_size, '\0');
  rtc_result = hiprtcGetCode(program, code.data());
  if (rtc_result != HIPRTC_SUCCESS) {
    const int result = fail_rtc("hiprtcGetCode", rtc_result, program);
    hiprtcDestroyProgram(&program);
    return result;
  }
  std::printf("HIPRTC compiled native gfx906 code object (%zu bytes)\n", code_size);
  hiprtcDestroyProgram(&program);

  int device_count = 0;
  hipError_t error = hipGetDeviceCount(&device_count);
  if (error == hipErrorNoDevice || device_count < 1) {
    std::fprintf(stderr, "HIPRTC module smoke: GPU-test-pending (no HIP device)\n");
    return 77;
  }
  if (error != hipSuccess) return fail_hip("hipGetDeviceCount", error);
  hipDeviceProp_t properties{};
  error = hipGetDeviceProperties(&properties, 0);
  if (error != hipSuccess) return fail_hip("hipGetDeviceProperties", error);
  if (!is_gfx906(properties)) {
    std::fprintf(stderr, "device 0 is %s/wave%d, expected gfx906/wave64\n", properties.gcnArchName,
                 properties.warpSize);
    return 2;
  }

  hipModule_t module = nullptr;
  hipFunction_t function = nullptr;
  error = hipModuleLoadData(&module, code.data());
  if (error == hipSuccess) error = hipModuleGetFunction(&function, module, "mi50_rtc_add");
  constexpr unsigned long long count = 1024;
  std::vector<float> host_input(count);
  std::vector<float> host_output(count, 0.0f);
  float* device_input = nullptr;
  float* device_output = nullptr;
  if (error == hipSuccess) error = hipMalloc(reinterpret_cast<void**>(&device_input), sizeof(float) * count);
  if (error == hipSuccess) error = hipMalloc(reinterpret_cast<void**>(&device_output), sizeof(float) * count);
  for (unsigned long long index = 0; index < count; ++index) host_input[index] = index * 0.5f;
  if (error == hipSuccess)
    error = hipMemcpy(device_input, host_input.data(), sizeof(float) * count, hipMemcpyHostToDevice);
  unsigned long long kernel_count = count;
  void* arguments[] = {&device_input, &device_output, &kernel_count};
  if (error == hipSuccess)
    error = hipModuleLaunchKernel(function, (count + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr,
                                  arguments, nullptr);
  if (error == hipSuccess) error = hipDeviceSynchronize();
  if (error == hipSuccess)
    error = hipMemcpy(host_output.data(), device_output, sizeof(float) * count, hipMemcpyDeviceToHost);
  if (error != hipSuccess) {
    hipFree(device_output);
    hipFree(device_input);
    if (module != nullptr) hipModuleUnload(module);
    return fail_hip("HIPRTC module launch", error);
  }
  for (unsigned long long index = 0; index < count; ++index) {
    if (std::fabs(host_output[index] - (host_input[index] + 3.0f)) > 1e-5f) {
      std::fprintf(stderr, "HIPRTC result mismatch at %llu\n", index);
      hipFree(device_output);
      hipFree(device_input);
      hipModuleUnload(module);
      return 3;
    }
  }
  hipFree(device_output);
  hipFree(device_input);
  hipModuleUnload(module);
  std::printf("MI50 HIPRTC module smoke passed: %s\n", properties.gcnArchName);
  return 0;
}
