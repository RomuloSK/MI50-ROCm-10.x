// Native rocFFT/rocRAND correctness smoke for MI50/gfx906.
//
// The FFT uses an impulse (all frequency bins should be one); rocRAND output
// is checked for the documented [0,1) range. Return 77 before hardware.

#include <hip/hip_runtime.h>
#include <rocfft/rocfft.h>
#include <rocrand/rocrand.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int fail_hip(const char* operation, hipError_t error) {
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(error));
  return 1;
}

template <typename Status>
int fail_status(const char* operation, Status status) {
  std::fprintf(stderr, "%s failed: status=%d\n", operation, static_cast<int>(status));
  return 1;
}

}  // namespace

int main() {
  int device_count = 0;
  hipError_t hip_error = hipGetDeviceCount(&device_count);
  if (hip_error == hipErrorNoDevice || device_count < 1) {
    std::fprintf(stderr, "FFT/RAND smoke: GPU-test-pending (no HIP device)\n");
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

  constexpr std::size_t sample_count = 256;
  float* random_device = nullptr;
  hip_error = hipMalloc(reinterpret_cast<void**>(&random_device), sample_count * sizeof(float));
  if (hip_error != hipSuccess) return fail_hip("rocRAND hipMalloc", hip_error);
  rocrand_generator generator = nullptr;
  rocrand_status random_status = rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT);
  if (random_status == ROCRAND_STATUS_SUCCESS)
    random_status = rocrand_set_seed(generator, 0x4d493530ULL);
  if (random_status == ROCRAND_STATUS_SUCCESS)
    random_status = rocrand_generate_uniform(generator, random_device, sample_count);
  if (random_status == ROCRAND_STATUS_SUCCESS) hip_error = hipDeviceSynchronize();
  std::vector<float> random_host(sample_count, 0.0f);
  if (random_status == ROCRAND_STATUS_SUCCESS && hip_error == hipSuccess)
    hip_error = hipMemcpy(random_host.data(), random_device, sample_count * sizeof(float),
                          hipMemcpyDeviceToHost);
  if (generator != nullptr) rocrand_destroy_generator(generator);
  hipFree(random_device);
  if (random_status != ROCRAND_STATUS_SUCCESS) return fail_status("rocRAND uniform", random_status);
  if (hip_error != hipSuccess) return fail_hip("rocRAND output", hip_error);
  for (std::size_t index = 0; index < random_host.size(); ++index) {
    if (!std::isfinite(random_host[index]) || random_host[index] < 0.0f || random_host[index] >= 1.0f) {
      std::fprintf(stderr, "rocRAND value out of range at %zu: %f\n", index, random_host[index]);
      return 3;
    }
  }

  rocfft_status fft_status = rocfft_setup();
  if (fft_status != rocfft_status_success) return fail_status("rocfft_setup", fft_status);
  rocfft_plan plan = nullptr;
  rocfft_plan_description description = nullptr;
  rocfft_execution_info execution_info = nullptr;
  size_t lengths[] = {sample_count};
  fft_status = rocfft_plan_description_create(&description);
  if (fft_status == rocfft_status_success)
    fft_status = rocfft_plan_create(&plan, rocfft_placement_inplace,
                                    rocfft_transform_type_complex_forward, rocfft_precision_single,
                                    1, lengths, 1, description);
  if (fft_status == rocfft_status_success)
    fft_status = rocfft_execution_info_create(&execution_info);

  std::vector<float2> fft_host(sample_count, float2{0.0f, 0.0f});
  fft_host[0].x = 1.0f;
  float2* fft_device = nullptr;
  if (fft_status == rocfft_status_success)
    hip_error = hipMalloc(reinterpret_cast<void**>(&fft_device), sample_count * sizeof(float2));
  if (fft_status == rocfft_status_success && hip_error == hipSuccess)
    hip_error = hipMemcpy(fft_device, fft_host.data(), sample_count * sizeof(float2), hipMemcpyHostToDevice);
  if (fft_status == rocfft_status_success && hip_error == hipSuccess) {
    void* input[] = {fft_device};
    void* output[] = {nullptr};
    fft_status = rocfft_execute(plan, input, output, execution_info);
  }
  if (fft_status == rocfft_status_success) hip_error = hipDeviceSynchronize();
  if (fft_status == rocfft_status_success && hip_error == hipSuccess)
    hip_error = hipMemcpy(fft_host.data(), fft_device, sample_count * sizeof(float2), hipMemcpyDeviceToHost);
  hipFree(fft_device);
  if (execution_info != nullptr) rocfft_execution_info_destroy(execution_info);
  if (plan != nullptr) rocfft_plan_destroy(plan);
  if (description != nullptr) rocfft_plan_description_destroy(description);
  rocfft_status cleanup_status = rocfft_cleanup();
  if (fft_status != rocfft_status_success) return fail_status("rocFFT execute", fft_status);
  if (cleanup_status != rocfft_status_success) return fail_status("rocfft_cleanup", cleanup_status);
  if (hip_error != hipSuccess) return fail_hip("rocFFT output", hip_error);
  for (std::size_t index = 0; index < fft_host.size(); ++index) {
    if (std::fabs(fft_host[index].x - 1.0f) > 1e-4f || std::fabs(fft_host[index].y) > 1e-4f) {
      std::fprintf(stderr, "rocFFT impulse mismatch at %zu: (%f,%f)\n", index, fft_host[index].x,
                   fft_host[index].y);
      return 4;
    }
  }
  std::printf("MI50 rocFFT/rocRAND smoke passed: %s\n", properties.gcnArchName);
  return 0;
}
