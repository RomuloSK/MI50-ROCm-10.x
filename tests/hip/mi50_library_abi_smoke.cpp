// Native gfx906 library ABI smoke.
//
// This deliberately performs small handle/version/plan operations across the
// supported ROCm math stack. It is not a performance test; the rocBLAS and
// RCCL tests cover numerical and collective correctness separately.

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <rocblas/rocblas.h>
#include <rocfft/rocfft.h>
#include <rocrand/rocrand.h>
#include <rocsolver/rocsolver.h>
#include <rocsparse/rocsparse.h>

#include <cstdio>
#include <cstring>

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
    std::fprintf(stderr, "library ABI smoke: GPU-test-pending (no HIP device)\n");
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

  miopenHandle_t miopen_handle = nullptr;
  miopenStatus_t miopen_status = miopenCreate(&miopen_handle);
  if (miopen_status != miopenStatusSuccess) return fail_status("miopenCreate", miopen_status);
  size_t miopen_major = 0;
  size_t miopen_minor = 0;
  size_t miopen_patch = 0;
  miopen_status = miopenGetVersion(&miopen_major, &miopen_minor, &miopen_patch);
  if (miopen_status == miopenStatusSuccess)
    miopen_status = miopenDestroy(miopen_handle);
  if (miopen_status != miopenStatusSuccess) return fail_status("MIOpen ABI", miopen_status);

  rocblas_handle rocblas_handle = nullptr;
  rocblas_status rocblas_status_value = rocblas_create_handle(&rocblas_handle);
  if (rocblas_status_value != rocblas_status_success)
    return fail_status("rocblas_create_handle", rocblas_status_value);

  rocsparse_handle rocsparse_handle_value = nullptr;
  rocsparse_status rocsparse_status_value = rocsparse_create_handle(&rocsparse_handle_value);
  if (rocsparse_status_value != rocsparse_status_success) {
    rocblas_destroy_handle(rocblas_handle);
    return fail_status("rocsparse_create_handle", rocsparse_status_value);
  }
  int rocsparse_version = 0;
  rocsparse_status_value = rocsparse_get_version(rocsparse_handle_value, &rocsparse_version);
  rocsparse_status destroy_sparse = rocsparse_destroy_handle(rocsparse_handle_value);
  rocblas_status destroy_blas = rocblas_destroy_handle(rocblas_handle);
  if (rocsparse_status_value != rocsparse_status_success)
    return fail_status("rocsparse_get_version", rocsparse_status_value);
  if (destroy_sparse != rocsparse_status_success)
    return fail_status("rocsparse_destroy_handle", destroy_sparse);
  if (destroy_blas != rocblas_status_success)
    return fail_status("rocblas_destroy_handle", destroy_blas);

  rocrand_generator random_generator = nullptr;
  rocrand_status random_status = rocrand_create_generator(&random_generator, ROCRAND_RNG_PSEUDO_DEFAULT);
  if (random_status != ROCRAND_STATUS_SUCCESS) return fail_status("rocrand_create_generator", random_status);
  random_status = rocrand_set_seed(random_generator, 0x4d493530ULL);
  rocrand_status destroy_random = rocrand_destroy_generator(random_generator);
  if (random_status != ROCRAND_STATUS_SUCCESS) return fail_status("rocrand_set_seed", random_status);
  if (destroy_random != ROCRAND_STATUS_SUCCESS) return fail_status("rocrand_destroy_generator", destroy_random);

  rocfft_status fft_status = rocfft_setup();
  if (fft_status != rocfft_status_success) return fail_status("rocfft_setup", fft_status);
  rocfft_plan_description description = nullptr;
  rocfft_plan plan = nullptr;
  fft_status = rocfft_plan_description_create(&description);
  size_t lengths[] = {8};
  if (fft_status == rocfft_status_success) {
    fft_status = rocfft_plan_create(&plan, rocfft_placement_inplace,
                                    rocfft_transform_type_complex_forward, rocfft_precision_single,
                                    1, lengths, 1, description);
  }
  if (plan != nullptr) rocfft_plan_destroy(plan);
  if (description != nullptr) rocfft_plan_description_destroy(description);
  rocfft_status cleanup_fft = rocfft_cleanup();
  if (fft_status != rocfft_status_success) return fail_status("rocfft plan", fft_status);
  if (cleanup_fft != rocfft_status_success) return fail_status("rocfft_cleanup", cleanup_fft);

  char solver_version[128] = {};
  rocblas_status solver_status = rocsolver_get_version_string(solver_version, sizeof(solver_version));
  if (solver_status != rocblas_status_success) return fail_status("rocsolver_get_version_string", solver_status);
  std::printf("MI50 library ABI smoke passed: %s, MIOpen %zu.%zu.%zu, rocSPARSE %d, rocSOLVER %s\n",
              properties.gcnArchName, miopen_major, miopen_minor, miopen_patch, rocsparse_version,
              solver_version);
  return 0;
}
