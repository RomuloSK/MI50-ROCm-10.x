// Native rocSPARSE CSR SpMV and rocSOLVER LU correctness smoke for gfx906.

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <rocsparse/rocsparse.h>

#include <algorithm>
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

// Exercise the ROCm 10 generic SpMV-v2 interface first.  Older or trimmed
// gfx906 builds may not ship a v2 kernel for a particular algorithm, so the
// caller deliberately falls back to the established v1 path only when the
// library returns NOT_IMPLEMENTED.
rocsparse_status run_spmv_v2(rocsparse_handle handle, rocsparse_spmat_descr matrix,
                             rocsparse_dnvec_descr vector_x, rocsparse_dnvec_descr vector_y,
                             float* device_y, float* host_y, hipError_t* hip_error) {
  rocsparse_spmv_descr descriptor = nullptr;
  rocsparse_status status = rocsparse_create_spmv_descr(&descriptor);
  rocsparse_spmv_alg algorithm = rocsparse_spmv_alg_default;
  rocsparse_operation operation = rocsparse_operation_none;
  rocsparse_datatype scalar_type = rocsparse_datatype_f32_r;
  rocsparse_datatype compute_type = rocsparse_datatype_f32_r;
  if (status == rocsparse_status_success)
    status = rocsparse_spmv_set_input(handle, descriptor, rocsparse_spmv_input_alg, &algorithm,
                                      sizeof(algorithm), nullptr);
  if (status == rocsparse_status_success)
    status = rocsparse_spmv_set_input(handle, descriptor, rocsparse_spmv_input_operation,
                                      &operation, sizeof(operation), nullptr);
  if (status == rocsparse_status_success)
    status = rocsparse_spmv_set_input(handle, descriptor,
                                      rocsparse_spmv_input_scalar_datatype, &scalar_type,
                                      sizeof(scalar_type), nullptr);
  if (status == rocsparse_status_success)
    status = rocsparse_spmv_set_input(handle, descriptor,
                                      rocsparse_spmv_input_compute_datatype, &compute_type,
                                      sizeof(compute_type), nullptr);

  size_t analysis_size = 0;
  size_t compute_size = 0;
  if (status == rocsparse_status_success)
    status = rocsparse_v2_spmv_buffer_size(handle, descriptor, matrix, vector_x, vector_y,
                                           rocsparse_v2_spmv_stage_analysis, &analysis_size,
                                           nullptr);
  if (status == rocsparse_status_success)
    status = rocsparse_v2_spmv_buffer_size(handle, descriptor, matrix, vector_x, vector_y,
                                           rocsparse_v2_spmv_stage_compute, &compute_size,
                                           nullptr);
  void* temporary = nullptr;
  const size_t temporary_size = analysis_size > compute_size ? analysis_size : compute_size;
  if (status == rocsparse_status_success && temporary_size > 0)
    *hip_error = hipMalloc(&temporary, temporary_size);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  if (status == rocsparse_status_success && *hip_error == hipSuccess)
    status = rocsparse_v2_spmv(handle, descriptor, &alpha, matrix, vector_x, &beta, vector_y,
                               rocsparse_v2_spmv_stage_analysis, analysis_size, temporary,
                               nullptr);
  if (status == rocsparse_status_success)
    status = rocsparse_v2_spmv(handle, descriptor, &alpha, matrix, vector_x, &beta, vector_y,
                               rocsparse_v2_spmv_stage_compute, compute_size, temporary,
                               nullptr);
  if (status == rocsparse_status_success) *hip_error = hipDeviceSynchronize();
  if (status == rocsparse_status_success && *hip_error == hipSuccess)
    *hip_error = hipMemcpy(host_y, device_y, 2 * sizeof(float), hipMemcpyDeviceToHost);
  if (temporary != nullptr) hipFree(temporary);
  if (descriptor != nullptr) rocsparse_destroy_spmv_descr(descriptor);
  return status;
}

}  // namespace

int main() {
  int device_count = 0;
  hipError_t hip_error = hipGetDeviceCount(&device_count);
  if (hip_error == hipErrorNoDevice || device_count < 1) {
    std::fprintf(stderr, "SPARSE/SOLVER smoke: GPU-test-pending (no HIP device)\n");
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

  // CSR matrix [[2, 1], [0, 3]] multiplied by [1, 2] gives [4, 6].
  const int rows = 2;
  const int cols = 2;
  const int nnz = 3;
  const int host_row_ptr[] = {0, 2, 3};
  const int host_col_ind[] = {0, 1, 1};
  const float host_values[] = {2.0f, 1.0f, 3.0f};
  const float host_x[] = {1.0f, 2.0f};
  float host_y[] = {0.0f, 0.0f};
  int* row_ptr = nullptr;
  int* col_ind = nullptr;
  float* values = nullptr;
  float* x = nullptr;
  float* y = nullptr;
  auto allocate = [&](void** pointer, std::size_t bytes) {
    if (hip_error == hipSuccess) hip_error = hipMalloc(pointer, bytes);
  };
  allocate(reinterpret_cast<void**>(&row_ptr), sizeof(host_row_ptr));
  allocate(reinterpret_cast<void**>(&col_ind), sizeof(host_col_ind));
  allocate(reinterpret_cast<void**>(&values), sizeof(host_values));
  allocate(reinterpret_cast<void**>(&x), sizeof(host_x));
  allocate(reinterpret_cast<void**>(&y), sizeof(host_y));
  if (hip_error == hipSuccess) hip_error = hipMemcpy(row_ptr, host_row_ptr, sizeof(host_row_ptr), hipMemcpyHostToDevice);
  if (hip_error == hipSuccess) hip_error = hipMemcpy(col_ind, host_col_ind, sizeof(host_col_ind), hipMemcpyHostToDevice);
  if (hip_error == hipSuccess) hip_error = hipMemcpy(values, host_values, sizeof(host_values), hipMemcpyHostToDevice);
  if (hip_error == hipSuccess) hip_error = hipMemcpy(x, host_x, sizeof(host_x), hipMemcpyHostToDevice);
  if (hip_error != hipSuccess) return fail_hip("rocSPARSE input setup", hip_error);

  rocsparse_handle sparse_handle = nullptr;
  rocsparse_spmat_descr matrix = nullptr;
  rocsparse_dnvec_descr vector_x = nullptr;
  rocsparse_dnvec_descr vector_y = nullptr;
  rocsparse_status sparse_status = rocsparse_create_handle(&sparse_handle);
  if (sparse_status == rocsparse_status_success)
    sparse_status = rocsparse_create_csr_descr(&matrix, rows, cols, nnz, row_ptr, col_ind, values,
                                               rocsparse_indextype_i32, rocsparse_indextype_i32,
                                               rocsparse_index_base_zero, rocsparse_datatype_f32_r);
  if (sparse_status == rocsparse_status_success)
    sparse_status = rocsparse_create_dnvec_descr(&vector_x, cols, x, rocsparse_datatype_f32_r);
  if (sparse_status == rocsparse_status_success)
    sparse_status = rocsparse_create_dnvec_descr(&vector_y, rows, y, rocsparse_datatype_f32_r);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const rocsparse_status v2_status =
      sparse_status == rocsparse_status_success
          ? run_spmv_v2(sparse_handle, matrix, vector_x, vector_y, y, host_y, &hip_error)
          : sparse_status;
  const char* sparse_path = "rocSPARSE v2";
  sparse_status = v2_status;
  if (sparse_status == rocsparse_status_not_implemented) {
    // Keep the mature v1 path as a real fallback for a trimmed gfx906 build.
    std::fill(host_y, host_y + 2, 0.0f);
    size_t buffer_size = 0;
    sparse_status = rocsparse_spmv(sparse_handle, rocsparse_operation_none, &alpha, matrix,
                                   vector_x, &beta, vector_y, rocsparse_datatype_f32_r,
                                   rocsparse_spmv_alg_default, rocsparse_spmv_stage_buffer_size,
                                   &buffer_size, nullptr);
    void* temporary = nullptr;
    if (sparse_status == rocsparse_status_success && buffer_size > 0)
      hip_error = hipMalloc(&temporary, buffer_size);
    if (sparse_status == rocsparse_status_success && hip_error == hipSuccess)
      sparse_status = rocsparse_spmv(sparse_handle, rocsparse_operation_none, &alpha, matrix,
                                     vector_x, &beta, vector_y, rocsparse_datatype_f32_r,
                                     rocsparse_spmv_alg_default, rocsparse_spmv_stage_preprocess,
                                     &buffer_size, temporary);
    if (sparse_status == rocsparse_status_success)
      sparse_status = rocsparse_spmv(sparse_handle, rocsparse_operation_none, &alpha, matrix,
                                     vector_x, &beta, vector_y, rocsparse_datatype_f32_r,
                                     rocsparse_spmv_alg_default, rocsparse_spmv_stage_compute,
                                     &buffer_size, temporary);
    if (sparse_status == rocsparse_status_success) hip_error = hipDeviceSynchronize();
    if (sparse_status == rocsparse_status_success && hip_error == hipSuccess)
      hip_error = hipMemcpy(host_y, y, sizeof(host_y), hipMemcpyDeviceToHost);
    if (temporary != nullptr) hipFree(temporary);
    sparse_path = "rocSPARSE v1 fallback";
  }
  if (vector_y != nullptr) rocsparse_destroy_dnvec_descr(vector_y);
  if (vector_x != nullptr) rocsparse_destroy_dnvec_descr(vector_x);
  if (matrix != nullptr) rocsparse_destroy_spmat_descr(matrix);
  if (sparse_handle != nullptr) rocsparse_destroy_handle(sparse_handle);
  hipFree(y);
  hipFree(x);
  hipFree(values);
  hipFree(col_ind);
  hipFree(row_ptr);
  if (sparse_status != rocsparse_status_success) return fail_status("rocSPARSE SpMV", sparse_status);
  if (hip_error != hipSuccess) return fail_hip("rocSPARSE output", hip_error);
  if (std::fabs(host_y[0] - 4.0f) > 1e-4f || std::fabs(host_y[1] - 6.0f) > 1e-4f) {
    std::fprintf(stderr, "rocSPARSE output mismatch: (%f,%f) expected (4,6)\n", host_y[0], host_y[1]);
    return 3;
  }

  // Column-major A=[[4,1],[2,3]]. No-pivot LU should produce U=(4,1;0,2.5), L21=.5.
  const float host_matrix[] = {4.0f, 2.0f, 1.0f, 3.0f};
  float host_factorized[4] = {};
  float* matrix_device = nullptr;
  rocblas_int* info_device = nullptr;
  hip_error = hipMalloc(reinterpret_cast<void**>(&matrix_device), sizeof(host_matrix));
  if (hip_error == hipSuccess) hip_error = hipMalloc(reinterpret_cast<void**>(&info_device), sizeof(rocblas_int));
  if (hip_error == hipSuccess) hip_error = hipMemcpy(matrix_device, host_matrix, sizeof(host_matrix), hipMemcpyHostToDevice);
  if (hip_error != hipSuccess) return fail_hip("rocSOLVER input setup", hip_error);
  rocblas_handle solver_handle = nullptr;
  rocblas_status solver_status = rocblas_create_handle(&solver_handle);
  rocblas_int host_info = -1;
  if (solver_status == rocblas_status_success)
    solver_status = rocsolver_sgetrf_npvt(solver_handle, 2, 2, matrix_device, 2, info_device);
  if (solver_status == rocblas_status_success) hip_error = hipDeviceSynchronize();
  if (solver_status == rocblas_status_success && hip_error == hipSuccess)
    hip_error = hipMemcpy(host_factorized, matrix_device, sizeof(host_factorized), hipMemcpyDeviceToHost);
  if (solver_status == rocblas_status_success && hip_error == hipSuccess)
    hip_error = hipMemcpy(&host_info, info_device, sizeof(host_info), hipMemcpyDeviceToHost);
  if (solver_handle != nullptr) rocblas_destroy_handle(solver_handle);
  hipFree(info_device);
  hipFree(matrix_device);
  if (solver_status != rocblas_status_success) return fail_status("rocSOLVER sgetrf", solver_status);
  if (hip_error != hipSuccess) return fail_hip("rocSOLVER output", hip_error);
  if (host_info != 0 || std::fabs(host_factorized[0] - 4.0f) > 1e-4f ||
      std::fabs(host_factorized[1] - 0.5f) > 1e-4f ||
      std::fabs(host_factorized[2] - 1.0f) > 1e-4f ||
      std::fabs(host_factorized[3] - 2.5f) > 1e-4f) {
    std::fprintf(stderr, "rocSOLVER LU mismatch: info=%d factors=(%f,%f,%f,%f)\n", host_info,
                 host_factorized[0], host_factorized[1], host_factorized[2], host_factorized[3]);
    return 4;
  }
  std::printf("MI50 rocSPARSE/rocSOLVER smoke passed: %s (%s)\n", properties.gcnArchName,
              sparse_path);
  return 0;
}
