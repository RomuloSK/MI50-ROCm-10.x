// Native MIOpen FP32 convolution smoke for MI50/gfx906.
//
// A small 5x5 convolution exercises descriptor setup, workspace sizing,
// algorithm selection and execution through the retained Vega20 database.
// Return 77 is reserved for the pre-hardware state.

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int fail_hip(const char* operation, hipError_t error) {
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(error));
  return 1;
}

int fail_miopen(const char* operation, miopenStatus_t status) {
  std::fprintf(stderr, "%s failed: MIOpen status=%d\n", operation, static_cast<int>(status));
  return 1;
}

}  // namespace

int main() {
  int device_count = 0;
  hipError_t hip_error = hipGetDeviceCount(&device_count);
  if (hip_error == hipErrorNoDevice || device_count < 1) {
    std::fprintf(stderr, "MIOpen smoke: GPU-test-pending (no HIP device)\n");
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

  constexpr int input_height = 5;
  constexpr int input_width = 5;
  constexpr int filter_height = 3;
  constexpr int filter_width = 3;
  constexpr int channels = 1;
  constexpr int output_channels = 1;
  constexpr int batch = 1;
  constexpr std::size_t input_elements = batch * channels * input_height * input_width;
  constexpr std::size_t filter_elements = output_channels * channels * filter_height * filter_width;
  constexpr std::size_t output_elements = batch * output_channels * input_height * input_width;
  std::vector<float> host_input(input_elements, 1.0f);
  std::vector<float> host_filter(filter_elements, 1.0f);
  std::vector<float> host_output(output_elements, 0.0f);
  float* device_input = nullptr;
  float* device_filter = nullptr;
  float* device_output = nullptr;

  hip_error = hipMalloc(reinterpret_cast<void**>(&device_input), input_elements * sizeof(float));
  if (hip_error == hipSuccess)
    hip_error = hipMalloc(reinterpret_cast<void**>(&device_filter), filter_elements * sizeof(float));
  if (hip_error == hipSuccess)
    hip_error = hipMalloc(reinterpret_cast<void**>(&device_output), output_elements * sizeof(float));
  if (hip_error == hipSuccess)
    hip_error = hipMemcpy(device_input, host_input.data(), input_elements * sizeof(float), hipMemcpyHostToDevice);
  if (hip_error == hipSuccess)
    hip_error = hipMemcpy(device_filter, host_filter.data(), filter_elements * sizeof(float), hipMemcpyHostToDevice);
  if (hip_error != hipSuccess) {
    hipFree(device_output);
    hipFree(device_filter);
    hipFree(device_input);
    return fail_hip("MIOpen input setup", hip_error);
  }

  miopenHandle_t handle = nullptr;
  miopenTensorDescriptor_t input_desc = nullptr;
  miopenTensorDescriptor_t filter_desc = nullptr;
  miopenTensorDescriptor_t output_desc = nullptr;
  miopenConvolutionDescriptor_t convolution_desc = nullptr;
  miopenStatus_t status = miopenCreate(&handle);
  if (status == miopenStatusSuccess) status = miopenCreateTensorDescriptor(&input_desc);
  if (status == miopenStatusSuccess) status = miopenCreateTensorDescriptor(&filter_desc);
  if (status == miopenStatusSuccess) status = miopenCreateTensorDescriptor(&output_desc);
  if (status == miopenStatusSuccess) status = miopenCreateConvolutionDescriptor(&convolution_desc);
  if (status == miopenStatusSuccess)
    status = miopenSet4dTensorDescriptor(input_desc, miopenFloat, batch, channels, input_height, input_width);
  if (status == miopenStatusSuccess)
    status = miopenSet4dTensorDescriptor(filter_desc, miopenFloat, output_channels, channels,
                                         filter_height, filter_width);
  if (status == miopenStatusSuccess)
    status = miopenInitConvolutionDescriptor(convolution_desc, miopenConvolution, 1, 1, 1, 1, 1, 1);
  int output_batch = 0;
  int output_channels_actual = 0;
  int output_height = 0;
  int output_width = 0;
  if (status == miopenStatusSuccess)
    status = miopenGetConvolutionForwardOutputDim(convolution_desc, input_desc, filter_desc,
                                                  &output_batch, &output_channels_actual,
                                                  &output_height, &output_width);
  if (status == miopenStatusSuccess &&
      (output_batch != batch || output_channels_actual != output_channels ||
       output_height != input_height || output_width != input_width)) {
    std::fprintf(stderr, "unexpected MIOpen output shape: %d,%d,%d,%d\n", output_batch,
                 output_channels_actual, output_height, output_width);
    status = miopenStatusBadParm;
  }
  if (status == miopenStatusSuccess)
    status = miopenSet4dTensorDescriptor(output_desc, miopenFloat, output_batch,
                                         output_channels_actual, output_height, output_width);

  size_t workspace_size = 0;
  if (status == miopenStatusSuccess)
    status = miopenConvolutionForwardGetWorkSpaceSize(handle, filter_desc, input_desc,
                                                      convolution_desc, output_desc, &workspace_size);
  void* workspace = nullptr;
  if (status == miopenStatusSuccess && workspace_size > 0)
    hip_error = hipMalloc(&workspace, workspace_size);
  if (status == miopenStatusSuccess && hip_error != hipSuccess)
    status = miopenStatusAllocFailed;

  int algorithm_count = 0;
  miopenConvAlgoPerf_t performance{};
  if (status == miopenStatusSuccess)
    status = miopenFindConvolutionForwardAlgorithm(
        handle, input_desc, device_input, filter_desc, device_filter, convolution_desc,
        output_desc, device_output, 1, &algorithm_count, &performance, workspace, workspace_size,
        false);
  if (status == miopenStatusSuccess && algorithm_count < 1) {
    std::fprintf(stderr, "MIOpen returned no forward convolution algorithm\n");
    status = miopenStatusNotImplemented;
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  if (status == miopenStatusSuccess)
    status = miopenConvolutionForward(handle, &alpha, input_desc, device_input, filter_desc,
                                      device_filter, convolution_desc, performance.fwd_algo, &beta,
                                      output_desc, device_output, workspace, workspace_size);
  if (status == miopenStatusSuccess) hip_error = hipDeviceSynchronize();
  if (status == miopenStatusSuccess && hip_error == hipSuccess)
    hip_error = hipMemcpy(host_output.data(), device_output, output_elements * sizeof(float),
                          hipMemcpyDeviceToHost);

  if (workspace != nullptr) hipFree(workspace);
  if (convolution_desc != nullptr) miopenDestroyConvolutionDescriptor(convolution_desc);
  if (output_desc != nullptr) miopenDestroyTensorDescriptor(output_desc);
  if (filter_desc != nullptr) miopenDestroyTensorDescriptor(filter_desc);
  if (input_desc != nullptr) miopenDestroyTensorDescriptor(input_desc);
  if (handle != nullptr) miopenDestroy(handle);
  hipFree(device_output);
  hipFree(device_filter);
  hipFree(device_input);
  if (status != miopenStatusSuccess) return fail_miopen("MIOpen convolution", status);
  if (hip_error != hipSuccess) return fail_hip("MIOpen output synchronization", hip_error);

  for (int row = 0; row < input_height; ++row) {
    for (int column = 0; column < input_width; ++column) {
      const int row_start = row == 0 ? 0 : row - 1;
      const int row_end = row == input_height - 1 ? input_height - 1 : row + 1;
      const int column_start = column == 0 ? 0 : column - 1;
      const int column_end = column == input_width - 1 ? input_width - 1 : column + 1;
      const float expected = static_cast<float>((row_end - row_start + 1) * (column_end - column_start + 1));
      const float observed = host_output[static_cast<std::size_t>(row * input_width + column)];
      if (std::fabs(observed - expected) > 1e-4f) {
        std::fprintf(stderr, "MIOpen output mismatch at (%d,%d): got=%f expected=%f\n", row, column,
                     observed, expected);
        return 3;
      }
    }
  }
  std::printf("MI50 MIOpen convolution smoke passed: %s\n", properties.gcnArchName);
  return 0;
}
