#include <hip/hip_runtime.h>

__global__ void mi50_noop_kernel(float *value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *value += 1.0f;
  }
}

int main() {
  // This binary is compile-only: it deliberately does not call hipInit or
  // launch the kernel, because pre-hardware CI must not claim runtime support.
  // Keep a reference to the kernel so the compiler emits a gfx906 code object.
  auto kernel = mi50_noop_kernel;
  return kernel == nullptr;
}
