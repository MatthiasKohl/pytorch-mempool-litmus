import os
import torch
from torch.utils import cpp_extension

CUSTOM_ALLOC_SOURCE_TEMPLATE = """
#include <cuda_runtime_api.h>
#include <iostream>
extern "C" {
void* customAlloc(size_t size, int device, void* stream) {
  std::cout << "Pool [::id::] alloc" << std::endl;
  void* ptr = nullptr;
  cudaMalloc(&ptr, size);
  return ptr;
}
void customFree(void* ptr, size_t size, int device, void* stream) {
  std::cout << "Pool [::id::] free" << std::endl;
  cudaFree(ptr);
}
}
"""

CUSTOM_ALLOCATORS = dict()


def make_custom_pool(pool_id: int) -> torch._C._CUDAPluggableAllocator:
    source = CUSTOM_ALLOC_SOURCE_TEMPLATE.replace("[::id::]", str(pool_id))
    custom_libname = f"custom_alloc{pool_id}"
    if custom_libname in CUSTOM_ALLOCATORS:
        return CUSTOM_ALLOCATORS[custom_libname][0].allocator()
    os.makedirs(f"./build_{pool_id}", exist_ok=True)
    custom_allocator = cpp_extension.load_inline(
        name=custom_libname,
        cpp_sources=source,
        with_cuda=True,
        is_python_module=False,
        build_directory=f"./build_{pool_id}",
    )
    allocator = torch.cuda.CUDAPluggableAllocator(
        f"./build_{pool_id}/{custom_libname}.so", "customAlloc", "customFree"
    )
    CUSTOM_ALLOCATORS[custom_libname] = (allocator, custom_allocator)
    return allocator.allocator()
