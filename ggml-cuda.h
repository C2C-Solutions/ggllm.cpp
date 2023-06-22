#pragma once
#include <cuda_runtime.h>
#include "ggml.h"


#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_CUDA_MAX_DEVICES       16

struct ggml_tensor_extra_gpu {
    void * data_device[GGML_CUDA_MAX_DEVICES]; // 1 pointer for each device for split tensors
};
typedef struct {
    int num_devices;
    int main_device_id;
    size_t total_vram;
    size_t total_free_vram;
    struct cudaDeviceProp device_props[GGML_CUDA_MAX_DEVICES];
    size_t device_vram_free[GGML_CUDA_MAX_DEVICES];
    size_t device_vram_total[GGML_CUDA_MAX_DEVICES];
} GPUStatus;
const GPUStatus* ggml_cuda_get_system_gpu_status();

void   ggml_init_cublas(void);
void   ggml_cuda_update_gpu_status(int device_id);
void   ggml_cuda_print_gpu_status(const GPUStatus *status);
void   ggml_cuda_set_tensor_split_prepare(const float * tensor_split, int num_devices);
void   ggml_cuda_set_tensor_split(const float * tensor_split);

void   ggml_cuda_mul(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
bool   ggml_cuda_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
size_t ggml_cuda_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
void   ggml_cuda_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize);

void   ggml_cuda_pool_free_all(int device_id);
// TODO: export these with GGML_API
void * ggml_cuda_host_malloc(size_t size);
void   ggml_cuda_host_free(void * ptr);

void   ggml_cuda_transform_tensor(void * data, struct ggml_tensor * tensor);

void   ggml_cuda_free_data(struct ggml_tensor * tensor);
void   ggml_cuda_assign_buffers(struct ggml_tensor * tensor);
void   ggml_cuda_assign_buffers_no_scratch(struct ggml_tensor * tensor);
void   ggml_cuda_set_main_device(int main_device);
void   ggml_cuda_set_scratch_size(size_t scratch_size);
void   ggml_cuda_free_scratch(void);
bool   ggml_cuda_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);

#ifdef  __cplusplus
}
#endif
