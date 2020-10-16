//
// Created by wei on 10/20/18.
//

#pragma once
#include <Cuda/Common/Palatte.h>
#include <math_constants.h>

#include "ScalableTSDFVolumeCudaDevice.cuh"

namespace open3d {
namespace cuda {

__global__ void CreateKernel(ScalableTSDFVolumeCudaDevice server) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= server.value_capacity_) return;

    int N = server.N_;
    const size_t offset = (N * N * N) * index;
    UniformTSDFVolumeCudaDevice &subvolume =
            server.hash_table_.memory_heap_value_.value_at(index);

    /** Assign dimension **/
    subvolume.N_ = server.N_;

    /** Assign memory **/
    subvolume.tsdf_ = &server.tsdf_memory_pool_[offset];
    subvolume.fg_ = &server.fg_memory_pool_[offset];
    subvolume.bg_ = &server.bg_memory_pool_[offset];
    subvolume.weight_ = &server.weight_memory_pool_[offset];
    subvolume.color_ = &server.color_memory_pool_[offset];

    /** Assign property **/
    subvolume.voxel_length_ = server.voxel_length_;
    subvolume.inv_voxel_length_ = server.inv_voxel_length_;
    subvolume.sdf_trunc_ = server.sdf_trunc_;
    subvolume.transform_volume_to_world_ = server.transform_volume_to_world_;
    subvolume.transform_world_to_volume_ = server.transform_world_to_volume_;
    subvolume.Initialize();
}

__host__ void ScalableTSDFVolumeCudaKernelCaller::Create(
        ScalableTSDFVolumeCuda &volume) {
    const dim3 threads(THREAD_1D_UNIT);
    const dim3 blocks(DIV_CEILING(volume.value_capacity_, THREAD_1D_UNIT));
    CreateKernel<<<blocks, threads>>>(*volume.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__ void TouchSubvolumesKernel(ScalableTSDFVolumeCudaDevice server,
                                      ImageCudaDevice<float, 1> depth,
                                      PinholeCameraIntrinsicCuda camera,
                                      TransformCuda transform_camera_to_world,
                                      int frame_id,
                                      ImageCudaDevice<uchar, 1> mask) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= depth.width_ || y >= depth.height_) return;
    if (mask.at(x, y, 0) <= 0) return;

    const Vector2i p = Vector2i(x, y);
    server.TouchSubvolume(p, depth, camera, transform_camera_to_world,
                          frame_id);
}

__host__ void ScalableTSDFVolumeCudaKernelCaller::TouchSubvolumes(
        ScalableTSDFVolumeCuda &volume,
        ImageCuda<float, 1> &depth,
        PinholeCameraIntrinsicCuda &camera,
        TransformCuda &transform_camera_to_world,
        int frame_id,
        ImageCuda<uchar, 1> &mask) {
    const dim3 blocks(DIV_CEILING(depth.width_, THREAD_2D_UNIT),
                      DIV_CEILING(depth.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    TouchSubvolumesKernel<<<blocks, threads>>>(
            *volume.device_, *depth.device_, camera, transform_camera_to_world, frame_id, *mask.device_);

    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__ void IntegrateSubvolumesKernel(
        ScalableTSDFVolumeCudaDevice server,
        RGBDImageCudaDevice rgbd,
        ImageCudaDevice<uchar, 1> mask_image,
        PinholeCameraIntrinsicCuda camera,
        TransformCuda transform_camera_to_world) {
    const size_t entry_idx = blockIdx.x;

    // 1070 supports up to 1024 threads per block
    // Each thread processes 4 blocks, so 1024 * 4 = 4096 = 16^3 can be achieved
    for (int workload = 0; workload < 4; ++workload) {
        const Vector3i Xlocal = Vector3i(threadIdx.x, threadIdx.y,
                                         threadIdx.z + blockDim.z * workload);

#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(entry_idx < server.active_subvolume_entry_array_.size() &&
               Xlocal(0) < server.N_ && Xlocal(1) < server.N_ &&
               Xlocal(2) < server.N_);
#endif

        HashEntry<Vector3i> &entry =
                server.active_subvolume_entry_array_.at(entry_idx);
#ifdef CUDA_DEBUG_ENABLE_ASSERTION
        assert(entry.internal_addr >= 0);
#endif
        server.Integrate(Xlocal, entry, rgbd, mask_image, camera,
                         transform_camera_to_world);
    }
}

__host__ void ScalableTSDFVolumeCudaKernelCaller::IntegrateSubvolumes(
        ScalableTSDFVolumeCuda &volume,
        RGBDImageCuda &rgbd,
        ImageCuda<uchar, 1> &mask_image,
        PinholeCameraIntrinsicCuda &camera,
        TransformCuda &transform_camera_to_world) {
    const dim3 blocks(volume.active_subvolume_entry_array_.size());
    const dim3 threads(volume.N_, volume.N_, volume.N_ / 4);
    printf("blocks: %d, threads: %d",
           volume.active_subvolume_entry_array_.size(), volume.N_);
    IntegrateSubvolumesKernel<<<blocks, threads>>>(
            *volume.device_, *rgbd.device_, *mask_image.device_, camera,
            transform_camera_to_world);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__ void GetSubvolumesInFrustumKernel(
        ScalableTSDFVolumeCudaDevice server,
        PinholeCameraIntrinsicCuda camera,
        TransformCuda transform_camera_to_world,
        int frame_id) {
    const int bucket_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (bucket_idx >= server.bucket_count_) return;

    auto &hash_table = server.hash_table_;

    int bucket_base_idx = bucket_idx * BUCKET_SIZE;
#pragma unroll 1
    for (size_t i = 0; i < BUCKET_SIZE; ++i) {
        HashEntry<Vector3i> &entry =
                hash_table.entry_array_.at(bucket_base_idx + i);
        if (!entry.IsEmpty()) {
            Vector3f X = server.voxelf_local_to_global(Vector3f(0), entry.key);
            if (camera.IsPointInFrustum(transform_camera_to_world.Inverse() *
                                        server.voxelf_to_world(X))) {
                server.ActivateSubvolume(entry);
                UniformTSDFVolumeCudaDevice *subvolume =
                        hash_table.GetValuePtrByInternalAddr(
                                entry.internal_addr);
                subvolume->last_visible_index_ = frame_id;
            }
        }
    }

    LinkedListCudaDevice<HashEntry<Vector3i>> &linked_list =
            hash_table.entry_list_array_.at(bucket_idx);
    int node_ptr = linked_list.head_node_ptr();
    while (node_ptr != NULLPTR_CUDA) {
        LinkedListNodeCuda<HashEntry<Vector3i>> &linked_list_node =
                linked_list.get_node(node_ptr);

        HashEntry<Vector3i> &entry = linked_list_node.data;
        Vector3f X = server.voxelf_local_to_global(Vector3f(0), entry.key);
        if (camera.IsPointInFrustum(transform_camera_to_world.Inverse() *
                                    server.voxelf_to_world(X))) {
            server.ActivateSubvolume(entry);
            UniformTSDFVolumeCudaDevice *subvolume =
                    hash_table.GetValuePtrByInternalAddr(entry.internal_addr);
            subvolume->last_visible_index_ = frame_id;
        }

        node_ptr = linked_list_node.next_node_ptr;
    }
}

__host__ void ScalableTSDFVolumeCudaKernelCaller::GetSubvolumesInFrustum(
        ScalableTSDFVolumeCuda &volume,
        PinholeCameraIntrinsicCuda &camera,
        TransformCuda &transform_camera_to_world,
        int frame_id) {
    const dim3 blocks(volume.bucket_count_);
    const dim3 threads(THREAD_1D_UNIT);
    GetSubvolumesInFrustumKernel<<<blocks, threads>>>(
            *volume.device_, camera, transform_camera_to_world, frame_id);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__ void GetAllSubvolumesKernel(ScalableTSDFVolumeCudaDevice server) {
    const int bucket_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (bucket_idx >= server.bucket_count_) return;

    auto &hash_table = server.hash_table_;

    int bucket_base_idx = bucket_idx * BUCKET_SIZE;
#pragma unroll 1
    for (size_t i = 0; i < BUCKET_SIZE; ++i) {
        HashEntry<Vector3i> &entry =
                hash_table.entry_array_.at(bucket_base_idx + i);
        if (entry.internal_addr != NULLPTR_CUDA) {
            server.ActivateSubvolume(entry);
        }
    }

    LinkedListCudaDevice<HashEntry<Vector3i>> &linked_list =
            hash_table.entry_list_array_.at(bucket_idx);
    int node_ptr = linked_list.head_node_ptr();
    while (node_ptr != NULLPTR_CUDA) {
        LinkedListNodeCuda<HashEntry<Vector3i>> &linked_list_node =
                linked_list.get_node(node_ptr);
        server.ActivateSubvolume(linked_list_node.data);
        node_ptr = linked_list_node.next_node_ptr;
    }
}

__host__ void ScalableTSDFVolumeCudaKernelCaller::GetAllSubvolumes(
        ScalableTSDFVolumeCuda &volume) {
    const dim3 blocks(volume.bucket_count_);
    const dim3 threads(THREAD_1D_UNIT);
    GetAllSubvolumesKernel<<<blocks, threads>>>(*volume.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__ void GetVisibleSubvolumesCountKernel(
        ScalableTSDFVolumeCudaDevice server,
        int *total_visible,
        int frame_id,
        int frame_threshold) {
    __shared__ int local_sum[THREAD_1D_UNIT];
    int tid = threadIdx.x;
    local_sum[tid] = 0;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= server.active_subvolume_entry_array_.size()) return;
    HashEntry<Vector3i> &entry = server.active_subvolume_entry_array_.at(idx);
    if (entry.internal_addr < 0) return;

    UniformTSDFVolumeCudaDevice *subvolume =
            server.hash_table_.GetValuePtrByInternalAddr(entry.internal_addr);
    int visible = subvolume->last_visible_index_ > (frame_id - frame_threshold)
                          ? 1
                          : 0;
    {
        local_sum[tid] = visible;
        __syncthreads();

        BlockReduceSum<int, THREAD_1D_UNIT>(tid, local_sum);
        if (tid == 0) atomicAdd(total_visible, local_sum[tid]);
        __syncthreads();
    }
}

__host__ void ScalableTSDFVolumeCudaKernelCaller::GetVisibleSubvolumesCount(
        const ScalableTSDFVolumeCuda &volume,
        int *total_visible,
        int frame_id,
        int frame_threshold) {
    const dim3 blocks(DIV_CEILING(volume.active_subvolume_entry_array_.size(),
                                  THREAD_1D_UNIT));
    const dim3 threads(THREAD_1D_UNIT);
    GetVisibleSubvolumesCountKernel<<<blocks, threads>>>(
            *volume.device_, total_visible, frame_id, frame_threshold);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__ void GetMinMaxBoundKernel(ScalableTSDFVolumeCudaDevice server,
                                     ArrayCudaDevice<int> num_valid_pts,
                                     ArrayCudaDevice<Vector3f> min_bounds,
                                     ArrayCudaDevice<Vector3f> max_bounds) {
    const size_t entry_idx = blockIdx.x;

    HashEntry<Vector3i> &entry =
            server.active_subvolume_entry_array_.at(entry_idx);

    UniformTSDFVolumeCudaDevice *subvolume =
            server.hash_table_.GetValuePtrByInternalAddr(entry.internal_addr);

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        Vector3f X_min = server.voxelf_local_to_global(Vector3f(0), entry.key);
        Vector3f X_max =
                server.voxelf_local_to_global(Vector3f(server.N_), entry.key);
        min_bounds[entry_idx] = server.voxelf_to_world(X_min);
        max_bounds[entry_idx] = server.voxelf_to_world(X_max);
    }

    // 1070 supports up to 1024 threads per block
    // Each thread processes 4 blocks, so 1024 * 4 = 4096 = 16^3 can be achieved
    for (int workload = 0; workload < 4; ++workload) {
        const Vector3i X000 = Vector3i(threadIdx.x, threadIdx.y,
                                       threadIdx.z + blockDim.z * workload);
        const Vector3i X001 = Vector3i(threadIdx.x + 1, threadIdx.y,
                                       threadIdx.z + blockDim.z * workload);
        const Vector3i X010 = Vector3i(threadIdx.x, threadIdx.y + 1,
                                       threadIdx.z + blockDim.z * workload);
        const Vector3i X100 = Vector3i(threadIdx.x, threadIdx.y,
                                       threadIdx.z + blockDim.z * workload + 1);

        float tsdf000 = subvolume->tsdf(X000);
        uchar weight000 = subvolume->weight(X000);
        if (weight000 == 0) continue;

        if (X001(0) < server.N_) {
            float tsdf_nb = subvolume->tsdf(X001);
            uchar weight_nb = subvolume->weight(X001);
            if (weight_nb > 0 && tsdf_nb * tsdf000 < 0) {
                atomicAdd(&num_valid_pts[entry_idx], 1);
            }
        }
        if (X010(1) < server.N_) {
            float tsdf_nb = subvolume->tsdf(X010);
            uchar weight_nb = subvolume->weight(X010);
            if (weight_nb > 0 && tsdf_nb * tsdf000 < 0) {
                atomicAdd(&num_valid_pts[entry_idx], 1);
            }
        }
        if (X100(2) < server.N_) {
            float tsdf_nb = subvolume->tsdf(X100);
            uchar weight_nb = subvolume->weight(X100);
            if (weight_nb > 0 && tsdf_nb * tsdf000 < 0) {
                atomicAdd(&num_valid_pts[entry_idx], 1);
            }
        }
    }
}

__host__ void ScalableTSDFVolumeCudaKernelCaller::GetMinMaxBound(
        ScalableTSDFVolumeCuda &volume,
        ArrayCuda<int> &num_valid_pts,
        ArrayCuda<Vector3f> &min_bounds,
        ArrayCuda<Vector3f> &max_bounds) {
    const dim3 blocks(volume.active_subvolume_entry_array_.size());
    const dim3 threads(volume.N_, volume.N_, volume.N_ / 4);
    GetMinMaxBoundKernel<<<blocks, threads>>>(
            *volume.device_, *num_valid_pts.device_, *min_bounds.device_,
            *max_bounds.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__ void RayCastingKernel(ScalableTSDFVolumeCudaDevice server,
                                 ImageCudaDevice<float, 3> vertex,
                                 ImageCudaDevice<float, 3> normal,
                                 ImageCudaDevice<uchar, 3> color,
                                 PinholeCameraIntrinsicCuda camera,
                                 TransformCuda transform_camera_to_world) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= vertex.width_ || y >= vertex.height_) return;

    Vector2i p = Vector2i(x, y);
    Vector3f v, n;
    Vector3b c;
    bool mask =
            server.RayCasting(p, v, n, c, camera, transform_camera_to_world);
    if (!mask) {
        vertex.at(x, y) = Vector3f(nanf("nan"));
        normal.at(x, y) = Vector3f(nanf("nan"));
        color.at(x, y) = Vector3b(0);
        return;
    }
    vertex.at(x, y) = v;
    normal.at(x, y) = n;
    color.at(x, y) = c;
}

__host__ void ScalableTSDFVolumeCudaKernelCaller::RayCasting(
        ScalableTSDFVolumeCuda &volume,
        ImageCuda<float, 3> &vertex,
        ImageCuda<float, 3> &normal,
        ImageCuda<uchar, 3> &color,
        PinholeCameraIntrinsicCuda &camera,
        TransformCuda &transform_camera_to_world) {
    const dim3 blocks(DIV_CEILING(vertex.width_, THREAD_2D_UNIT),
                      DIV_CEILING(vertex.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    RayCastingKernel<<<blocks, threads>>>(*volume.device_, *vertex.device_,
                                          *normal.device_, *color.device_,
                                          camera, transform_camera_to_world);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__ void VolumeRenderingKernel(ScalableTSDFVolumeCudaDevice server,
                                      ImageCudaDevice<float, 3> vertex,
                                      PinholeCameraIntrinsicCuda camera,
                                      TransformCuda transform_camera_to_world) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= vertex.width_ || y >= vertex.height_) return;

    Vector2i p = Vector2i(x, y);
    Vector3f v = server.VolumeRendering(p, camera, transform_camera_to_world);
    vertex.at(x, y) = v;
}

__host__ void ScalableTSDFVolumeCudaKernelCaller::VolumeRendering(
        ScalableTSDFVolumeCuda &volume,
        ImageCuda<float, 3> &image,
        PinholeCameraIntrinsicCuda &camera,
        TransformCuda &transform_camera_to_world) {
    const dim3 blocks(DIV_CEILING(image.width_, THREAD_2D_UNIT),
                      DIV_CEILING(image.height_, THREAD_2D_UNIT));
    const dim3 threads(THREAD_2D_UNIT, THREAD_2D_UNIT);
    VolumeRenderingKernel<<<blocks, threads>>>(
            *volume.device_, *image.device_, camera, transform_camera_to_world);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}

__global__ void DownSampleKernel(ScalableTSDFVolumeCudaDevice volume,
                                 ScalableTSDFVolumeCudaDevice volume_down) {
    HashEntry<Vector3i> &entry =
            volume.active_subvolume_entry_array_[blockIdx.x];

    UniformTSDFVolumeCudaDevice *subvolume = volume.QuerySubvolume(entry.key);
    UniformTSDFVolumeCudaDevice *subvolume_down =
            volume_down.QuerySubvolume(entry.key);

    assert(subvolume != nullptr && subvolume_down != nullptr);

    int x = 2 * threadIdx.x, y = 2 * threadIdx.y, z = 2 * threadIdx.z;

    float sum_tsdf = 0;
    float sum_weight = 0;
    Vector3f sum_color = Vector3f(0);
    for (int i = 0; i < 8; ++i) {
        int idx = subvolume->IndexOf(
                Vector3i(x + (i & 4), y + (i & 2), z + (i & 1)));
        sum_tsdf += subvolume->tsdf_[idx];
        sum_weight += (float)subvolume->weight_[idx];

        const Vector3b &color = subvolume->color_[idx];
        sum_color(0) += (float)color(0);
        sum_color(1) += (float)color(1);
        sum_color(2) += (float)color(2);
    }

    int idx = subvolume_down->IndexOf(
            Vector3i(threadIdx.x, threadIdx.y, threadIdx.z));

    subvolume_down->tsdf_[idx] = 0.125f * sum_tsdf;
    subvolume_down->weight_[idx] = uchar(0.125f * sum_weight);

    sum_color *= 0.125f;
    subvolume_down->color_[idx] = sum_color.template saturate_cast<uchar>();
}

void ScalableTSDFVolumeCudaKernelCaller::DownSample(
        ScalableTSDFVolumeCuda &volume, ScalableTSDFVolumeCuda &volume_down) {
    const dim3 blocks(volume.active_subvolume_entry_array_.size());
    const dim3 threads(volume.N_ / 2, volume.N_ / 2, volume.N_ / 2);
    DownSampleKernel<<<blocks, threads>>>(*volume.device_,
                                          *volume_down.device_);
    CheckCuda(cudaDeviceSynchronize());
    CheckCuda(cudaGetLastError());
}
}  // namespace cuda
}  // namespace open3d
