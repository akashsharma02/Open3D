//
// Created by wei on 11/9/18.
//

#include <Open3D/Utility/Console.h>
#include <cuda_runtime.h>

#include <Cuda/Container/HashTableCudaHost.hpp>
#include <iostream>

#include "ScalableTSDFVolumeCuda.h"

namespace open3d {
namespace cuda {
/**
 * Client end
 */

ScalableTSDFVolumeCuda::ScalableTSDFVolumeCuda() {
    N_ = -1;
    bucket_count_ = -1;
    value_capacity_ = -1;
}

ScalableTSDFVolumeCuda::ScalableTSDFVolumeCuda(
        int N,
        float voxel_length,
        float sdf_trunc,
        const TransformCuda &transform_volume_to_world,
        int bucket_count,
        int value_capacity) {
    voxel_length_ = voxel_length;
    sdf_trunc_ = sdf_trunc;
    transform_volume_to_world_ = transform_volume_to_world;

    Create(N, bucket_count, value_capacity);
}

ScalableTSDFVolumeCuda::ScalableTSDFVolumeCuda(
        const ScalableTSDFVolumeCuda &other) {
    N_ = other.N_;
    device_ = other.device_;
    hash_table_ = other.hash_table_;
    active_subvolume_entry_array_ = other.active_subvolume_entry_array_;

    bucket_count_ = other.bucket_count_;
    value_capacity_ = other.value_capacity_;

    voxel_length_ = other.voxel_length_;
    sdf_trunc_ = other.sdf_trunc_;
    transform_volume_to_world_ = other.transform_volume_to_world_;
}

ScalableTSDFVolumeCuda &ScalableTSDFVolumeCuda::operator=(
        const ScalableTSDFVolumeCuda &other) {
    if (this != &other) {
        Release();

        N_ = other.N_;
        device_ = other.device_;
        hash_table_ = other.hash_table_;
        active_subvolume_entry_array_ = other.active_subvolume_entry_array_;

        bucket_count_ = other.bucket_count_;
        value_capacity_ = other.value_capacity_;

        voxel_length_ = other.voxel_length_;
        sdf_trunc_ = other.sdf_trunc_;
        transform_volume_to_world_ = other.transform_volume_to_world_;
    }

    return *this;
}

ScalableTSDFVolumeCuda::~ScalableTSDFVolumeCuda() { Release(); }

void ScalableTSDFVolumeCuda::Create(int N,
                                    int bucket_count,
                                    int value_capacity) {
    assert(bucket_count > 0 && value_capacity > 0);

    if (device_ != nullptr) {
        utility::LogError(
                "[ScalableTSDFVolumeCuda] Already created, "
                "abort!");
        return;
    }

    N_ = N;
    bucket_count_ = bucket_count;
    value_capacity_ = value_capacity;
    device_ = std::make_shared<ScalableTSDFVolumeCudaDevice>();
    hash_table_.Create(bucket_count, value_capacity);
    active_subvolume_entry_array_.Create(value_capacity);

    /** Comparing to 512^3, we can hold (sparsely) at most (512^2) 8^3 cubes.
     *  That is 262144. **/
    const int NNN = N_ * N_ * N_;
    CheckCuda(cudaMalloc(&device_->tsdf_memory_pool_,
                         sizeof(float) * NNN * value_capacity));
    CheckCuda(cudaMalloc(&device_->fg_memory_pool_,
                         sizeof(uint16_t) * NNN * value_capacity));
    CheckCuda(cudaMalloc(&device_->bg_memory_pool_,
                         sizeof(uint16_t) * NNN * value_capacity));
    CheckCuda(cudaMalloc(&device_->weight_memory_pool_,
                         sizeof(uchar) * NNN * value_capacity));
    CheckCuda(cudaMalloc(&device_->color_memory_pool_,
                         sizeof(Vector3b) * NNN * value_capacity));

    CheckCuda(cudaMalloc(&device_->active_subvolume_indices_,
                         sizeof(int) * value_capacity));
    UpdateDevice();
    Reset();

    ScalableTSDFVolumeCudaKernelCaller::Create(*this);
}

void ScalableTSDFVolumeCuda::Reset() {
    assert(device_ != nullptr);

    const int NNN = N_ * N_ * N_;
    CheckCuda(cudaMemset(device_->tsdf_memory_pool_, 0,
                         sizeof(float) * NNN * value_capacity_));
    CheckCuda(cudaMemset(device_->weight_memory_pool_, 0,
                         sizeof(uchar) * NNN * value_capacity_));
    CheckCuda(cudaMemset(device_->color_memory_pool_, 0,
                         sizeof(Vector3b) * NNN * value_capacity_));
}

void ScalableTSDFVolumeCuda::Release() {
    if (device_ != nullptr && device_.use_count() == 1) {
        CheckCuda(cudaFree(device_->tsdf_memory_pool_));
        CheckCuda(cudaFree(device_->fg_memory_pool_));
        CheckCuda(cudaFree(device_->bg_memory_pool_));
        CheckCuda(cudaFree(device_->weight_memory_pool_));
        CheckCuda(cudaFree(device_->color_memory_pool_));
        CheckCuda(cudaFree(device_->active_subvolume_indices_));
    }

    device_ = nullptr;
    hash_table_.Release();
    active_subvolume_entry_array_.Release();
}

void ScalableTSDFVolumeCuda::UpdateDevice() {
    if (device_ != nullptr) {
        device_->N_ = N_;

        device_->hash_table_ = *hash_table_.device_;
        device_->active_subvolume_entry_array_ =
                *active_subvolume_entry_array_.device_;

        device_->bucket_count_ = bucket_count_;
        device_->value_capacity_ = value_capacity_;

        device_->voxel_length_ = voxel_length_;
        device_->inv_voxel_length_ = 1.0f / voxel_length_;
        device_->sdf_trunc_ = sdf_trunc_;
        device_->transform_volume_to_world_ = transform_volume_to_world_;
        device_->transform_world_to_volume_ =
                transform_volume_to_world_.Inverse();
    }
}

std::vector<Vector3i> ScalableTSDFVolumeCuda::DownloadKeys() {
    assert(device_ != nullptr);

    auto keys = hash_table_.DownloadKeys();
    return std::move(keys);
}

Eigen::Vector3d ScalableTSDFVolumeCuda::GetMinBound() {
    std::vector<Vector3i> keys = DownloadKeys();
    int max_int = std::numeric_limits<int>::max();
    Vector3i min_bound(max_int, max_int, max_int);

    for (auto &key : keys) {
        for (int d = 0; d < 3; ++d) {
            min_bound(d) = std::min(key(d), min_bound(d));
        }
    }

    Vector3f min_boundf((min_bound(0) * N_ + 0.5f) * voxel_length_,
                        (min_bound(1) * N_ + 0.5f) * voxel_length_,
                        (min_bound(2) * N_ + 0.5f) * voxel_length_);
    Vector3f min_boundw = transform_volume_to_world_ * min_boundf;

    return Eigen::Vector3d(min_boundw(0), min_boundw(1), min_boundw(2));
}

Eigen::Vector3d ScalableTSDFVolumeCuda::GetMaxBound() {
    std::vector<Vector3i> keys = DownloadKeys();
    int min_int = std::numeric_limits<int>::min();
    Vector3i max_bound(min_int, min_int, min_int);

    for (auto &key : keys) {
        for (int d = 0; d < 3; ++d) {
            max_bound(d) = std::max(key(d), max_bound(d));
        }
    }
    Vector3f max_boundf((max_bound(0) * N_ + N_ + 0.5f) * voxel_length_,
                        (max_bound(1) * N_ + N_ + 0.5f) * voxel_length_,
                        (max_bound(2) * N_ + N_ + 0.5f) * voxel_length_);
    Vector3f max_boundw = transform_volume_to_world_ * max_boundf;

    return Eigen::Vector3d(max_boundw(0), max_boundw(1), max_boundw(2));
}

std::pair<Eigen::Vector3d, Eigen::Vector3d>
ScalableTSDFVolumeCuda::GetMinMaxBound(int num_valid_pts_thr) {
    ResetActiveSubvolumeIndices();
    GetAllSubvolumes();

    Eigen::Vector3f min_boundf(1e5, 1e5, 1e5), max_boundf(-1e5, -1e5, -1e5);

    int num_active_subvolumes = active_subvolume_entry_array_.size();
    ArrayCuda<int> valid_pts_count(num_active_subvolumes);
    ArrayCuda<Vector3f> min_bounds(num_active_subvolumes);
    ArrayCuda<Vector3f> max_bounds(num_active_subvolumes);

    ScalableTSDFVolumeCudaKernelCaller::GetMinMaxBound(*this, valid_pts_count,
                                                       min_bounds, max_bounds);

    std::vector<int> valid_pts_count_cpu = valid_pts_count.DownloadAll();
    std::vector<Vector3f> min_bounds_cpu = min_bounds.DownloadAll();
    std::vector<Vector3f> max_bounds_cpu = max_bounds.DownloadAll();

    for (int i = 0; i < valid_pts_count_cpu.size(); ++i) {
        if (valid_pts_count_cpu[i] > num_valid_pts_thr) {
            for (int j = 0; j < 3; ++j) {
                min_boundf(j) = std::min(min_boundf(j), min_bounds_cpu[i](j));
                max_boundf(j) = std::max(max_boundf(j), max_bounds_cpu[i](j));
            }
        }
    }

    return std::make_pair(min_boundf.cast<double>(), max_boundf.cast<double>());
}

std::pair<std::vector<Vector3i>, std::vector<ScalableTSDFVolumeCpuData>>
ScalableTSDFVolumeCuda::DownloadVolumes() {
    assert(device_ != nullptr);

    auto key_value_pairs = hash_table_.DownloadKeyValuePairs();
    std::vector<Vector3i> &keys = key_value_pairs.first;
    std::vector<UniformTSDFVolumeCudaDevice> &subvolumes_device =
            key_value_pairs.second;

    assert(keys.size() == subvolumes_device.size());

    std::vector<ScalableTSDFVolumeCpuData> subvolumes;
    subvolumes.resize(subvolumes_device.size());

    for (int i = 0; i < subvolumes.size(); ++i) {
        auto &subvolume = subvolumes[i];
        const size_t NNN = N_ * N_ * N_;
        subvolume.tsdf_.resize(NNN);
        subvolume.weight_.resize(NNN);
        subvolume.color_.resize(NNN);

        CheckCuda(cudaMemcpy(subvolume.tsdf_.data(), subvolumes_device[i].tsdf_,
                             sizeof(float) * NNN, cudaMemcpyDeviceToHost));
        CheckCuda(cudaMemcpy(subvolume.weight_.data(),
                             subvolumes_device[i].weight_, sizeof(uchar) * NNN,
                             cudaMemcpyDeviceToHost));
        CheckCuda(cudaMemcpy(subvolume.color_.data(),
                             subvolumes_device[i].color_,
                             sizeof(Vector3b) * NNN, cudaMemcpyDeviceToHost));
    }

    return std::make_pair(std::move(keys), std::move(subvolumes));
}

/** We can easily download occupied subvolumes in parallel
 * However, uploading is not guaranteed to be correct
 * due to thread conflicts **/

std::vector<int> ScalableTSDFVolumeCuda::UploadKeys(
        std::vector<Vector3i> &keys) {
    std::vector<Vector3i> keys_to_attempt = keys;
    std::vector<int> value_addrs(keys.size());
    std::vector<int> index_map(keys.size());
    for (int i = 0; i < index_map.size(); ++i) {
        index_map[i] = i;
    }

    const int kTotalAttempt = 10;
    int attempt = 0;
    while (attempt++ < kTotalAttempt) {
        hash_table_.ResetLocks();
        std::vector<int> ret_value_addrs = hash_table_.New(keys_to_attempt);

        std::vector<int> new_index_map;
        std::vector<Vector3i> new_keys_to_attempt;
        for (int i = 0; i < keys_to_attempt.size(); ++i) {
            int addr = ret_value_addrs[i];
            /** Failed to allocate due to thread locks **/
            if (addr < 0) {
                new_index_map.emplace_back(index_map[i]);
                new_keys_to_attempt.emplace_back(keys_to_attempt[i]);
            } else {
                value_addrs[index_map[i]] = addr;
            }
        }

        utility::LogInfo("{} / {} subvolume info uploaded",
                         keys_to_attempt.size() - new_keys_to_attempt.size(),
                         keys_to_attempt.size());

        if (new_keys_to_attempt.empty()) {
            break;
        }

        std::swap(index_map, new_index_map);
        std::swap(keys_to_attempt, new_keys_to_attempt);
    }

    if (attempt == kTotalAttempt) {
        utility::LogWarning(
                "Reach maximum attempts, "
                "{} subvolumes may fail to be inserted!",
                keys_to_attempt.size());
    }

    return std::move(value_addrs);
}

bool ScalableTSDFVolumeCuda::UploadVolumes(
        std::vector<Vector3i> &keys,
        std::vector<ScalableTSDFVolumeCpuData> &values) {
    auto value_addrs = UploadKeys(keys);

    const int NNN = (N_ * N_ * N_);
    bool ret = true;
    for (int i = 0; i < value_addrs.size(); ++i) {
        int addr = value_addrs[i];

        if (addr < 0) {
            ret = false;
            continue;
        }

        const int offset = NNN * addr;
        CheckCuda(cudaMemcpy(&device_->tsdf_memory_pool_[offset],
                             values[i].tsdf_.data(), sizeof(float) * NNN,
                             cudaMemcpyHostToDevice));
        CheckCuda(cudaMemcpy(&device_->weight_memory_pool_[offset],
                             values[i].weight_.data(), sizeof(uchar) * NNN,
                             cudaMemcpyHostToDevice));
        CheckCuda(cudaMemcpy(&device_->color_memory_pool_[offset],
                             values[i].color_.data(), sizeof(Vector3b) * NNN,
                             cudaMemcpyHostToDevice));
    }
    return ret;
}

void ScalableTSDFVolumeCuda::TouchSubvolumes(
        ImageCuda<float, 1> &depth,
        PinholeCameraIntrinsicCuda &camera,
        TransformCuda &transform_camera_to_world,
        int frame_id,
        ImageCuda<uchar, 1> &mask_image) {
    assert(device_ != nullptr);

    ScalableTSDFVolumeCudaKernelCaller::TouchSubvolumes(
            *this, depth, camera, transform_camera_to_world, frame_id, mask_image);
}

void ScalableTSDFVolumeCuda::GetSubvolumesInFrustum(
        PinholeCameraIntrinsicCuda &camera,
        TransformCuda &transform_camera_to_world,
        int frame_id) {
    assert(device_ != nullptr);

    ScalableTSDFVolumeCudaKernelCaller::GetSubvolumesInFrustum(
            *this, camera, transform_camera_to_world, frame_id);
}

void ScalableTSDFVolumeCuda::GetAllSubvolumes() {
    assert(device_ != nullptr);
    ScalableTSDFVolumeCudaKernelCaller::GetAllSubvolumes(*this);
}

int ScalableTSDFVolumeCuda::GetVisibleSubvolumesCount(
        int frame_id, int frame_threshold) const {
    assert(device_ != nullptr);

    int *total_visible;
    CheckCuda(cudaMalloc(&total_visible, sizeof(int)));
    CheckCuda(cudaMemset(total_visible, 0, sizeof(int)));

    ScalableTSDFVolumeCudaKernelCaller::GetVisibleSubvolumesCount(
            *this, total_visible, frame_id, frame_threshold);

    int visible_count;
    CheckCuda(cudaMemcpy(&visible_count, total_visible, sizeof(int),
                         cudaMemcpyDeviceToHost));
    utility::LogDebug("Visible count: {}", visible_count);
    return visible_count;
}

int ScalableTSDFVolumeCuda::GetTotalAllocatedSubvolumesCount() const {
    assert(device_ != nullptr);
    return hash_table_.memory_heap_value_.HeapCounter();
}

void ScalableTSDFVolumeCuda::IntegrateSubvolumes(
        RGBDImageCuda &rgbd,
        ImageCuda<uchar, 1> &mask_image,
        PinholeCameraIntrinsicCuda &camera,
        TransformCuda &transform_camera_to_world) {
    assert(device_ != nullptr);

    ScalableTSDFVolumeCudaKernelCaller::IntegrateSubvolumes(
            *this, rgbd, mask_image, camera, transform_camera_to_world);
}

void ScalableTSDFVolumeCuda::ResetActiveSubvolumeIndices() {
    assert(device_ != nullptr);

    CheckCuda(cudaMemset(device_->active_subvolume_indices_, 0xff,
                         sizeof(int) * value_capacity_));
}

void ScalableTSDFVolumeCuda::Integrate(
        RGBDImageCuda &rgbd,
        PinholeCameraIntrinsicCuda &camera,
        TransformCuda &transform_camera_to_world,
        int frame_id,
        const ImageCuda<uchar, 1> &r_mask_image) {
    assert(device_ != nullptr);

    hash_table_.ResetLocks();
    ImageCuda<uchar, 1> mask_image;
    if (r_mask_image.width_ <= 0 || r_mask_image.height_ <= 0 ||
        r_mask_image.device_ == nullptr) {
        mask_image.Create(rgbd.depth_.width_, rgbd.depth_.height_, 1);
    } else {
        mask_image = r_mask_image;
    }

    active_subvolume_entry_array_.set_iterator(0);
    TouchSubvolumes(rgbd.depth_, camera, transform_camera_to_world, frame_id, mask_image);

    ResetActiveSubvolumeIndices();
    GetSubvolumesInFrustum(camera, transform_camera_to_world, frame_id);
    utility::LogDebug("Active subvolumes in volume: {}",
                      active_subvolume_entry_array_.size());

    IntegrateSubvolumes(rgbd, mask_image, camera, transform_camera_to_world);
}

void ScalableTSDFVolumeCuda::RayCasting(
        ImageCuda<float, 3> &vertex,
        ImageCuda<float, 3> &normal,
        ImageCuda<uchar, 3> &color,
        PinholeCameraIntrinsicCuda &camera,
        TransformCuda &transform_camera_to_world) {
    assert(device_ != nullptr);

    ScalableTSDFVolumeCudaKernelCaller::RayCasting(
            *this, vertex, normal, color, camera, transform_camera_to_world);
}

void ScalableTSDFVolumeCuda::VolumeRendering(
        ImageCuda<float, 3> &image,
        PinholeCameraIntrinsicCuda &camera,
        TransformCuda &transform_camera_to_world) {
    assert(device_ != nullptr);

    ScalableTSDFVolumeCudaKernelCaller::VolumeRendering(
            *this, image, camera, transform_camera_to_world);
}

ScalableTSDFVolumeCuda ScalableTSDFVolumeCuda::DownSample() {
    ScalableTSDFVolumeCuda volume_down(N_ / 2, voxel_length_ * 2,
                                       sdf_trunc_ * 2);

    auto keys = DownloadKeys();
    volume_down.UploadKeys(keys);

    GetAllSubvolumes();
    ScalableTSDFVolumeCudaKernelCaller::DownSample(*this, volume_down);

    return volume_down;
}
}  // namespace cuda
}  // namespace open3d
