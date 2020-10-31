//
// Created by wei on 11/9/18.
//

#pragma once

#include "Cuda/Geometry/ImageCuda.h"
#include "Open3D/Utility/Console.h"
#include "RGBDOdometryCuda.h"

namespace open3d {
namespace cuda {
/**
 * Client end
 * TODO: Think about how do we use device_ ... we don't want copy
 * constructors for such a large system...
 */
template <size_t N>
RGBDOdometryCuda<N>::RGBDOdometryCuda() : device_(nullptr) {}

template <size_t N>
RGBDOdometryCuda<N>::~RGBDOdometryCuda() {
    Release();
}

template <size_t N>
void RGBDOdometryCuda<N>::SetParameters(const odometry::OdometryOption &option,
                                        float sigma,
                                        OdometryType odometry_type) {
    assert(option_.iteration_number_per_pyramid_level_.size() == N);
    option_ = option;
    sigma_ = sigma;
    odometry_type_ = odometry_type;
}

template <size_t N>
void RGBDOdometryCuda<N>::SetIntrinsics(
        camera::PinholeCameraIntrinsic intrinsics) {
    intrinsics_ = intrinsics;
}

template <size_t N>
bool RGBDOdometryCuda<N>::Create(int width, int height) {
    assert(width > 0 && height > 0);

    if (device_ != nullptr) {
        if (source_depth_[0].width_ != width ||
            source_depth_[0].height_ != height) {
            utility::LogError(
                    "[RGBDOdometryCuda] Incompatible image size, "
                    "width: {} vs {}, height: {} vs {}, "
                    "@Create aborted.",
                    source_depth_[0].width_, width, source_depth_[0].height_,
                    height);
            return false;
        }
        return true;
    }

    device_ = std::make_shared<RGBDOdometryCudaDevice<N>>();

    //! TODO(Akash): Conditional allocation can be dangerous
    if (odometry_type_ == OdometryType::FRAME_TO_FRAME) {
        source_input_.Create(width, height);
        target_input_.Create(width, height);

        for (int i = 0; i < N; ++i) {
            source_depth_[i].Create(width >> i, height >> i);
            source_intensity_[i].Create(width >> i, height >> i);
            target_depth_[i].Create(width >> i, height >> i);
            target_intensity_[i].Create(width >> i, height >> i);

            target_depth_dx_[i].Create(width >> i, height >> i);
            target_depth_dy_[i].Create(width >> i, height >> i);
            target_intensity_dx_[i].Create(width >> i, height >> i);
            target_intensity_dy_[i].Create(width >> i, height >> i);
        }
    } else {
        source_input_.Create(width, height);
        target_input_vertex_.Create(width, height);
        target_input_normal_.Create(width, height);
        target_input_color_.Create(width, height);

        for (int i = 0; i < N; ++i) {
            source_vertex_[i].Create(width >> i, height >> i);
            source_normal_[i].Create(width >> i, height >> i);
            source_intensity_[i].Create(width >> i, height >> i);
            target_vertex_[i].Create(width >> i, height >> i);
            target_normal_[i].Create(width >> i, height >> i);
            target_intensity_[i].Create(width >> i, height >> i);

            //! Normals will provide the gradient for depth component
            target_intensity_dx_[i].Create(width >> i, height >> i);
            target_intensity_dy_[i].Create(width >> i, height >> i);
        }
    }

    results_.Create(29);  // 21 + 6 + 2
    correspondences_.Create(width * height);

    transform_source_to_target_ = Eigen::Matrix4d::Identity();

    UpdateDevice();
    return true;
}

template <size_t N>
void RGBDOdometryCuda<N>::Release() {
    source_input_.Release();
    target_input_.Release();

    for (int i = 0; i < N; ++i) {
        source_depth_[i].Release();
        source_intensity_[i].Release();
        target_depth_[i].Release();
        target_intensity_[i].Release();

        target_depth_dx_[i].Release();
        target_depth_dy_[i].Release();
        target_intensity_dx_[i].Release();
        target_intensity_dy_[i].Release();
    }

    results_.Release();
    correspondences_.Release();

    device_ = nullptr;
}

template <size_t N>
void RGBDOdometryCuda<N>::UpdateSigma(float sigma) {
    device_->sigma_ = sigma;
    device_->sqrt_coeff_D_ = sqrtf(sigma);
    device_->sqrt_coeff_I_ = sqrtf(1 - sigma);
}

template <size_t N>
void RGBDOdometryCuda<N>::UpdateDevice() {
    if (device_ != nullptr) {
        device_->odometry_type_ = odometry_type_;
        if (odometry_type_ == FRAME_TO_FRAME) {
            source_input_.UpdateDevice();
            device_->source_input_ = *source_input_.device_;

            target_input_.UpdateDevice();
            device_->target_input_ = *target_input_.device_;

            for (int i = 0; i < N; ++i) {
                source_depth_[i].UpdateDevice();
                device_->source_depth_[i] = *source_depth_[i].device_;
                source_intensity_[i].UpdateDevice();
                device_->source_intensity_[i] = *source_intensity_[i].device_;

                target_depth_[i].UpdateDevice();
                device_->target_depth_[i] = *target_depth_[i].device_;
                target_intensity_[i].UpdateDevice();
                device_->target_intensity_[i] = *target_intensity_[i].device_;

                target_depth_dx_[i].UpdateDevice();
                device_->target_depth_dx_[i] = *target_depth_dx_[i].device_;
                target_depth_dy_[i].UpdateDevice();
                device_->target_depth_dy_[i] = *target_depth_dy_[i].device_;

                target_intensity_dx_[i].UpdateDevice();
                device_->target_intensity_dx_[i] =
                        *target_intensity_dx_[i].device_;
                target_intensity_dy_[i].UpdateDevice();
                device_->target_intensity_dy_[i] =
                        *target_intensity_dy_[i].device_;
            }
        } else {
            source_input_.UpdateDevice();
            device_->source_input_ = *source_input_.device_;

            target_input_vertex_.UpdateDevice();
            target_input_normal_.UpdateDevice();
            target_input_color_.UpdateDevice();
            device_->target_input_vertex_ = *target_input_vertex_.device_;
            device_->target_input_normal_ = *target_input_normal_.device_;
            device_->target_input_color_ = *target_input_color_.device_;

            for (int i = 0; i < N; ++i) {
                source_vertex_[i].UpdateDevice();
                device_->source_vertex_[i] = *source_vertex_[i].device_;
                source_normal_[i].UpdateDevice();
                device_->source_normal_[i] = *source_normal_[i].device_;
                source_intensity_[i].UpdateDevice();
                device_->source_intensity_[i] = *source_intensity_[i].device_;

                target_vertex_[i].UpdateDevice();
                device_->target_vertex_[i] = *target_vertex_[i].device_;
                target_normal_[i].UpdateDevice();
                device_->target_normal_[i] = *target_normal_[i].device_;
                target_intensity_[i].UpdateDevice();
                device_->target_intensity_[i] = *target_intensity_[i].device_;

                target_intensity_dx_[i].UpdateDevice();
                device_->target_intensity_dx_[i] =
                        *target_intensity_dx_[i].device_;
                target_intensity_dy_[i].UpdateDevice();
                device_->target_intensity_dy_[i] =
                        *target_intensity_dy_[i].device_;
            }
        }
        device_->results_ = *results_.device_;
        device_->correspondences_ = *correspondences_.device_;

        /** Update parameters **/
        device_->min_depth_ = (float)option_.min_depth_;
        device_->max_depth_ = (float)option_.max_depth_;
        device_->max_depth_diff_ = (float)option_.max_depth_diff_;

        UpdateSigma(sigma_);

        device_->intrinsics_[0] = PinholeCameraIntrinsicCuda(intrinsics_);
        for (size_t i = 1; i < N; ++i) {
            device_->intrinsics_[i] = device_->intrinsics_[i - 1].Downsample();
        }
        device_->transform_source_to_target_.FromEigen(
                transform_source_to_target_);
    }
}

template <size_t N>
void RGBDOdometryCuda<N>::Initialize(RGBDImageCuda &source,
                                     RGBDImageCuda &target) {
    assert(source.width_ == target.width_);
    assert(source.height_ == target.height_);

    bool success = Create(source.width_, source.height_);
    if (!success) {
        utility::LogError(
                "[RGBDOdometryCuda] create failed, "
                "@PrepareData aborted.");
        return;
    }

    source_input_.CopyFrom(source);
    target_input_.CopyFrom(target);

    /** Preprocess: truncate depth to nan values, then perform Gaussian **/
    ImageCudaf source_depth_preprocessed, source_intensity_preprocessed;
    ImageCudaf target_depth_preprocessed, target_intensity_preprocessed;
    source_depth_preprocessed.Create(source.width_, source.height_);
    source_intensity_preprocessed.Create(source.width_, source.height_);
    target_depth_preprocessed.Create(source.width_, source.height_);
    target_intensity_preprocessed.Create(source.width_, source.height_);
    RGBDOdometryCudaKernelCaller<N>::PreprocessInput(
            *this, source_depth_preprocessed, source_intensity_preprocessed,
            target_depth_preprocessed, target_intensity_preprocessed);

    /** Preprocess: Smooth **/
    source_depth_preprocessed.Gaussian(source_depth_[0], Gaussian3x3);
    source_intensity_preprocessed.Gaussian(source_intensity_[0], Gaussian3x3);
    target_depth_preprocessed.Gaussian(target_depth_[0], Gaussian3x3);
    target_intensity_preprocessed.Gaussian(target_intensity_[0], Gaussian3x3);

    /** Preprocess: normalize intensity between pair (source_[0], target_[0])
     * **/
    device_->transform_source_to_target_.FromEigen(transform_source_to_target_);
    correspondences_.set_iterator(0);
    RGBDOdometryCudaKernelCaller<N>::NormalizeIntensity(*this);

    /* Downsample */
    for (int i = 1; i < N; ++i) {
        source_depth_[i - 1].Downsample(source_depth_[i], BoxFilter);
        target_depth_[i - 1].Downsample(target_depth_[i], BoxFilter);

        auto tmp = source_intensity_[i - 1].Gaussian(Gaussian3x3);
        tmp.Downsample(source_intensity_[i], BoxFilter);
        tmp = target_intensity_[i - 1].Gaussian(Gaussian3x3);
        tmp.Downsample(target_intensity_[i], BoxFilter);
    }

    /* Compute gradients */
    for (int i = 0; i < N; ++i) {
        target_depth_[i].Sobel(target_depth_dx_[i], target_depth_dy_[i]);
        target_intensity_[i].Sobel(target_intensity_dx_[i],
                                   target_intensity_dy_[i]);
    }

    UpdateDevice();
}

template <size_t N>
void RGBDOdometryCuda<N>::Initialize(RGBDImageCuda &source,
                                     ImageCuda<float, 3> &target_vertex,
                                     ImageCuda<float, 3> &target_normal,
                                     ImageCuda<uchar, 3> &target_color) {
    assert((source.width_ == target_vertex.width_) &&
           (source.width_ == target_normal.width_) &&
           (source.width_ == target_color.width_));
    assert((source.height_ == target_vertex.height_) &&
           (source.height_ == target_normal.height_) &&
           (source.width_ == target_color.width_));

    if (odometry_type_ == OdometryType::FRAME_TO_FRAME) {
        utility::LogError(
                "[RGBDOdometryCuda] Frame to Frame requires Initialize() with "
                "two RGBDImages, "
                "@Initialize aborted.");
        return;
    }

    bool success = Create(source.width_, source.height_);
    if (!success) {
        utility::LogError(
                "[RGBDOdometryCuda] create failed, "
                "@PrepareData aborted.");
        return;
    }

    source_input_.CopyFrom(source);
    target_input_vertex_.CopyFrom(target_vertex);
    target_input_normal_.CopyFrom(target_normal);
    target_input_color_.CopyFrom(target_color);

    /** Preprocess: truncate depth to nan values. Only required for source,
     * since vertex and normals already have NAN **/
    ImageCudaf source_depth_preprocessed, source_intensity_preprocessed;
    ImageCudaf target_intensity_preprocessed;

    source_depth_preprocessed.Create(source.width_, source.height_);
    source_intensity_preprocessed.Create(source.width_, source.height_);
    target_intensity_preprocessed.Create(source.width_, source.height_);

    RGBDOdometryCudaKernelCaller<N>::PreprocessInput(
            *this, source_depth_preprocessed, source_intensity_preprocessed,
            target_intensity_preprocessed);

    /** Preprocess: Smooth **/
    auto bilateral_depth = source_depth_preprocessed.Bilateral();
    bilateral_depth.GetVertexMap(source_vertex_[0], device_->intrinsics_[0]);
    source_vertex_[0].GetNormalMap(source_normal_[0]);
    source_intensity_preprocessed.Gaussian(source_intensity_[0], Gaussian3x3);

    target_vertex_[0].CopyFrom(target_input_vertex_);
    target_normal_[0].CopyFrom(target_input_normal_);
    target_intensity_preprocessed.Gaussian(target_intensity_[0], Gaussian3x3);

    /** Preprocess: normalize intensity between pair (source_[0], target_[0])
     * **/
    device_->transform_source_to_target_.FromEigen(transform_source_to_target_);
    correspondences_.set_iterator(0);
    RGBDOdometryCudaKernelCaller<N>::NormalizeIntensity(*this);

    /* Downsample */
    for (int i = 1; i < N; ++i) {
        source_vertex_[i - 1].Downsample(source_vertex_[i], BoxFilter);
        source_normal_[i - 1].Downsample(source_normal_[i], BoxFilterNormalize);
        target_vertex_[i - 1].Downsample(target_vertex_[i], BoxFilter);
        target_normal_[i - 1].Downsample(target_normal_[i], BoxFilterNormalize);

        auto tmp = source_intensity_[i - 1].Gaussian(Gaussian3x3);
        tmp.Downsample(source_intensity_[i], BoxFilter);
        tmp = target_intensity_[i - 1].Gaussian(Gaussian3x3);
        tmp.Downsample(target_intensity_[i], BoxFilter);
    }

    /* Compute gradients */
    for (int i = 0; i < N; ++i) {
        //! target depth gradients are the target normals
        target_intensity_[i].Sobel(target_intensity_dx_[i],
                                   target_intensity_dy_[i]);
    }

    UpdateDevice();
}

template <size_t N>
std::tuple<bool, Eigen::Matrix4d, float> RGBDOdometryCuda<N>::DoSingleIteration(
        size_t level, int iter) {
    results_.Memset(0);
    correspondences_.set_iterator(0);

    device_->transform_source_to_target_.FromEigen(transform_source_to_target_);

    utility::Timer timer;
    timer.Start();
    RGBDOdometryCudaKernelCaller<N>::DoSingleIteration(*this, level);
    timer.Stop();

    std::vector<float> results = results_.DownloadAll();

    Eigen::Matrix6d JtJ;
    Eigen::Vector6d Jtr;
    float loss, inliers;
    ExtractResults(results, JtJ, Jtr, loss, inliers);
    utility::LogDebug(
            "> Level {}, iter {}: loss = {}, avg loss = {}, "
            "inliers = {}",
            level, iter, loss, loss / inliers, inliers);
    utility::LogDebug("JtJ: \n{}\n, Jtr: \n{}\n", JtJ, Jtr);
    bool is_success;
    Eigen::Matrix4d extrinsic;
    std::tie(is_success, extrinsic) =
            utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JtJ, Jtr);

    return std::make_tuple(is_success, extrinsic, loss / inliers);
}

template <size_t N>
std::tuple<bool, Eigen::Matrix4d, std::vector<std::vector<float>>>
RGBDOdometryCuda<N>::ComputeMultiScale() {
    bool is_success;
    Eigen::Matrix4d delta;
    float loss;

    std::vector<std::vector<float>> losses;
    for (int level = (int)(N - 1); level >= 0; --level) {
        std::vector<float> losses_on_level;

        for (int iter = 0;
             iter < option_.iteration_number_per_pyramid_level_[N - 1 - level];
             ++iter) {
            std::tie(is_success, delta, loss) =
                    DoSingleIteration((size_t)level, iter);
            transform_source_to_target_ = delta * transform_source_to_target_;
            losses_on_level.emplace_back(loss);

            if (!is_success) {
                utility::LogWarning("[ComputeOdometry] no solution!");
                return std::make_tuple(false, Eigen::Matrix4d::Identity(),
                                       losses);
            }
        }

        losses.emplace_back(losses_on_level);
    }

    return std::make_tuple(true, transform_source_to_target_, losses);
}

template <size_t N>
Eigen::Matrix6d RGBDOdometryCuda<N>::ComputeInformationMatrix() {
    results_.Memset(0);

    RGBDOdometryCudaKernelCaller<N>::ComputeInformationMatrix(*this);
    std::vector<float> results = results_.DownloadAll();

    Eigen::Matrix6d JtJ;
    Eigen::Vector6d Jtr;  // dummy
    float loss, inliers;  // dummy
    ExtractResults(results, JtJ, Jtr, loss, inliers);

    return JtJ;
}
}  // namespace cuda
}  // namespace open3d
