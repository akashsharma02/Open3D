#include <Cuda/Integration/RGBDToTSDF.h>
#include <Cuda/Open3DCuda.h>
#include <Cuda/Visualization/Visualizer/VisualizerWithCudaModule.h>
#include <Open3D/Open3D.h>
#include <Open3D/Visualization/Visualizer/Visualizer.h>

#include "../ReconstructionSystem/DatasetConfig.h"

using namespace open3d;
using namespace open3d::registration;
using namespace open3d::geometry;
using namespace open3d::io;
using namespace open3d::utility;
using namespace open3d::visualization;

cuda::ScalableTSDFVolumeCuda ReadTSDFVolume(const std::string &filename,
                                            DatasetConfig &config) {
    auto tsdf_volume = io::ReadScalableTSDFVolumeFromBIN(filename);
    return tsdf_volume;
}

int Test(const std::string &source_path,
         const Eigen::Matrix4d &init_source_to_target,
         DatasetConfig &config) {
    auto source = ReadTSDFVolume(source_path, config);
}

int main(int argc, char **argv) {
    DatasetConfig config;
    std::string config_path =
            argc > 1 ? argv[1]
                     : kDefaultDatasetConfigDir + "/stanford/lounge.json";
    bool is_success = io::ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;

    int fragment_id = 0;
    const int begin = fragment_id * config.n_frames_per_fragment_;
    const int end = std::min((fragment_id + 1) * config.n_frames_per_fragment_,
                             (int)config.color_files_.size());
    PoseGraph pose_graph;
    ReadPoseGraph(config.GetPoseGraphFileForFragment(fragment_id, true),
                  pose_graph);

    cuda::PinholeCameraIntrinsicCuda intrinsic(config.intrinsic_);
    cuda::TransformCuda trans = cuda::TransformCuda::Identity();
    cuda::ScalableTSDFVolumeCuda tsdf_volume(
            16, (float)config.tsdf_cubic_size_ / 512,
            (float)config.tsdf_truncation_, config.max_depth_, trans, 40000, 60000);
    cuda::RGBDImageCuda rgbd((float)config.max_depth_,
                             (float)config.depth_factor_);

    Eigen::Matrix4d pose, prev_pose;
    for (int i = begin; i < begin + 13; ++i) {
        utility::LogInfo("Integrating frame {} ...", i);

        Image depth, color;
        ReadImage(config.depth_files_[i], depth);
        ReadImage(config.color_files_[i], color);
        rgbd.Upload(depth, color);

        /* Use ground truth trajectory */
        if (i - begin <= 10) {
            pose = pose_graph.nodes_[i - begin].pose_;
            trans.FromEigen(pose);
        } else {
            trans.FromEigen(prev_pose);
            auto result =
                    RGBDToTSDFRegistration(rgbd, tsdf_volume, intrinsic, trans);
            pose = std::get<1>(result);
            trans.FromEigen(pose);

            std::cout << "------\n";
            std::cout << pose << "\n";
            std::cout << pose_graph.nodes_[i - begin].pose_ << "\n";
            std::cout << "------\n";
        }
        tsdf_volume.Integrate(rgbd, intrinsic, trans);
        prev_pose = pose;
    }

    tsdf_volume.GetAllSubvolumes();
    cuda::ScalableMeshVolumeCuda mesher(
            cuda::VertexWithNormalAndColor, 16,
            tsdf_volume.active_subvolume_entry_array_.size(), 40000000,
            40000000);
    mesher.MarchingCubes(tsdf_volume);
    auto mesh = mesher.mesh().Download();
    visualization::DrawGeometries({mesh});
}
