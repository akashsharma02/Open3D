//
// Created by wei on 3/31/19.
//

#include <Cuda/Open3DCuda.h>
#include <Open3D/Open3D.h>

#include "../ReconstructionSystem/DatasetConfig.h"

using namespace open3d;
using namespace open3d::registration;
using namespace open3d::geometry;
using namespace open3d::io;
using namespace open3d::utility;

void IntegrateAndWriteFragment(int fragment_id, DatasetConfig &config) {
    PoseGraph pose_graph;
    ReadPoseGraph(config.GetPoseGraphFileForFragment(fragment_id, true),
                  pose_graph);

    int voxel_resolution = 256;
    float voxel_length = config.voxel_size_;

    cuda::PinholeCameraIntrinsicCuda intrinsic(config.intrinsic_);
    cuda::TransformCuda trans = cuda::TransformCuda::Identity();

    float offset = -(voxel_length * voxel_resolution)/2;
    cuda::TransformCuda extrinsics = cuda::TransformCuda::Identity();
    extrinsics.SetTranslation(cuda::Vector3f(offset, offset, offset));

    cuda::UniformTSDFVolumeCuda tsdf_volume(voxel_resolution, voxel_length,
                                            3 * voxel_length, extrinsics);

    cuda::RGBDImageCuda rgbd((float)config.max_depth_,
                             (float)config.depth_factor_);


    const int begin = fragment_id * config.n_frames_per_fragment_;
    const int end = std::min((fragment_id + 1) * config.n_frames_per_fragment_,
                             (int)config.color_files_.size());

    Timer timer;
    timer.Start();
    for (int i = begin; i < end; ++i) {
        LogDebug("Integrating frame {} ...\n", i);

        Image depth, color;
        ReadImage(config.depth_files_[i], depth);
        ReadImage(config.color_files_[i], color);
        rgbd.Upload(depth, color);

        /* Use ground truth trajectory */
        Eigen::Matrix4d pose = pose_graph.nodes_[i - begin].pose_;
        trans.FromEigen(pose);

        tsdf_volume.Integrate(rgbd, intrinsic, trans);
    }
    timer.Stop();
    utility::LogInfo("Integration takes {} ms\n", timer.GetDuration());

    cuda::UniformMeshVolumeCuda mesher(cuda::VertexWithNormalAndColor,
                                       voxel_resolution, 4000000, 8000000);
    WriteUniformTSDFVolumeToBIN("tmp.bin", tsdf_volume, true);
    mesher.MarchingCubes(tsdf_volume);
    auto mesh = mesher.mesh().Download();
    visualization::DrawGeometries({mesh});
}

void ReadFragment(int fragment_id, DatasetConfig &config) {
    PoseGraph pose_graph;
    ReadPoseGraph(config.GetPoseGraphFileForFragment(fragment_id, true),
                  pose_graph);

    Timer timer;
    timer.Start();
    auto tsdf_volume = io::ReadUniformTSDFVolumeFromBIN("tmp.bin", true);
    timer.Stop();
    utility::LogInfo("Read takes {} ms\n", timer.GetDuration());

    cuda::UniformMeshVolumeCuda mesher(cuda::VertexWithNormalAndColor,
                                       tsdf_volume.N_, 4000000, 8000000);
    mesher.MarchingCubes(tsdf_volume);
    auto mesh = mesher.mesh().Download();
    visualization::DrawGeometries({mesh});
}

int main(int argc, char **argv) {
    DatasetConfig config;
    std::string config_path =
            argc > 1 ? argv[1]
                     : kDefaultDatasetConfigDir + "/bundlefusion/copyroom.json";
    bool is_success = io::ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;
    config.GetFragmentFiles();

    for (int i = 0; i < 1; ++i) {  // config.fragment_files_.size(); ++i) {
        utility::LogInfo("{}\n", i);
        IntegrateAndWriteFragment(i, config);
        ReadFragment(i, config);
    }
}
