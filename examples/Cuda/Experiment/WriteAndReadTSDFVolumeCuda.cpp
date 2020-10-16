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

    float voxel_length = config.tsdf_cubic_size_ / 512.0;

    cuda::PinholeCameraIntrinsicCuda intrinsic(config.intrinsic_);
    cuda::TransformCuda trans = cuda::TransformCuda::Identity();
    cuda::ScalableTSDFVolumeCuda tsdf_volume(
            16, voxel_length, (float)config.tsdf_truncation_, trans);

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

    timer.Start();
    std::string filename = config.GetBinFileForFragment(fragment_id);
    io::WriteScalableTSDFVolumeToBIN("target-high.bin", tsdf_volume, true);
    timer.Stop();
    utility::LogInfo("Write TSDF takes {} ms\n", timer.GetDuration());

    tsdf_volume.GetAllSubvolumes();
    cuda::ScalableMeshVolumeCuda mesher(
            cuda::VertexWithNormalAndColor, 16,
            tsdf_volume.active_subvolume_entry_array_.size());
    mesher.MarchingCubes(tsdf_volume);

    auto no_constraint_minmax = tsdf_volume.GetMinMaxBound(0);
    auto constraint_minmax = tsdf_volume.GetMinMaxBound(400);

    auto no_constraint_aabb = std::make_shared<AxisAlignedBoundingBox>(
            no_constraint_minmax.first, no_constraint_minmax.second);
    auto constraint_aabb = std::make_shared<AxisAlignedBoundingBox>(
            constraint_minmax.first, constraint_minmax.second);

    auto mesh = mesher.mesh().Download();
    std::cout << "vertex count = " << mesh->vertices_.size() << "\n";
    visualization::DrawGeometries({mesh, constraint_aabb, no_constraint_aabb});
    //
    //    PointCloud pcl;
    //    pcl.points_ = mesh->vertices_;
    //    pcl.normals_ = mesh->vertex_normals_;
    //    pcl.colors_ = mesh->vertex_colors_;
    //
    //    /** Write original fragments **/
    //    timer.Start();
    //    WritePointCloudToPLY("test.ply", pcl);
    //    timer.Stop();
    //    utility::LogInfo("Write ply takes {} ms\n", timer.GetDuration());
}

void ReadFragment(int fragment_id, DatasetConfig &config) {
    PoseGraph pose_graph;
    ReadPoseGraph(config.GetPoseGraphFileForFragment(fragment_id, true),
                  pose_graph);

    float voxel_length = config.tsdf_cubic_size_ / 512.0;

    // cuda::TransformCuda trans = cuda::TransformCuda::Identity();
    // cuda::ScalableTSDFVolumeCuda tsdf_volume(
    //         16, voxel_length, (float)config.tsdf_truncation_, trans);

    Timer timer;
    timer.Start();
    std::string filename = config.GetBinFileForFragment(fragment_id);
    cuda::ScalableTSDFVolumeCuda tsdf_volume =
            io::ReadScalableTSDFVolumeFromBIN("target-high.bin", true);
    std::cout << tsdf_volume.GetMinBound() << "\n";
    std::cout << tsdf_volume.GetMaxBound() << "\n";
    timer.Stop();
    utility::LogInfo("Read takes {} ms\n", timer.GetDuration());
    auto aabb = std::make_shared<AxisAlignedBoundingBox>(
            tsdf_volume.GetMinBound(), tsdf_volume.GetMaxBound());

    tsdf_volume.GetAllSubvolumes();
    cuda::ScalableMeshVolumeCuda mesher(
            cuda::VertexWithNormalAndColor, 16,
            tsdf_volume.active_subvolume_entry_array_.size());
    mesher.MarchingCubes(tsdf_volume);
    auto mesh = mesher.mesh().Download();
    std::cout << mesh->GetMinBound() << "\n";
    std::cout << mesh->GetMaxBound() << "\n";
    visualization::DrawGeometries({mesh, aabb});
}

int main(int argc, char **argv) {
    DatasetConfig config;
    std::string config_path =
            argc > 1 ? argv[1]
                     : kDefaultDatasetConfigDir + "/stanford/lounge.json";
    bool is_success = io::ReadIJsonConvertible(config_path, config);
    if (!is_success) return 1;
    config.GetFragmentFiles();

    for (int i = 1; i < 2; ++i) {  // config.fragment_files_.size(); ++i) {
        utility::LogInfo("{}\n", i);
        IntegrateAndWriteFragment(i, config);
        ReadFragment(i, config);
    }
}
