//
// Created by wei on 3/28/19.
//

#include "ScalableTSDFVolumeCudaIO.h"

#include "ZlibIO.h"

namespace open3d {
namespace io {
bool WriteScalableTSDFVolumeToBIN(
        const std::string &filename,
        cuda::ScalableTSDFVolumeCuda &volume,
        std::pair<std::vector<cuda::Vector3i>,
                  std::vector<cuda::ScalableTSDFVolumeCpuData>> key_value,
        bool use_zlib) {
    auto keys = key_value.first;
    auto values = key_value.second;
    assert(keys.size() == values.size());

    FILE *fid = fopen(filename.c_str(), "wb");
    if (fid == NULL) {
        utility::LogWarning("Write BIN failed: unable to open file: %s\n",
                            filename.c_str());
        return false;
    }

    /** metadata **/
    int num_volumes = keys.size(), volume_size = volume.N_;
    if (fwrite(&num_volumes, sizeof(int), 1, fid) < 1) {
        utility::LogWarning("Write BIN failed: unable to write num volumes\n");
        return false;
    }
    if (fwrite(&volume.N_, sizeof(int), 1, fid) < 1) {
        utility::LogWarning("Write BIN failed: unable to write volume size\n");
        return false;
    }
    if (fwrite(&volume.bucket_count_, sizeof(int), 1, fid) < 1) {
        utility::LogWarning("Write BIN failed: unable to write bucket count\n");
        return false;
    }
    if (fwrite(&volume.value_capacity_, sizeof(int), 1, fid) < 1) {
        utility::LogWarning(
                "Write BIN failed: unable to read write capacity\n");
        return false;
    }

    float voxel_length, sdf_trunc, max_depth;
    if (fwrite(&volume.voxel_length_, sizeof(float), 1, fid) < 1) {
        utility::LogWarning("Write BIN failed: unable to write voxel length\n");
        return false;
    }
    if (fwrite(&volume.sdf_trunc_, sizeof(float), 1, fid) < 1) {
        utility::LogWarning("Write BIN failed: unable to write sdf_trunc\n");
        return false;
    }
    if (fwrite(&volume.max_depth_, sizeof(float), 1, fid) < 1) {
        utility::LogWarning("Write BIN failed: unable to write max_depth\n");
        return false;
    }

    Eigen::Matrix4d transform = volume.transform_volume_to_world_.ToEigen();
    if (fwrite(transform.data(), sizeof(double), 16, fid) < 1) {
        utility::LogWarning("Write BIN failed: unable to write transform\n");
        return false;
    }

    /** keys **/
    if (!Write(keys, fid, "keys")) {
        return false;
    }

    /** subvolumes **/
    std::vector<uchar> compressed_buf;
    if (use_zlib) {
        compressed_buf.resize(/* provide some redundancy */
                              volume.N_ * volume.N_ * volume.N_ *
                              sizeof(float) * 2);
    }

    utility::LogInfo("Writing {} subvolumes.\n", keys.size());
    for (auto &subvolume : values) {
        auto &tsdf = subvolume.tsdf_;
        auto &weight = subvolume.weight_;
        auto &color = subvolume.color_;
        auto &fg = subvolume.fg_;
        auto &bg = subvolume.bg_;

        if (!use_zlib) {
            if (!Write(tsdf, fid, "TSDF")) {
                return false;
            }
            if (!Write(weight, fid, "weight")) {
                return false;
            }
            if (!Write(color, fid, "color")) {
                return false;
            }
            if (!Write(color, fid, "fg")) {
                return false;
            }
            if (!Write(color, fid, "bg")) {
                return false;
            }
        } else {
            if (!CompressAndWrite(compressed_buf, tsdf, fid, "TSDF")) {
                return false;
            }
            if (!CompressAndWrite(compressed_buf, weight, fid, "weight")) {
                return false;
            }
            if (!CompressAndWrite(compressed_buf, color, fid, "color")) {
                return false;
            }
            if (!CompressAndWrite(compressed_buf, fg, fid, "fg")) {
                return false;
            }
            if (!CompressAndWrite(compressed_buf, bg, fid, "bg")) {
                return false;
            }
        }
    }

    fclose(fid);
    return true;
}
bool WriteScalableTSDFVolumeToBIN(const std::string &filename,
                                  cuda::ScalableTSDFVolumeCuda &volume,
                                  bool use_zlib) {
    auto key_value = volume.DownloadVolumes();
    WriteScalableTSDFVolumeToBIN(filename, volume, key_value, use_zlib);
}

cuda::ScalableTSDFVolumeCuda ReadScalableTSDFVolumeFromBIN(
        const std::string &filename, bool use_zlib, int batch_size) {
    FILE *fid = fopen(filename.c_str(), "rb");
    if (fid == NULL) {
        utility::LogWarning("Read BIN failed: unable to open file: %s\n",
                            filename.c_str());
        throw std::runtime_error("ScalableTSDFVolume Read from bin error");
    }

    /** metadata **/
    int num_volumes, volume_size, bucket_count, value_capacity;
    if (fread(&num_volumes, sizeof(int), 1, fid) < 1) {
        utility::LogWarning("Read BIN failed: unable to read num volumes\n");
        throw std::runtime_error("ScalableTSDFVolume Read from bin error");
    }
    if (fread(&volume_size, sizeof(int), 1, fid) < 1) {
        utility::LogWarning("Read BIN failed: unable to read volume size\n");
        throw std::runtime_error("ScalableTSDFVolume Read from bin error");
    }
    if (fread(&bucket_count, sizeof(int), 1, fid) < 1) {
        utility::LogWarning("Read BIN failed: unable to read bucket count\n");
        throw std::runtime_error("ScalableTSDFVolume Read from bin error");
    }
    if (fread(&value_capacity, sizeof(int), 1, fid) < 1) {
        utility::LogWarning("Read BIN failed: unable to read value capacity\n");
        throw std::runtime_error("ScalableTSDFVolume Read from bin error");
    }

    float voxel_length, sdf_trunc, max_depth;
    if (fread(&voxel_length, sizeof(float), 1, fid) < 1) {
        utility::LogWarning("Read BIN failed: unable to read voxel length\n");
        throw std::runtime_error("ScalableTSDFVolume Read from bin error");
    }
    if (fread(&sdf_trunc, sizeof(float), 1, fid) < 1) {
        utility::LogWarning("Read BIN failed: unable to read sdf_trunc\n");
        throw std::runtime_error("ScalableTSDFVolume Read from bin error");
    }
    if (fread(&max_depth, sizeof(float), 1, fid) < 1) {
        utility::LogWarning("Read BIN failed: unable to read sdf_trunc\n");
        throw std::runtime_error("ScalableTSDFVolume Read from bin error");
    }

    Eigen::Matrix4d transform;
    if (fread(transform.data(), sizeof(double), 16, fid) < 1) {
        utility::LogWarning("Read BIN failed: unable to read transform\n");
        throw std::runtime_error("ScalableTSDFVolume Read from bin error");
    }

    cuda::TransformCuda transform_volume_to_world;
    transform_volume_to_world.FromEigen(transform);

    auto volume = cuda::ScalableTSDFVolumeCuda(
            volume_size, voxel_length, sdf_trunc, max_depth, transform_volume_to_world,
            bucket_count, value_capacity);

    int N = volume.N_;
    int NNN = N * N * N;

    /** keys **/
    std::vector<cuda::Vector3i> keys(num_volumes);
    if (!Read(keys, fid, "keys")) {
        throw std::runtime_error("ScalableTSDFVolume Read from bin error");
    }

    /** values **/
    std::vector<float> tsdf_buf(NNN);
    std::vector<uchar> weight_buf(NNN);
    std::vector<cuda::Vector3b> color_buf(NNN);
    std::vector<uint16_t> fg_buf(NNN);
    std::vector<uint16_t> bg_buf(NNN);

    std::vector<int> failed_key_indices;
    std::vector<cuda::Vector3i> failed_keys;
    std::vector<cuda::ScalableTSDFVolumeCpuData> failed_subvolumes;

    /** Updating (key, value) pairs in parallel is tricky:
     * - thread lock can forbid some block to be allocated;
     * - split all the pairs them into batches can increase success rate;
     * - we should retry to insert stubborn failure pairs until they get in. **/
    if (batch_size <= 0) {
        batch_size = int(volume.hash_table_.bucket_count_ * 0.2f);
    }
    int num_batches = (num_volumes + batch_size - 1) / batch_size;

    std::vector<uchar> compressed_buf(volume.N_ * volume.N_ * volume.N_ *
                                      sizeof(float) * 2);
    for (int batch = 0; batch < num_batches; ++batch) {
        std::vector<cuda::Vector3i> batch_keys;
        std::vector<cuda::ScalableTSDFVolumeCpuData> batch_subvolumes;

        int begin = batch * batch_size;
        int end = std::min((batch + 1) * batch_size, num_volumes);

        for (int i = begin; i < end; ++i) {
            if (!use_zlib) {
                if (!Read(tsdf_buf, fid, "TSDF")) {
                    throw std::runtime_error(
                            "ScalableTSDFVolume Read from bin error");
                }
                if (!Read(weight_buf, fid, "weight")) {
                    throw std::runtime_error(
                            "ScalableTSDFVolume Read from bin error");
                }
                if (!Read(color_buf, fid, "color")) {
                    throw std::runtime_error(
                            "ScalableTSDFVolume Read from bin error");
                }
                if (!Read(color_buf, fid, "fg")) {
                    throw std::runtime_error(
                            "ScalableTSDFVolume Read from bin error");
                }
                if (!Read(color_buf, fid, "fg")) {
                    throw std::runtime_error(
                            "ScalableTSDFVolume Read from bin error");
                }
            } else {
                if (!ReadAndUncompress(compressed_buf, tsdf_buf, fid, "TSDF")) {
                    throw std::runtime_error(
                            "ScalableTSDFVolume Read from bin error");
                }
                if (!ReadAndUncompress(compressed_buf, weight_buf, fid,
                                       "weight")) {
                    throw std::runtime_error(
                            "ScalableTSDFVolume Read from bin error");
                }
                if (!ReadAndUncompress(compressed_buf, color_buf, fid,
                                       "uchar")) {
                    throw std::runtime_error(
                            "ScalableTSDFVolume Read from bin error");
                }
                if (!ReadAndUncompress(compressed_buf, fg_buf, fid,
                                       "uint16_t")) {
                    throw std::runtime_error(
                            "ScalableTSDFVolume Read from bin error");
                }
                if (!ReadAndUncompress(compressed_buf, bg_buf, fid,
                                       "uint16_t")) {
                    throw std::runtime_error(
                            "ScalableTSDFVolume Read from bin error");
                }
            }
            batch_keys.emplace_back(keys[i]);
            batch_subvolumes.emplace_back(cuda::ScalableTSDFVolumeCpuData(
                    tsdf_buf, weight_buf, color_buf, fg_buf, bg_buf));
        }

        volume.UploadVolumes(batch_keys, batch_subvolumes);
    }
    fclose(fid);

    return volume;
}
}  // namespace io
}  // namespace open3d
