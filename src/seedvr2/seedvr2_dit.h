#ifndef SEEDVR2_DIT_H
#define SEEDVR2_DIT_H

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <net.h>

#include "adaptive_window_attention.h"

namespace seedvr2 {

struct DiTBlockInputs
{
    ncnn::Mat video;
    ncnn::Mat text;
    std::vector<std::array<int, 3> > video_shapes;
    std::vector<int> text_lengths;
    std::array<ncnn::Mat, 6> modulation;
};

struct DiTInputs
{
    // Flattened T-H-W tokens with 33 channels, before the official 1x2x2
    // patchification. Samples are concatenated in video_shapes order.
    ncnn::Mat video;
    // Flattened text tokens with 5120 channels.
    ncnn::Mat text;
    std::vector<std::array<int, 3> > video_shapes;
    std::vector<int> text_lengths;
    std::vector<float> timesteps;
};

int split_dit_inputs_by_sample(const DiTInputs& inputs,
                               std::vector<DiTInputs>& samples);
int split_dit_block_inputs_by_sample(
    const DiTBlockInputs& inputs,
    std::vector<DiTBlockInputs>& samples);

class SeedVR2DiTBlocks
{
public:
    SeedVR2DiTBlocks();
    ~SeedVR2DiTBlocks();

    int load(const std::string& model_directory, int gpu_id = 0,
             bool use_bf16_storage = false);
    int forward_frontend(const DiTInputs& inputs,
                         DiTBlockInputs& outputs) const;
    int forward(const DiTInputs& inputs, ncnn::Mat& video_output,
                ncnn::Mat& text_output) const;
    int forward(const DiTBlockInputs& inputs, ncnn::Mat& video_output,
                ncnn::Mat& text_output) const;
    void clear();

private:
    struct Block;
    int record_blocks(const std::vector<std::array<int, 3> >& video_shapes,
                      const std::vector<int>& text_lengths,
                      ncnn::VkMat& video, ncnn::VkMat& text,
                      const ncnn::VkMat& video_shape,
                      const ncnn::VkMat& text_length,
                      const std::array<ncnn::VkMat, 6>& modulation,
                      ncnn::VkCompute& command,
                      ncnn::VkAllocator* blob_allocator,
                      ncnn::VkAllocator* staging_allocator) const;

    ncnn::Net frontend;
    ncnn::Net tail;
    std::vector<std::unique_ptr<Block> > blocks;
    int gpu_id;
    bool bf16_storage;
    bool frontend_loaded;
    bool tail_loaded;
};

} // namespace seedvr2

#endif // SEEDVR2_DIT_H
