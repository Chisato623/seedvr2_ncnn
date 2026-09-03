#ifndef SEEDVR2_PIPELINE_H
#define SEEDVR2_PIPELINE_H

#include <cstdint>
#include <string>

#include <net.h>

#include "seedvr2_dit.h"
#include "seedvr2_vae.h"

namespace seedvr2 {

struct PipelineOptions
{
    int steps = 1;
    float cfg_scale = 1.f;
    float cfg_rescale = 0.f;
    uint64_t seed = 666;
    ncnn::Mat initial_noise;
};

class SeedVR2Pipeline
{
public:
    int load(const std::string& dit_directory,
             const std::string& vae_directory, int gpu_id = 0);
    int load_staged(const std::string& dit_directory,
                    const std::string& vae_directory, int gpu_id = 0);
    int restore(const ncnn::Mat& input_video,
                const ncnn::Mat& positive_text,
                const ncnn::Mat& negative_text,
                ncnn::Mat& output_video,
                const PipelineOptions& options = PipelineOptions());
    void clear();

private:
    int predict(const ncnn::Mat& latent, const ncnn::Mat& condition,
                const ncnn::Mat& text, float timestep,
                ncnn::Mat& prediction);
    int denoise(ncnn::Mat& latent, const ncnn::Mat& condition,
                const ncnn::Mat& positive_text,
                const ncnn::Mat& negative_text,
                const PipelineOptions& options);
    int load_vae_stage();
    void clear_vae_stage();

    SeedVR2DiTBlocks dit;
    SeedVR2VAE vae;
    std::string dit_directory;
    std::string vae_directory;
    int gpu_id = -1;
    bool staged = false;
    bool vae_stage_active = false;
    bool loaded = false;
};

} // namespace seedvr2

#endif // SEEDVR2_PIPELINE_H
