#ifndef SEEDVR2_VAE_H
#define SEEDVR2_VAE_H

#include <string>

#include <net.h>

namespace seedvr2 {

// Dynamic spatiotemporal runtime for the official SeedVR2 VAE. Encoder inputs
// longer than five frames are sliced as 5 + 4n and decoder inputs longer than
// two latent frames are sliced as 2 + 1n. Video inputs must be padded to a
// 4n+1 frame count before long-video encoding. Spatial dimensions must be
// divisible by eight and the model must use a dynamic-spatiotemporal graph.
class SeedVR2VAE
{
public:
    SeedVR2VAE();
    ~SeedVR2VAE();

    int load(const std::string& model_directory, int gpu_id = 0);
    int encode_moments(const ncnn::Mat& video, ncnn::Mat& moments) const;
    int encode(const ncnn::Mat& video, ncnn::Mat& latent) const;
    int decode(const ncnn::Mat& latent, ncnn::Mat& video) const;
    void clear();

    static constexpr float scaling_factor() { return 0.9152f; }

private:
    ncnn::Net encoder;
    ncnn::Net decoder;
    ncnn::VkAllocator* memory_vkallocator;
    int gpu_id;
    bool loaded;
};

} // namespace seedvr2

#endif // SEEDVR2_VAE_H
