#ifndef SEEDVR2_VAE_DOWNSAMPLE3D_H
#define SEEDVR2_VAE_DOWNSAMPLE3D_H

#include <layer.h>

#if NCNN_VULKAN
#include <command.h>
#include <pipeline.h>
#endif

namespace seedvr2 {

class VAEDownsample3D : public ncnn::Layer
{
public:
    VAEDownsample3D();
    virtual ~VAEDownsample3D();

    virtual int load_model(const ncnn::ModelBin& mb);
    virtual int create_pipeline(const ncnn::Option& opt);
    virtual int destroy_pipeline(const ncnn::Option& opt);
    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob,
                        const ncnn::Option& opt) const;

    void release_completed_temporal_memory() const;
    void reset_temporal_memory() const;

#if NCNN_VULKAN
    virtual int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt);
    virtual int forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob,
                        ncnn::VkCompute& cmd, const ncnn::Option& opt) const;
#endif

private:
    int channels;
    int pad_right;
    int pad_bottom;
    ncnn::Layer* convolution;
#if NCNN_VULKAN
    ncnn::Pipeline* pipeline_pad;
#endif
};

ncnn::Layer* vae_downsample3d_layer_creator(void* userdata);

} // namespace seedvr2

#endif // SEEDVR2_VAE_DOWNSAMPLE3D_H
