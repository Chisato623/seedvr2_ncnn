#ifndef SEEDVR2_DECOMPOSED_CONV3D_H
#define SEEDVR2_DECOMPOSED_CONV3D_H

#include <vector>

#include <layer.h>

#if NCNN_VULKAN
#include <command.h>
#include <pipeline.h>
#endif

namespace seedvr2 {

enum class MemoryState
{
    DISABLED,
    INITIALIZING,
    ACTIVE
};

MemoryState get_vae_memory_state();
void set_vae_memory_state(MemoryState state);
#if NCNN_VULKAN
ncnn::VkAllocator* get_vae_memory_vkallocator();
void set_vae_memory_vkallocator(ncnn::VkAllocator* allocator);
#endif

// Exact Conv3D decomposition used by the experimental SeedVR2 VAE exporter.
// NCNN stores [B,C,T,H,W] as a batched 4D Mat with T in the physical depth
// dimension. This layer gathers one input frame per temporal kernel, delegates
// spatial work to NCNN Convolution, and accumulates the frame results.
class DecomposedConv3D : public ncnn::Layer
{
public:
    DecomposedConv3D();
    virtual ~DecomposedConv3D();

    virtual int load_param(const ncnn::ParamDict& pd);
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
    int num_output;
    int num_input;
    int kernel_w;
    int kernel_h;
    int kernel_t;
    int stride_t;
    int dilation_t;
    int head_frames;
    int stride_w;
    int stride_h;
    int dilation_w;
    int dilation_h;
    int pad_w;
    int pad_h;
    int groups;
    bool bias_term;

    ncnn::Mat bias_data;
    std::vector<ncnn::Mat> spatial_weights;
    mutable ncnn::Mat temporal_memory;
    std::vector<ncnn::Layer*> spatial_convolutions;

#if NCNN_VULKAN
    std::vector<ncnn::Layer*> spatial_convolutions_vulkan;
    ncnn::Pipeline* pipeline_gather_frame;
    ncnn::Pipeline* pipeline_zero;
    ncnn::Pipeline* pipeline_accumulate_frame;
    mutable ncnn::VkMat temporal_memory_vulkan;
    mutable ncnn::VkMat previous_temporal_memory_vulkan;
#endif
};

ncnn::Layer* decomposed_conv3d_layer_creator(void* userdata);

} // namespace seedvr2

#endif // SEEDVR2_DECOMPOSED_CONV3D_H
