#ifndef SEEDVR2_SHARED_LINEAR_H
#define SEEDVR2_SHARED_LINEAR_H

#include <layer.h>

#if NCNN_VULKAN
#include <command.h>
#endif

namespace seedvr2 {

// Applies one serialized linear projection independently to two token streams.
// This avoids materializing Concat/Crop buffers while keeping one weight copy.
class SharedLinear : public ncnn::Layer
{
public:
    SharedLinear();
    virtual ~SharedLinear();

    virtual int load_param(const ncnn::ParamDict& pd);
    virtual int load_model(const ncnn::ModelBin& mb);
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                        std::vector<ncnn::Mat>& top_blobs,
                        const ncnn::Option& opt) const;

#if NCNN_VULKAN
    virtual int create_pipeline(const ncnn::Option& opt);
    virtual int destroy_pipeline(const ncnn::Option& opt);
    virtual int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt);
    virtual int forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                        std::vector<ncnn::VkMat>& top_blobs,
                        ncnn::VkCompute& cmd,
                        const ncnn::Option& opt) const;
#endif

private:
    int num_output;
    int num_input;
    bool bias_term;
    ncnn::Mat weight_data;
    ncnn::Mat bias_data;
    ncnn::Layer* projection;
#if NCNN_VULKAN
    bool pipeline_created;
#endif
};

ncnn::Layer* shared_linear_layer_creator(void* userdata);

} // namespace seedvr2

#endif // SEEDVR2_SHARED_LINEAR_H
