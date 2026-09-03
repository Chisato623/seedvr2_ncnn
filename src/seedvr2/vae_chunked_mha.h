#ifndef SEEDVR2_VAE_CHUNKED_MHA_H
#define SEEDVR2_VAE_CHUNKED_MHA_H

#include <layer.h>

#if NCNN_VULKAN
#include <command.h>
#include <pipeline.h>
#endif

namespace seedvr2 {

class VAEChunkedMHA : public ncnn::Layer
{
public:
    VAEChunkedMHA();
    virtual ~VAEChunkedMHA();

    virtual int load_param(const ncnn::ParamDict& pd);
    virtual int load_model(const ncnn::ModelBin& mb);
    virtual int create_pipeline(const ncnn::Option& opt);
    virtual int destroy_pipeline(const ncnn::Option& opt);
    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob,
                        const ncnn::Option& opt) const;

#if NCNN_VULKAN
    virtual int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt);
    virtual int forward(const ncnn::VkMat& bottom_blob,
                        ncnn::VkMat& top_blob, ncnn::VkCompute& cmd,
                        const ncnn::Option& opt) const;
#endif

private:
    int configure_attention(ncnn::Layer* layer, bool kv_cache,
                            const ncnn::Option& opt) const;

    int embed_dim;
    int num_heads;
    int weight_data_size;
    int kdim;
    int vdim;
    float scale;
    int chunk_size;
    ncnn::Mat weights[8];
    ncnn::Layer* cpu_attention;
#if NCNN_VULKAN
    ncnn::Layer* vk_attention;
    ncnn::Pipeline* pipeline_copy_rows;
#endif
};

ncnn::Layer* vae_chunked_mha_layer_creator(void* userdata);

} // namespace seedvr2

#endif // SEEDVR2_VAE_CHUNKED_MHA_H
