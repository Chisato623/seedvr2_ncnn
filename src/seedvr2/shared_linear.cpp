#include "shared_linear.h"

#include <modelbin.h>

namespace seedvr2 {

SharedLinear::SharedLinear()
    : num_output(0), num_input(0), bias_term(false), projection(0)
#if NCNN_VULKAN
      , pipeline_created(false)
#endif
{
    one_blob_only = false;
    support_inplace = false;
#if NCNN_VULKAN
    support_vulkan = true;
    support_vulkan_packing = true;
    support_vulkan_any_packing = true;
#endif
}

SharedLinear::~SharedLinear()
{
    delete projection;
}

int SharedLinear::load_param(const ncnn::ParamDict& pd)
{
    num_output = pd.get(0, 0);
    num_input = pd.get(1, 0);
    bias_term = pd.get(2, 0) != 0;
    if (num_output <= 0 || num_input <= 0)
        return -1;

    delete projection;
#if NCNN_VULKAN
    projection = ncnn::create_layer_vulkan("Gemm");
#else
    projection = ncnn::create_layer("Gemm");
#endif
    if (!projection)
        return -1;

    // A[token, input] * B[output, input]^T + C[output].
    ncnn::ParamDict gemm_params;
    gemm_params.set(0, 1.f);
    gemm_params.set(1, 1.f);
    gemm_params.set(2, 0);
    gemm_params.set(3, 1);
    gemm_params.set(4, 0);
    gemm_params.set(5, 1);
    gemm_params.set(6, 1);
    gemm_params.set(7, 0);
    gemm_params.set(8, num_output);
    gemm_params.set(9, num_input);
    // -1 is NCNN's constant-empty C sentinel for a bias-free projection.
    gemm_params.set(10, bias_term ? 4 : -1);
    return projection->load_param(gemm_params);
}

int SharedLinear::load_model(const ncnn::ModelBin& mb)
{
    if (!projection)
        return -1;

    weight_data = mb.load(num_input * num_output, 1);
    if (weight_data.empty())
        return -100;
    weight_data = weight_data.reshape(num_input, num_output);
    if (weight_data.empty())
        return -100;
    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

    ncnn::Mat model_weights[2] = {weight_data, bias_data};
    return projection->load_model(ncnn::ModelBinFromMatArray(model_weights));
}

int SharedLinear::forward(const std::vector<ncnn::Mat>& bottom_blobs,
                          std::vector<ncnn::Mat>& top_blobs,
                          const ncnn::Option& opt) const
{
    if (!projection || bottom_blobs.size() != 2 || top_blobs.size() != 2 ||
        bottom_blobs[0].dims != 2 || bottom_blobs[1].dims != 2 ||
        bottom_blobs[0].w != num_input || bottom_blobs[1].w != num_input)
        return -1;

    int ret = projection->forward(bottom_blobs[0], top_blobs[0], opt);
    if (ret == 0)
        ret = projection->forward(bottom_blobs[1], top_blobs[1], opt);
    return ret;
}

#if NCNN_VULKAN
int SharedLinear::create_pipeline(const ncnn::Option& opt)
{
    if (!opt.use_vulkan_compute)
        return 0;
    if (!projection || !vkdev)
        return -1;
    projection->vkdev = vkdev;
    const int ret = projection->create_pipeline(opt);
    if (ret == 0)
        pipeline_created = true;
    return ret;
}

int SharedLinear::destroy_pipeline(const ncnn::Option& opt)
{
    if (!pipeline_created || !projection)
        return 0;
    const int ret = projection->destroy_pipeline(opt);
    pipeline_created = false;
    return ret;
}

int SharedLinear::upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt)
{
    if (!projection || !pipeline_created)
        return -1;
    return projection->upload_model(cmd, opt);
}

int SharedLinear::forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                          std::vector<ncnn::VkMat>& top_blobs,
                          ncnn::VkCompute& cmd,
                          const ncnn::Option& opt) const
{
    if (!projection || !pipeline_created || bottom_blobs.size() != 2 ||
        top_blobs.size() != 2 || bottom_blobs[0].dims != 2 ||
        bottom_blobs[1].dims != 2 || bottom_blobs[0].w != num_input ||
        bottom_blobs[1].w != num_input)
        return -1;

    int ret = projection->forward(bottom_blobs[0], top_blobs[0], cmd, opt);
    if (ret == 0)
        ret = projection->forward(bottom_blobs[1], top_blobs[1], cmd, opt);
    return ret;
}
#endif

ncnn::Layer* shared_linear_layer_creator(void* /*userdata*/)
{
    return new SharedLinear;
}

} // namespace seedvr2
