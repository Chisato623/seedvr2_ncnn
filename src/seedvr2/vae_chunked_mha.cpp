#include "vae_chunked_mha.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include <modelbin.h>

#if NCNN_VULKAN
#include <gpu.h>
#include <pipeline.h>
#endif

namespace seedvr2 {

namespace {

#if NCNN_VULKAN
const char copy_rows_shader[] = R"glsl(
#version 450
layout(binding = 0) readonly buffer source_blob { sfp source_data[]; };
layout(binding = 1) writeonly buffer destination_blob { sfp destination_data[]; };
layout(push_constant) uniform parameter
{
    int width;
    int rows;
    int source_row;
    int destination_row;
} p;
void main()
{
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);
    if (x >= p.width || y >= p.rows) return;
    const int src = (p.source_row + y) * p.width + x;
    const int dst = (p.destination_row + y) * p.width + x;
    buffer_st1(destination_data, dst, buffer_ld1(source_data, src));
}
)glsl";

int create_copy_pipeline(const ncnn::Option& opt,
                         const ncnn::VulkanDevice* device,
                         ncnn::Pipeline*& pipeline)
{
    std::vector<uint32_t> spirv;
    int ret = ncnn::compile_spirv_module(copy_rows_shader, opt, spirv);
    if (ret != 0)
        return ret;
    pipeline = new ncnn::Pipeline(device);
    pipeline->set_local_size_xyz(16, 8, 1);
    ret = pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t),
                           std::vector<ncnn::vk_specialization_type>());
    if (ret != 0)
    {
        delete pipeline;
        pipeline = 0;
    }
    return ret;
}

void record_copy_rows(const ncnn::Pipeline* pipeline,
                      const ncnn::VkMat& source,
                      ncnn::VkMat& destination,
                      int rows, int source_row, int destination_row,
                      ncnn::VkCompute& cmd)
{
    std::vector<ncnn::VkMat> bindings(2);
    bindings[0] = source;
    bindings[1] = destination;
    std::vector<ncnn::vk_constant_type> constants(4);
    constants[0].i = source.w;
    constants[1].i = rows;
    constants[2].i = source_row;
    constants[3].i = destination_row;
    ncnn::VkMat dispatcher;
    dispatcher.w = source.w;
    dispatcher.h = rows;
    dispatcher.c = 1;
    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
}
#endif

} // namespace

VAEChunkedMHA::VAEChunkedMHA()
    : embed_dim(0), num_heads(0), weight_data_size(0), kdim(0), vdim(0),
      scale(0.f), chunk_size(128), cpu_attention(0)
#if NCNN_VULKAN
      , vk_attention(0), pipeline_copy_rows(0)
#endif
{
    one_blob_only = true;
    support_inplace = false;
#if NCNN_VULKAN
    support_vulkan = true;
#endif
}

VAEChunkedMHA::~VAEChunkedMHA()
{
    delete cpu_attention;
#if NCNN_VULKAN
    delete vk_attention;
    delete pipeline_copy_rows;
#endif
}

int VAEChunkedMHA::load_param(const ncnn::ParamDict& pd)
{
    embed_dim = pd.get(0, 0);
    num_heads = pd.get(1, 1);
    weight_data_size = pd.get(2, 0);
    kdim = pd.get(3, embed_dim);
    vdim = pd.get(4, embed_dim);
    scale = pd.get(6, 1.f / std::sqrt(static_cast<float>(embed_dim / num_heads)));
    chunk_size = pd.get(20, 128);
    if (embed_dim <= 0 || num_heads <= 0 || embed_dim % num_heads != 0 ||
        weight_data_size <= 0 || weight_data_size % embed_dim != 0 ||
        kdim <= 0 || vdim <= 0 || chunk_size <= 0)
        return -1;
    return 0;
}

int VAEChunkedMHA::load_model(const ncnn::ModelBin& mb)
{
    const int qdim = weight_data_size / embed_dim;
    weights[0] = mb.load(embed_dim * qdim, 0);
    weights[1] = mb.load(embed_dim, 1);
    weights[2] = mb.load(embed_dim * kdim, 0);
    weights[3] = mb.load(embed_dim, 1);
    weights[4] = mb.load(embed_dim * vdim, 0);
    weights[5] = mb.load(embed_dim, 1);
    weights[6] = mb.load(qdim * embed_dim, 0);
    weights[7] = mb.load(qdim, 1);
    for (const ncnn::Mat& weight : weights)
    {
        if (weight.empty())
            return -100;
    }
    return 0;
}

int VAEChunkedMHA::configure_attention(ncnn::Layer* layer, bool kv_cache,
                                       const ncnn::Option& opt) const
{
    if (!layer)
        return -1;
    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, weight_data_size);
    pd.set(3, kdim);
    pd.set(4, vdim);
    pd.set(5, 0);
    pd.set(6, scale);
    pd.set(7, kv_cache ? 1 : 0);
    int ret = layer->load_param(pd);
    if (ret == 0)
        ret = layer->load_model(ncnn::ModelBinFromMatArray(weights));
    if (ret == 0)
        ret = layer->create_pipeline(opt);
    return ret;
}

int VAEChunkedMHA::create_pipeline(const ncnn::Option& opt)
{
    ncnn::Option cpu_opt = opt;
    cpu_opt.use_vulkan_compute = false;
    cpu_attention = ncnn::create_layer_cpu("MultiHeadAttention");
    int ret = configure_attention(cpu_attention, false, cpu_opt);
#if NCNN_VULKAN
    if (ret == 0 && opt.use_vulkan_compute)
    {
        vk_attention = ncnn::create_layer_vulkan("MultiHeadAttention");
        if (vk_attention)
            vk_attention->vkdev = vkdev;
        ret = configure_attention(vk_attention, true, opt);
        if (ret == 0)
            ret = create_copy_pipeline(opt, vkdev, pipeline_copy_rows);
    }
#endif
    return ret;
}

int VAEChunkedMHA::destroy_pipeline(const ncnn::Option& opt)
{
    if (cpu_attention)
        cpu_attention->destroy_pipeline(opt);
#if NCNN_VULKAN
    if (vk_attention)
        vk_attention->destroy_pipeline(opt);
#endif
    return 0;
}

int VAEChunkedMHA::forward(const ncnn::Mat& bottom_blob,
                           ncnn::Mat& top_blob,
                           const ncnn::Option& opt) const
{
    if (!cpu_attention || bottom_blob.dims != 2 ||
        bottom_blob.w != weight_data_size / embed_dim)
        return -1;
    std::vector<ncnn::Mat> bottoms(1, bottom_blob);
    std::vector<ncnn::Mat> tops(1);
    const int ret = cpu_attention->forward(bottoms, tops, opt);
    if (ret == 0)
        top_blob = tops[0];
    return ret;
}

#if NCNN_VULKAN
int VAEChunkedMHA::upload_model(ncnn::VkTransfer& cmd,
                                const ncnn::Option& opt)
{
    return vk_attention ? vk_attention->upload_model(cmd, opt) : -1;
}

int VAEChunkedMHA::forward(const ncnn::VkMat& bottom_blob,
                           ncnn::VkMat& top_blob,
                           ncnn::VkCompute& cmd,
                           const ncnn::Option& opt) const
{
    if (!vk_attention || !pipeline_copy_rows || bottom_blob.dims != 2 ||
        bottom_blob.w != weight_data_size / embed_dim ||
        bottom_blob.elempack != 1)
        return -1;
    const int sequence = bottom_blob.h;
    top_blob.create(bottom_blob.w, sequence, bottom_blob.elemsize, 1,
                    opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    ncnn::VkMat key_cache;
    ncnn::VkMat value_cache;
    for (int start = 0; start < sequence; start += chunk_size)
    {
        const int count = std::min(chunk_size, sequence - start);
        ncnn::VkMat query(bottom_blob.w, count, bottom_blob.elemsize, 1,
                          opt.workspace_vkallocator);
        if (query.empty())
            return -100;
        record_copy_rows(pipeline_copy_rows, bottom_blob, query, count,
                         start, 0, cmd);

        std::vector<ncnn::VkMat> bottoms(5);
        bottoms[0] = query;
        bottoms[1] = bottom_blob;
        bottoms[2] = bottom_blob;
        bottoms[3] = key_cache;
        bottoms[4] = value_cache;
        std::vector<ncnn::VkMat> tops(3);
        ncnn::Option inner_opt = opt;
        int ret = vk_attention->forward(bottoms, tops, cmd, inner_opt);
        if (ret != 0)
            return ret;
        if (tops[0].dims != 2 || tops[0].w != top_blob.w ||
            tops[0].h * tops[0].elempack != count)
            return -1;
        if (tops[0].elempack != 1)
        {
            ncnn::VkMat unpacked;
            vkdev->convert_packing(tops[0], unpacked, 1, cmd, inner_opt);
            tops[0] = unpacked;
        }
        record_copy_rows(pipeline_copy_rows, tops[0], top_blob, count, 0,
                         start, cmd);
        if (key_cache.empty())
        {
            key_cache = tops[1];
            value_cache = tops[2];
        }
        ret = cmd.submit_and_wait();
        if (ret != 0)
            return ret;
        ret = cmd.reset();
        if (ret != 0)
            return ret;
    }
    return 0;
}
#endif

ncnn::Layer* vae_chunked_mha_layer_creator(void*)
{
    return new VAEChunkedMHA;
}

} // namespace seedvr2
