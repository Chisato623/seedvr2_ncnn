#include "vae_downsample3d.h"

#include "decomposed_conv3d.h"

#include <cstring>

#if NCNN_VULKAN
#include <gpu.h>
#include <pipeline.h>
#endif

namespace seedvr2 {

namespace {

#if NCNN_VULKAN
const char pad_shader[] = R"glsl(
#version 450
layout(binding = 0) readonly buffer input_blob { sfp input_data[]; };
layout(binding = 1) writeonly buffer output_blob { sfp output_data[]; };
layout(push_constant) uniform parameter
{
    int input_w;
    int input_h;
    int frames;
    int channels;
    int input_cstep;
    int output_cstep;
    int output_w;
    int output_h;
} p;
void main()
{
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);
    const int q = int(gl_GlobalInvocationID.z);
    if (x >= p.output_w || y >= p.output_h || q >= p.channels * p.frames) return;
    const int channel = q / p.frames;
    const int frame = q - channel * p.frames;
    const int dst = channel * p.output_cstep + frame * p.output_w * p.output_h +
                    y * p.output_w + x;
    if (x < p.input_w && y < p.input_h)
    {
        const int src = channel * p.input_cstep + frame * p.input_w * p.input_h +
                        y * p.input_w + x;
        buffer_st1(output_data, dst, buffer_ld1(input_data, src));
    }
    else
    {
        buffer_st1(output_data, dst, afp(0.f));
    }
}
)glsl";

int create_pad_pipeline(const ncnn::Option& opt,
                        const ncnn::VulkanDevice* device,
                        ncnn::Pipeline*& pipeline)
{
    std::vector<uint32_t> spirv;
    int ret = ncnn::compile_spirv_module(pad_shader, opt, spirv);
    if (ret != 0) return ret;
    pipeline = new ncnn::Pipeline(device);
    pipeline->set_local_size_xyz(8, 8, 1);
    ret = pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t),
                           std::vector<ncnn::vk_specialization_type>());
    if (ret != 0)
    {
        delete pipeline;
        pipeline = 0;
    }
    return ret;
}
#endif

} // namespace

VAEDownsample3D::VAEDownsample3D()
    : channels(0), pad_right(0), pad_bottom(0), convolution(0)
#if NCNN_VULKAN
    , pipeline_pad(0)
#endif
{
    one_blob_only = true;
    support_inplace = false;
#if NCNN_VULKAN
    support_vulkan = true;
#endif
}

VAEDownsample3D::~VAEDownsample3D()
{
    delete convolution;
#if NCNN_VULKAN
    delete pipeline_pad;
#endif
}

int VAEDownsample3D::load_model(const ncnn::ModelBin& mb)
{
    const ncnn::Mat config = mb.load(3, 1);
    if (config.empty()) return -100;
    const float* values = config;
    channels = static_cast<int>(values[0]);
    pad_right = static_cast<int>(values[1]);
    pad_bottom = static_cast<int>(values[2]);
    if (channels <= 0 || pad_right < 0 || pad_bottom < 0 ||
        (pad_right == 0 && pad_bottom == 0))
        return -1;

    convolution = new DecomposedConv3D;
    return convolution->load_model(mb);
}

int VAEDownsample3D::create_pipeline(const ncnn::Option& opt)
{
#if NCNN_VULKAN
    if (convolution) convolution->vkdev = vkdev;
#endif
    int ret = convolution ? convolution->create_pipeline(opt) : -1;
#if NCNN_VULKAN
    if (ret == 0 && opt.use_vulkan_compute)
        ret = create_pad_pipeline(opt, vkdev, pipeline_pad);
#endif
    return ret;
}

int VAEDownsample3D::destroy_pipeline(const ncnn::Option& opt)
{
    return convolution ? convolution->destroy_pipeline(opt) : 0;
}

void VAEDownsample3D::release_completed_temporal_memory() const
{
    if (convolution)
        static_cast<DecomposedConv3D*>(convolution)
            ->release_completed_temporal_memory();
}

void VAEDownsample3D::reset_temporal_memory() const
{
    if (convolution)
        static_cast<DecomposedConv3D*>(convolution)->reset_temporal_memory();
}

int VAEDownsample3D::forward(const ncnn::Mat& bottom_blob,
                             ncnn::Mat& top_blob,
                             const ncnn::Option& opt) const
{
    if (!convolution || bottom_blob.dims != 4 || bottom_blob.n != 1 ||
        bottom_blob.c != channels || bottom_blob.elempack != 1 ||
        bottom_blob.elemsize != sizeof(float))
        return -1;

    ncnn::Mat padded(bottom_blob.w + pad_right, bottom_blob.h + pad_bottom,
                     bottom_blob.d, channels, bottom_blob.elemsize, 1, 1,
                     opt.workspace_allocator);
    if (padded.empty()) return -100;
    padded.fill(0.f);
    for (int channel = 0; channel < channels; channel++)
    {
        for (int frame = 0; frame < bottom_blob.d; frame++)
        {
            const ncnn::Mat source = bottom_blob.channel(channel).depth(frame);
            ncnn::Mat destination = padded.channel(channel).depth(frame);
            for (int y = 0; y < bottom_blob.h; y++)
            {
                std::memcpy(destination.row(y), source.row(y),
                            static_cast<size_t>(bottom_blob.w) * sizeof(float));
            }
        }
    }
    return convolution->forward(padded, top_blob, opt);
}

#if NCNN_VULKAN
int VAEDownsample3D::upload_model(ncnn::VkTransfer& cmd,
                                  const ncnn::Option& opt)
{
    return convolution ? convolution->upload_model(cmd, opt) : -1;
}

int VAEDownsample3D::forward(const ncnn::VkMat& bottom_blob,
                             ncnn::VkMat& top_blob,
                             ncnn::VkCompute& cmd,
                             const ncnn::Option& opt) const
{
    if (!convolution || !pipeline_pad || bottom_blob.dims != 4 ||
        bottom_blob.n != 1 || bottom_blob.c * bottom_blob.elempack != channels)
        return -1;

    ncnn::VkMat unpacked;
    if (bottom_blob.elempack == 1)
        unpacked = bottom_blob;
    else
        vkdev->convert_packing(bottom_blob, unpacked, 1, cmd, opt);
    if (unpacked.empty()) return -100;

    const size_t scalar_elemsize = bottom_blob.elemsize / bottom_blob.elempack;
    ncnn::VkMat padded(bottom_blob.w + pad_right, bottom_blob.h + pad_bottom,
                       bottom_blob.d, channels, scalar_elemsize, 1, 1,
                       opt.workspace_vkallocator);
    if (padded.empty()) return -100;

    std::vector<ncnn::VkMat> bindings(2);
    bindings[0] = unpacked;
    bindings[1] = padded;
    std::vector<ncnn::vk_constant_type> constants(8);
    constants[0].i = bottom_blob.w;
    constants[1].i = bottom_blob.h;
    constants[2].i = bottom_blob.d;
    constants[3].i = channels;
    constants[4].i = static_cast<int>(unpacked.cstep);
    constants[5].i = static_cast<int>(padded.cstep);
    constants[6].i = padded.w;
    constants[7].i = padded.h;
    ncnn::VkMat dispatcher;
    dispatcher.w = padded.w;
    dispatcher.h = padded.h;
    dispatcher.c = channels * bottom_blob.d;
    cmd.record_pipeline(pipeline_pad, bindings, constants, dispatcher);
    return convolution->forward(padded, top_blob, cmd, opt);
}
#endif

ncnn::Layer* vae_downsample3d_layer_creator(void*)
{
    return new VAEDownsample3D;
}

} // namespace seedvr2
