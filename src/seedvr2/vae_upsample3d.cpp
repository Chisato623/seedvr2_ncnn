#include "vae_upsample3d.h"

#include <algorithm>
#include <climits>
#include <cstdlib>

#include "decomposed_conv3d.h"

#if NCNN_VULKAN
#include <allocator.h>
#include <gpu.h>
#include <pipeline.h>
#endif

namespace seedvr2 {

namespace {

#if NCNN_VULKAN
const char shuffle_shader[] = R"glsl(
#version 450
layout(binding = 0) readonly buffer packed_blob { sfp packed_data[]; };
layout(binding = 1) writeonly buffer output_blob { sfp output_data[]; };
layout(push_constant) uniform parameter
{
    int input_w;
    int input_h;
    int input_t;
    int channels;
    int temporal_ratio;
    int spatial_ratio;
    int remove_head;
    int packed_cstep;
    int output_cstep;
    int output_w;
    int output_h;
    int output_t;
    int y_offset;
    int x_offset;
    int z;
    int packed_base_index;
    int output_base_index;
} p;
void main()
{
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);
    const int q = int(gl_GlobalInvocationID.z);
    if (x >= p.input_w || y >= p.input_h || q >= p.channels * p.input_t) return;
    const int channel = q / p.input_t;
    const int frame = q - channel * p.input_t;
    const int source_time = frame * p.temporal_ratio + p.z;
    if (p.remove_head != 0 && source_time == 1) return;
    const int output_time = p.remove_head != 0 && source_time > 1
                                ? source_time - 1 : source_time;
    if (output_time >= p.output_t) return;
    const int src = p.packed_base_index + channel * p.packed_cstep +
                    frame * p.input_w * p.input_h + y * p.input_w + x;
    const int dst = p.output_base_index + channel * p.output_cstep +
                    output_time * p.output_w * p.output_h +
                    (y * p.spatial_ratio + p.y_offset) * p.output_w +
                    x * p.spatial_ratio + p.x_offset;
    buffer_st1(output_data, dst, buffer_ld1(packed_data, src));
}
)glsl";

int create_shuffle_pipeline(const ncnn::Option& opt,
                            const ncnn::VulkanDevice* device,
                            ncnn::Pipeline*& pipeline)
{
    std::vector<uint32_t> spirv;
    int ret = ncnn::compile_spirv_module(shuffle_shader, opt, spirv);
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

size_t shuffle_descriptor_limit_bytes()
{
    const size_t hard_limit = 0x7ffff000u;
    const char* value = std::getenv("SEEDVR2_VK_DESCRIPTOR_LIMIT_BYTES");
    if (!value || !value[0])
        return hard_limit;

    char* end = 0;
    const unsigned long long requested = std::strtoull(value, &end, 10);
    if (!end || end == value || *end != '\0' || requested < 4096u)
        return hard_limit;
    return static_cast<size_t>(std::min<unsigned long long>(
        requested, hard_limit));
}

size_t shuffle_channel_view_capacity(const ncnn::VkMat& value, int first,
                                     int max_count, size_t descriptor_limit,
                                     size_t buffer_offset_alignment,
                                     size_t storage_word_bytes)
{
    if (max_count <= 0 || value.elemsize == 0 || value.cstep == 0 ||
        descriptor_limit <= 1 || buffer_offset_alignment == 0)
        return 0;

    const size_t target = value.buffer_offset() +
                          static_cast<size_t>(first) * value.cstep *
                              value.elemsize;
    const size_t aligned = target / buffer_offset_alignment *
                           buffer_offset_alignment;
    if (aligned > target || aligned < value.data->offset)
        return 0;
    const size_t prefix_bytes = target - aligned;
    if (prefix_bytes % value.elemsize != 0 ||
        prefix_bytes >= descriptor_limit - 1)
        return 0;

    const size_t max_view_bytes = descriptor_limit - 1;
    size_t capacity_elements = (max_view_bytes - prefix_bytes) /
                               value.elemsize;
    size_t capacity = capacity_elements / value.cstep;
    while (capacity > 0 &&
           ncnn::alignSize(prefix_bytes + capacity * value.cstep * value.elemsize,
                           static_cast<int>(storage_word_bytes)) > max_view_bytes)
        capacity--;
    return std::min(static_cast<size_t>(max_count), capacity);
}

ncnn::VkMat shuffle_channel_view(const ncnn::VkMat& value, int first,
                                 int count, size_t buffer_offset_alignment,
                                 size_t storage_word_bytes, int& base_index)
{
    const size_t target = value.buffer_offset() +
                          static_cast<size_t>(first) * value.cstep *
                              value.elemsize;
    const size_t aligned = target / buffer_offset_alignment *
                           buffer_offset_alignment;
    const size_t prefix_bytes = target - aligned;
    base_index = static_cast<int>(prefix_bytes / value.elemsize);
    const size_t view_bytes = ncnn::alignSize(
        prefix_bytes + static_cast<size_t>(count) * value.cstep *
            value.elemsize, static_cast<int>(storage_word_bytes));
    const size_t view_elems = view_bytes / value.elemsize;
    const int view_w = static_cast<int>(std::min(
        view_elems, static_cast<size_t>(INT_MAX)));
    // Retain the source allocation for delayed Vulkan records.  Constructing
    // an external VkMat from data directly leaves refcount at zero.
    ncnn::VkMat view = value;
    view.dims = 1;
    view.w = view_w;
    view.h = 1;
    view.d = 1;
    view.c = 1;
    view.cstep = view_elems;
    view.n = 1;
    view.nstep = view_elems;
    view.offset = aligned - value.data->offset;
    return view;
}
#endif

} // namespace

VAEUpsample3D::VAEUpsample3D()
    : channels(0), temporal_ratio(0), spatial_ratio(0),
      remove_temporal_head(false), post_conv(0), upscale_conv(0)
#if NCNN_VULKAN
      , pipeline_shuffle(0)
#endif
{
    one_blob_only = true;
    support_inplace = false;
#if NCNN_VULKAN
    support_vulkan = true;
#endif
}

VAEUpsample3D::~VAEUpsample3D()
{
    delete post_conv;
    delete upscale_conv;
#if NCNN_VULKAN
    delete pipeline_shuffle;
#endif
}

int VAEUpsample3D::load_model(const ncnn::ModelBin& mb)
{
    const ncnn::Mat config = mb.load(4, 1);
    if (config.empty())
        return -100;
    const float* values = config;
    channels = static_cast<int>(values[0]);
    temporal_ratio = static_cast<int>(values[1]);
    spatial_ratio = static_cast<int>(values[2]);
    remove_temporal_head = static_cast<int>(values[3]) != 0;
    if (channels <= 0 || (temporal_ratio != 1 && temporal_ratio != 2) ||
        (spatial_ratio != 1 && spatial_ratio != 2) ||
        remove_temporal_head != (temporal_ratio == 2))
        return -1;

    // PNNX attributes are lexical: a_config, b_post_conv.*, c_upscale_conv.*.
    post_conv = new DecomposedConv3D;
    upscale_conv = new DecomposedConv3D;
    int ret = post_conv->load_model(mb);
    if (ret == 0)
        ret = upscale_conv->load_model(mb);
    return ret;
}

int VAEUpsample3D::create_pipeline(const ncnn::Option& opt)
{
#if NCNN_VULKAN
    if (post_conv) post_conv->vkdev = vkdev;
    if (upscale_conv) upscale_conv->vkdev = vkdev;
#endif
    int ret = post_conv ? post_conv->create_pipeline(opt) : -1;
    if (ret == 0)
        ret = upscale_conv ? upscale_conv->create_pipeline(opt) : -1;
#if NCNN_VULKAN
    if (ret == 0 && opt.use_vulkan_compute)
        ret = create_shuffle_pipeline(opt, vkdev, pipeline_shuffle);
#endif
    return ret;
}

int VAEUpsample3D::destroy_pipeline(const ncnn::Option& opt)
{
    if (post_conv)
        post_conv->destroy_pipeline(opt);
    if (upscale_conv)
        upscale_conv->destroy_pipeline(opt);
    return 0;
}

void VAEUpsample3D::release_completed_temporal_memory() const
{
    if (post_conv)
        static_cast<DecomposedConv3D*>(post_conv)
            ->release_completed_temporal_memory();
    if (upscale_conv)
        static_cast<DecomposedConv3D*>(upscale_conv)
            ->release_completed_temporal_memory();
}

void VAEUpsample3D::reset_temporal_memory() const
{
    if (post_conv)
        static_cast<DecomposedConv3D*>(post_conv)->reset_temporal_memory();
    if (upscale_conv)
        static_cast<DecomposedConv3D*>(upscale_conv)->reset_temporal_memory();
}

int VAEUpsample3D::forward(const ncnn::Mat& bottom_blob,
                           ncnn::Mat& top_blob,
                           const ncnn::Option& opt) const
{
    if (!post_conv || !upscale_conv || bottom_blob.dims != 4 ||
        bottom_blob.c != channels || bottom_blob.n != 1 ||
        bottom_blob.elempack != 1)
        return -1;

    ncnn::Mat packed;
    int ret = upscale_conv->forward(bottom_blob, packed, opt);
    const int ratio = temporal_ratio * spatial_ratio * spatial_ratio;
    if (ret != 0 || packed.dims != 4 || packed.c != channels * ratio ||
        packed.elempack != 1 || packed.elemsize != sizeof(float))
        return ret != 0 ? ret : -1;

    const int shuffled_t = packed.d * temporal_ratio;
    const bool remove_head = remove_temporal_head &&
                             get_vae_memory_state() != MemoryState::ACTIVE;
    const int output_t = shuffled_t - (remove_head ? 1 : 0);
    ncnn::Mat shuffled(packed.w * spatial_ratio, packed.h * spatial_ratio,
                       output_t, channels, packed.elemsize, 1, 1,
                       opt.workspace_allocator);
    if (shuffled.empty())
        return -100;

    for (int channel = 0; channel < channels; channel++)
    {
        for (int frame = 0; frame < packed.d; frame++)
        {
            for (int z = 0; z < temporal_ratio; z++)
            {
                const int source_time = frame * temporal_ratio + z;
                if (remove_head && source_time == 1)
                    continue;
                const int output_time = remove_head && source_time > 1
                                            ? source_time - 1
                                            : source_time;
                for (int y_offset = 0; y_offset < spatial_ratio; y_offset++)
                {
                    for (int x_offset = 0; x_offset < spatial_ratio; x_offset++)
                    {
                        const int packed_channel =
                            (((y_offset * spatial_ratio + x_offset) *
                               temporal_ratio + z) * channels) + channel;
                        const float* source = packed.channel(packed_channel).depth(frame);
                        float* destination = shuffled.channel(channel).depth(output_time);
                        for (int y = 0; y < packed.h; y++)
                        {
                            for (int x = 0; x < packed.w; x++)
                            {
                                destination[(y * spatial_ratio + y_offset) * shuffled.w +
                                            x * spatial_ratio + x_offset] =
                                    source[y * packed.w + x];
                            }
                        }
                    }
                }
            }
        }
    }
    return post_conv->forward(shuffled, top_blob, opt);
}

#if NCNN_VULKAN
int VAEUpsample3D::upload_model(ncnn::VkTransfer& cmd,
                                const ncnn::Option& opt)
{
    int ret = post_conv ? post_conv->upload_model(cmd, opt) : -1;
    if (ret == 0)
        ret = upscale_conv ? upscale_conv->upload_model(cmd, opt) : -1;
    return ret;
}

int VAEUpsample3D::forward(const ncnn::VkMat& bottom_blob,
                           ncnn::VkMat& top_blob,
                           ncnn::VkCompute& cmd,
                           const ncnn::Option& opt) const
{
    if (!post_conv || !upscale_conv || !pipeline_shuffle ||
        bottom_blob.dims != 4 || bottom_blob.c * bottom_blob.elempack != channels ||
        bottom_blob.n != 1)
        return -1;

    const size_t scalar_elemsize =
        bottom_blob.elemsize / bottom_blob.elempack;
    const size_t output_frame_bytes =
        static_cast<size_t>(bottom_blob.w) * spatial_ratio *
        bottom_blob.h * spatial_ratio * channels * scalar_elemsize;
    const bool bounded_command_memory =
        output_frame_bytes >= 256u * 1024u * 1024u;
    ncnn::VkBlobAllocator packed_scratch_vkallocator(vkdev);
    ncnn::VkBlobAllocator shuffled_scratch_vkallocator(vkdev);
    ncnn::Option upscale_opt = opt;
    if (bounded_command_memory)
    {
        upscale_opt.blob_vkallocator = &packed_scratch_vkallocator;
        upscale_opt.workspace_vkallocator = &packed_scratch_vkallocator;
    }
    ncnn::VkMat packed;
    int ret = upscale_conv->forward(bottom_blob, packed, cmd, upscale_opt);
    const int ratio = temporal_ratio * spatial_ratio * spatial_ratio;
    if (ret != 0 || packed.dims != 4 || packed.c != channels * ratio ||
        packed.elempack != 1)
        return ret != 0 ? ret : -1;
    const int shuffled_t = packed.d * temporal_ratio;
    const bool remove_head = remove_temporal_head &&
                             get_vae_memory_state() != MemoryState::ACTIVE;
    const int output_t = shuffled_t - (remove_head ? 1 : 0);
    ncnn::VkAllocator* shuffled_vkallocator = bounded_command_memory
        ? static_cast<ncnn::VkAllocator*>(&shuffled_scratch_vkallocator)
        : opt.workspace_vkallocator;
    ncnn::VkMat shuffled(packed.w * spatial_ratio, packed.h * spatial_ratio,
                         output_t, channels, packed.elemsize, 1, 1,
                         shuffled_vkallocator);
    if (shuffled.empty()) return -100;
    const int channel_chunk =
        channels;
    const size_t descriptor_limit = std::min(
        shuffle_descriptor_limit_bytes(),
        static_cast<size_t>(vkdev->info.physicalDeviceProperties()
                                .limits.maxStorageBufferRange));
    const size_t buffer_offset_alignment = std::max(
        static_cast<size_t>(1), vkdev->info.buffer_offset_alignment());

    {
        std::vector<ncnn::VkMat> bindings(2);
        std::vector<ncnn::vk_constant_type> constants(17);
        for (int y_offset = 0; y_offset < spatial_ratio; y_offset++)
        {
            for (int x_offset = 0; x_offset < spatial_ratio; x_offset++)
            {
                for (int z = 0; z < temporal_ratio; z++)
                {
                    const int group = (y_offset * spatial_ratio + x_offset) *
                                      temporal_ratio + z;
                    for (int first = 0; first < channels; )
                    {
                        const size_t packed_capacity =
                            shuffle_channel_view_capacity(
                                packed, group * channels + first,
                                channels - first, descriptor_limit,
                                buffer_offset_alignment, packed.elemsize);
                        const size_t output_capacity =
                            shuffle_channel_view_capacity(
                                shuffled, first, channels - first,
                                descriptor_limit, buffer_offset_alignment,
                                shuffled.elemsize);
                        const size_t capacity = std::min(
                            packed_capacity, output_capacity);
                        if (capacity == 0)
                            return -1;
                        const int count = static_cast<int>(capacity);
                        int packed_base_index = 0;
                        int output_base_index = 0;
                        bindings[0] = shuffle_channel_view(
                            packed, group * channels + first, count,
                            buffer_offset_alignment, packed.elemsize,
                            packed_base_index);
                        bindings[1] = shuffle_channel_view(
                            shuffled, first, count, buffer_offset_alignment,
                            shuffled.elemsize, output_base_index);
                        constants[0].i = packed.w;
                        constants[1].i = packed.h;
                        constants[2].i = packed.d;
                        constants[3].i = count;
                        constants[4].i = temporal_ratio;
                        constants[5].i = spatial_ratio;
                        constants[6].i = remove_head ? 1 : 0;
                        constants[7].i = static_cast<int>(packed.cstep);
                        constants[8].i = static_cast<int>(shuffled.cstep);
                        constants[9].i = shuffled.w;
                        constants[10].i = shuffled.h;
                        constants[11].i = shuffled.d;
                        constants[12].i = y_offset;
                        constants[13].i = x_offset;
                        constants[14].i = z;
                        constants[15].i = packed_base_index;
                        constants[16].i = output_base_index;
                        ncnn::VkMat dispatcher;
                        dispatcher.w = packed.w;
                        dispatcher.h = packed.h;
                        dispatcher.c = count * packed.d;
                        cmd.record_pipeline(
                            pipeline_shuffle, bindings, constants, dispatcher);
                        first += count;
                    }
                }
            }
        }
    }
    if (bounded_command_memory)
    {
        ret = cmd.submit_and_wait();
        if (ret != 0)
            return ret;
        ret = cmd.reset();
        if (ret != 0)
            return ret;
        packed.release();
        packed_scratch_vkallocator.clear();
    }

    ret = post_conv->forward(shuffled, top_blob, cmd, opt);
    if (bounded_command_memory && ret == 0)
    {
        // Do not depend on the nested convolution's current submission
        // threshold before releasing its multi-gigabyte source tensor.
        ret = cmd.submit_and_wait();
        if (ret == 0)
            ret = cmd.reset();
    }
    if (bounded_command_memory && ret == 0)
    {
        shuffled.release();
        shuffled_scratch_vkallocator.clear();
    }
    return ret;
}
#endif

ncnn::Layer* vae_upsample3d_layer_creator(void*)
{
    return new VAEUpsample3D;
}

} // namespace seedvr2
