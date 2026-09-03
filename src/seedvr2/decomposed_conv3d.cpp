#include "decomposed_conv3d.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#include <modelbin.h>

#if NCNN_VULKAN
#include <allocator.h>
#include <gpu.h>
#include <pipeline.h>
#endif

namespace seedvr2 {

namespace {

thread_local MemoryState current_memory_state = MemoryState::DISABLED;
#if NCNN_VULKAN
thread_local ncnn::VkAllocator* current_memory_vkallocator = 0;
#endif

#if NCNN_VULKAN
const char gather_frame_shader[] = R"glsl(
#version 450
layout(binding = 0) readonly buffer input_blob { sfp input_data[]; };
layout(binding = 1) writeonly buffer frame_blob { sfp frame_data[]; };
layout(push_constant) uniform parameter
{
    int width;
    int height;
    int channels;
    int source_frame;
    int input_cstep;
    int output_cstep;
    int destination_frame;
    int input_base_index;
    int output_base_index;
} p;
void main()
{
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);
    const int c = int(gl_GlobalInvocationID.z);
    if (x >= p.width || y >= p.height || c >= p.channels) return;
    const int spatial = p.width * p.height;
    const int src = p.input_base_index + c * p.input_cstep +
                    p.source_frame * spatial + y * p.width + x;
    const int dst = p.output_base_index + c * p.output_cstep +
                    p.destination_frame * spatial +
                    y * p.width + x;
    buffer_st1(frame_data, dst, buffer_ld1(input_data, src));
}
)glsl";

const char zero_output_shader[] = R"glsl(
#version 450
layout(binding = 0) writeonly buffer output_blob { sfp output_data[]; };
layout(push_constant) uniform parameter
{
    int count;
    int base_index;
} p;
void main()
{
    const int i = int(gl_GlobalInvocationID.x);
    if (i < p.count) buffer_st1(output_data, p.base_index + i, afp(0.f));
}
)glsl";

const char accumulate_frame_shader[] = R"glsl(
#version 450
layout(binding = 0) readonly buffer frame_blob { sfp frame_data[]; };
layout(binding = 1) buffer output_blob { sfp output_data[]; };
layout(push_constant) uniform parameter
{
    int width;
    int height;
    int channels;
    int output_frame;
    int frame_cstep;
    int output_cstep;
    int frame_base_index;
    int output_base_index;
} p;
void main()
{
    const int x = int(gl_GlobalInvocationID.x);
    const int y = int(gl_GlobalInvocationID.y);
    const int c = int(gl_GlobalInvocationID.z);
    if (x >= p.width || y >= p.height || c >= p.channels) return;
    const int spatial = p.width * p.height;
    const int src = p.frame_base_index + c * p.frame_cstep + y * p.width + x;
    const int dst = p.output_base_index + c * p.output_cstep +
                    p.output_frame * spatial + y * p.width + x;
    const float value = float(buffer_ld1(output_data, dst)) + float(buffer_ld1(frame_data, src));
    buffer_st1(output_data, dst, afp(value));
}
)glsl";

int create_shader_pipeline(const char* source, const ncnn::Option& opt,
                           const ncnn::VulkanDevice* device,
                           int local_x, int local_y, int local_z,
                           ncnn::Pipeline*& pipeline)
{
    std::vector<uint32_t> spirv;
    int ret = ncnn::compile_spirv_module(source, opt, spirv);
    if (ret != 0) return ret;
    pipeline = new ncnn::Pipeline(device);
    pipeline->set_local_size_xyz(local_x, local_y, local_z);
    ret = pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t),
                           std::vector<ncnn::vk_specialization_type>());
    if (ret != 0)
    {
        delete pipeline;
        pipeline = 0;
    }
    return ret;
}

size_t descriptor_limit_bytes()
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

size_t channel_view_capacity(const ncnn::VkMat& value, int first,
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

ncnn::VkMat channel_view(const ncnn::VkMat& value, int first, int count,
                         size_t buffer_offset_alignment,
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
    // Keep a reference to the source allocation while delayed command
    // records may still access this view.  An external VkMat constructor has
    // no refcount ownership and could let a temporary convolution result be
    // recycled before the command buffer executes.
    ncnn::VkMat view = value;
    view.dims = 1;
    view.w = static_cast<int>(std::min(view_elems,
                                       static_cast<size_t>(INT_MAX)));
    view.h = 1;
    view.d = 1;
    view.c = 1;
    view.cstep = view_elems;
    view.n = 1;
    view.nstep = view_elems;
    view.offset = aligned - value.data->offset;
    return view;
}

int record_zero_channels(const ncnn::VkMat& output, int channels,
                         const ncnn::Pipeline* pipeline,
                         size_t descriptor_limit,
                         size_t buffer_offset_alignment,
                         ncnn::VkCompute& cmd)
{
    const size_t storage_word_bytes = output.elemsize;
    for (int first = 0; first < channels; )
    {
        const size_t capacity = channel_view_capacity(
            output, first, channels - first, descriptor_limit,
            buffer_offset_alignment, storage_word_bytes);
        if (capacity == 0)
            return -1;
        const int count = static_cast<int>(capacity);
        int output_base_index = 0;
        const ncnn::VkMat output_view = channel_view(
            output, first, count, buffer_offset_alignment,
            storage_word_bytes, output_base_index);
        const size_t logical_count = static_cast<size_t>(count) * output.cstep;
        if (logical_count > static_cast<size_t>(INT_MAX))
            return -1;
        std::vector<ncnn::VkMat> bindings(1, output_view);
        std::vector<ncnn::vk_constant_type> constants(2);
        constants[0].i = static_cast<int>(logical_count);
        constants[1].i = output_base_index;
        ncnn::VkMat dispatcher;
        dispatcher.w = constants[0].i;
        dispatcher.h = 1;
        dispatcher.c = 1;
        cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
        first += count;
    }
    return 0;
}

int record_gather_channels(const ncnn::VkMat& input,
                           const ncnn::VkMat& output, int channels,
                           int source_frame, int destination_frame,
                           const ncnn::Pipeline* pipeline,
                           size_t descriptor_limit,
                           size_t buffer_offset_alignment,
                           ncnn::VkCompute& cmd)
{
    const size_t input_word_bytes = input.elemsize;
    const size_t output_word_bytes = output.elemsize;
    for (int first = 0; first < channels; )
    {
        const size_t input_capacity = channel_view_capacity(
            input, first, channels - first, descriptor_limit,
            buffer_offset_alignment, input_word_bytes);
        const size_t output_capacity = channel_view_capacity(
            output, first, channels - first, descriptor_limit,
            buffer_offset_alignment, output_word_bytes);
        const size_t capacity = std::min(input_capacity, output_capacity);
        if (capacity == 0)
            return -1;
        const int count = static_cast<int>(capacity);
        int input_base_index = 0;
        int output_base_index = 0;
        const ncnn::VkMat input_view = channel_view(
            input, first, count, buffer_offset_alignment,
            input_word_bytes, input_base_index);
        const ncnn::VkMat output_view = channel_view(
            output, first, count, buffer_offset_alignment,
            output_word_bytes, output_base_index);
        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = input_view;
        bindings[1] = output_view;
        std::vector<ncnn::vk_constant_type> constants(9);
        constants[0].i = input.w;
        constants[1].i = input.h;
        constants[2].i = count;
        constants[3].i = source_frame;
        constants[4].i = static_cast<int>(input.cstep);
        constants[5].i = static_cast<int>(output.cstep);
        constants[6].i = destination_frame;
        constants[7].i = input_base_index;
        constants[8].i = output_base_index;
        ncnn::VkMat dispatcher;
        dispatcher.w = input.w;
        dispatcher.h = input.h;
        dispatcher.c = count;
        cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
        first += count;
    }
    return 0;
}

int record_accumulate_channels(const ncnn::VkMat& frame,
                               const ncnn::VkMat& output, int channels,
                               int output_frame,
                               const ncnn::Pipeline* pipeline,
                               size_t descriptor_limit,
                               size_t buffer_offset_alignment,
                               ncnn::VkCompute& cmd)
{
    const size_t frame_word_bytes = frame.elemsize;
    const size_t output_word_bytes = output.elemsize;
    for (int first = 0; first < channels; )
    {
        const size_t frame_capacity = channel_view_capacity(
            frame, first, channels - first, descriptor_limit,
            buffer_offset_alignment, frame_word_bytes);
        const size_t output_capacity = channel_view_capacity(
            output, first, channels - first, descriptor_limit,
            buffer_offset_alignment, output_word_bytes);
        const size_t capacity = std::min(frame_capacity, output_capacity);
        if (capacity == 0)
            return -1;
        const int count = static_cast<int>(capacity);
        int frame_base_index = 0;
        int output_base_index = 0;
        const ncnn::VkMat frame_view = channel_view(
            frame, first, count, buffer_offset_alignment,
            frame_word_bytes, frame_base_index);
        const ncnn::VkMat output_view = channel_view(
            output, first, count, buffer_offset_alignment,
            output_word_bytes, output_base_index);
        std::vector<ncnn::VkMat> bindings(2);
        bindings[0] = frame_view;
        bindings[1] = output_view;
        std::vector<ncnn::vk_constant_type> constants(8);
        constants[0].i = frame.w;
        constants[1].i = frame.h;
        constants[2].i = count;
        constants[3].i = output_frame;
        constants[4].i = static_cast<int>(frame.cstep);
        constants[5].i = static_cast<int>(output.cstep);
        constants[6].i = frame_base_index;
        constants[7].i = output_base_index;
        ncnn::VkMat dispatcher;
        dispatcher.w = frame.w;
        dispatcher.h = frame.h;
        dispatcher.c = count;
        cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
        first += count;
    }
    return 0;
}
#endif

} // namespace

MemoryState get_vae_memory_state()
{
    return current_memory_state;
}

void set_vae_memory_state(MemoryState state)
{
    current_memory_state = state;
}

#if NCNN_VULKAN
ncnn::VkAllocator* get_vae_memory_vkallocator()
{
    return current_memory_vkallocator;
}

void set_vae_memory_vkallocator(ncnn::VkAllocator* allocator)
{
    current_memory_vkallocator = allocator;
}
#endif

DecomposedConv3D::DecomposedConv3D()
    : num_output(0), num_input(0), kernel_w(0), kernel_h(0), kernel_t(0),
      stride_t(1), dilation_t(1), head_frames(0), stride_w(1), stride_h(1),
      dilation_w(1), dilation_h(1), pad_w(0), pad_h(0), groups(1),
      bias_term(false)
#if NCNN_VULKAN
      , pipeline_gather_frame(0), pipeline_zero(0), pipeline_accumulate_frame(0)
#endif
{
    one_blob_only = true;
    support_inplace = false;
#if NCNN_VULKAN
    support_vulkan = true;
#endif
}

DecomposedConv3D::~DecomposedConv3D()
{
    for (size_t i = 0; i < spatial_convolutions.size(); i++)
        delete spatial_convolutions[i];
#if NCNN_VULKAN
    for (size_t i = 0; i < spatial_convolutions_vulkan.size(); i++)
        delete spatial_convolutions_vulkan[i];
    delete pipeline_gather_frame;
    delete pipeline_zero;
    delete pipeline_accumulate_frame;
#endif
}

int DecomposedConv3D::load_param(const ncnn::ParamDict& pd)
{
    num_output = pd.get(0, 0);
    num_input = pd.get(1, 0);
    kernel_w = pd.get(2, 0);
    kernel_h = pd.get(3, 0);
    kernel_t = pd.get(4, 0);
    stride_t = pd.get(5, 1);
    dilation_t = pd.get(6, 1);
    head_frames = pd.get(7, 0);
    stride_w = pd.get(8, 1);
    stride_h = pd.get(9, 1);
    dilation_w = pd.get(10, 1);
    dilation_h = pd.get(11, 1);
    pad_w = pd.get(12, 0);
    pad_h = pd.get(13, 0);
    groups = pd.get(14, 1);
    bias_term = pd.get(15, 0) != 0;

    return 0;
}

int DecomposedConv3D::load_model(const ncnn::ModelBin& mb)
{
    const ncnn::Mat config = mb.load(16, 1);
    if (config.empty())
        return -100;
    const float* values = config;
    num_output = static_cast<int>(values[0]);
    num_input = static_cast<int>(values[1]);
    kernel_w = static_cast<int>(values[2]);
    kernel_h = static_cast<int>(values[3]);
    kernel_t = static_cast<int>(values[4]);
    stride_t = static_cast<int>(values[5]);
    dilation_t = static_cast<int>(values[6]);
    head_frames = static_cast<int>(values[7]);
    stride_w = static_cast<int>(values[8]);
    stride_h = static_cast<int>(values[9]);
    dilation_w = static_cast<int>(values[10]);
    dilation_h = static_cast<int>(values[11]);
    pad_w = static_cast<int>(values[12]);
    pad_h = static_cast<int>(values[13]);
    groups = static_cast<int>(values[14]);
    bias_term = static_cast<int>(values[15]) != 0;
    if (num_output <= 0 || num_input <= 0 || kernel_w <= 0 || kernel_h <= 0 ||
        kernel_t <= 0 || stride_t <= 0 || dilation_t <= 0 || stride_w <= 0 ||
        stride_h <= 0 || dilation_w <= 0 || dilation_h <= 0 || groups <= 0 ||
        num_input % groups != 0 || num_output % groups != 0 || head_frames < 0)
        return -1;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
        {
            std::fprintf(stderr, "DecomposedConv3D bias load failed (%d)\n",
                         num_output);
            return -100;
        }
    }

    const int weight_data_size = num_output * (num_input / groups) *
                                 kernel_w * kernel_h;
    spatial_convolutions.reserve(kernel_t);
    spatial_weights.reserve(kernel_t);
    for (int temporal_index = 0; temporal_index < kernel_t; temporal_index++)
    {
        // PNNX moduleop attributes are stored as raw FP32 even when fp16=1.
        ncnn::Mat weight = mb.load(weight_data_size, 1);
        if (weight.empty())
        {
            std::fprintf(stderr,
                         "DecomposedConv3D weight %d load failed (%d)\n",
                         temporal_index, weight_data_size);
            return -100;
        }
        spatial_weights.push_back(weight);

        ncnn::Layer* convolution = ncnn::create_layer_cpu("Convolution");
        if (!convolution)
            return -1;

        ncnn::ParamDict params;
        params.set(0, num_output);
        params.set(1, kernel_w);
        params.set(11, kernel_h);
        params.set(2, dilation_w);
        params.set(12, dilation_h);
        params.set(3, stride_w);
        params.set(13, stride_h);
        params.set(4, pad_w);
        params.set(15, pad_w);
        params.set(14, pad_h);
        params.set(16, pad_h);
        params.set(5, temporal_index == 0 && bias_term ? 1 : 0);
        params.set(6, weight_data_size);
        int ret = convolution->load_param(params);
        if (ret != 0)
        {
            delete convolution;
            return ret;
        }

        ncnn::Mat model_weights[2];
        model_weights[0] = weight;
        model_weights[1] = bias_data;
        ret = convolution->load_model(ncnn::ModelBinFromMatArray(model_weights));
        if (ret != 0)
        {
            delete convolution;
            return ret;
        }
        spatial_convolutions.push_back(convolution);
    }
    return 0;
}

int DecomposedConv3D::create_pipeline(const ncnn::Option& opt)
{
    for (size_t i = 0; i < spatial_convolutions.size(); i++)
    {
        const int ret = spatial_convolutions[i]->create_pipeline(opt);
        if (ret != 0)
            return ret;
    }
#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        spatial_convolutions_vulkan.reserve(kernel_t);
        const int weight_data_size = num_output * (num_input / groups) *
                                     kernel_w * kernel_h;
        for (int temporal_index = 0; temporal_index < kernel_t; temporal_index++)
        {
            ncnn::Layer* convolution = ncnn::create_layer_vulkan("Convolution");
            if (!convolution) return -1;
            convolution->vkdev = vkdev;
            ncnn::ParamDict params;
            params.set(0, num_output);
            params.set(1, kernel_w);
            params.set(11, kernel_h);
            params.set(2, dilation_w);
            params.set(12, dilation_h);
            params.set(3, stride_w);
            params.set(13, stride_h);
            params.set(4, pad_w);
            params.set(15, pad_w);
            params.set(14, pad_h);
            params.set(16, pad_h);
            params.set(5, temporal_index == 0 && bias_term ? 1 : 0);
            params.set(6, weight_data_size);
            int ret = convolution->load_param(params);
            ncnn::Mat weights[2];
            weights[0] = spatial_weights[temporal_index];
            weights[1] = bias_data;
            if (ret == 0)
                ret = convolution->load_model(ncnn::ModelBinFromMatArray(weights));
            if (ret == 0)
                ret = convolution->create_pipeline(opt);
            if (ret != 0)
            {
                delete convolution;
                return ret;
            }
            spatial_convolutions_vulkan.push_back(convolution);
        }
        int ret = create_shader_pipeline(gather_frame_shader, opt, vkdev,
                                         8, 8, 1, pipeline_gather_frame);
        if (ret == 0)
            ret = create_shader_pipeline(zero_output_shader, opt, vkdev,
                                         256, 1, 1, pipeline_zero);
        if (ret == 0)
            ret = create_shader_pipeline(accumulate_frame_shader, opt, vkdev,
                                         8, 8, 1, pipeline_accumulate_frame);
        if (ret != 0) return ret;
    }
#endif
    return 0;
}

int DecomposedConv3D::destroy_pipeline(const ncnn::Option& opt)
{
    for (size_t i = 0; i < spatial_convolutions.size(); i++)
        spatial_convolutions[i]->destroy_pipeline(opt);
#if NCNN_VULKAN
    for (size_t i = 0; i < spatial_convolutions_vulkan.size(); i++)
        spatial_convolutions_vulkan[i]->destroy_pipeline(opt);
#endif
    return 0;
}

void DecomposedConv3D::release_completed_temporal_memory() const
{
#if NCNN_VULKAN
    previous_temporal_memory_vulkan.release();
#endif
}

void DecomposedConv3D::reset_temporal_memory() const
{
    temporal_memory.release();
#if NCNN_VULKAN
    temporal_memory_vulkan.release();
    previous_temporal_memory_vulkan.release();
#endif
}

int DecomposedConv3D::forward(const ncnn::Mat& bottom_blob,
                              ncnn::Mat& top_blob,
                              const ncnn::Option& opt) const
{
    const MemoryState memory_state = get_vae_memory_state();
    if (memory_state != MemoryState::ACTIVE)
        temporal_memory.release();
    if (bottom_blob.dims != 4 || bottom_blob.c * bottom_blob.elempack != num_input ||
        bottom_blob.n != 1 ||
        static_cast<int>(spatial_convolutions.size()) != kernel_t)
    {
        std::fprintf(stderr,
                     "DecomposedConv3D input mismatch dims=%d c=%d/%d pack=%d n=%d kernels=%zu/%d\n",
                     bottom_blob.dims, bottom_blob.c * bottom_blob.elempack, num_input,
                     bottom_blob.elempack, bottom_blob.n,
                     spatial_convolutions.size(), kernel_t);
        return -1;
    }

    const bool use_memory = memory_state == MemoryState::ACTIVE &&
                            !temporal_memory.empty();
    if (use_memory &&
        (temporal_memory.w != bottom_blob.w ||
         temporal_memory.h != bottom_blob.h ||
         temporal_memory.c != bottom_blob.c ||
         temporal_memory.elempack != bottom_blob.elempack ||
         temporal_memory.elemsize != bottom_blob.elemsize))
        return -1;

    const int prefix_t = use_memory ? temporal_memory.d : head_frames;
    const int padded_t = bottom_blob.d + prefix_t;
    const int kernel_extent = dilation_t * (kernel_t - 1) + 1;
    if (padded_t < kernel_extent)
        return -1;
    const int output_t = (padded_t - kernel_extent) / stride_t + 1;

    ncnn::Mat input_frame(bottom_blob.w, bottom_blob.h, num_input,
                          bottom_blob.elemsize, 1, opt.workspace_allocator);
    if (input_frame.empty())
        return -100;

    bool output_created = false;
    for (int output_index = 0; output_index < output_t; output_index++)
    {
        for (int temporal_index = 0; temporal_index < kernel_t; temporal_index++)
        {
            const int padded_index = output_index * stride_t +
                                     temporal_index * dilation_t;
            const ncnn::Mat* source_blob = &bottom_blob;
            int source_index = padded_index - prefix_t;
            if (use_memory && padded_index < prefix_t)
            {
                source_blob = &temporal_memory;
                source_index = padded_index;
            }
            else if (!use_memory && padded_index < prefix_t)
            {
                source_index = 0;
            }
            if (source_index < 0 || source_index >= source_blob->d)
            {
                std::fprintf(stderr,
                             "DecomposedConv3D source frame out of range %d/%d\n",
                             source_index, bottom_blob.d);
                return -1;
            }

            for (int channel = 0; channel < num_input; channel++)
            {
                const ncnn::Mat source = source_blob->channel(channel).depth(source_index);
                ncnn::Mat destination = input_frame.channel(channel);
                std::memcpy(destination.data, source.data,
                            static_cast<size_t>(bottom_blob.w) * bottom_blob.h *
                                bottom_blob.elemsize);
            }

            ncnn::Mat current;
            const int ret = spatial_convolutions[temporal_index]->forward(
                input_frame, current, opt);
            if (ret != 0)
            {
                std::fprintf(stderr,
                             "DecomposedConv3D spatial convolution %d failed (%d)\n",
                             temporal_index, ret);
                return ret;
            }

            if (!output_created)
            {
                top_blob.create(current.w, current.h, output_t, current.c,
                                current.elemsize, current.elempack, 1,
                                opt.blob_allocator);
                if (top_blob.empty())
                    return -100;
                top_blob.fill(0.f);
                output_created = true;
            }
            if (current.w != top_blob.w || current.h != top_blob.h ||
                current.c != top_blob.c || current.elempack != 1 ||
                current.elemsize != sizeof(float))
            {
                std::fprintf(stderr,
                             "DecomposedConv3D output mismatch current=(%d,%d,%d,%zu,%d) top=(%d,%d,%d)\n",
                             current.w, current.h, current.c, current.elemsize,
                             current.elempack, top_blob.w, top_blob.h,
                             top_blob.c);
                return -1;
            }

            for (int channel = 0; channel < num_output; channel++)
            {
                const float* source = current.channel(channel);
                float* destination = top_blob.channel(channel).depth(output_index);
                const int count = current.w * current.h;
                for (int i = 0; i < count; i++)
                    destination[i] += source[i];
            }
        }
    }
    const int cache_frames = kernel_t - stride_t;
    if (memory_state != MemoryState::DISABLED && cache_frames > 0)
    {
        if (cache_frames > padded_t)
            return -1;
        ncnn::Mat next_memory(bottom_blob.w, bottom_blob.h, cache_frames,
                              bottom_blob.c, bottom_blob.elemsize,
                              bottom_blob.elempack, 1);
        if (next_memory.empty())
            return -100;
        for (int destination_frame = 0; destination_frame < cache_frames;
             destination_frame++)
        {
            const int padded_index = padded_t - cache_frames + destination_frame;
            const ncnn::Mat* source_blob = &bottom_blob;
            int source_index = padded_index - prefix_t;
            if (use_memory && padded_index < prefix_t)
            {
                source_blob = &temporal_memory;
                source_index = padded_index;
            }
            else if (!use_memory && padded_index < prefix_t)
            {
                source_index = 0;
            }
            if (source_index < 0 || source_index >= source_blob->d)
                return -1;
            for (int channel = 0; channel < bottom_blob.c; channel++)
            {
                const ncnn::Mat source =
                    source_blob->channel(channel).depth(source_index);
                ncnn::Mat destination =
                    next_memory.channel(channel).depth(destination_frame);
                std::memcpy(destination.data, source.data,
                            static_cast<size_t>(bottom_blob.w) * bottom_blob.h *
                                bottom_blob.elemsize);
            }
        }
        temporal_memory = next_memory;
    }
    return output_created ? 0 : -1;
}

#if NCNN_VULKAN
int DecomposedConv3D::upload_model(ncnn::VkTransfer& cmd,
                                   const ncnn::Option& opt)
{
    if (static_cast<int>(spatial_convolutions_vulkan.size()) != kernel_t)
        return -1;
    for (int i = 0; i < kernel_t; i++)
    {
        const int ret = spatial_convolutions_vulkan[i]->upload_model(cmd, opt);
        if (ret != 0) return ret;
    }
    return 0;
}

int DecomposedConv3D::forward(const ncnn::VkMat& bottom_blob,
                              ncnn::VkMat& top_blob,
                              ncnn::VkCompute& cmd,
                              const ncnn::Option& opt) const
{
    const MemoryState memory_state = get_vae_memory_state();
    previous_temporal_memory_vulkan.release();
    if (memory_state != MemoryState::ACTIVE)
        temporal_memory_vulkan.release();
    if (bottom_blob.dims != 4 || bottom_blob.c != num_input ||
        bottom_blob.elempack != 1 || bottom_blob.n != 1 ||
        static_cast<int>(spatial_convolutions_vulkan.size()) != kernel_t ||
        !pipeline_gather_frame || !pipeline_zero || !pipeline_accumulate_frame)
    {
        std::fprintf(stderr,
                     "DecomposedConv3D Vulkan input mismatch dims=%d c=%d/%d pack=%d n=%d kernels=%zu/%d pipelines=%d%d%d\n",
                     bottom_blob.dims, bottom_blob.c, num_input,
                     bottom_blob.elempack, bottom_blob.n,
                     spatial_convolutions_vulkan.size(), kernel_t,
                     pipeline_gather_frame != 0, pipeline_zero != 0,
                     pipeline_accumulate_frame != 0);
        return -1;
    }
    const bool use_memory = memory_state == MemoryState::ACTIVE &&
                            !temporal_memory_vulkan.empty();
    if (use_memory &&
        (temporal_memory_vulkan.w != bottom_blob.w ||
         temporal_memory_vulkan.h != bottom_blob.h ||
         temporal_memory_vulkan.c != bottom_blob.c ||
         temporal_memory_vulkan.elemsize != bottom_blob.elemsize ||
         temporal_memory_vulkan.elempack != 1))
        return -1;
    const int prefix_t = use_memory ? temporal_memory_vulkan.d : head_frames;
    const int padded_t = bottom_blob.d + prefix_t;
    const int kernel_extent = dilation_t * (kernel_t - 1) + 1;
    if (padded_t < kernel_extent) return -1;
    const int output_t = (padded_t - kernel_extent) / stride_t + 1;
    const int output_w = (bottom_blob.w + pad_w * 2 -
        dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    const int output_h = (bottom_blob.h + pad_h * 2 -
        dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    if (output_w <= 0 || output_h <= 0) return -1;

    ncnn::VkMat unpacked_input;
    if (bottom_blob.elempack == 1)
        unpacked_input = bottom_blob;
    else
        vkdev->convert_packing(bottom_blob, unpacked_input, 1, cmd, opt);
    if (unpacked_input.empty()) return -100;
    const size_t scalar_elemsize = bottom_blob.elemsize / bottom_blob.elempack;

    // Production frames need bounded scratch allocators. VkBlobAllocator
    // reuses freed ranges but retains every VkDeviceMemory block until clear(),
    // so mixing multi-GiB convolution temporaries with persistent graph blobs
    // eventually leaves no contiguous block for the next layer output.
    const size_t frame_bytes =
        static_cast<size_t>(output_w) * output_h * num_output *
        scalar_elemsize;
    const bool bounded_command_memory = frame_bytes >= 256u * 1024u * 1024u;
    ncnn::VkBlobAllocator frame_scratch_vkallocator(vkdev);
    ncnn::VkBlobAllocator kernel_scratch_vkallocator(vkdev);
    ncnn::VkAllocator* input_frame_vkallocator = bounded_command_memory
        ? static_cast<ncnn::VkAllocator*>(&frame_scratch_vkallocator)
        : opt.workspace_vkallocator;
    ncnn::VkMat input_frame(bottom_blob.w, bottom_blob.h, num_input,
                            scalar_elemsize, 1, input_frame_vkallocator);
    top_blob.create(output_w, output_h, output_t, num_output,
                    scalar_elemsize, 1, 1, opt.blob_vkallocator);
    if (input_frame.empty() || top_blob.empty()) return -100;
    if (input_frame.buffer_capacity() > 0x7ffff000u ||
        frame_bytes > 0x7ffff000u)
    {
        std::fprintf(stderr,
                     "DecomposedConv3D frame exceeds safe Vulkan descriptor range input=%zu output=%zu\n",
                     input_frame.buffer_capacity(), frame_bytes);
        return -1;
    }

    ncnn::Option convolution_opt = opt;
    if (bounded_command_memory)
    {
        convolution_opt.blob_vkallocator = &kernel_scratch_vkallocator;
        convolution_opt.workspace_vkallocator = &kernel_scratch_vkallocator;
    }
    const size_t descriptor_limit = std::min(
        descriptor_limit_bytes(),
        static_cast<size_t>(vkdev->info.physicalDeviceProperties()
                                .limits.maxStorageBufferRange));
    const size_t buffer_offset_alignment = std::max(
        static_cast<size_t>(1), vkdev->info.buffer_offset_alignment());
    int ret = record_zero_channels(top_blob, num_output, pipeline_zero,
                                   descriptor_limit,
                                   buffer_offset_alignment, cmd);
    if (ret != 0)
        return ret;

    for (int output_index = 0; output_index < output_t; output_index++)
    {
        for (int temporal_index = 0; temporal_index < kernel_t; temporal_index++)
        {
            const int padded_index = output_index * stride_t +
                                     temporal_index * dilation_t;
            const ncnn::VkMat* source_blob = &unpacked_input;
            int source_index = padded_index - prefix_t;
            if (use_memory && padded_index < prefix_t)
            {
                source_blob = &temporal_memory_vulkan;
                source_index = padded_index;
            }
            else if (!use_memory && padded_index < prefix_t)
            {
                source_index = 0;
            }
            if (source_index < 0 || source_index >= source_blob->d) return -1;
            ret = record_gather_channels(
                *source_blob, input_frame, num_input, source_index, 0,
                pipeline_gather_frame, descriptor_limit,
                buffer_offset_alignment, cmd);
            if (ret != 0)
                return ret;

            {
                ncnn::VkMat convolution_input = input_frame;
                ncnn::VkMat input_frame_packed;
                if (num_input % 4 == 0)
                {
                    vkdev->convert_packing(input_frame, input_frame_packed, 4,
                                           cmd, convolution_opt);
                    if (input_frame_packed.empty()) return -100;
                    convolution_input = input_frame_packed;
                }

                ncnn::VkMat current_packed;
                ret = spatial_convolutions_vulkan[temporal_index]->forward(
                    convolution_input, current_packed, cmd, convolution_opt);
                if (ret != 0)
                {
                    std::fprintf(stderr,
                                 "DecomposedConv3D Vulkan convolution %d failed (%d)\n",
                                 temporal_index, ret);
                    return ret;
                }
                ncnn::VkMat current;
                if (current_packed.elempack == 1)
                    current = current_packed;
                else
                    vkdev->convert_packing(current_packed, current, 1, cmd,
                                           convolution_opt);
                if (current.empty()) return -100;
                if (current.dims != 3 || current.w != output_w ||
                    current.h != output_h || current.c != num_output ||
                    current.elempack != 1)
                {
                    std::fprintf(stderr,
                                 "DecomposedConv3D Vulkan output mismatch dims=%d w=%d/%d h=%d/%d c=%d/%d pack=%d\n",
                                 current.dims, current.w, output_w, current.h,
                                 output_h, current.c, num_output,
                                 current.elempack);
                    return -1;
                }
                ret = record_accumulate_channels(
                    current, top_blob, num_output, output_index,
                    pipeline_accumulate_frame, descriptor_limit,
                    buffer_offset_alignment, cmd);
                if (ret != 0)
                    return ret;

                if (bounded_command_memory)
                {
                    ret = cmd.submit_and_wait();
                    if (ret != 0)
                        return ret;
                    ret = cmd.reset();
                    if (ret != 0)
                        return ret;
                }
            }
            if (bounded_command_memory)
                kernel_scratch_vkallocator.clear();
        }
    }

    if (bounded_command_memory)
    {
        input_frame.release();
        frame_scratch_vkallocator.clear();
    }

    const int cache_frames = kernel_t - stride_t;
    if (memory_state != MemoryState::DISABLED && cache_frames > 0)
    {
        if (cache_frames > padded_t)
            return -1;
        ncnn::VkAllocator* memory_allocator = get_vae_memory_vkallocator();
        if (!memory_allocator)
            memory_allocator = opt.blob_vkallocator;
        ncnn::VkMat next_memory(bottom_blob.w, bottom_blob.h, cache_frames,
                                num_input, scalar_elemsize, 1, 1,
                                memory_allocator);
        if (next_memory.empty())
            return -100;
        for (int destination_frame = 0; destination_frame < cache_frames;
             destination_frame++)
        {
            const int padded_index = padded_t - cache_frames + destination_frame;
            const ncnn::VkMat* source_blob = &unpacked_input;
            int source_index = padded_index - prefix_t;
            if (use_memory && padded_index < prefix_t)
            {
                source_blob = &temporal_memory_vulkan;
                source_index = padded_index;
            }
            else if (!use_memory && padded_index < prefix_t)
            {
                source_index = 0;
            }
            if (source_index < 0 || source_index >= source_blob->d)
                return -1;
            const int cache_ret = record_gather_channels(
                *source_blob, next_memory, num_input, source_index,
                destination_frame, pipeline_gather_frame, descriptor_limit,
                buffer_offset_alignment, cmd);
            if (cache_ret != 0)
                return cache_ret;
        }
        previous_temporal_memory_vulkan = temporal_memory_vulkan;
        temporal_memory_vulkan = next_memory;

        // The bounded path may hand its input to a caller-owned scratch
        // allocator. Complete the cache copy before that allocator is cleared.
        if (bounded_command_memory)
        {
            const int ret = cmd.submit_and_wait();
            if (ret != 0)
                return ret;
            const int reset_ret = cmd.reset();
            if (reset_ret != 0)
                return reset_ret;
        }
    }
    return 0;
}
#endif

ncnn::Layer* decomposed_conv3d_layer_creator(void*)
{
    return new DecomposedConv3D;
}

} // namespace seedvr2
