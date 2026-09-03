#include "seedvr2_dit.h"
#include "shared_linear.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>

#include <gpu.h>
#include <pipeline.h>

namespace seedvr2 {

namespace {

const char finite_check_shader[] = R"glsl(
#version 450

layout(binding = 0) readonly buffer input_blob { sfp input_data[]; };
layout(binding = 1) buffer flag_blob { uint flag_data[]; };

layout(push_constant) uniform parameter
{
    int count;
    int flag_index;
} p;

void main()
{
    const int index = int(gl_GlobalInvocationID.x);
    if (index >= p.count)
        return;
    const float value = float(buffer_ld1(input_data, index));
    if (isnan(value) || isinf(value))
        atomicOr(flag_data[p.flag_index], 1u);
}
)glsl";

int create_finite_check_pipeline(const ncnn::Option& option,
                                 const ncnn::VulkanDevice* device,
                                 ncnn::Pipeline*& pipeline)
{
    std::vector<uint32_t> spirv;
    int ret = ncnn::compile_spirv_module(finite_check_shader, option, spirv);
    if (ret != 0)
        return ret;
    pipeline = new ncnn::Pipeline(device);
    pipeline->set_local_size_xyz(256, 1, 1);
    ret = pipeline->create(spirv.data(), spirv.size() * sizeof(uint32_t),
                           std::vector<ncnn::vk_specialization_type>());
    if (ret != 0)
    {
        delete pipeline;
        pipeline = 0;
    }
    return ret;
}

int write_float_mat(const char* path, const ncnn::Mat& value)
{
    if (!path || !*path || value.empty() || value.elemsize != sizeof(float))
        return -1;
    std::FILE* file = std::fopen(path, "wb");
    if (!file)
        return -1;
    const size_t count = value.total() * value.n;
    const size_t written = std::fwrite(value.data, sizeof(float), count, file);
    std::fclose(file);
    return written == count ? 0 : -1;
}

std::vector<int> parse_checkpoint_blocks(const char* value)
{
    std::vector<int> blocks;
    if (!value || !*value)
        return blocks;

    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ','))
    {
        char* end = 0;
        const long index = std::strtol(item.c_str(), &end, 10);
        if (end == item.c_str() || *end != '\0' || index < 0 || index >= 32)
        {
            blocks.clear();
            return blocks;
        }
        blocks.push_back(static_cast<int>(index));
    }
    return blocks;
}

int upload_pack1(const ncnn::Mat& source, ncnn::VkMat& destination,
                 ncnn::VulkanDevice* device, ncnn::VkCompute& command,
                 const ncnn::Option& option)
{
    ncnn::VkMat uploaded;
    command.record_upload(source, uploaded, option);
    if (uploaded.empty())
        return -100;
    if (uploaded.elempack == 1)
    {
        destination = uploaded;
        return 0;
    }
    device->convert_packing(uploaded, destination, 1, command, option);
    return destination.empty() ? -100 : 0;
}

int convert_pack1(const ncnn::VkMat& source, ncnn::VkMat& destination,
                  ncnn::VulkanDevice* device, ncnn::VkCompute& command,
                  const ncnn::Option& option)
{
    if (source.elempack == 1)
    {
        destination = source;
        return 0;
    }
    device->convert_packing(source, destination, 1, command, option);
    return destination.empty() ? -100 : 0;
}

int make_shape_mats(const std::vector<std::array<int, 3> >& video_shapes,
                    const std::vector<int>& text_lengths,
                    ncnn::Mat& video_shape, ncnn::Mat& text_length)
{
    if (video_shapes.empty() || video_shapes.size() != text_lengths.size())
        return -1;

    video_shape.create(3, static_cast<int>(video_shapes.size()));
    text_length.create(1, static_cast<int>(text_lengths.size()));
    if (video_shape.empty() || text_length.empty())
        return -100;
    for (size_t sample = 0; sample < video_shapes.size(); sample++)
    {
        const std::array<int, 3>& shape = video_shapes[sample];
        if (shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 ||
            text_lengths[sample] <= 0)
            return -1;
        for (int axis = 0; axis < 3; axis++)
            video_shape.row(static_cast<int>(sample))[axis] =
                static_cast<float>(shape[axis]);
        text_length.row(static_cast<int>(sample))[0] =
            static_cast<float>(text_lengths[sample]);
    }
    return 0;
}

int patchify_video(const ncnn::Mat& source,
                   const std::vector<std::array<int, 3> >& input_shapes,
                   ncnn::Mat& destination,
                   std::vector<std::array<int, 3> >& output_shapes)
{
    if (source.dims != 2 || source.w != 33 || source.elemsize != 4u)
        return -1;

    size_t expected_tokens = 0;
    size_t patch_tokens = 0;
    output_shapes.clear();
    output_shapes.reserve(input_shapes.size());
    for (size_t sample = 0; sample < input_shapes.size(); sample++)
    {
        const std::array<int, 3>& shape = input_shapes[sample];
        if (shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 ||
            shape[1] % 2 != 0 || shape[2] % 2 != 0)
            return -1;
        expected_tokens += static_cast<size_t>(shape[0]) * shape[1] * shape[2];
        patch_tokens += static_cast<size_t>(shape[0]) * (shape[1] / 2) *
                        (shape[2] / 2);
        output_shapes.push_back({shape[0], shape[1] / 2, shape[2] / 2});
    }
    if (expected_tokens != static_cast<size_t>(source.h))
        return -1;

    destination.create(132, static_cast<int>(patch_tokens));
    if (destination.empty())
        return -100;

    size_t input_offset = 0;
    size_t output_offset = 0;
    for (size_t sample = 0; sample < input_shapes.size(); sample++)
    {
        const int frames = input_shapes[sample][0];
        const int height = input_shapes[sample][1];
        const int width = input_shapes[sample][2];
        for (int frame = 0; frame < frames; frame++)
        {
            for (int patch_y = 0; patch_y < height / 2; patch_y++)
            {
                for (int patch_x = 0; patch_x < width / 2; patch_x++)
                {
                    float* output = destination.row(static_cast<int>(output_offset++));
                    for (int y = 0; y < 2; y++)
                    {
                        for (int x = 0; x < 2; x++)
                        {
                            const size_t token = input_offset +
                                (static_cast<size_t>(frame) * height + patch_y * 2 + y) *
                                    width + patch_x * 2 + x;
                            const float* input = source.row(static_cast<int>(token));
                            const int feature_offset = (y * 2 + x) * 33;
                            for (int channel = 0; channel < 33; channel++)
                                output[feature_offset + channel] = input[channel];
                        }
                    }
                }
            }
        }
        input_offset += static_cast<size_t>(frames) * height * width;
    }
    return 0;
}

int make_timestep_embedding(const std::vector<float>& timesteps,
                            ncnn::Mat& embedding)
{
    if (timesteps.empty())
        return -1;
    const int half_dim = 128;
    embedding.create(half_dim * 2, static_cast<int>(timesteps.size()));
    if (embedding.empty())
        return -100;
    for (size_t sample = 0; sample < timesteps.size(); sample++)
    {
        float* output = embedding.row(static_cast<int>(sample));
        for (int index = 0; index < half_dim; index++)
        {
            const float exponent = -std::log(10000.f) * index / half_dim;
            const float phase = timesteps[sample] * std::exp(exponent);
            output[index] = std::sin(phase);
            output[index + half_dim] = std::cos(phase);
        }
    }
    return 0;
}

int unpatchify_prediction(
    const ncnn::Mat& source,
    const std::vector<std::array<int, 3> >& input_shapes,
    const std::vector<std::array<int, 3> >& patchified_shapes,
    ncnn::Mat& destination)
{
    if (source.dims != 2 || source.w != 64 || source.elemsize != 4u ||
        input_shapes.size() != patchified_shapes.size())
        return -1;

    size_t expected_patches = 0;
    size_t output_tokens = 0;
    for (size_t sample = 0; sample < input_shapes.size(); sample++)
    {
        const std::array<int, 3>& input_shape = input_shapes[sample];
        const std::array<int, 3>& patch_shape = patchified_shapes[sample];
        if (patch_shape[0] != input_shape[0] ||
            patch_shape[1] * 2 != input_shape[1] ||
            patch_shape[2] * 2 != input_shape[2])
            return -1;
        expected_patches += static_cast<size_t>(patch_shape[0]) *
                            patch_shape[1] * patch_shape[2];
        output_tokens += static_cast<size_t>(input_shape[0]) *
                         input_shape[1] * input_shape[2];
    }
    if (expected_patches != static_cast<size_t>(source.h))
        return -1;

    destination.create(16, static_cast<int>(output_tokens));
    if (destination.empty())
        return -100;

    size_t patch_offset = 0;
    size_t output_offset = 0;
    for (size_t sample = 0; sample < input_shapes.size(); sample++)
    {
        const int frames = input_shapes[sample][0];
        const int height = input_shapes[sample][1];
        const int width = input_shapes[sample][2];
        const int patch_height = patchified_shapes[sample][1];
        const int patch_width = patchified_shapes[sample][2];
        for (int frame = 0; frame < frames; frame++)
        {
            for (int patch_y = 0; patch_y < patch_height; patch_y++)
            {
                for (int patch_x = 0; patch_x < patch_width; patch_x++)
                {
                    const size_t patch_index = patch_offset +
                        (static_cast<size_t>(frame) * patch_height + patch_y) *
                            patch_width + patch_x;
                    const float* input = source.row(static_cast<int>(patch_index));
                    for (int y = 0; y < 2; y++)
                    {
                        for (int x = 0; x < 2; x++)
                        {
                            const size_t token = output_offset +
                                (static_cast<size_t>(frame) * height + patch_y * 2 + y) *
                                    width + patch_x * 2 + x;
                            float* output = destination.row(static_cast<int>(token));
                            const int feature_offset = (y * 2 + x) * 16;
                            for (int channel = 0; channel < 16; channel++)
                                output[channel] = input[feature_offset + channel];
                        }
                    }
                }
            }
        }
        patch_offset += static_cast<size_t>(frames) * patch_height * patch_width;
        output_offset += static_cast<size_t>(frames) * height * width;
    }
    return 0;
}

int append_row_mats(const std::vector<ncnn::Mat>& values,
                    ncnn::Mat& output)
{
    if (values.empty())
        return -1;

    const int width = values[0].w;
    const size_t elemsize = values[0].elemsize;
    const int elempack = values[0].elempack;
    size_t rows = 0;
    for (size_t i = 0; i < values.size(); i++)
    {
        const ncnn::Mat& value = values[i];
        if (value.dims != 2 || value.w != width ||
            value.elemsize != elemsize || value.elempack != elempack ||
            value.h <= 0)
            return -1;
        rows += static_cast<size_t>(value.h);
        if (rows > static_cast<size_t>(std::numeric_limits<int>::max()))
            return -1;
    }

    output.create(width, static_cast<int>(rows), elemsize, elempack);
    if (output.empty())
        return -100;

    size_t row_offset = 0;
    for (size_t i = 0; i < values.size(); i++)
    {
        const ncnn::Mat& value = values[i];
        const size_t bytes = static_cast<size_t>(value.w) * value.h *
                             value.elemsize;
        std::memcpy(output.row(static_cast<int>(row_offset)), value.data,
                    bytes);
        row_offset += static_cast<size_t>(value.h);
    }
    return 0;
}

} // namespace

int split_dit_inputs_by_sample(const DiTInputs& inputs,
                               std::vector<DiTInputs>& samples)
{
    samples.clear();
    const size_t batch = inputs.video_shapes.size();
    if (batch == 0 || inputs.text_lengths.size() != batch ||
        inputs.timesteps.size() != batch || inputs.video.dims != 2 ||
        inputs.video.w != 33 || inputs.video.elemsize != sizeof(float) ||
        inputs.video.elempack != 1 || inputs.text.dims != 2 ||
        inputs.text.w != 5120 || inputs.text.elemsize != sizeof(float) ||
        inputs.text.elempack != 1)
        return -1;

    size_t video_offset = 0;
    size_t text_offset = 0;
    samples.reserve(batch);
    for (size_t sample_index = 0; sample_index < batch; sample_index++)
    {
        const std::array<int, 3>& shape = inputs.video_shapes[sample_index];
        const int text_length = inputs.text_lengths[sample_index];
        if (shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 ||
            text_length <= 0 || !std::isfinite(inputs.timesteps[sample_index]))
        {
            samples.clear();
            return -1;
        }
        const size_t video_tokens = static_cast<size_t>(shape[0]) *
                                    static_cast<size_t>(shape[1]) *
                                    static_cast<size_t>(shape[2]);
        if (video_tokens > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            video_offset + video_tokens > static_cast<size_t>(inputs.video.h) ||
            text_offset + static_cast<size_t>(text_length) >
                static_cast<size_t>(inputs.text.h))
        {
            samples.clear();
            return -1;
        }

        DiTInputs sample;
        sample.video = inputs.video.row_range(static_cast<int>(video_offset),
                                              static_cast<int>(video_tokens));
        sample.text = inputs.text.row_range(static_cast<int>(text_offset),
                                            text_length);
        sample.video_shapes.push_back(shape);
        sample.text_lengths.push_back(text_length);
        sample.timesteps.push_back(inputs.timesteps[sample_index]);
        samples.push_back(sample);
        video_offset += video_tokens;
        text_offset += static_cast<size_t>(text_length);
    }

    if (video_offset != static_cast<size_t>(inputs.video.h) ||
        text_offset != static_cast<size_t>(inputs.text.h))
    {
        samples.clear();
        return -1;
    }
    return 0;
}

int split_dit_block_inputs_by_sample(
    const DiTBlockInputs& inputs,
    std::vector<DiTBlockInputs>& samples)
{
    samples.clear();
    const size_t batch = inputs.video_shapes.size();
    if (batch == 0 || inputs.text_lengths.size() != batch ||
        inputs.video.dims != 2 || inputs.video.w != 2560 ||
        inputs.video.elemsize != sizeof(float) || inputs.video.elempack != 1 ||
        inputs.text.dims != 2 || inputs.text.w != 2560 ||
        inputs.text.elemsize != sizeof(float) || inputs.text.elempack != 1)
        return -1;
    for (int i = 0; i < 6; i++)
    {
        const ncnn::Mat& modulation = inputs.modulation[i];
        if (modulation.dims != 2 || modulation.w != 2560 ||
            modulation.h != static_cast<int>(batch) ||
            modulation.elemsize != sizeof(float) || modulation.elempack != 1)
            return -1;
    }

    size_t video_offset = 0;
    size_t text_offset = 0;
    samples.reserve(batch);
    for (size_t sample_index = 0; sample_index < batch; sample_index++)
    {
        const std::array<int, 3>& shape = inputs.video_shapes[sample_index];
        const int text_length = inputs.text_lengths[sample_index];
        if (shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 ||
            text_length <= 0)
        {
            samples.clear();
            return -1;
        }
        const size_t video_tokens = static_cast<size_t>(shape[0]) *
                                    static_cast<size_t>(shape[1]) *
                                    static_cast<size_t>(shape[2]);
        if (video_tokens > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            video_offset + video_tokens > static_cast<size_t>(inputs.video.h) ||
            text_offset + static_cast<size_t>(text_length) >
                static_cast<size_t>(inputs.text.h))
        {
            samples.clear();
            return -1;
        }

        DiTBlockInputs sample;
        sample.video = inputs.video.row_range(static_cast<int>(video_offset),
                                              static_cast<int>(video_tokens));
        sample.text = inputs.text.row_range(static_cast<int>(text_offset),
                                            text_length);
        sample.video_shapes.push_back(shape);
        sample.text_lengths.push_back(text_length);
        for (int i = 0; i < 6; i++)
            sample.modulation[i] = inputs.modulation[i].row_range(
                static_cast<int>(sample_index), 1);
        samples.push_back(sample);
        video_offset += video_tokens;
        text_offset += static_cast<size_t>(text_length);
    }

    if (video_offset != static_cast<size_t>(inputs.video.h) ||
        text_offset != static_cast<size_t>(inputs.text.h))
    {
        samples.clear();
        return -1;
    }
    return 0;
}

struct SeedVR2DiTBlocks::Block
{
    AdaptiveWindowRuntimeState runtime_state;
    ncnn::Net net;
};

SeedVR2DiTBlocks::SeedVR2DiTBlocks()
    : gpu_id(-1), bf16_storage(false), frontend_loaded(false),
      tail_loaded(false)
{
}

SeedVR2DiTBlocks::~SeedVR2DiTBlocks()
{
    clear();
}

void SeedVR2DiTBlocks::clear()
{
    frontend.clear();
    tail.clear();
    frontend_loaded = false;
    tail_loaded = false;
    for (size_t i = 0; i < blocks.size(); i++)
        blocks[i]->net.clear();
    blocks.clear();
    if (gpu_id >= 0)
        ncnn::destroy_gpu_instance();
    gpu_id = -1;
    bf16_storage = false;
}

int SeedVR2DiTBlocks::load(const std::string& model_directory, int new_gpu_id,
                           bool use_bf16_storage)
{
    clear();
    ncnn::create_gpu_instance();
    if (new_gpu_id < 0 || new_gpu_id >= ncnn::get_gpu_count())
    {
        ncnn::destroy_gpu_instance();
        return -1;
    }
    gpu_id = new_gpu_id;
    bf16_storage = use_bf16_storage;
    blocks.reserve(32);
    const bool disable_cooperative_matrix =
        std::getenv("SEEDVR2_DIT_DISABLE_COOPERATIVE_MATRIX") != 0;
    const bool disable_shader_local_memory =
        std::getenv("SEEDVR2_DIT_DISABLE_SHADER_LOCAL_MEMORY") != 0;
    const bool disable_subgroup_ops =
        std::getenv("SEEDVR2_DIT_DISABLE_SUBGROUP_OPS") != 0;
    if (disable_cooperative_matrix)
        std::fprintf(stderr, "SeedVR2 DiT cooperative-matrix kernels disabled\n");
    if (disable_shader_local_memory)
        std::fprintf(stderr, "SeedVR2 DiT shader-local-memory kernels disabled\n");
    if (disable_subgroup_ops)
        std::fprintf(stderr, "SeedVR2 DiT subgroup kernels disabled\n");

    frontend.opt.use_vulkan_compute = true;
    // FP16 activations diverge from the official model after block 5 because
    // the same large AdaSingle scale is reused by all 32 blocks. Keep the
    // exported weights in FP16, but execute and store activations in FP32.
    frontend.opt.use_fp16_storage = false;
    frontend.opt.use_fp16_packed = false;
    frontend.opt.use_bf16_storage = bf16_storage;
    frontend.opt.use_bf16_packed = false;
    frontend.opt.use_fp16_arithmetic = false;
    frontend.opt.use_cooperative_matrix = !disable_cooperative_matrix;
    frontend.opt.use_shader_local_memory = !disable_shader_local_memory;
    frontend.opt.use_subgroup_ops = !disable_subgroup_ops;
    frontend.set_vulkan_device(gpu_id);
    const std::string frontend_stem = model_directory + "/frontend/seedvr2_frontend.ncnn";
    int ret = frontend.load_param((frontend_stem + ".param").c_str());
    if (ret == 0)
        ret = frontend.load_model((frontend_stem + ".bin").c_str());
    if (ret != 0)
    {
        std::fprintf(stderr, "failed to load SeedVR2 DiT frontend (%d)\n", ret);
        clear();
        return ret;
    }
    frontend_loaded = true;

    tail.opt.use_vulkan_compute = true;
    tail.opt.use_fp16_storage = false;
    tail.opt.use_fp16_packed = false;
    tail.opt.use_bf16_storage = bf16_storage;
    tail.opt.use_bf16_packed = false;
    tail.opt.use_fp16_arithmetic = false;
    tail.opt.use_cooperative_matrix = !disable_cooperative_matrix;
    tail.opt.use_shader_local_memory = !disable_shader_local_memory;
    tail.opt.use_subgroup_ops = !disable_subgroup_ops;
    tail.set_vulkan_device(gpu_id);
    const std::string tail_stem = model_directory + "/tail/seedvr2_tail.ncnn";
    ret = tail.load_param((tail_stem + ".param").c_str());
    if (ret == 0)
        ret = tail.load_model((tail_stem + ".bin").c_str());
    if (ret != 0)
    {
        std::fprintf(stderr, "failed to load SeedVR2 DiT tail (%d)\n", ret);
        clear();
        return ret;
    }
    tail_loaded = true;

    for (int index = 0; index < 32; index++)
    {
        std::ostringstream block_name;
        block_name << "block_" << std::setfill('0') << std::setw(2) << index;
        const std::string stem = "seedvr2_" + block_name.str();
        const std::string directory = model_directory + "/" + block_name.str();

        std::unique_ptr<Block> block(new Block);
        block->net.opt.use_vulkan_compute = true;
        block->net.opt.use_fp16_storage = false;
        block->net.opt.use_fp16_packed = false;
        block->net.opt.use_bf16_storage = bf16_storage;
        block->net.opt.use_bf16_packed = false;
        block->net.opt.use_fp16_arithmetic = false;
        block->net.opt.use_cooperative_matrix = !disable_cooperative_matrix;
        block->net.opt.use_shader_local_memory = !disable_shader_local_memory;
        block->net.opt.use_subgroup_ops = !disable_subgroup_ops;
        block->net.set_vulkan_device(gpu_id);
        block->net.register_custom_layer(
            "AdaptiveWindowAttention", adaptive_window_attention_layer_creator,
            0, &block->runtime_state);
        block->net.register_custom_layer(
            "SharedLinear", shared_linear_layer_creator);

        ret = block->net.load_param((directory + "/" + stem + ".ncnn.param").c_str());
        if (ret == 0)
            ret = block->net.load_model((directory + "/" + stem + ".ncnn.bin").c_str());
        if (ret != 0)
        {
            std::fprintf(stderr, "failed to load SeedVR2 DiT block %d (%d)\n", index, ret);
            clear();
            return ret;
        }
        blocks.push_back(std::move(block));
    }
    return 0;
}

int SeedVR2DiTBlocks::record_blocks(
    const std::vector<std::array<int, 3> >& video_shapes,
    const std::vector<int>& text_lengths,
    ncnn::VkMat& video, ncnn::VkMat& text,
    const ncnn::VkMat& video_shape, const ncnn::VkMat& text_length,
    const std::array<ncnn::VkMat, 6>& modulation,
    ncnn::VkCompute& command, ncnn::VkAllocator* blob_allocator,
    ncnn::VkAllocator* staging_allocator,
    const std::vector<int>& checkpoint_blocks,
    std::vector<ncnn::Mat>& checkpoint_videos,
    std::vector<ncnn::Mat>& checkpoint_texts) const
{
    if (checkpoint_blocks.size() != checkpoint_videos.size() ||
        checkpoint_blocks.size() != checkpoint_texts.size())
        return -1;
    const bool check_finite = std::getenv("SEEDVR2_DIT_CHECK_FINITE") != 0;
    ncnn::Pipeline* finite_check_pipeline = 0;
    ncnn::Option check_option;
    if (check_finite)
    {
        check_option.use_vulkan_compute = true;
        check_option.use_packing_layout = false;
        check_option.use_fp16_storage = false;
        check_option.use_fp16_packed = false;
        check_option.use_bf16_storage = bf16_storage;
        check_option.use_bf16_packed = false;
        check_option.use_fp16_arithmetic = false;
        check_option.blob_vkallocator = blob_allocator;
        check_option.workspace_vkallocator = blob_allocator;
        check_option.staging_vkallocator = staging_allocator;
        const int pipeline_ret = create_finite_check_pipeline(
            check_option, ncnn::get_gpu_device(gpu_id), finite_check_pipeline);
        if (pipeline_ret != 0)
            return pipeline_ret;
    }

    int ret = 0;
    if (check_finite)
    {
        ncnn::Mat flags(2);
        std::memset(flags.data, 0, flags.total() * flags.elemsize);
        ncnn::VkMat flags_gpu;
        ncnn::Option flag_option = check_option;
        flag_option.use_bf16_storage = false;
        flag_option.use_bf16_packed = false;
        command.record_upload(flags, flags_gpu, flag_option);
        std::vector<ncnn::VkMat> bindings(2);
        bindings[1] = flags_gpu;
        std::vector<ncnn::vk_constant_type> constants(2);
        ncnn::VkMat dispatcher;
        bindings[0] = video;
        constants[0].i = static_cast<int>(video.total() * video.elempack);
        constants[1].i = 0;
        dispatcher.w = constants[0].i;
        dispatcher.h = 1;
        dispatcher.c = 1;
        command.record_pipeline(finite_check_pipeline, bindings, constants,
                                dispatcher);
        bindings[0] = text;
        constants[0].i = static_cast<int>(text.total() * text.elempack);
        constants[1].i = 1;
        dispatcher.w = constants[0].i;
        command.record_pipeline(finite_check_pipeline, bindings, constants,
                                dispatcher);
        command.record_download(flags_gpu, flags, flag_option);
        ret = command.submit_and_wait();
        command.reset();
        if (ret == 0)
        {
            const unsigned int* flag_values =
                static_cast<const unsigned int*>(flags.data);
            std::fprintf(stderr,
                         "SeedVR2 DiT frontend finite video=%d text=%d\n",
                         flag_values[0] == 0 ? 1 : 0,
                         flag_values[1] == 0 ? 1 : 0);
            if (flag_values[0] != 0 || flag_values[1] != 0)
                ret = -1;
            else if (std::getenv("SEEDVR2_DIT_CHECK_FRONTEND_ONLY"))
                ret = -2;
        }
    }
    for (size_t index = 0; ret == 0 && index < blocks.size(); index++)
    {
        Block& block = *blocks[index];
        block.runtime_state.video_shapes = video_shapes;
        block.runtime_state.text_lengths = text_lengths;
        ncnn::Extractor extractor = block.net.create_extractor();
        extractor.set_blob_vkallocator(blob_allocator);
        extractor.set_workspace_vkallocator(blob_allocator);
        extractor.set_staging_vkallocator(staging_allocator);
        ret = extractor.input("in0", video);
        if (ret == 0) ret = extractor.input("in1", text);
        if (ret == 0) ret = extractor.input("in2", video_shape);
        if (ret == 0) ret = extractor.input("in3", text_length);
        for (int i = 0; ret == 0 && i < 6; i++)
        {
            const std::string name = "in" + std::to_string(i + 4);
            ret = extractor.input(name.c_str(), modulation[i]);
        }
        ncnn::VkMat next_video;
        ncnn::VkMat next_text;
        ncnn::VkMat block0_qkv_input;
        std::array<ncnn::VkMat, 9> block0_qkv_input_trace;
        std::array<ncnn::VkMat, 4> block0_attention_boundaries;
        const bool check_block0_attention =
            check_finite && index == 0 &&
            std::getenv("SEEDVR2_DIT_CHECK_BLOCK0_ATTENTION");
        int block0_qkv_chunk_count = 0;
        if (check_block0_attention)
        {
            if (std::getenv("SEEDVR2_DIT_CHECK_QKV_CHUNKS"))
            {
                const char* trace_names[] = {
                    "1", "40", "41", "44", "45", "46", "47", "48", "50"};
                for (int i = 0; ret == 0 && i < 9; i++)
                    ret = extractor.extract(trace_names[i],
                                            block0_qkv_input_trace[i], command);
                ret = extractor.extract("50", block0_qkv_input, command);
                if (ret == 0 && std::getenv("SEEDVR2_DIT_LOG_TENSORS"))
                {
                    for (int i = 0; i < 9; i++)
                    {
                        const ncnn::VkMat& value = block0_qkv_input_trace[i];
                        std::fprintf(stderr,
                                     "SeedVR2 DiT block 0 input trace %s "
                                     "dims=%d w=%d h=%d elemsize=%zu "
                                     "elempack=%d total=%zu range=%zu "
                                     "offset=%zu\n",
                                     trace_names[i], value.dims, value.w,
                                     value.h, value.elemsize, value.elempack,
                                     value.total(), value.buffer_capacity(),
                                     value.buffer_offset());
                    }
                }
            }
            const char* names[] = {"62", "63", "64", "65"};
            for (int i = 0; ret == 0 && i < 4; i++)
                ret = extractor.extract(names[i],
                                        block0_attention_boundaries[i],
                                        command);
            if (ret == 0 && std::getenv("SEEDVR2_DIT_LOG_TENSORS"))
            {
                for (int i = 0; i < 4; i++)
                {
                    const ncnn::VkMat& value = block0_attention_boundaries[i];
                    std::fprintf(
                        stderr,
                        "SeedVR2 DiT block 0 blob %s dims=%d w=%d h=%d "
                        "d=%d c=%d elemsize=%zu elempack=%d total=%zu "
                        "range=%zu offset=%zu\n",
                        names[i], value.dims, value.w, value.h, value.d,
                        value.c, value.elemsize, value.elempack, value.total(),
                        value.buffer_capacity(), value.buffer_offset());
                }
            }
#if NCNN_BATCH
            if (ret == 0 && std::getenv("SEEDVR2_DIT_CHECK_QKV_CHUNKS"))
            {
                const ncnn::VkMat& video_qkv = block0_attention_boundaries[0];
                const int qkv_tokens = video_qkv.h * video_qkv.elempack;
                block0_qkv_chunk_count = (qkv_tokens + 16383) / 16384;
            }
#endif
        }
        if (ret == 0) ret = extractor.extract("out0", next_video, command);
        if (ret == 0) ret = extractor.extract("out1", next_text, command);
        if (ret != 0)
        {
            std::fprintf(stderr, "SeedVR2 DiT block %zu failed (%d)\n", index, ret);
            break;
        }
        video = next_video;
        text = next_text;

        if (check_finite)
        {
            const int block0_trace_count = block0_qkv_chunk_count ? 9 : 0;
            std::array<ncnn::Mat, 3> block0_small_trace_cpu;
            ncnn::Mat flags(check_block0_attention
                                ? 6 + block0_qkv_chunk_count * 2 +
                                      block0_trace_count
                                : 2);
            std::memset(flags.data, 0, flags.total() * flags.elemsize);
            ncnn::VkMat flags_gpu;
            ncnn::Option flag_option = check_option;
            flag_option.use_bf16_storage = false;
            flag_option.use_bf16_packed = false;
            command.record_upload(flags, flags_gpu, flag_option);

            std::vector<ncnn::VkMat> bindings(2);
            bindings[1] = flags_gpu;
            std::vector<ncnn::vk_constant_type> constants(2);
            ncnn::VkMat dispatcher;
            bindings[0] = video;
            constants[0].i = static_cast<int>(video.total() * video.elempack);
            constants[1].i = 0;
            dispatcher.w = constants[0].i;
            dispatcher.h = 1;
            dispatcher.c = 1;
            command.record_pipeline(finite_check_pipeline, bindings, constants,
                                    dispatcher);

            bindings[0] = text;
            constants[0].i = static_cast<int>(text.total() * text.elempack);
            constants[1].i = 1;
            dispatcher.w = constants[0].i;
            command.record_pipeline(finite_check_pipeline, bindings, constants,
                                    dispatcher);
            if (check_block0_attention)
            {
                for (int i = 0; i < 4; i++)
                {
                    bindings[0] = block0_attention_boundaries[i];
                    constants[0].i = static_cast<int>(
                        block0_attention_boundaries[i].total() *
                        block0_attention_boundaries[i].elempack);
                    constants[1].i = i + 2;
                    dispatcher.w = constants[0].i;
                    command.record_pipeline(finite_check_pipeline, bindings,
                                            constants, dispatcher);
                }
#if NCNN_BATCH
                const ncnn::VkMat& video_qkv = block0_attention_boundaries[0];
                const int qkv_tokens = video_qkv.h * video_qkv.elempack;
                for (int chunk = 0; chunk < block0_qkv_chunk_count; chunk++)
                {
                    const int token_offset = chunk * 16384;
                    const int token_count = std::min(16384, qkv_tokens - token_offset);
                    ncnn::VkMat qkv_chunk = video_qkv;
                    qkv_chunk.h = token_count / qkv_chunk.elempack;
                    qkv_chunk.cstep = ncnn::alignSize(
                        static_cast<size_t>(qkv_chunk.w) * qkv_chunk.h *
                            qkv_chunk.elemsize,
                        16) /
                                      qkv_chunk.elemsize;
                    qkv_chunk.nstep = qkv_chunk.total();
                    qkv_chunk.offset = video_qkv.offset +
                                       static_cast<size_t>(
                                           token_offset / qkv_chunk.elempack) *
                                           qkv_chunk.w * qkv_chunk.elemsize;
                    bindings[0] = qkv_chunk;
                    constants[0].i = static_cast<int>(
                        qkv_chunk.total() * qkv_chunk.elempack);
                    constants[1].i = chunk + 6;
                    dispatcher.w = constants[0].i;
                    command.record_pipeline(finite_check_pipeline, bindings,
                                            constants, dispatcher);

                    ncnn::VkMat input_chunk = block0_qkv_input;
                    input_chunk.h = token_count / input_chunk.elempack;
                    input_chunk.cstep = ncnn::alignSize(
                        static_cast<size_t>(input_chunk.w) * input_chunk.h *
                            input_chunk.elemsize,
                        16) /
                                        input_chunk.elemsize;
                    input_chunk.nstep = input_chunk.total();
                    input_chunk.offset = block0_qkv_input.offset +
                                         static_cast<size_t>(
                                             token_offset /
                                             input_chunk.elempack) *
                                             input_chunk.w *
                                             input_chunk.elemsize;
                    bindings[0] = input_chunk;
                    constants[0].i = static_cast<int>(
                        input_chunk.total() * input_chunk.elempack);
                    constants[1].i = chunk + 6 + block0_qkv_chunk_count;
                    dispatcher.w = constants[0].i;
                    command.record_pipeline(finite_check_pipeline, bindings,
                                            constants, dispatcher);
                }
                for (int i = 0; i < block0_trace_count; i++)
                {
                    bindings[0] = block0_qkv_input_trace[i];
                    constants[0].i = static_cast<int>(
                        block0_qkv_input_trace[i].total() *
                        block0_qkv_input_trace[i].elempack);
                    constants[1].i = 6 + block0_qkv_chunk_count * 2 + i;
                    dispatcher.w = constants[0].i;
                    command.record_pipeline(finite_check_pipeline, bindings,
                                            constants, dispatcher);
                }
                if (block0_trace_count)
                {
                    command.record_download(block0_qkv_input_trace[2],
                                            block0_small_trace_cpu[0],
                                            check_option);
                    command.record_download(block0_qkv_input_trace[3],
                                            block0_small_trace_cpu[1],
                                            check_option);
                    command.record_download(block0_qkv_input_trace[4],
                                            block0_small_trace_cpu[2],
                                            check_option);
                }
#endif
            }
            command.record_download(flags_gpu, flags, flag_option);
            ret = command.submit_and_wait();
            command.reset();
            if (ret != 0)
                break;
            const unsigned int* flag_values =
                static_cast<const unsigned int*>(flags.data);
            std::fprintf(stderr,
                         "SeedVR2 DiT block %zu finite video=%d text=%d\n",
                         index, flag_values[0] == 0 ? 1 : 0,
                         flag_values[1] == 0 ? 1 : 0);
            if (check_block0_attention)
            {
                std::fprintf(
                    stderr,
                    "SeedVR2 DiT block 0 attention finite "
                    "video_qkv=%d text_qkv=%d video_out=%d text_out=%d\n",
                    flag_values[2] == 0 ? 1 : 0,
                    flag_values[3] == 0 ? 1 : 0,
                    flag_values[4] == 0 ? 1 : 0,
                    flag_values[5] == 0 ? 1 : 0);
                for (int chunk = 0; chunk < block0_qkv_chunk_count; chunk++)
                {
                    const ncnn::VkMat& video_qkv = block0_attention_boundaries[0];
                    const int token_offset = chunk * 16384;
                    const int token_count = std::min(
                        16384,
                        video_qkv.h * video_qkv.elempack - token_offset);
                    const size_t byte_offset = video_qkv.buffer_offset() +
                                               static_cast<size_t>(
                                                   token_offset /
                                                   video_qkv.elempack) *
                                                   video_qkv.w *
                                                   video_qkv.elemsize;
                    const size_t byte_range =
                        static_cast<size_t>(token_count /
                                            video_qkv.elempack) *
                        video_qkv.w * video_qkv.elemsize;
                    std::fprintf(stderr,
                                 "SeedVR2 DiT block 0 video QKV chunk %d "
                                 "tokens=%d:%d input_finite=%d output_finite=%d "
                                 "offset=%zu range=%zu\n",
                                 chunk, token_offset,
                                 token_offset + token_count,
                                 flag_values[chunk + 6 +
                                             block0_qkv_chunk_count] == 0
                                     ? 1
                                     : 0,
                                 flag_values[chunk + 6] == 0 ? 1 : 0,
                                 byte_offset, byte_range);
                }
                if (block0_trace_count)
                {
                    const char* trace_names[] = {
                        "1", "40", "41", "44", "45", "46", "47", "48", "50"};
                    for (int i = 0; i < block0_trace_count; i++)
                    {
                        std::fprintf(
                            stderr,
                            "SeedVR2 DiT block 0 input trace %s finite=%d\n",
                            trace_names[i],
                            flag_values[6 + block0_qkv_chunk_count * 2 + i] == 0
                                ? 1
                                : 0);
                    }
                    const char* small_names[] = {"41", "44", "45"};
                    for (int i = 0; i < 3; i++)
                    {
                        const ncnn::Mat& value = block0_small_trace_cpu[i];
                        const float* values = static_cast<const float*>(value.data);
                        const size_t count = value.total() * value.elempack;
                        float minimum = INFINITY;
                        float maximum = -INFINITY;
                        size_t nonpositive = 0;
                        size_t nonfinite = 0;
                        size_t first_nonfinite = count;
                        for (size_t j = 0; j < count; j++)
                        {
                            const float scalar = values[j];
                            if (!std::isfinite(scalar))
                            {
                                nonfinite++;
                                if (first_nonfinite == count)
                                    first_nonfinite = j;
                                continue;
                            }
                            minimum = std::min(minimum, scalar);
                            maximum = std::max(maximum, scalar);
                            if (scalar <= 0.f)
                                nonpositive++;
                        }
                        std::fprintf(stderr,
                                     "SeedVR2 DiT block 0 input trace %s "
                                     "values=%zu min=%.9g max=%.9g "
                                     "nonpositive=%zu nonfinite=%zu "
                                     "first_nonfinite=%zu\n",
                                     small_names[i], count, minimum, maximum,
                                     nonpositive, nonfinite, first_nonfinite);
                    }
                }
            }
            if (flag_values[0] != 0 || flag_values[1] != 0)
            {
                ret = -1;
                break;
            }
        }
        for (size_t checkpoint = 0; checkpoint < checkpoint_blocks.size(); checkpoint++)
        {
            if (static_cast<int>(index) != checkpoint_blocks[checkpoint])
                continue;
            ncnn::Option option;
            option.use_vulkan_compute = true;
            option.use_packing_layout = false;
            option.use_fp16_storage = false;
            option.use_fp16_packed = false;
            option.use_bf16_storage = bf16_storage;
            option.use_bf16_packed = false;
            option.use_fp16_arithmetic = false;
            option.blob_vkallocator = blob_allocator;
            option.workspace_vkallocator = blob_allocator;
            option.staging_vkallocator = staging_allocator;
            command.record_download(video, checkpoint_videos[checkpoint], option);
            command.record_download(text, checkpoint_texts[checkpoint], option);
        }
    }
    delete finite_check_pipeline;
    return ret;
}

int SeedVR2DiTBlocks::forward(const DiTBlockInputs& inputs,
                              ncnn::Mat& video_output,
                              ncnn::Mat& text_output) const
{
    if (blocks.size() != 32)
        return -1;

    std::vector<DiTBlockInputs> samples;
    int ret = split_dit_block_inputs_by_sample(inputs, samples);
    if (ret != 0)
        return ret;
    if (samples.size() > 1)
    {
        std::vector<ncnn::Mat> video_samples(samples.size());
        std::vector<ncnn::Mat> text_samples(samples.size());
        for (size_t i = 0; i < samples.size(); i++)
        {
            ret = forward(samples[i], video_samples[i], text_samples[i]);
            if (ret != 0)
                return ret;
        }
        ret = append_row_mats(video_samples, video_output);
        if (ret == 0)
            ret = append_row_mats(text_samples, text_output);
        return ret;
    }

    ncnn::Mat video_shape;
    ncnn::Mat text_length;
    ret = make_shape_mats(inputs.video_shapes, inputs.text_lengths,
                          video_shape, text_length);
    if (ret != 0)
        return ret;

    ncnn::VulkanDevice* device = ncnn::get_gpu_device(gpu_id);
    ncnn::VkAllocator* blob_allocator = device->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = device->acquire_staging_allocator();
    ncnn::Option option;
    option.use_vulkan_compute = true;
    option.use_packing_layout = false;
    option.use_fp16_storage = false;
    option.use_fp16_packed = false;
    option.use_bf16_storage = bf16_storage;
    option.use_bf16_packed = false;
    option.use_fp16_arithmetic = false;
    option.blob_vkallocator = blob_allocator;
    option.workspace_vkallocator = blob_allocator;
    option.staging_vkallocator = staging_allocator;

    ret = 0;
    {
        ncnn::VkCompute command(device);
        ncnn::VkMat video;
        ncnn::VkMat text;
        ncnn::VkMat video_shape_gpu;
        ncnn::VkMat text_length_gpu;
        std::array<ncnn::VkMat, 6> modulation;
        ret = upload_pack1(inputs.video, video, device, command, option);
        if (ret == 0) ret = upload_pack1(inputs.text, text, device, command, option);
        if (ret == 0) ret = upload_pack1(video_shape, video_shape_gpu, device, command, option);
        if (ret == 0) ret = upload_pack1(text_length, text_length_gpu, device, command, option);
        for (int i = 0; ret == 0 && i < 6; i++)
            ret = upload_pack1(inputs.modulation[i], modulation[i], device, command, option);

        std::vector<int> checkpoint_blocks;
        std::vector<ncnn::Mat> checkpoint_videos;
        std::vector<ncnn::Mat> checkpoint_texts;
        if (ret == 0)
            ret = record_blocks(inputs.video_shapes, inputs.text_lengths,
                                video, text, video_shape_gpu, text_length_gpu,
                                modulation, command, blob_allocator,
                                staging_allocator, checkpoint_blocks,
                                checkpoint_videos, checkpoint_texts);

        if (ret == 0)
        {
            command.record_download(video, video_output, option);
            command.record_download(text, text_output, option);
            ret = command.submit_and_wait();
        }
    }

    device->reclaim_blob_allocator(blob_allocator);
    device->reclaim_staging_allocator(staging_allocator);
    return ret;
}

int SeedVR2DiTBlocks::forward_frontend(const DiTInputs& inputs,
                                       DiTBlockInputs& outputs) const
{
    if (!frontend_loaded)
        return -1;

    std::vector<DiTInputs> samples;
    int ret = split_dit_inputs_by_sample(inputs, samples);
    if (ret != 0)
        return ret;
    if (samples.size() > 1)
    {
        std::vector<DiTBlockInputs> sample_outputs(samples.size());
        std::vector<ncnn::Mat> video_samples(samples.size());
        std::vector<ncnn::Mat> text_samples(samples.size());
        std::array<std::vector<ncnn::Mat>, 6> modulation_samples;
        DiTBlockInputs combined;
        for (size_t i = 0; i < samples.size(); i++)
        {
            ret = forward_frontend(samples[i], sample_outputs[i]);
            if (ret != 0)
                return ret;
            video_samples[i] = sample_outputs[i].video;
            text_samples[i] = sample_outputs[i].text;
            combined.video_shapes.push_back(sample_outputs[i].video_shapes[0]);
            combined.text_lengths.push_back(sample_outputs[i].text_lengths[0]);
            for (int j = 0; j < 6; j++)
                modulation_samples[j].push_back(sample_outputs[i].modulation[j]);
        }
        ret = append_row_mats(video_samples, combined.video);
        if (ret == 0)
            ret = append_row_mats(text_samples, combined.text);
        for (int i = 0; ret == 0 && i < 6; i++)
            ret = append_row_mats(modulation_samples[i], combined.modulation[i]);
        if (ret == 0)
            outputs = combined;
        return ret;
    }

    ncnn::Mat patchified_video;
    std::vector<std::array<int, 3> > patchified_shapes;
    ret = patchify_video(inputs.video, inputs.video_shapes,
                         patchified_video, patchified_shapes);
    if (ret != 0)
        return ret;
    ncnn::Mat timestep_embedding;
    ret = make_timestep_embedding(inputs.timesteps, timestep_embedding);
    if (ret != 0)
        return ret;

    ncnn::VulkanDevice* device = ncnn::get_gpu_device(gpu_id);
    ncnn::VkAllocator* blob_allocator = device->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = device->acquire_staging_allocator();
    ncnn::Option option;
    option.use_vulkan_compute = true;
    option.use_packing_layout = false;
    option.use_fp16_storage = false;
    option.use_fp16_packed = false;
    option.use_bf16_storage = bf16_storage;
    option.use_bf16_packed = false;
    option.use_fp16_arithmetic = false;
    option.blob_vkallocator = blob_allocator;
    option.workspace_vkallocator = blob_allocator;
    option.staging_vkallocator = staging_allocator;
    {
        ncnn::VkCompute command(device);
        ncnn::VkMat patchified_video_gpu;
        ncnn::VkMat text_embedding_gpu;
        ncnn::VkMat timestep_embedding_gpu;
        ret = upload_pack1(patchified_video, patchified_video_gpu, device,
                           command, option);
        if (ret == 0)
            ret = upload_pack1(inputs.text, text_embedding_gpu, device,
                               command, option);
        if (ret == 0)
            ret = upload_pack1(timestep_embedding, timestep_embedding_gpu,
                               device, command, option);

        std::array<ncnn::VkMat, 8> packed;
        std::array<ncnn::VkMat, 8> unpacked;
        if (ret == 0)
        {
            ncnn::Extractor extractor = frontend.create_extractor();
            extractor.set_blob_vkallocator(blob_allocator);
            extractor.set_workspace_vkallocator(blob_allocator);
            extractor.set_staging_vkallocator(staging_allocator);
            ret = extractor.input("in0", patchified_video_gpu);
            if (ret == 0) ret = extractor.input("in1", text_embedding_gpu);
            if (ret == 0) ret = extractor.input("in2", timestep_embedding_gpu);
            for (int i = 0; ret == 0 && i < 8; i++)
            {
                const std::string name = "out" + std::to_string(i);
                ret = extractor.extract(name.c_str(), packed[i], command);
            }
            for (int i = 0; ret == 0 && i < 8; i++)
                ret = convert_pack1(packed[i], unpacked[i], device, command, option);
        }
        if (ret == 0)
        {
            command.record_download(unpacked[0], outputs.video, option);
            command.record_download(unpacked[1], outputs.text, option);
            for (int i = 0; i < 6; i++)
                command.record_download(unpacked[i + 2], outputs.modulation[i], option);
            ret = command.submit_and_wait();
        }
    }

    outputs.video_shapes = patchified_shapes;
    outputs.text_lengths = inputs.text_lengths;
    device->reclaim_blob_allocator(blob_allocator);
    device->reclaim_staging_allocator(staging_allocator);
    return ret;
}

int SeedVR2DiTBlocks::forward(const DiTInputs& inputs,
                              ncnn::Mat& video_output,
                              ncnn::Mat& text_output) const
{
    if (!frontend_loaded || !tail_loaded || blocks.size() != 32)
        return -1;

    // Flattened token streams cannot broadcast a [batch, dim] AdaSingle value
    // without mixing samples. Dispatch each validated sample through the shared
    // loaded Vulkan networks, then restore the flattened batch order.
    std::vector<DiTInputs> samples;
    int ret = split_dit_inputs_by_sample(inputs, samples);
    if (ret != 0)
        return ret;
    if (samples.size() > 1)
    {
        std::vector<ncnn::Mat> video_samples(samples.size());
        std::vector<ncnn::Mat> text_samples(samples.size());
        for (size_t i = 0; i < samples.size(); i++)
        {
            ret = forward(samples[i], video_samples[i], text_samples[i]);
            if (ret != 0)
                return ret;
        }
        ret = append_row_mats(video_samples, video_output);
        if (ret == 0)
            ret = append_row_mats(text_samples, text_output);
        return ret;
    }

    ncnn::Mat patchified_video;
    std::vector<std::array<int, 3> > patchified_shapes;
    ret = patchify_video(inputs.video, inputs.video_shapes,
                         patchified_video, patchified_shapes);
    if (ret != 0)
        return ret;
    ncnn::Mat timestep_embedding;
    ret = make_timestep_embedding(inputs.timesteps, timestep_embedding);
    if (ret != 0)
        return ret;
    ncnn::Mat video_shape;
    ncnn::Mat text_length;
    ret = make_shape_mats(patchified_shapes, inputs.text_lengths,
                          video_shape, text_length);
    if (ret != 0)
        return ret;

    ncnn::VulkanDevice* device = ncnn::get_gpu_device(gpu_id);
    ncnn::VkAllocator* blob_allocator = device->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = device->acquire_staging_allocator();
    ncnn::Option option;
    option.use_vulkan_compute = true;
    option.use_packing_layout = false;
    option.use_fp16_storage = false;
    option.use_fp16_packed = false;
    option.use_bf16_storage = bf16_storage;
    option.use_bf16_packed = false;
    option.use_fp16_arithmetic = false;
    option.blob_vkallocator = blob_allocator;
    option.workspace_vkallocator = blob_allocator;
    option.staging_vkallocator = staging_allocator;
    const char* checkpoint_path = std::getenv("SEEDVR2_DIT_DUMP_BLOCK");
    const char* checkpoint_index_value =
        std::getenv("SEEDVR2_DIT_DUMP_BLOCK_INDEX");
    std::vector<int> checkpoint_blocks;
    std::vector<std::string> checkpoint_paths;
    std::vector<std::string> checkpoint_text_paths;
    if (checkpoint_path && checkpoint_index_value)
    {
        checkpoint_blocks.push_back(std::atoi(checkpoint_index_value));
        checkpoint_paths.push_back(checkpoint_path);
        checkpoint_text_paths.push_back(std::string(checkpoint_path) + ".text.f32");
    }
    const char* checkpoint_list = std::getenv("SEEDVR2_DIT_DUMP_BLOCKS");
    const char* checkpoint_directory = std::getenv("SEEDVR2_DIT_DUMP_DIR");
    if (checkpoint_list && checkpoint_directory)
    {
        const std::vector<int> parsed = parse_checkpoint_blocks(checkpoint_list);
        if (parsed.empty())
            return -1;
        for (size_t i = 0; i < parsed.size(); i++)
        {
            checkpoint_blocks.push_back(parsed[i]);
            checkpoint_paths.push_back(std::string(checkpoint_directory) +
                                       "/ncnn_block" + std::to_string(parsed[i]) +
                                       ".f32");
            checkpoint_text_paths.push_back(std::string(checkpoint_directory) +
                                            "/ncnn_block" + std::to_string(parsed[i]) +
                                            "_text.f32");
        }
    }
    std::vector<ncnn::Mat> checkpoint_videos(checkpoint_blocks.size());
    std::vector<ncnn::Mat> checkpoint_texts(checkpoint_blocks.size());
    std::array<ncnn::Mat, 6> checkpoint_modulations;

    {
        ncnn::VkCompute command(device);
        ncnn::VkMat patchified_video_gpu;
        ncnn::VkMat text_embedding_gpu;
        ncnn::VkMat timestep_embedding_gpu;
        ncnn::VkMat video_shape_gpu;
        ncnn::VkMat text_length_gpu;
        ret = upload_pack1(patchified_video, patchified_video_gpu, device,
                           command, option);
        if (ret == 0)
            ret = upload_pack1(inputs.text, text_embedding_gpu, device,
                               command, option);
        if (ret == 0)
            ret = upload_pack1(timestep_embedding, timestep_embedding_gpu,
                               device, command, option);
        if (ret == 0)
            ret = upload_pack1(video_shape, video_shape_gpu, device,
                               command, option);
        if (ret == 0)
            ret = upload_pack1(text_length, text_length_gpu, device,
                               command, option);

        ncnn::VkMat video;
        ncnn::VkMat text;
        std::array<ncnn::VkMat, 6> modulation;
        if (ret == 0)
        {
            ncnn::Extractor extractor = frontend.create_extractor();
            extractor.set_blob_vkallocator(blob_allocator);
            extractor.set_workspace_vkallocator(blob_allocator);
            extractor.set_staging_vkallocator(staging_allocator);
            ret = extractor.input("in0", patchified_video_gpu);
            if (ret == 0) ret = extractor.input("in1", text_embedding_gpu);
            if (ret == 0) ret = extractor.input("in2", timestep_embedding_gpu);

            ncnn::VkMat packed_video;
            ncnn::VkMat packed_text;
            std::array<ncnn::VkMat, 6> packed_modulation;
            if (ret == 0) ret = extractor.extract("out0", packed_video, command);
            if (ret == 0) ret = extractor.extract("out1", packed_text, command);
            for (int i = 0; ret == 0 && i < 6; i++)
            {
                const std::string name = "out" + std::to_string(i + 2);
                ret = extractor.extract(name.c_str(), packed_modulation[i], command);
            }
            if (ret == 0)
                ret = convert_pack1(packed_video, video, device, command, option);
            if (ret == 0)
                ret = convert_pack1(packed_text, text, device, command, option);
            for (int i = 0; ret == 0 && i < 6; i++)
                ret = convert_pack1(packed_modulation[i], modulation[i], device,
                                    command, option);
        }

        if (ret == 0)
            ret = record_blocks(patchified_shapes, inputs.text_lengths,
                                video, text, video_shape_gpu, text_length_gpu,
                                modulation, command, blob_allocator,
                                staging_allocator, checkpoint_blocks,
                                checkpoint_videos, checkpoint_texts);
        ncnn::VkMat patchified_prediction;
        if (ret == 0)
        {
            ncnn::Extractor extractor = tail.create_extractor();
            extractor.set_blob_vkallocator(blob_allocator);
            extractor.set_workspace_vkallocator(blob_allocator);
            extractor.set_staging_vkallocator(staging_allocator);
            ret = extractor.input("in0", video);
            if (ret == 0) ret = extractor.input("in1", modulation[0]);
            if (ret == 0) ret = extractor.input("in2", modulation[1]);
            if (ret == 0)
                ret = extractor.extract("out0", patchified_prediction, command);
        }
        ncnn::Mat patchified_prediction_cpu;
        if (ret == 0)
        {
            command.record_download(patchified_prediction,
                                    patchified_prediction_cpu, option);
            command.record_download(text, text_output, option);
            if (checkpoint_list && checkpoint_directory)
            {
                for (int i = 0; i < 6; i++)
                    command.record_download(modulation[i], checkpoint_modulations[i], option);
            }
            ret = command.submit_and_wait();
        }
        if (ret == 0)
            ret = unpatchify_prediction(patchified_prediction_cpu,
                                        inputs.video_shapes, patchified_shapes,
                                        video_output);
        for (size_t i = 0; ret == 0 && i < checkpoint_blocks.size(); i++)
        {
            if (write_float_mat(checkpoint_paths[i].c_str(), checkpoint_videos[i]) != 0)
                ret = -1;
            if (ret == 0 &&
                write_float_mat(checkpoint_text_paths[i].c_str(), checkpoint_texts[i]) != 0)
                ret = -1;
        }
        if (ret == 0 && checkpoint_list && checkpoint_directory)
        {
            for (int i = 0; ret == 0 && i < 6; i++)
            {
                const std::string path = std::string(checkpoint_directory) +
                                         "/ncnn_modulation" + std::to_string(i) +
                                         ".f32";
                if (write_float_mat(path.c_str(), checkpoint_modulations[i]) != 0)
                    ret = -1;
            }
        }
    }

    device->reclaim_blob_allocator(blob_allocator);
    device->reclaim_staging_allocator(staging_allocator);
    return ret;
}

} // namespace seedvr2
