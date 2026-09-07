#include "seedvr2_vae.h"

#include "decomposed_conv3d.h"
#include "vae_downsample3d.h"
#include "vae_chunked_mha.h"
#include "vae_upsample3d.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace seedvr2 {

namespace {

bool environment_enabled(const char* name)
{
    const char* value = std::getenv(name);
    return value && value[0] && std::strcmp(value, "0") != 0;
}

int environment_index(const char* name)
{
    const char* value = std::getenv(name);
    return value && value[0] ? std::atoi(value) : -1;
}

const char* memory_state_name(MemoryState state)
{
    switch (state)
    {
    case MemoryState::DISABLED: return "disabled";
    case MemoryState::INITIALIZING: return "initializing";
    case MemoryState::ACTIVE: return "active";
    }
    return "unknown";
}

bool should_trace_blob(int layer_index, int blob_index)
{
    if (!environment_enabled("SEEDVR2_VAE_TRACE_BLOBS"))
        return false;
    const int requested_layer = environment_index("SEEDVR2_VAE_TRACE_LAYER");
    const int requested_blob = environment_index("SEEDVR2_VAE_TRACE_BLOB");
    return (requested_layer < 0 || requested_layer == layer_index) &&
           (requested_blob < 0 || requested_blob == blob_index);
}

int dump_layer_index()
{
    return environment_index("SEEDVR2_VAE_DUMP_LAYER");
}

bool dump_layer_requested(int layer_index)
{
    const char* list = std::getenv("SEEDVR2_VAE_DUMP_LAYERS");
    if (list && list[0])
    {
        const char* cursor = list;
        while (*cursor)
        {
            char* end = 0;
            const long value = std::strtol(cursor, &end, 10);
            if (end != cursor && value == layer_index)
                return true;
            if (!end || end == cursor)
                break;
            cursor = *end == ',' ? end + 1 : end;
        }
    }
    return dump_layer_index() == layer_index;
}

int dump_mat_f32(const char* path, const ncnn::Mat& value)
{
    if (!path || !path[0] || value.empty() || value.elempack != 1 ||
        value.elemsize != sizeof(float))
        return -1;

    std::FILE* file = std::fopen(path, "wb");
    if (!file)
        return -1;
    size_t written = 0;
    bool valid = true;
    if (value.dims == 4 && value.n == 1)
    {
        // Reshape views can retain a source cstep.  Serialize logical rows
        // instead of assuming that value.data is a contiguous CTHW array.
        for (int channel = 0; channel < value.c && valid; channel++)
        {
            for (int depth = 0; depth < value.d && valid; depth++)
            {
                const ncnn::Mat plane = value.channel(channel).depth(depth);
                for (int row = 0; row < value.h && valid; row++)
                {
                    const size_t count = static_cast<size_t>(value.w);
                    const size_t wrote = std::fwrite(
                        plane.row(row), sizeof(float), count, file);
                    written += wrote;
                    valid = wrote == count;
                }
            }
        }
    }
    else
    {
        const size_t count = value.total() * value.n;
        written = std::fwrite(value.data, sizeof(float), count, file);
        valid = written == count;
    }
    const int close_ret = std::fclose(file);
    return valid && close_ret == 0 ? 0 : -1;
}

bool release_completed_temporal_memory(const ncnn::Layer* layer)
{
    if (const DecomposedConv3D* convolution =
            dynamic_cast<const DecomposedConv3D*>(layer))
    {
        convolution->release_completed_temporal_memory();
        return true;
    }
    if (const VAEDownsample3D* downsample =
            dynamic_cast<const VAEDownsample3D*>(layer))
    {
        downsample->release_completed_temporal_memory();
        return true;
    }
    if (const VAEUpsample3D* upsample =
            dynamic_cast<const VAEUpsample3D*>(layer))
    {
        upsample->release_completed_temporal_memory();
        return true;
    }
    return false;
}

void finish_temporal_memory(const ncnn::Net& net,
                            ncnn::VkAllocator* memory_vkallocator)
{
    const std::vector<ncnn::Layer*>& layers = net.layers();
    for (const ncnn::Layer* layer : layers)
    {
        if (const DecomposedConv3D* convolution =
                dynamic_cast<const DecomposedConv3D*>(layer))
            convolution->reset_temporal_memory();
        else if (const VAEDownsample3D* downsample =
                     dynamic_cast<const VAEDownsample3D*>(layer))
            downsample->reset_temporal_memory();
        else if (const VAEUpsample3D* upsample =
                     dynamic_cast<const VAEUpsample3D*>(layer))
            upsample->reset_temporal_memory();
    }
    if (memory_vkallocator)
        static_cast<ncnn::VkBlobAllocator*>(memory_vkallocator)->trim();
    set_vae_memory_state(MemoryState::DISABLED);
    set_vae_memory_vkallocator(0);
}

void configure_net(ncnn::Net& net, int gpu_id)
{
    const bool fp16_storage =
        environment_enabled("SEEDVR2_VAE_FP16_STORAGE");
    net.opt.use_vulkan_compute = true;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_storage = fp16_storage;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_arithmetic = false;
    // Winograd expands production-resolution 3x3 activations by 16x/36x and
    // can request a single workspace larger than an 80 GiB device. SGEMM's
    // im2col indexing also exceeds 32-bit ranges at multi-megapixel 512-channel
    // shapes, producing NaNs. Use the exact direct Vulkan convolution path.
    net.opt.use_winograd_convolution = false;
    net.opt.use_sgemm_convolution = false;
    net.set_vulkan_device(gpu_id);
    net.register_custom_layer("DecomposedConv3d",
                              decomposed_conv3d_layer_creator);
    net.register_custom_layer("ExportUpsample3D",
                              vae_upsample3d_layer_creator);
    net.register_custom_layer("ExportDownsample3D",
                              vae_downsample3d_layer_creator);
    net.register_custom_layer("SeedVR2ChunkedMHA",
                              vae_chunked_mha_layer_creator);
}

bool is_video_shape(const ncnn::Mat& value)
{
    return value.dims == 4 && value.w > 0 && value.h > 0 &&
           value.w % 8 == 0 && value.h % 8 == 0 && value.d > 0 &&
           value.c == 3 && value.n == 1 &&
           value.elempack == 1 && value.elemsize == sizeof(float);
}

bool is_latent_shape(const ncnn::Mat& value, int channels)
{
    return value.dims == 4 && value.w > 0 && value.h > 0 &&
           value.d > 0 && value.c == channels && value.n == 1 &&
           value.elempack == 1 && value.elemsize == sizeof(float);
}

int temporal_slice(const ncnn::Mat& input, int start, int count,
                   ncnn::Mat& output)
{
    if (start < 0 || count <= 0 || start + count > input.d)
        return -1;
    output.create(input.w, input.h, count, input.c, input.elemsize,
                  input.elempack, 1);
    if (output.empty())
        return -100;
    const size_t frame_bytes = static_cast<size_t>(input.w) * input.h *
                               input.elemsize;
    for (int channel = 0; channel < input.c; channel++)
    {
        for (int frame = 0; frame < count; frame++)
        {
            const ncnn::Mat source =
                input.channel(channel).depth(start + frame);
            ncnn::Mat destination = output.channel(channel).depth(frame);
            std::memcpy(destination.data, source.data, frame_bytes);
        }
    }
    return 0;
}

int concatenate_temporal(const std::vector<ncnn::Mat>& segments,
                         ncnn::Mat& output)
{
    if (segments.empty())
        return -1;
    const ncnn::Mat& first = segments.front();
    int total_frames = 0;
    for (const ncnn::Mat& segment : segments)
    {
        if (segment.dims != 4 || segment.w != first.w ||
            segment.h != first.h || segment.c != first.c ||
            segment.n != 1 || segment.elempack != first.elempack ||
            segment.elemsize != first.elemsize)
            return -1;
        total_frames += segment.d;
    }
    output.create(first.w, first.h, total_frames, first.c, first.elemsize,
                  first.elempack, 1);
    if (output.empty())
        return -100;
    const size_t frame_bytes = static_cast<size_t>(first.w) * first.h *
                               first.elemsize;
    int output_frame = 0;
    for (const ncnn::Mat& segment : segments)
    {
        for (int channel = 0; channel < first.c; channel++)
        {
            for (int frame = 0; frame < segment.d; frame++)
            {
                const ncnn::Mat source = segment.channel(channel).depth(frame);
                ncnn::Mat destination =
                    output.channel(channel).depth(output_frame + frame);
                std::memcpy(destination.data, source.data, frame_bytes);
            }
        }
        output_frame += segment.d;
    }
    return 0;
}

int extract_segment(const ncnn::Net& net, const ncnn::Mat& input,
                    MemoryState memory_state,
                    ncnn::VkAllocator* memory_vkallocator,
                    int segment_index, int segment_start,
                    ncnn::Mat& output)
{
    set_vae_memory_state(memory_state);
    set_vae_memory_vkallocator(memory_vkallocator);
    const ncnn::VulkanDevice* device = net.vulkan_device();
    if (!device)
    {
        std::fprintf(stderr, "SeedVR2 VAE failure segment=%d stage=device\n",
                     segment_index);
        return -1;
    }
    ncnn::VkAllocator* blob_allocator = device->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = device->acquire_staging_allocator();
    if (!blob_allocator || !staging_allocator)
    {
        if (blob_allocator)
            device->reclaim_blob_allocator(blob_allocator);
        if (staging_allocator)
            device->reclaim_staging_allocator(staging_allocator);
        set_vae_memory_vkallocator(0);
        std::fprintf(stderr,
                     "SeedVR2 VAE failure segment=%d stage=acquire_allocator\n",
                     segment_index);
        return -100;
    }
    // VulkanDevice always creates VkBlobAllocator instances for this pool.
    // trim() returns completely unused blocks to Vulkan after a synchronized
    // checkpoint instead of retaining multi-gigabyte decoder activations.
    ncnn::VkBlobAllocator* graph_allocator =
        static_cast<ncnn::VkBlobAllocator*>(blob_allocator);
    int ret = 0;
    {
        ncnn::Extractor extractor = net.create_extractor();
        extractor.set_light_mode(true);
        extractor.set_blob_vkallocator(blob_allocator);
        extractor.set_workspace_vkallocator(blob_allocator);
        extractor.set_staging_vkallocator(staging_allocator);
        ret = extractor.input("in0", input);
        if (ret != 0)
            std::fprintf(stderr,
                         "SeedVR2 VAE failure segment=%d stage=input ret=%d\n",
                         segment_index, ret);
        const char* check_layer_value = std::getenv("SEEDVR2_VAE_CHECK_LAYER");
        const int check_layer = check_layer_value
                                    ? std::atoi(check_layer_value) : -1;
        const char* dump_path = std::getenv("SEEDVR2_VAE_DUMP_PATH");
        const std::vector<ncnn::Layer*>& layers = net.layers();
        for (size_t layer_index = 0; layer_index < layers.size(); layer_index++)
        {
            const ncnn::Layer* layer = layers[layer_index];
            if (ret != 0)
                break;
            for (int blob_index : layer->tops)
            {
                {
                    ncnn::VkCompute command(device);
                    ncnn::VkMat checkpoint;
                    ret = extractor.extract(blob_index, checkpoint, command);
                    if (ret == 0)
                        ret = command.submit_and_wait();
                    if (ret != 0)
                        std::fprintf(stderr,
                                     "SeedVR2 VAE failure segment=%d "
                                     "layer=%zu name=%s blob=%d "
                                     "stage=checkpoint ret=%d\n",
                                     segment_index, layer_index,
                                     layer->name.c_str(), blob_index, ret);
                    const bool inspect_layer =
                        static_cast<int>(layer_index) == check_layer ||
                        dump_layer_requested(static_cast<int>(layer_index));
                    if (ret == 0 && inspect_layer)
                    {
                        ncnn::Mat checked;
                        ret = extractor.extract(blob_index, checked);
                        bool finite = ret == 0 && !checked.empty();
                        const float* values = checked;
                        for (size_t index = 0;
                             finite && index < checked.total() * checked.n;
                             index++)
                            finite = std::isfinite(values[index]);
                        if (!finite)
                        {
                            std::fprintf(stderr,
                                         "SeedVR2 VAE failure segment=%d "
                                         "layer=%zu name=%s blob=%d "
                                         "stage=nonfinite\n",
                                         segment_index, layer_index,
                                         layer->name.c_str(), blob_index);
                            ret = -1;
                        }
                        if (ret == 0 && dump_layer_requested(static_cast<int>(layer_index)) &&
                            dump_path && dump_path[0])
                        {
                            std::string resolved_path = dump_path;
                            if (std::getenv("SEEDVR2_VAE_DUMP_LAYERS"))
                            {
                                resolved_path += ".layer" +
                                                 std::to_string(layer_index) +
                                                 ".blob" +
                                                 std::to_string(blob_index) +
                                                 ".f32";
                            }
                            const int dump_ret =
                                dump_mat_f32(resolved_path.c_str(), checked);
                            if (dump_ret != 0)
                                ret = dump_ret;
                        }
                    }
                }
                // submit_and_wait() has completed and both command/checkpoint
                // are gone, so blocks returned by light-mode execution are no
                // longer referenced by queued Vulkan work.
                if (release_completed_temporal_memory(layer) &&
                    memory_vkallocator)
                    static_cast<ncnn::VkBlobAllocator*>(memory_vkallocator)
                        ->trim();
                graph_allocator->trim();
                if (ret != 0)
                    break;
            }
        }
        if (ret == 0)
            ret = extractor.extract("out0", output);
        if (ret != 0)
            std::fprintf(stderr,
                         "SeedVR2 VAE failure segment=%d stage=output ret=%d\n",
                         segment_index, ret);
    }
    graph_allocator->trim();
    device->reclaim_blob_allocator(blob_allocator);
    device->reclaim_staging_allocator(staging_allocator);
    set_vae_memory_vkallocator(0);
    return ret;
}

int extract_temporal_slices(const ncnn::Net& net, const ncnn::Mat& input,
                            int first_size, int active_size,
                            ncnn::VkAllocator* memory_vkallocator,
                            ncnn::Mat& output)
{
    // A prior failed run may not have reached every stateful layer again.
    finish_temporal_memory(net, memory_vkallocator);
    if (input.d <= first_size)
    {
        const int ret = extract_segment(
            net, input, MemoryState::DISABLED, memory_vkallocator, 0, 0,
            output);
        finish_temporal_memory(net, memory_vkallocator);
        return ret;
    }

    std::vector<ncnn::Mat> outputs;
    int start = 0;
    int segment_index = 0;
    MemoryState state = MemoryState::INITIALIZING;
    while (start < input.d)
    {
        const int limit = start == 0 ? first_size : active_size;
        const int count = std::min(limit, input.d - start);
        ncnn::Mat segment;
        int ret = temporal_slice(input, start, count, segment);
        ncnn::Mat segment_output;
        if (ret == 0)
            ret = extract_segment(net, segment, state, memory_vkallocator,
                                  segment_index, start, segment_output);
        if (ret != 0)
        {
            finish_temporal_memory(net, memory_vkallocator);
            return ret;
        }
        outputs.push_back(segment_output);
        start += count;
        segment_index++;
        state = MemoryState::ACTIVE;
    }
    const int ret = concatenate_temporal(outputs, output);
    finish_temporal_memory(net, memory_vkallocator);
    return ret;
}

int moments_to_latent(const ncnn::Mat& moments, ncnn::Mat& latent,
                      const VAEEncodeOptions& options)
{
    if (!is_latent_shape(moments, 32))
        return -1;
    latent.create(moments.w, moments.h, moments.d, 16);
    if (latent.empty())
        return -100;
    std::mt19937_64 generator(options.seed);
    std::normal_distribution<float> normal(0.f, 1.f);
    const int count = moments.d * moments.h * moments.w;
    for (int channel = 0; channel < 16; channel++)
    {
        const float* mean = moments.channel(channel);
        const float* logvar = moments.channel(channel + 16);
        float* output = latent.channel(channel);
        for (int index = 0; index < count; index++)
        {
            float value = mean[index];
            if (options.sample_posterior)
            {
                const float clamped =
                    std::max(-30.f, std::min(20.f, logvar[index]));
                value += std::exp(0.5f * clamped) * normal(generator);
            }
            output[index] = value * SeedVR2VAE::scaling_factor();
        }
    }
    return 0;
}

} // namespace

SeedVR2VAE::SeedVR2VAE()
    : memory_vkallocator(0), gpu_id(-1), loaded(false)
{
}

SeedVR2VAE::~SeedVR2VAE()
{
    clear();
}

void SeedVR2VAE::clear()
{
    encoder.clear();
    decoder.clear();
    if (memory_vkallocator && gpu_id >= 0)
        ncnn::get_gpu_device(gpu_id)->reclaim_blob_allocator(
            memory_vkallocator);
    memory_vkallocator = 0;
    gpu_id = -1;
    loaded = false;
}

int SeedVR2VAE::load(const std::string& model_directory, int new_gpu_id)
{
    clear();
    if (new_gpu_id < 0 || new_gpu_id >= ncnn::get_gpu_count())
        return -1;
    gpu_id = new_gpu_id;
    memory_vkallocator =
        ncnn::get_gpu_device(gpu_id)->acquire_blob_allocator();
    if (!memory_vkallocator)
    {
        clear();
        return -100;
    }
    configure_net(encoder, gpu_id);
    configure_net(decoder, gpu_id);

    const std::string encoder_stem =
        model_directory + "/seedvr2_vae_encoder.ncnn";
    int ret = encoder.load_param((encoder_stem + ".param").c_str());
    if (ret == 0)
        ret = encoder.load_model((encoder_stem + ".bin").c_str());
    if (ret != 0)
    {
        std::fprintf(stderr, "failed to load SeedVR2 VAE encoder (%d)\n", ret);
        clear();
        return ret;
    }

    const std::string decoder_stem =
        model_directory + "/seedvr2_vae_decoder.ncnn";
    ret = decoder.load_param((decoder_stem + ".param").c_str());
    if (ret == 0)
        ret = decoder.load_model((decoder_stem + ".bin").c_str());
    if (ret != 0)
    {
        std::fprintf(stderr, "failed to load SeedVR2 VAE decoder (%d)\n", ret);
        clear();
        return ret;
    }
    loaded = true;
    return 0;
}

int SeedVR2VAE::encode_moments(const ncnn::Mat& video,
                               ncnn::Mat& moments) const
{
    if (!loaded || !is_video_shape(video) ||
        (video.d > 5 && (video.d - 1) % 4 != 0))
        return -1;
    int ret = extract_temporal_slices(encoder, video, 5, 4,
                                      memory_vkallocator, moments);
    const int expected_frames = (video.d - 1) / 4 + 1;
    if (ret == 0 && (!is_latent_shape(moments, 32) ||
                     moments.w != video.w / 8 || moments.h != video.h / 8 ||
                     moments.d != expected_frames))
        ret = -1;
    return ret;
}

int SeedVR2VAE::encode(const ncnn::Mat& video, ncnn::Mat& latent,
                       const VAEEncodeOptions& options) const
{
    ncnn::Mat moments;
    int ret = encode_moments(video, moments);
    if (ret != 0)
        return ret;

    return moments_to_latent(moments, latent, options);
}

int SeedVR2VAE::decode(const ncnn::Mat& latent, ncnn::Mat& video) const
{
    if (!loaded || !is_latent_shape(latent, 16))
        return -1;
    ncnn::Mat unscaled = latent.clone();
    if (unscaled.empty())
        return -100;
    float* values = unscaled;
    for (size_t index = 0; index < unscaled.total(); index++)
        values[index] /= scaling_factor();

    const int first_slice_size =
        environment_enabled("SEEDVR2_VAE_DECODER_FIRST_ONE") ? 1 : 2;
    int ret = extract_temporal_slices(decoder, unscaled, first_slice_size, 1,
                                      memory_vkallocator, video);
    const int expected_frames = (latent.d - 1) * 4 + 1;
    if (ret == 0 && (!is_video_shape(video) ||
                     video.w != latent.w * 8 || video.h != latent.h * 8 ||
                     video.d != expected_frames))
        ret = -1;
    return ret;
}

} // namespace seedvr2
