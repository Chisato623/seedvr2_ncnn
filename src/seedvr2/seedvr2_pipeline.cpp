#include "seedvr2_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>

namespace seedvr2 {

namespace {

bool is_latent(const ncnn::Mat& value)
{
    return value.dims == 4 && value.w > 0 && value.h > 0 &&
           value.w % 2 == 0 && value.h % 2 == 0 &&
           value.d > 0 && value.c == 16 && value.n == 1 &&
           value.elempack == 1 && value.elemsize == sizeof(float);
}

bool is_text(const ncnn::Mat& value)
{
    return value.dims == 2 && value.w == 5120 && value.h > 0 &&
           value.elempack == 1 && value.elemsize == sizeof(float);
}

bool is_video(const ncnn::Mat& value)
{
    return value.dims == 4 && value.w > 0 && value.h > 0 &&
           value.w % 16 == 0 && value.h % 16 == 0 && value.d > 0 &&
           value.c == 3 && value.n == 1 && value.elempack == 1 &&
           value.elemsize == sizeof(float);
}

bool all_finite(const ncnn::Mat& value)
{
    if (value.empty() || value.elemsize != sizeof(float) ||
        value.elempack != 1)
        return false;
    const float* values = value;
    for (size_t index = 0; index < value.total() * value.n; index++)
    {
        if (!std::isfinite(values[index]))
            return false;
    }
    return true;
}

int copy_video_frames(const ncnn::Mat& source, int output_frames,
                      bool repeat_last, ncnn::Mat& destination)
{
    if (!is_video(source) || output_frames <= 0 ||
        (!repeat_last && output_frames > source.d))
        return -1;
    destination.create(source.w, source.h, output_frames, source.c,
                       source.elemsize, source.elempack, 1);
    if (destination.empty())
        return -100;
    const size_t frame_bytes = static_cast<size_t>(source.w) * source.h *
                               source.elemsize;
    for (int channel = 0; channel < source.c; channel++)
    {
        for (int frame = 0; frame < output_frames; frame++)
        {
            const int source_frame =
                repeat_last ? std::min(frame, source.d - 1) : frame;
            const ncnn::Mat input =
                source.channel(channel).depth(source_frame);
            ncnn::Mat output = destination.channel(channel).depth(frame);
            std::memcpy(output.data, input.data, frame_bytes);
        }
    }
    return 0;
}

int pad_video_frames(const ncnn::Mat& source, ncnn::Mat& destination)
{
    if (!is_video(source))
        return -1;
    const int padded_frames = source.d == 1
                                  ? 1
                                  : ((source.d - 1 + 3) / 4) * 4 + 1;
    if (padded_frames == source.d)
    {
        destination = source;
        return 0;
    }
    return copy_video_frames(source, padded_frames, true, destination);
}

void apply_cfg(const ncnn::Mat& positive, const ncnn::Mat& negative,
               float scale, float rescale, ncnn::Mat& output)
{
    output.create(positive.w, positive.h);
    const size_t count = positive.total();
    const float* pos = positive;
    const float* neg = negative;
    float* cfg = output;
    for (size_t index = 0; index < count; index++)
        cfg[index] = neg[index] + scale * (pos[index] - neg[index]);

    if (rescale == 0.f || count < 2)
        return;
    double pos_mean = 0.0;
    double cfg_mean = 0.0;
    for (size_t index = 0; index < count; index++)
    {
        pos_mean += pos[index];
        cfg_mean += cfg[index];
    }
    pos_mean /= count;
    cfg_mean /= count;
    double pos_variance = 0.0;
    double cfg_variance = 0.0;
    for (size_t index = 0; index < count; index++)
    {
        const double p = pos[index] - pos_mean;
        const double c = cfg[index] - cfg_mean;
        pos_variance += p * p;
        cfg_variance += c * c;
    }
    const double pos_std = std::sqrt(pos_variance / (count - 1));
    const double cfg_std = std::sqrt(cfg_variance / (count - 1));
    if (cfg_std == 0.0)
        return;
    const float factor = static_cast<float>(
        rescale * pos_std / cfg_std + (1.f - rescale));
    for (size_t index = 0; index < count; index++)
        cfg[index] *= factor;
}

} // namespace

void SeedVR2Pipeline::clear()
{
    vae.clear();
    dit.clear();
    if (vae_stage_active)
        ncnn::destroy_gpu_instance();
    vae_stage_active = false;
    dit_directory.clear();
    vae_directory.clear();
    gpu_id = -1;
    staged = false;
    loaded = false;
}

int SeedVR2Pipeline::load(const std::string& dit_directory,
                          const std::string& vae_directory, int gpu_id)
{
    clear();
    int ret = dit.load(dit_directory, gpu_id);
    if (ret == 0)
        ret = vae.load(vae_directory, gpu_id);
    if (ret != 0)
    {
        clear();
        return ret;
    }
    this->dit_directory = dit_directory;
    this->vae_directory = vae_directory;
    this->gpu_id = gpu_id;
    loaded = true;
    return 0;
}

int SeedVR2Pipeline::load_staged(const std::string& new_dit_directory,
                                 const std::string& new_vae_directory,
                                 int new_gpu_id)
{
    clear();
    ncnn::create_gpu_instance();
    const bool valid_gpu = new_gpu_id >= 0 &&
                           new_gpu_id < ncnn::get_gpu_count();
    ncnn::destroy_gpu_instance();
    if (!valid_gpu || new_dit_directory.empty() || new_vae_directory.empty())
        return -1;
    dit_directory = new_dit_directory;
    vae_directory = new_vae_directory;
    gpu_id = new_gpu_id;
    staged = true;
    loaded = true;
    return 0;
}

int SeedVR2Pipeline::load_vae_stage()
{
    ncnn::create_gpu_instance();
    vae_stage_active = true;
    const int ret = vae.load(vae_directory, gpu_id);
    if (ret != 0)
        clear_vae_stage();
    return ret;
}

void SeedVR2Pipeline::clear_vae_stage()
{
    vae.clear();
    if (vae_stage_active)
        ncnn::destroy_gpu_instance();
    vae_stage_active = false;
}

int SeedVR2Pipeline::predict(const ncnn::Mat& latent,
                             const ncnn::Mat& condition,
                             const ncnn::Mat& text, float timestep,
                             ncnn::Mat& prediction)
{
    if (!is_latent(latent) || !is_latent(condition) || !is_text(text) ||
        latent.w != condition.w || latent.h != condition.h ||
        latent.d != condition.d)
        return -1;
    const int frames = latent.d;
    const int height = latent.h;
    const int width = latent.w;
    ncnn::Mat video(33, frames * height * width);
    if (video.empty())
        return -100;
    for (int frame = 0; frame < frames; frame++)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                const int token = (frame * height + y) * width + x;
                float* row = video.row(token);
                for (int channel = 0; channel < 16; channel++)
                {
                    const int offset = (frame * height + y) * width + x;
                    row[channel] = static_cast<const float*>(
                        latent.channel(channel))[offset];
                    row[channel + 16] = static_cast<const float*>(
                        condition.channel(channel))[offset];
                }
                row[32] = 1.f;
            }
        }
    }

    DiTInputs inputs;
    inputs.video = video;
    inputs.text = text;
    inputs.video_shapes.push_back({frames, height, width});
    inputs.text_lengths.push_back(text.h);
    inputs.timesteps.push_back(timestep);
    ncnn::Mat text_output;
    const int ret = dit.forward(inputs, prediction, text_output);
    return ret;
}

int SeedVR2Pipeline::restore(const ncnn::Mat& input_video,
                             const ncnn::Mat& positive_text,
                             const ncnn::Mat& negative_text,
                             ncnn::Mat& output_video,
                             const PipelineOptions& options)
{
    if (!loaded || options.steps <= 0 || !is_video(input_video) ||
        !is_text(positive_text) ||
        (options.cfg_scale != 1.f && !is_text(negative_text)))
        return -1;

    ncnn::Mat padded_video;
    int ret = pad_video_frames(input_video, padded_video);
    if (ret != 0)
        return ret;

    ncnn::Mat condition;
    if (staged)
    {
        ret = load_vae_stage();
        if (ret != 0)
        {
            std::fprintf(stderr,
                         "SeedVR2 staged VAE encode load failed (%d)\n",
                         ret);
            return ret;
        }
    }

    ret = vae.encode(padded_video, condition);
    if (staged)
        clear_vae_stage();
    if (ret != 0)
    {
        std::fprintf(stderr, "SeedVR2 VAE encode failed (%d)\n", ret);
        return ret;
    }
    if (!all_finite(condition))
    {
        std::fprintf(stderr, "SeedVR2 VAE encode produced non-finite values\n");
        return -1;
    }
    ncnn::Mat latent;
    if (!options.initial_noise.empty())
    {
        if (!is_latent(options.initial_noise) ||
            options.initial_noise.w != condition.w ||
            options.initial_noise.h != condition.h ||
            options.initial_noise.d != condition.d)
            return -1;
        latent = options.initial_noise.clone();
    }
    else
    {
        latent.create(condition.w, condition.h, condition.d, condition.c);
        if (!latent.empty())
        {
            std::mt19937_64 generator(options.seed + 1);
            std::normal_distribution<float> normal(0.f, 1.f);
            float* latent_values = latent;
            for (size_t index = 0; index < latent.total(); index++)
                latent_values[index] = normal(generator);
        }
    }
    if (latent.empty())
        return -100;

    if (staged)
    {
        const uint64_t patch_tokens =
            static_cast<uint64_t>(condition.d) * (condition.h / 2) *
            (condition.w / 2);
        // v3 manifests split QKV into separate q/k/v projections, so the
        // largest DiT activation is the SwiGLU gate/up projection
        // [tokens, 3456]. Older manifests use one fused [tokens, 7680] QKV
        // activation. Buffers above the driver limit produce invalid data,
        // so the fallback must key on the model's actual largest tensor.
        uint64_t activation_bytes_per_token = 3u * 2560u * sizeof(float);
        {
            const std::string manifest_path = dit_directory + "/manifest.json";
            std::FILE* manifest_file = std::fopen(manifest_path.c_str(), "rb");
            if (manifest_file)
            {
                char manifest_text[8192];
                const size_t manifest_read =
                    std::fread(manifest_text, 1, sizeof(manifest_text) - 1,
                               manifest_file);
                std::fclose(manifest_file);
                manifest_text[manifest_read] = '\0';
                if (std::strstr(manifest_text, "seedvr2-ncnn-dit-blocks-v3"))
                    activation_bytes_per_token = 3456u * sizeof(float);
            }
        }
        const uint64_t fp32_qkv_bytes =
            patch_tokens * activation_bytes_per_token;
        bool use_bf16_storage =
            fp32_qkv_bytes > std::numeric_limits<uint32_t>::max();
        const char* force_fp32_env = std::getenv("SEEDVR2_DIT_FP32_STORAGE");
        if (force_fp32_env && force_fp32_env[0] == '1' && use_bf16_storage)
        {
            std::fprintf(stderr,
                         "SeedVR2 DiT FP32 storage forced by "
                         "SEEDVR2_DIT_FP32_STORAGE=1\n");
            use_bf16_storage = false;
        }
        if (use_bf16_storage)
        {
            std::fprintf(stderr,
                         "SeedVR2 DiT using BF16 activation storage: "
                         "FP32 QKV buffer would require %llu bytes\n",
                         static_cast<unsigned long long>(fp32_qkv_bytes));
        }
        ret = dit.load(dit_directory, gpu_id, use_bf16_storage);
        if (ret != 0)
        {
            std::fprintf(stderr, "SeedVR2 staged DiT load failed (%d)\n", ret);
            return ret;
        }
    }
    ret = denoise(latent, condition, positive_text, negative_text, options);
    if (staged)
        dit.clear();
    if (ret != 0)
    {
        std::fprintf(stderr, "SeedVR2 DiT denoise failed (%d)\n", ret);
        return ret;
    }
    if (!all_finite(latent))
    {
        std::fprintf(stderr, "SeedVR2 DiT denoise produced non-finite values\n");
        return -1;
    }
    if (staged)
    {
        ret = load_vae_stage();
        if (ret != 0)
        {
            std::fprintf(stderr, "SeedVR2 staged VAE decode load failed (%d)\n", ret);
            return ret;
        }
    }
    ncnn::Mat padded_output;
    ret = vae.decode(latent, padded_output);
    if (staged)
        clear_vae_stage();
    if (ret != 0)
    {
        std::fprintf(stderr, "SeedVR2 VAE decode failed (%d)\n", ret);
        return ret;
    }
    if (!all_finite(padded_output))
    {
        std::fprintf(stderr, "SeedVR2 VAE decode produced non-finite values\n");
        return -1;
    }
    if (padded_output.d == input_video.d)
    {
        output_video = padded_output;
        return 0;
    }
    return copy_video_frames(padded_output, input_video.d, false,
                             output_video);
}

int SeedVR2Pipeline::denoise(ncnn::Mat& latent,
                             const ncnn::Mat& condition,
                             const ncnn::Mat& positive_text,
                             const ncnn::Mat& negative_text,
                             const PipelineOptions& options)
{
    for (int step = 0; step < options.steps; step++)
    {
        const float timestep = 1000.f * (options.steps - step) / options.steps;
        const float next_timestep =
            1000.f * (options.steps - step - 1) / options.steps;
        ncnn::Mat positive_prediction;
        int ret = predict(latent, condition, positive_text, timestep,
                          positive_prediction);
        if (ret != 0)
            return ret;

        ncnn::Mat prediction = positive_prediction;
        ncnn::Mat guided;
        if (options.cfg_scale != 1.f)
        {
            ncnn::Mat negative_prediction;
            ret = predict(latent, condition, negative_text, timestep,
                          negative_prediction);
            if (ret != 0)
                return ret;
            apply_cfg(positive_prediction, negative_prediction,
                      options.cfg_scale, options.cfg_rescale, guided);
            prediction = guided;
        }
        if (prediction.dims != 2 || prediction.w != 16 ||
            prediction.h != latent.d * latent.h * latent.w)
            return -1;

        const float delta = (next_timestep - timestep) / 1000.f;
        for (int frame = 0; frame < latent.d; frame++)
        {
            for (int y = 0; y < latent.h; y++)
            {
                for (int x = 0; x < latent.w; x++)
                {
                    const int token = (frame * latent.h + y) * latent.w + x;
                    const float* row = prediction.row(token);
                    const int offset = token;
                    for (int channel = 0; channel < 16; channel++)
                        static_cast<float*>(latent.channel(channel))[offset] +=
                            delta * row[channel];
                }
            }
        }
    }
    return 0;
}

} // namespace seedvr2
