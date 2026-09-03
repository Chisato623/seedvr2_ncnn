#include "seedvr2/seedvr2_dit.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

void fill_deterministic(ncnn::Mat& value, float frequency, bool cosine)
{
    float* data = value;
    for (size_t i = 0; i < value.total(); i++)
    {
        const float phase = static_cast<float>(i + 1) * frequency;
        data[i] = cosine ? std::cos(phase) : std::sin(phase);
    }
}

void print_summary(const char* name, const ncnn::Mat& value)
{
    const float* data = value;
    float minimum = data[0];
    float maximum = data[0];
    double sum = 0.0;
    for (size_t i = 0; i < value.total(); i++)
    {
        minimum = std::min(minimum, data[i]);
        maximum = std::max(maximum, data[i]);
        sum += data[i];
    }
    std::printf("%s: shape=(%d,%d), min=%g, max=%g, mean=%g\n",
                name, value.h, value.w, minimum, maximum,
                sum / static_cast<double>(value.total()));
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr,
                     "usage: %s MODEL_DIRECTORY [--gpu ID] [--smoke]\n",
                     argv[0]);
        return 2;
    }

    int gpu_id = 0;
    bool smoke = false;
    for (int i = 2; i < argc; i++)
    {
        if (std::strcmp(argv[i], "--smoke") == 0)
        {
            smoke = true;
        }
        else if (std::strcmp(argv[i], "--gpu") == 0 && i + 1 < argc)
        {
            gpu_id = std::atoi(argv[++i]);
        }
        else
        {
            std::fprintf(stderr, "unknown or incomplete option: %s\n", argv[i]);
            return 2;
        }
    }

    seedvr2::SeedVR2DiTBlocks runtime;
    int ret = runtime.load(argv[1], gpu_id);
    if (ret != 0)
        return ret;

    std::fprintf(stderr, "Loaded SeedVR2 frontend, 32 DiT blocks, and tail on Vulkan GPU %d.\n",
                 gpu_id);
    if (!smoke)
        return 0;

    seedvr2::DiTInputs inputs;
    inputs.video.create(33, 32);
    inputs.text.create(5120, 4);
    if (inputs.video.empty() || inputs.text.empty())
        return -100;
    fill_deterministic(inputs.video, 0.0011f, false);
    fill_deterministic(inputs.text, 0.0017f, true);
    inputs.video_shapes = {{1, 4, 8}};
    inputs.text_lengths = {4};
    inputs.timesteps = {500.f};

    ncnn::Mat prediction;
    ncnn::Mat text;
    const std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();
    ret = runtime.forward(inputs, prediction, text);
    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start;
    if (ret != 0)
    {
        std::fprintf(stderr, "SeedVR2 DiT inference failed (%d).\n", ret);
        return ret;
    }

    print_summary("prediction", prediction);
    print_summary("text", text);
    std::printf("elapsed: %.3f s\n", elapsed.count());
    return 0;
}
