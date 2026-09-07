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

int read_mat(const std::string& path, int width, int height, ncnn::Mat& value)
{
    FILE* file = std::fopen(path.c_str(), "rb");
    if (!file)
        return -1;
    value.create(width, height);
    const size_t expected = static_cast<size_t>(width) * height;
    const size_t count = std::fread(value.data, sizeof(float), expected, file);
    const int trailing = std::fgetc(file);
    std::fclose(file);
    return count == expected && trailing == EOF ? 0 : -1;
}

int write_mat(const std::string& path, const ncnn::Mat& value)
{
    FILE* file = std::fopen(path.c_str(), "wb");
    if (!file)
        return -1;
    const size_t count = std::fwrite(value.data, value.elemsize, value.total(), file);
    std::fclose(file);
    return count == value.total() ? 0 : -1;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr,
                     "usage: %s MODEL_DIRECTORY [--gpu ID] [--smoke]\n"
                     "       %s MODEL_DIRECTORY --compare INPUT_DIR OUTPUT_DIR\n",
                     argv[0], argv[0]);
        return 2;
    }

    int gpu_id = 0;
    bool smoke = false;
    bool compare = false;
    std::string compare_input_dir;
    std::string compare_output_dir;
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
        else if (std::strcmp(argv[i], "--compare") == 0 && i + 2 < argc)
        {
            compare = true;
            compare_input_dir = argv[++i];
            compare_output_dir = argv[++i];
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
    if (!smoke && !compare)
        return 0;

    seedvr2::DiTInputs inputs;
    if (compare)
    {
        if (read_mat(compare_input_dir + "/video.bin", 33, 32, inputs.video) != 0 ||
            read_mat(compare_input_dir + "/text.bin", 5120, 4, inputs.text) != 0)
            return 3;
    }
    else
    {
        inputs.video.create(33, 32);
        inputs.text.create(5120, 4);
        if (inputs.video.empty() || inputs.text.empty())
            return -100;
        fill_deterministic(inputs.video, 0.0011f, false);
        fill_deterministic(inputs.text, 0.0017f, true);
    }
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
    if (compare)
    {
        if (write_mat(compare_output_dir + "/ncnn_out0.bin", prediction) != 0 ||
            write_mat(compare_output_dir + "/ncnn_out1.bin", text) != 0)
            return 4;
    }
    return 0;
}
