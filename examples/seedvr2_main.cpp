#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "seedvr2/seedvr2_pipeline.h"
#include "seedvr2/seedvr2_video_io.h"

namespace {

namespace fs = std::filesystem;

struct Arguments
{
    std::string dit;
    std::string vae;
    std::string input;
    std::string output;
    std::string positive_text;
    std::string negative_text;
    std::string noise;
    std::string output_f32;
    int gpu = 0;
    int steps = 1;
    int sp_size = 1;
    float cfg_scale = 1.f;
    float cfg_rescale = 0.f;
    uint64_t seed = 666;
    int width = 1280;
    int height = 720;
    double output_fps = 0.0;
    bool low_memory = true;
    bool output_is_directory = false;
};

void usage(const char* program)
{
    std::fprintf(stderr,
        "usage: %s --dit DIR --vae DIR --video-path INPUT "
        "--output-dir OUTPUT --text-pos POS.f32 "
        "[--text-neg NEG.f32] [--gpu 0] [--steps 1] "
        "[--cfg-scale 1] [--cfg-rescale 0] [--seed 666] "
        "[--res-w 1280] [--res-h 720] [--sp-size 1] "
        "[--out-fps FPS] [--noise INITIAL_NOISE.f32] "
        "[--output-f32 RESTORED_RGB.f32] "
        "[--no-low-memory]\n"
        "       legacy aliases: --input, --output, --width, --height\n",
        program);
}

bool parse(int argc, char** argv, Arguments& args)
{
    for (int index = 1; index < argc; index++)
    {
        const std::string key = argv[index];
        if (key == "--low-memory")
        {
            args.low_memory = true;
            continue;
        }
        if (key == "--no-low-memory")
        {
            args.low_memory = false;
            continue;
        }
        if (index + 1 >= argc)
            return false;
        const std::string value = argv[++index];
        if (key == "--dit") args.dit = value;
        else if (key == "--vae") args.vae = value;
        else if (key == "--input" || key == "--video-path") args.input = value;
        else if (key == "--output") args.output = value;
        else if (key == "--output-dir")
        {
            args.output = value;
            args.output_is_directory = true;
        }
        else if (key == "--text-pos") args.positive_text = value;
        else if (key == "--text-neg") args.negative_text = value;
        else if (key == "--noise") args.noise = value;
        else if (key == "--output-f32") args.output_f32 = value;
        else if (key == "--gpu") args.gpu = std::atoi(value.c_str());
        else if (key == "--steps") args.steps = std::atoi(value.c_str());
        else if (key == "--sp-size") args.sp_size = std::atoi(value.c_str());
        else if (key == "--cfg-scale") args.cfg_scale = std::strtof(value.c_str(), nullptr);
        else if (key == "--cfg-rescale") args.cfg_rescale = std::strtof(value.c_str(), nullptr);
        else if (key == "--seed") args.seed = std::strtoull(value.c_str(), nullptr, 10);
        else if (key == "--width" || key == "--res-w") args.width = std::atoi(value.c_str());
        else if (key == "--height" || key == "--res-h") args.height = std::atoi(value.c_str());
        else if (key == "--out-fps") args.output_fps = std::strtod(value.c_str(), nullptr);
        else return false;
    }
    return !args.dit.empty() && !args.vae.empty() && !args.input.empty() &&
           !args.output.empty() && !args.positive_text.empty() &&
           args.steps > 0 && args.gpu >= 0 && args.sp_size == 1 &&
           args.width > 0 && args.height > 0 && args.output_fps >= 0.0 &&
           (args.cfg_scale == 1.f || !args.negative_text.empty());
}

std::string lowercase_extension(const fs::path& path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) {
                       return static_cast<char>(std::tolower(value));
                   });
    return extension;
}

bool is_image(const fs::path& path)
{
    const std::string extension = lowercase_extension(path);
    return extension == ".jpg" || extension == ".jpeg" ||
           extension == ".png" || extension == ".bmp" ||
           extension == ".tiff" || extension == ".tif" ||
           extension == ".webp";
}

bool is_video(const fs::path& path)
{
    const std::string extension = lowercase_extension(path);
    return extension == ".mp4" || extension == ".mkv" ||
           extension == ".mov" || extension == ".webm" ||
           extension == ".avi" || extension == ".m4v" ||
           extension == ".mpg" || extension == ".mpeg";
}

int read_text(const std::string& path, ncnn::Mat& value)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        return -1;
    const std::streamsize bytes = stream.tellg();
    const std::streamsize row_bytes = 5120 * sizeof(float);
    if (bytes <= 0 || bytes % row_bytes != 0)
        return -1;
    value.create(5120, static_cast<int>(bytes / row_bytes));
    if (value.empty())
        return -100;
    stream.seekg(0);
    return stream.read(static_cast<char*>(value.data), bytes) ? 0 : -1;
}

int read_exact(const std::string& path, ncnn::Mat& value,
               int width, int height, int frames, int channels)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        return -1;
    const size_t expected = static_cast<size_t>(width) * height * frames *
                            channels * sizeof(float);
    if (stream.tellg() != static_cast<std::streamsize>(expected))
        return -1;
    value.create(width, height, frames, channels);
    if (value.empty())
        return -100;
    stream.seekg(0);
    return stream.read(static_cast<char*>(value.data), expected) ? 0 : -1;
}

int write_video_f32(const std::string& path, const ncnn::Mat& value)
{
    if (value.dims != 4 || value.w <= 0 || value.h <= 0 || value.d <= 0 ||
        value.c != 3 || value.elempack != 1 ||
        value.elemsize != sizeof(float))
        return -1;

    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
        return -1;
    const std::streamsize row_bytes =
        static_cast<std::streamsize>(value.w) * sizeof(float);
    for (int channel = 0; channel < value.c; channel++)
    {
        for (int frame = 0; frame < value.d; frame++)
        {
            const ncnn::Mat plane = value.channel(channel).depth(frame);
            for (int row = 0; row < value.h; row++)
            {
                stream.write(reinterpret_cast<const char*>(plane.row(row)),
                             row_bytes);
                if (!stream)
                    return -1;
            }
        }
    }
    return 0;
}

int process_file(const fs::path& input_path, const fs::path& output_path,
                 const Arguments& args, const ncnn::Mat& positive,
                 const ncnn::Mat& negative)
{
    const bool image_input = is_image(input_path);
    ncnn::Mat input;
    double fps = 0.0;
    if (seedvr2::read_video(input_path.string(), args.width, args.height,
                            input, fps) != 0)
        return 3;
    if (image_input && input.d != 1)
    {
        std::fprintf(stderr, "image input decoded to %d frames: %s\n",
                     input.d, input_path.string().c_str());
        return 3;
    }

    ncnn::Mat initial_noise;
    if (!args.noise.empty())
    {
        const int padded_frames = input.d == 1
                                      ? 1
                                      : ((input.d - 1 + 3) / 4) * 4 + 1;
        const int latent_frames = (padded_frames - 1) / 4 + 1;
        if (read_exact(args.noise, initial_noise, input.w / 8, input.h / 8,
                       latent_frames, 16) != 0)
        {
            std::fprintf(stderr,
                         "failed to read initial noise for %dx%dx%d input\n",
                         input.d, input.h, input.w);
            return 4;
        }
    }

    seedvr2::SeedVR2Pipeline pipeline;
    int ret = args.low_memory
                  ? pipeline.load_staged(args.dit, args.vae, args.gpu)
                  : pipeline.load(args.dit, args.vae, args.gpu);
    if (ret != 0)
        return 5;
    seedvr2::PipelineOptions options;
    options.steps = args.steps;
    options.cfg_scale = args.cfg_scale;
    options.cfg_rescale = args.cfg_rescale;
    options.seed = args.seed;
    options.initial_noise = initial_noise;
    ncnn::Mat output;
    ret = pipeline.restore(input, positive, negative, output, options);
    if (ret != 0)
    {
        std::fprintf(stderr, "SeedVR2 inference failed for %s (%d)\n",
                     input_path.string().c_str(), ret);
        return 6;
    }
    if (!args.output_f32.empty())
    {
        const fs::path raw_path(args.output_f32);
        if (!raw_path.parent_path().empty())
        {
            std::error_code error;
            fs::create_directories(raw_path.parent_path(), error);
            if (error)
                return 7;
        }
        if (write_video_f32(args.output_f32, output) != 0)
        {
            std::fprintf(stderr, "failed to write raw FP32 output: %s\n",
                         args.output_f32.c_str());
            return 7;
        }
    }
    const double selected_fps = args.output_fps > 0.0
                                    ? args.output_fps : fps;
    ret = image_input
              ? seedvr2::write_image(output_path.string(), output)
              : seedvr2::write_video(output_path.string(), output,
                                     selected_fps, input_path.string());
    if (ret != 0)
        return 7;
    std::fprintf(stdout, "wrote %s\n", output_path.string().c_str());
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    Arguments args;
    if (!parse(argc, argv, args))
    {
        usage(argv[0]);
        return 2;
    }

    ncnn::Mat positive;
    ncnn::Mat negative;
    if (read_text(args.positive_text, positive) != 0 ||
        (!args.negative_text.empty() &&
         read_text(args.negative_text, negative) != 0))
    {
        std::fprintf(stderr, "failed to read text embedding\n");
        return 4;
    }

    const fs::path input_path(args.input);
    const fs::path output_path(args.output);
    std::error_code error;
    if (fs::is_regular_file(input_path, error))
    {
        fs::path selected_output = output_path;
        if (args.output_is_directory || fs::is_directory(output_path, error))
        {
            fs::create_directories(output_path, error);
            selected_output /= input_path.filename();
        }
        else if (!output_path.parent_path().empty())
            fs::create_directories(output_path.parent_path(), error);
        if (error)
            return 7;
        if (is_image(input_path))
            selected_output.replace_extension(".png");
        return process_file(input_path, selected_output, args,
                            positive, negative);
    }
    if (!fs::is_directory(input_path, error))
    {
        std::fprintf(stderr, "input path does not exist: %s\n",
                     input_path.string().c_str());
        return 3;
    }
    if (!args.output_f32.empty())
    {
        std::fprintf(stderr,
                     "--output-f32 requires a single input file\n");
        return 2;
    }
    fs::create_directories(output_path, error);
    if (error)
        return 7;

    std::vector<fs::path> inputs;
    for (const fs::directory_entry& entry : fs::directory_iterator(input_path))
    {
        if (entry.is_regular_file() &&
            (is_image(entry.path()) || is_video(entry.path())))
            inputs.push_back(entry.path());
    }
    std::sort(inputs.begin(), inputs.end());
    if (inputs.empty())
    {
        std::fprintf(stderr, "input directory contains no supported media\n");
        return 3;
    }
    for (const fs::path& path : inputs)
    {
        fs::path selected_output = output_path / path.filename();
        if (is_image(path))
            selected_output.replace_extension(".png");
        const int ret = process_file(path, selected_output, args,
                                     positive, negative);
        if (ret != 0)
            return ret;
    }
    return 0;
}
