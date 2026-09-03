#include "seedvr2_video_io.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

namespace seedvr2 {

namespace {

std::string av_error(int code)
{
    char buffer[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(code, buffer, sizeof(buffer));
    return buffer;
}

bool is_video_shape(const ncnn::Mat& value)
{
    return value.dims == 4 && value.w > 0 && value.h > 0 &&
           value.w % 2 == 0 && value.h % 2 == 0 && value.d > 0 &&
           value.c == 3 && value.n == 1 &&
           value.elempack == 1 && value.elemsize == sizeof(float);
}

int receive_frames(AVCodecContext* decoder, SwsContext* scaler,
                   AVFrame* frame, int resized_width, int resized_height,
                   int crop_left, int crop_top, int output_width,
                   int output_height,
                   std::vector<std::vector<float> >& output_frames)
{
    while (true)
    {
        const int ret = avcodec_receive_frame(decoder, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return 0;
        if (ret < 0)
            return ret;
        std::vector<uint8_t> rgb(
            static_cast<size_t>(resized_width) * resized_height * 3);
        uint8_t* destinations[4] = {rgb.data(), nullptr, nullptr, nullptr};
        int destination_strides[4] = {resized_width * 3, 0, 0, 0};
        sws_scale(scaler, frame->data, frame->linesize, 0, decoder->height,
                  destinations, destination_strides);
        std::vector<float> output(
            static_cast<size_t>(output_width) * output_height * 3);
        for (int channel = 0; channel < 3; channel++)
        {
            float* output_channel = output.data() +
                static_cast<size_t>(channel) * output_width * output_height;
            for (int y = 0; y < output_height; y++)
            {
                for (int x = 0; x < output_width; x++)
                {
                    const int source =
                        ((y + crop_top) * resized_width + x + crop_left) * 3;
                    output_channel[y * output_width + x] =
                        rgb[source + channel] / 127.5f - 1.f;
                }
            }
        }
        output_frames.push_back(std::move(output));
        av_frame_unref(frame);
    }
}

bool supports_media_remux(const std::string& path)
{
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos)
        return false;
    std::string extension = path.substr(dot);
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) {
                       return static_cast<char>(std::tolower(value));
                   });
    return extension == ".mp4" || extension == ".mov" ||
           extension == ".webm";
}

bool is_mov_container(const std::string& path)
{
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos)
        return false;
    std::string extension = path.substr(dot);
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) {
                       return static_cast<char>(std::tolower(value));
                   });
    return extension == ".mp4" || extension == ".mov";
}

int copy_stream_metadata(const AVStream* source, AVStream* destination)
{
    av_dict_copy(&destination->metadata, source->metadata, 0);
    destination->disposition = source->disposition;
    destination->id = source->id;
    destination->sample_aspect_ratio = source->sample_aspect_ratio;
    for (int index = 0; index < source->nb_side_data; index++)
    {
        const AVPacketSideData& side_data = source->side_data[index];
        uint8_t* output = av_stream_new_side_data(
            destination, side_data.type, side_data.size);
        if (!output)
            return AVERROR(ENOMEM);
        std::memcpy(output, side_data.data, side_data.size);
    }
    return 0;
}

struct RemuxState
{
    AVFormatContext* input = nullptr;
    AVPacket* packet = nullptr;
    std::vector<int> stream_mapping;
    int64_t origin_us = 0;
    int64_t target_duration_us = 0;
    int64_t last_audio_end_us = AV_NOPTS_VALUE;
    bool packet_ready = false;
    bool eof = false;
    bool has_audio = false;
    bool active = false;
};

void close_remux(RemuxState& state)
{
    av_packet_free(&state.packet);
    avformat_close_input(&state.input);
    state.stream_mapping.clear();
    state.packet_ready = false;
    state.eof = true;
    state.active = false;
}

int prepare_remux(const std::string& source_path, AVFormatContext* output,
                  AVStream* output_video, int64_t target_duration_us,
                  RemuxState& state)
{
    int result = avformat_open_input(&state.input, source_path.c_str(), nullptr,
                                     nullptr);
    if (result < 0)
    {
        std::fprintf(stderr,
                     "SeedVR2 could not open source media for remux (%s); "
                     "writing video only: %s\n",
                     av_error(result).c_str(), source_path.c_str());
        close_remux(state);
        return 1;
    }
    result = avformat_find_stream_info(state.input, nullptr);
    if (result < 0)
    {
        std::fprintf(stderr,
                     "SeedVR2 could not inspect source media for remux (%s); "
                     "writing video only: %s\n",
                     av_error(result).c_str(), source_path.c_str());
        close_remux(state);
        return 1;
    }
    const int source_video = av_find_best_stream(
        state.input, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (source_video < 0)
    {
        std::fprintf(stderr,
                     "SeedVR2 source media has no video stream; writing video "
                     "only: %s\n", source_path.c_str());
        close_remux(state);
        return 1;
    }

    const AVStream* input_video = state.input->streams[source_video];
    av_dict_copy(&output->metadata, state.input->metadata, 0);
    result = copy_stream_metadata(input_video, output_video);
    if (result < 0)
    {
        close_remux(state);
        return result;
    }

    if (input_video->start_time != AV_NOPTS_VALUE)
        state.origin_us = av_rescale_q(input_video->start_time,
                                       input_video->time_base,
                                       AV_TIME_BASE_Q);
    else if (state.input->start_time != AV_NOPTS_VALUE)
        state.origin_us = state.input->start_time;
    state.target_duration_us = target_duration_us;
    state.stream_mapping.assign(state.input->nb_streams, -1);

    for (unsigned int index = 0; index < state.input->nb_streams; index++)
    {
        const AVStream* input_stream = state.input->streams[index];
        if (input_stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            continue;
        const int supported = avformat_query_codec(
            output->oformat, input_stream->codecpar->codec_id,
            FF_COMPLIANCE_NORMAL);
        if (supported <= 0)
        {
            std::fprintf(stderr,
                         "SeedVR2 cannot streamcopy source stream %u codec %s "
                         "to %s; omitting that stream\n",
                         index,
                         avcodec_get_name(input_stream->codecpar->codec_id),
                         output->oformat->name);
            continue;
        }
        AVStream* output_stream = avformat_new_stream(output, nullptr);
        if (!output_stream)
        {
            close_remux(state);
            return AVERROR(ENOMEM);
        }
        result = avcodec_parameters_copy(output_stream->codecpar,
                                         input_stream->codecpar);
        if (result < 0)
        {
            close_remux(state);
            return result;
        }
        output_stream->codecpar->codec_tag = 0;
        output_stream->time_base = input_stream->time_base;
        output_stream->avg_frame_rate = input_stream->avg_frame_rate;
        result = copy_stream_metadata(input_stream, output_stream);
        if (result < 0)
        {
            close_remux(state);
            return result;
        }
        state.stream_mapping[index] = output_stream->index;
        if (input_stream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO)
            state.has_audio = true;
    }
    state.packet = av_packet_alloc();
    if (!state.packet)
    {
        close_remux(state);
        return AVERROR(ENOMEM);
    }
    state.active = true;
    return 0;
}

int read_next_remux_packet(RemuxState& state, AVFormatContext* output)
{
    while (!state.eof)
    {
        av_packet_unref(state.packet);
        const int result = av_read_frame(state.input, state.packet);
        if (result == AVERROR_EOF)
        {
            state.eof = true;
            return 0;
        }
        if (result < 0)
            return result;
        const int input_index = state.packet->stream_index;
        if (input_index < 0 ||
            input_index >= static_cast<int>(state.stream_mapping.size()))
            continue;
        const int output_index = state.stream_mapping[input_index];
        if (output_index < 0)
            continue;

        const AVStream* input_stream = state.input->streams[input_index];
        AVStream* output_stream = output->streams[output_index];
        const int64_t origin = av_rescale_q(state.origin_us, AV_TIME_BASE_Q,
                                            input_stream->time_base);
        if (state.packet->pts != AV_NOPTS_VALUE)
            state.packet->pts -= origin;
        if (state.packet->dts != AV_NOPTS_VALUE)
            state.packet->dts -= origin;
        av_packet_rescale_ts(state.packet, input_stream->time_base,
                             output_stream->time_base);
        state.packet->stream_index = output_index;
        state.packet->pos = -1;

        const int64_t timestamp = state.packet->pts != AV_NOPTS_VALUE
                                      ? state.packet->pts : state.packet->dts;
        const int64_t limit = av_rescale_q(state.target_duration_us,
                                           AV_TIME_BASE_Q,
                                           output_stream->time_base);
        if (timestamp != AV_NOPTS_VALUE && timestamp >= limit)
            continue;
        if (timestamp != AV_NOPTS_VALUE && state.packet->duration > 0 &&
            timestamp + state.packet->duration > limit)
            state.packet->duration = std::max<int64_t>(0, limit - timestamp);

        if (input_stream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO &&
            timestamp != AV_NOPTS_VALUE)
        {
            const int64_t end = timestamp +
                std::max<int64_t>(0, state.packet->duration);
            state.last_audio_end_us = std::max(
                state.last_audio_end_us == AV_NOPTS_VALUE
                    ? std::numeric_limits<int64_t>::min()
                    : state.last_audio_end_us,
                av_rescale_q(end, output_stream->time_base, AV_TIME_BASE_Q));
        }
        state.packet_ready = true;
        return 0;
    }
    return 0;
}

int write_remux_packets_until(RemuxState& state, AVFormatContext* output,
                              int64_t limit_us)
{
    if (!state.active)
        return 0;
    while (true)
    {
        if (!state.packet_ready)
        {
            const int result = read_next_remux_packet(state, output);
            if (result < 0 || state.eof)
                return result;
        }
        AVStream* stream = output->streams[state.packet->stream_index];
        const int64_t timestamp = state.packet->dts != AV_NOPTS_VALUE
                                      ? state.packet->dts : state.packet->pts;
        const int64_t packet_us = timestamp == AV_NOPTS_VALUE
                                      ? std::numeric_limits<int64_t>::min()
                                      : av_rescale_q(timestamp, stream->time_base,
                                                     AV_TIME_BASE_Q);
        if (packet_us > limit_us)
            return 0;
        const int result = av_interleaved_write_frame(output, state.packet);
        state.packet_ready = false;
        av_packet_unref(state.packet);
        if (result < 0)
            return result;
    }
}

int drain_encoder(AVFormatContext* format, AVCodecContext* encoder,
                  AVStream* stream, AVPacket* packet, RemuxState* remux)
{
    while (true)
    {
        const int ret = avcodec_receive_packet(encoder, packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return 0;
        if (ret < 0)
            return ret;
        if (packet->duration <= 0 && encoder->framerate.num > 0 &&
            encoder->framerate.den > 0)
        {
            packet->duration = std::max<int64_t>(
                1, av_rescale_q(1, av_inv_q(encoder->framerate),
                                encoder->time_base));
        }
        av_packet_rescale_ts(packet, encoder->time_base, stream->time_base);
        packet->stream_index = stream->index;
        if (remux && remux->active)
        {
            const int64_t timestamp = packet->dts != AV_NOPTS_VALUE
                                          ? packet->dts : packet->pts;
            const int64_t limit_us = timestamp == AV_NOPTS_VALUE
                                         ? 0
                                         : av_rescale_q(timestamp,
                                                        stream->time_base,
                                                        AV_TIME_BASE_Q);
            const int remux_ret = write_remux_packets_until(
                *remux, format, limit_us);
            if (remux_ret < 0)
                return remux_ret;
        }
        const int write_ret = av_interleaved_write_frame(format, packet);
        av_packet_unref(packet);
        if (write_ret < 0)
            return write_ret;
    }
}

} // namespace

int read_video(const std::string& path, int target_width, int target_height,
               ncnn::Mat& video, double& fps)
{
    if (target_width <= 0 || target_height <= 0)
        return -1;
    AVFormatContext* format = nullptr;
    AVCodecContext* decoder = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* frame = nullptr;
    SwsContext* scaler = nullptr;
    int resized_width = 0;
    int resized_height = 0;
    int output_width = 0;
    int output_height = 0;
    int crop_left = 0;
    int crop_top = 0;
    int result = avformat_open_input(&format, path.c_str(), nullptr, nullptr);
    int stream_index = -1;
    if (result >= 0)
        result = avformat_find_stream_info(format, nullptr);
    if (result >= 0)
    {
        stream_index = av_find_best_stream(format, AVMEDIA_TYPE_VIDEO,
                                           -1, -1, nullptr, 0);
        if (stream_index < 0)
            result = stream_index;
    }
    if (result >= 0)
    {
        AVStream* stream = format->streams[stream_index];
        const AVCodec* codec =
            avcodec_find_decoder(stream->codecpar->codec_id);
        if (!codec)
            result = AVERROR_DECODER_NOT_FOUND;
        else
        {
            decoder = avcodec_alloc_context3(codec);
            if (!decoder)
                result = AVERROR(ENOMEM);
        }
        if (result >= 0)
            result = avcodec_parameters_to_context(decoder, stream->codecpar);
        if (result >= 0)
            result = avcodec_open2(decoder, codec, nullptr);
        if (result >= 0)
        {
            const double target_area =
                static_cast<double>(target_width) * target_height;
            const double source_area =
                static_cast<double>(decoder->width) * decoder->height;
            const double scale = std::sqrt(target_area / source_area);
            resized_width = std::max(1, static_cast<int>(
                std::nearbyint(decoder->width * scale)));
            resized_height = std::max(1, static_cast<int>(
                std::nearbyint(decoder->height * scale)));
            output_width = resized_width - resized_width % 16;
            output_height = resized_height - resized_height % 16;
            if (output_width <= 0 || output_height <= 0)
            {
                std::fprintf(stderr,
                             "SeedVR2 resized video is too small: %dx%d\n",
                             resized_width, resized_height);
                result = AVERROR(EINVAL);
            }
            crop_left = static_cast<int>(std::nearbyint(
                (resized_width - output_width) / 2.0));
            crop_top = static_cast<int>(std::nearbyint(
                (resized_height - output_height) / 2.0));
        }
        AVRational rate = av_guess_frame_rate(format, stream, nullptr);
        fps = rate.num > 0 && rate.den > 0 ? av_q2d(rate) : 24.0;
    }
    if (result >= 0)
    {
        scaler = sws_getContext(decoder->width, decoder->height,
                                decoder->pix_fmt,
                                resized_width, resized_height,
                                AV_PIX_FMT_RGB24, SWS_BICUBIC,
                                nullptr, nullptr, nullptr);
        packet = av_packet_alloc();
        frame = av_frame_alloc();
        if (!scaler || !packet || !frame)
            result = AVERROR(ENOMEM);
    }

    std::vector<std::vector<float> > frames;
    while (result >= 0 && av_read_frame(format, packet) >= 0)
    {
        if (packet->stream_index == stream_index)
        {
            result = avcodec_send_packet(decoder, packet);
            if (result >= 0)
                result = receive_frames(
                    decoder, scaler, frame, resized_width, resized_height,
                    crop_left, crop_top, output_width, output_height, frames);
        }
        av_packet_unref(packet);
    }
    if (result >= 0)
    {
        result = avcodec_send_packet(decoder, nullptr);
        if (result >= 0)
            result = receive_frames(
                decoder, scaler, frame, resized_width, resized_height,
                crop_left, crop_top, output_width, output_height, frames);
    }
    if (result >= 0 && frames.empty())
    {
        std::fprintf(stderr, "SeedVR2 input contains no video frames\n");
        result = AVERROR(EINVAL);
    }
    if (result >= 0)
    {
        video.create(output_width, output_height,
                     static_cast<int>(frames.size()), 3);
        if (video.empty())
            result = AVERROR(ENOMEM);
        const size_t channel_values =
            static_cast<size_t>(output_width) * output_height;
        for (size_t frame_index = 0;
             result >= 0 && frame_index < frames.size(); frame_index++)
        {
            for (int channel = 0; channel < 3; channel++)
            {
                ncnn::Mat output = video.channel(channel).depth(
                    static_cast<int>(frame_index));
                std::memcpy(output.data,
                            frames[frame_index].data() +
                                channel * channel_values,
                            channel_values * sizeof(float));
            }
        }
    }
    if (result < 0)
        std::fprintf(stderr, "failed to read video: %s\n",
                     av_error(result).c_str());

    sws_freeContext(scaler);
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&decoder);
    avformat_close_input(&format);
    return result < 0 ? -1 : 0;
}

int read_fixed_video(const std::string& path, ncnn::Mat& video, double& fps)
{
    const int ret = read_video(path, 32, 32, video, fps);
    if (ret != 0)
        return ret;
    if (video.w != 32 || video.h != 32 || video.d != 5)
    {
        std::fprintf(stderr,
                     "SeedVR2 fixed runtime requires 5x32x32, got %dx%dx%d\n",
                     video.d, video.h, video.w);
        video.release();
        return -1;
    }
    return 0;
}

static int write_video_impl(const std::string& path, const ncnn::Mat& video,
                            double fps,
                            const std::string* source_media_path)
{
    if (!is_video_shape(video) || !(fps > 0.0))
        return -1;
    AVFormatContext* format = nullptr;
    AVCodecContext* encoder = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    SwsContext* scaler = nullptr;
    RemuxState remux;
    int result = avformat_alloc_output_context2(&format, nullptr, nullptr,
                                                 path.c_str());
    if (result >= 0 && !format)
        result = AVERROR(EINVAL);
    const AVCodec* codec = nullptr;
    AVStream* stream = nullptr;
    if (result >= 0)
    {
        codec = avcodec_find_encoder(format->oformat->video_codec);
        if (!codec)
            codec = avcodec_find_encoder(AV_CODEC_ID_MPEG4);
        if (!codec)
            result = AVERROR_ENCODER_NOT_FOUND;
    }
    if (result >= 0)
    {
        stream = avformat_new_stream(format, nullptr);
        encoder = avcodec_alloc_context3(codec);
        if (!stream || !encoder)
            result = AVERROR(ENOMEM);
    }
    if (result >= 0)
    {
        const AVRational frame_rate = av_d2q(fps, 100000);
        encoder->codec_id = codec->id;
        encoder->codec_type = AVMEDIA_TYPE_VIDEO;
        encoder->width = video.w;
        encoder->height = video.h;
        encoder->pix_fmt = AV_PIX_FMT_YUV420P;
        encoder->framerate = frame_rate;
        encoder->time_base = av_inv_q(frame_rate);
        // Keep a floor for tiny regression clips, then scale quality with the
        // actual output size. A fixed 500 kbit/s severely distorts 720p/1080p
        // model comparisons even when the restored FP32 tensors are aligned.
        encoder->bit_rate = std::max<int64_t>(
            500000, static_cast<int64_t>(std::llround(
                        static_cast<double>(video.w) * video.h * fps * 0.35)));
        encoder->gop_size = 12;
        if (format->oformat->flags & AVFMT_GLOBALHEADER)
            encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        if (encoder->priv_data)
            av_opt_set(encoder->priv_data, "preset", "medium", 0);
        result = avcodec_open2(encoder, codec, nullptr);
        if (result >= 0)
            result = avcodec_parameters_from_context(stream->codecpar, encoder);
        stream->time_base = encoder->time_base;
        stream->avg_frame_rate = encoder->framerate;
    }
    if (result >= 0 && source_media_path)
    {
        const int64_t target_duration_us = static_cast<int64_t>(std::llround(
            static_cast<double>(video.d) / fps * AV_TIME_BASE));
        const int remux_result = prepare_remux(
            *source_media_path, format, stream, target_duration_us, remux);
        if (remux_result < 0)
            result = remux_result;
    }
    if (result >= 0 && !(format->oformat->flags & AVFMT_NOFILE))
        result = avio_open(&format->pb, path.c_str(), AVIO_FLAG_WRITE);
    if (result >= 0)
    {
        AVDictionary* muxer_options = nullptr;
        if (source_media_path && remux.active && is_mov_container(path))
            av_dict_set(&muxer_options, "movflags", "use_metadata_tags", 0);
        result = avformat_write_header(format, &muxer_options);
        av_dict_free(&muxer_options);
    }
    if (result >= 0)
    {
        frame = av_frame_alloc();
        packet = av_packet_alloc();
        if (!frame || !packet)
            result = AVERROR(ENOMEM);
    }
    if (result >= 0)
    {
        frame->format = encoder->pix_fmt;
        frame->width = video.w;
        frame->height = video.h;
        result = av_frame_get_buffer(frame, 32);
        scaler = sws_getContext(video.w, video.h, AV_PIX_FMT_RGB24,
                                video.w, video.h, encoder->pix_fmt,
                                SWS_BILINEAR,
                                nullptr, nullptr, nullptr);
        if (!scaler && result >= 0)
            result = AVERROR(ENOMEM);
    }

    const int pixels = video.w * video.h;
    std::vector<uint8_t> rgb(static_cast<size_t>(pixels) * 3);
    for (int frame_index = 0;
         result >= 0 && frame_index < video.d; frame_index++)
    {
        for (int channel = 0; channel < 3; channel++)
        {
            const float* input = video.channel(channel).depth(frame_index);
            for (int index = 0; index < pixels; index++)
            {
                const float value = std::max(-1.f, std::min(1.f, input[index]));
                rgb[index * 3 + channel] = static_cast<uint8_t>(
                    std::lround((value + 1.f) * 127.5f));
            }
        }
        result = av_frame_make_writable(frame);
        if (result < 0)
            break;
        const uint8_t* sources[4] = {rgb.data(), nullptr, nullptr, nullptr};
        int source_strides[4] = {video.w * 3, 0, 0, 0};
        sws_scale(scaler, sources, source_strides, 0, video.h,
                  frame->data, frame->linesize);
        frame->pts = frame_index;
        result = avcodec_send_frame(encoder, frame);
        if (result >= 0)
            result = drain_encoder(format, encoder, stream, packet, &remux);
    }
    if (result >= 0)
    {
        result = avcodec_send_frame(encoder, nullptr);
        if (result >= 0)
            result = drain_encoder(format, encoder, stream, packet, &remux);
    }
    if (result >= 0 && remux.active)
        result = write_remux_packets_until(remux, format,
                                           remux.target_duration_us);
    if (result >= 0)
        result = av_write_trailer(format);
    if (result < 0)
        std::fprintf(stderr, "failed to write video: %s\n",
                     av_error(result).c_str());

    sws_freeContext(scaler);
    av_packet_free(&packet);
    av_frame_free(&frame);
    avcodec_free_context(&encoder);
    if (result >= 0 && remux.has_audio &&
        (remux.last_audio_end_us == AV_NOPTS_VALUE ||
         remux.last_audio_end_us + 50000 < remux.target_duration_us))
    {
        std::fprintf(stderr,
                     "SeedVR2 source audio ends before the restored video; "
                     "streamcopy cannot synthesize silence, so the output "
                     "keeps the shorter audio track\n");
    }
    close_remux(remux);
    if (format)
    {
        if (!(format->oformat->flags & AVFMT_NOFILE) && format->pb)
            avio_closep(&format->pb);
        avformat_free_context(format);
    }
    return result < 0 ? -1 : 0;
}

int write_video(const std::string& path, const ncnn::Mat& video, double fps)
{
    return write_video_impl(path, video, fps, nullptr);
}

int write_video(const std::string& path, const ncnn::Mat& video, double fps,
                const std::string& source_media_path)
{
    if (!supports_media_remux(path))
    {
        std::fprintf(stderr,
                     "SeedVR2 media remux supports MP4, MOV, and WebM; "
                     "writing video-only output for %s\n", path.c_str());
        return write_video_impl(path, video, fps, nullptr);
    }
    return write_video_impl(path, video, fps, &source_media_path);
}

int write_image(const std::string& path, const ncnn::Mat& image)
{
    if (!is_video_shape(image) || image.d != 1)
        return -1;

    AVFormatContext* format = nullptr;
    AVCodecContext* encoder = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    SwsContext* scaler = nullptr;
    int result = avformat_alloc_output_context2(&format, nullptr, nullptr,
                                                 path.c_str());
    if (result >= 0 && !format)
        result = AVERROR(EINVAL);
    const AVCodec* codec = nullptr;
    AVStream* stream = nullptr;
    if (result >= 0)
    {
        codec = avcodec_find_encoder(AV_CODEC_ID_PNG);
        if (!codec)
            result = AVERROR_ENCODER_NOT_FOUND;
    }
    if (result >= 0)
    {
        stream = avformat_new_stream(format, nullptr);
        encoder = avcodec_alloc_context3(codec);
        if (!stream || !encoder)
            result = AVERROR(ENOMEM);
    }
    if (result >= 0)
    {
        AVPixelFormat pixel_format = AV_PIX_FMT_NONE;
        const AVPixelFormat preferences[] = {
            AV_PIX_FMT_RGB24, AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUVJ420P};
        for (AVPixelFormat preference : preferences)
        {
            if (!codec->pix_fmts)
            {
                pixel_format = preference;
                break;
            }
            for (const AVPixelFormat* supported = codec->pix_fmts;
                 *supported != AV_PIX_FMT_NONE; supported++)
            {
                if (*supported == preference)
                {
                    pixel_format = preference;
                    break;
                }
            }
            if (pixel_format != AV_PIX_FMT_NONE)
                break;
        }
        if (pixel_format == AV_PIX_FMT_NONE && codec->pix_fmts)
            pixel_format = codec->pix_fmts[0];
        if (pixel_format == AV_PIX_FMT_NONE)
            result = AVERROR(EINVAL);
        encoder->codec_id = codec->id;
        encoder->codec_type = AVMEDIA_TYPE_VIDEO;
        encoder->width = image.w;
        encoder->height = image.h;
        encoder->pix_fmt = pixel_format;
        encoder->time_base = AVRational{1, 1};
        encoder->framerate = AVRational{1, 1};
        if (format->oformat->flags & AVFMT_GLOBALHEADER)
            encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        if (result >= 0)
            result = avcodec_open2(encoder, codec, nullptr);
        if (result >= 0)
            result = avcodec_parameters_from_context(stream->codecpar,
                                                       encoder);
        stream->time_base = encoder->time_base;
        stream->avg_frame_rate = encoder->framerate;
    }
    if (result >= 0 && !(format->oformat->flags & AVFMT_NOFILE))
        result = avio_open(&format->pb, path.c_str(), AVIO_FLAG_WRITE);
    if (result >= 0)
        result = avformat_write_header(format, nullptr);
    if (result >= 0)
    {
        frame = av_frame_alloc();
        packet = av_packet_alloc();
        if (!frame || !packet)
            result = AVERROR(ENOMEM);
    }
    if (result >= 0)
    {
        frame->format = encoder->pix_fmt;
        frame->width = image.w;
        frame->height = image.h;
        frame->pts = 0;
        result = av_frame_get_buffer(frame, 32);
        scaler = sws_getContext(image.w, image.h, AV_PIX_FMT_RGB24,
                                image.w, image.h, encoder->pix_fmt,
                                SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!scaler && result >= 0)
            result = AVERROR(ENOMEM);
    }

    const int pixels = image.w * image.h;
    std::vector<uint8_t> rgb(static_cast<size_t>(pixels) * 3);
    if (result >= 0)
    {
        for (int channel = 0; channel < 3; channel++)
        {
            const float* input = image.channel(channel).depth(0);
            for (int index = 0; index < pixels; index++)
            {
                const float value = std::max(-1.f,
                                             std::min(1.f, input[index]));
                rgb[index * 3 + channel] = static_cast<uint8_t>(
                    std::lround((value + 1.f) * 127.5f));
            }
        }
        const uint8_t* sources[4] = {rgb.data(), nullptr, nullptr, nullptr};
        int source_strides[4] = {image.w * 3, 0, 0, 0};
        sws_scale(scaler, sources, source_strides, 0, image.h,
                  frame->data, frame->linesize);
        result = avcodec_send_frame(encoder, frame);
        if (result >= 0)
            result = drain_encoder(format, encoder, stream, packet, nullptr);
    }
    if (result >= 0)
    {
        result = avcodec_send_frame(encoder, nullptr);
        if (result >= 0)
            result = drain_encoder(format, encoder, stream, packet, nullptr);
    }
    if (result >= 0)
        result = av_write_trailer(format);
    if (result < 0)
        std::fprintf(stderr, "failed to write image: %s\n",
                     av_error(result).c_str());

    sws_freeContext(scaler);
    av_packet_free(&packet);
    av_frame_free(&frame);
    avcodec_free_context(&encoder);
    if (format)
    {
        if (!(format->oformat->flags & AVFMT_NOFILE) && format->pb)
            avio_closep(&format->pb);
        avformat_free_context(format);
    }
    return result < 0 ? -1 : 0;
}

int write_fixed_video(const std::string& path, const ncnn::Mat& video,
                      double fps)
{
    if (!is_video_shape(video) || video.w != 32 || video.h != 32 ||
        video.d != 5)
        return -1;
    return write_video(path, video, fps);
}

} // namespace seedvr2
