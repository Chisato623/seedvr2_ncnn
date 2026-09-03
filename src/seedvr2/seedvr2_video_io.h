#ifndef SEEDVR2_VIDEO_IO_H
#define SEEDVR2_VIDEO_IO_H

#include <string>

#include <net.h>

namespace seedvr2 {

// Applies the official inference spatial transform: resize to the requested
// target area while preserving aspect ratio, then center-crop each axis to a
// multiple of 16. All decoded frames are retained.
int read_video(const std::string& path, int target_width, int target_height,
               ncnn::Mat& video, double& fps);

// Writes a dynamic [3,T,H,W] normalized RGB tensor to the container selected
// by path. Width and height must be positive even values.
int write_video(const std::string& path, const ncnn::Mat& video, double fps);

// Replaces the video stream while remuxing compatible non-video streams and
// metadata from source_media_path. MP4, MOV, and WebM outputs are supported;
// other containers fall back to video-only output with a warning.
int write_video(const std::string& path, const ncnn::Mat& video, double fps,
                const std::string& source_media_path);

// Writes the first frame of a normalized [3,1,H,W] RGB tensor using the
// image codec selected from the output filename extension.
int write_image(const std::string& path, const ncnn::Mat& image);

// Reads exactly five 32x32 frames and normalizes RGB from [0,255] to [-1,1].
int read_fixed_video(const std::string& path, ncnn::Mat& video, double& fps);

// Writes [3,5,32,32] normalized RGB to a video container selected by path.
int write_fixed_video(const std::string& path, const ncnn::Mat& video,
                      double fps);

} // namespace seedvr2

#endif // SEEDVR2_VIDEO_IO_H
