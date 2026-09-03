#ifndef SEEDVR2_ADAPTIVE_WINDOW_ATTENTION_H
#define SEEDVR2_ADAPTIVE_WINDOW_ATTENTION_H

#include <array>
#include <vector>

#include <layer.h>

#if NCNN_VULKAN
#include <command.h>
#include <pipeline.h>
#endif

namespace seedvr2 {

struct WindowSlice
{
    int sample_index;
    int t0;
    int t1;
    int h0;
    int h1;
    int w0;
    int w1;
    int partition_offset;
    int token_count;
};

struct AdaptiveWindowPlan
{
    std::vector<WindowSlice> windows;
    std::vector<int> sample_window_count;
    std::vector<int> video_indices;
    std::vector<int> inverse_video_indices;
    std::vector<std::array<int, 3> > local_positions;
};

struct AdaptiveWindowRuntimeState
{
    std::vector<std::array<int, 3> > video_shapes;
    std::vector<int> text_lengths;
};

// Reproduces SeedVR2's 720p adaptive window construction. The window tuple is
// a target number of windows, not a fixed token extent.
int build_adaptive_window_plan(const std::vector<std::array<int, 3> >& video_shapes,
                               const std::array<int, 3>& target_windows,
                               bool shifted,
                               AdaptiveWindowPlan& plan);

class AdaptiveWindowAttention : public ncnn::Layer
{
public:
    AdaptiveWindowAttention();
    explicit AdaptiveWindowAttention(AdaptiveWindowRuntimeState* runtime_state);
    virtual ~AdaptiveWindowAttention();

    virtual int load_param(const ncnn::ParamDict& pd);
    virtual int load_model(const ncnn::ModelBin& mb);
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                        std::vector<ncnn::Mat>& top_blobs,
                        const ncnn::Option& opt) const;

#if NCNN_VULKAN
    virtual int create_pipeline(const ncnn::Option& opt);
    virtual int destroy_pipeline(const ncnn::Option& opt);
    virtual int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt);
    virtual int forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                        std::vector<ncnn::VkMat>& top_blobs,
                        ncnn::VkCompute& cmd,
                        const ncnn::Option& opt) const;
#endif

    // Test/runtime setup helper. Exported NCNN models use load_param/load_model.
    void configure(int heads, int head_dim, const std::array<int, 3>& target_windows,
                   bool shifted, float epsilon, int rope_pairs_per_axis,
                   bool shape_values_are_int32 = false,
                   bool official_bf16_attention = false);
    int set_weights(const std::vector<float>& video_norm_q,
                    const std::vector<float>& video_norm_k,
                    const std::vector<float>& text_norm_q,
                    const std::vector<float>& text_norm_k,
                    const std::vector<float>& rope_freqs);
    void set_runtime_shapes(const std::vector<std::array<int, 3> >& video_shapes,
                            const std::vector<int>& text_lengths);

private:
    int heads;
    int head_dim;
    std::array<int, 3> target_windows;
    bool shifted;
    float epsilon;
    int rope_pairs_per_axis;
    bool shared_norm;
    bool shape_values_are_int32;
    bool official_bf16_attention;

    ncnn::Mat video_norm_q;
    ncnn::Mat video_norm_k;
    ncnn::Mat text_norm_q;
    ncnn::Mat text_norm_k;
    ncnn::Mat rope_freqs;

    AdaptiveWindowRuntimeState owned_runtime_state;
    AdaptiveWindowRuntimeState* runtime_state;

#if NCNN_VULKAN
    ncnn::Layer* sdpa_vulkan;
    ncnn::Pipeline* pipeline_prepare_qkv;
    ncnn::Pipeline* pipeline_prepare_qkv_split;
    ncnn::Pipeline* pipeline_zero;
    ncnn::Pipeline* pipeline_scatter;
    ncnn::VkMat norm_weights_gpu;
    ncnn::VkMat rope_freqs_gpu;
#endif
};

ncnn::Layer* adaptive_window_attention_layer_creator(void* userdata);

} // namespace seedvr2

#endif // SEEDVR2_ADAPTIVE_WINDOW_ATTENTION_H
