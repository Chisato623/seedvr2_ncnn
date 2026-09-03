#include "adaptive_window_attention.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

#include <modelbin.h>

#if NCNN_VULKAN
#include <command.h>
#include <gpu.h>
#include <pipeline.h>
#endif

namespace seedvr2 {

namespace {

#if NCNN_VULKAN
const char prepare_qkv_shader[] = R"glsl(
#version 450

layout(binding = 0) readonly buffer video_qkv_blob { sfp video_qkv_data[]; };
layout(binding = 1) readonly buffer text_qkv_blob { sfp text_qkv_data[]; };
layout(binding = 2) readonly buffer norm_blob { sfp norm_data[]; };
layout(binding = 3) readonly buffer rope_blob { sfp rope_data[]; };
layout(binding = 4) readonly buffer index_blob { float index_data[]; };
layout(binding = 5) readonly buffer position_blob { float position_data[]; };
layout(binding = 6) writeonly buffer query_blob { sfp query_data[]; };
layout(binding = 7) writeonly buffer key_blob { sfp key_data[]; };
layout(binding = 8) writeonly buffer value_blob { sfp value_data[]; };

layout(push_constant) uniform parameter
{
    int qkv_w;
    int inner_dim;
    int head_dim;
    int text_offset;
    int text_length;
    int plan_offset;
    int video_length;
    int sequence_length;
    int output_cstep;
    int rope_pairs;
    float epsilon;
    int official_bf16_attention;
} p;

float round_bf16(float value)
{
    uint bits = floatBitsToUint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u)
        return value;
    bits += 0x00007fffu + ((bits >> 16) & 1u);
    return uintBitsToFloat(bits & 0xffff0000u);
}

shared float q_values[256];
shared float k_values[256];
shared float q_squares[256];
shared float k_squares[256];

void main()
{
    const int d = int(gl_LocalInvocationID.x);
    const int token = int(gl_GlobalInvocationID.y);
    const int head = int(gl_GlobalInvocationID.z);
    if (d >= p.head_dim || token >= p.sequence_length)
        return;

    int source_offset;
    int norm_q_offset;
    int norm_k_offset;
    ivec3 position;
    if (token < p.video_length)
    {
        const int plan_index = p.plan_offset + token;
        const int source_token = int(index_data[plan_index]);
        source_offset = source_token * p.qkv_w;
        norm_q_offset = 0;
        norm_k_offset = p.head_dim;
        position = ivec3(
            int(position_data[plan_index * 3]) + p.text_length,
            int(position_data[plan_index * 3 + 1]),
            int(position_data[plan_index * 3 + 2]));
    }
    else
    {
        const int text_token = token - p.video_length;
        source_offset = (p.text_offset + text_token) * p.qkv_w;
        norm_q_offset = p.head_dim * 2;
        norm_k_offset = p.head_dim * 3;
        position = ivec3(text_token);
    }

    const int head_offset = head * p.head_dim + d;
    float q;
    float k;
    float v;
    if (token < p.video_length)
    {
        q = float(buffer_ld1(video_qkv_data, source_offset + head_offset));
        k = float(buffer_ld1(video_qkv_data, source_offset + p.inner_dim + head_offset));
        v = float(buffer_ld1(video_qkv_data, source_offset + p.inner_dim * 2 + head_offset));
    }
    else
    {
        q = float(buffer_ld1(text_qkv_data, source_offset + head_offset));
        k = float(buffer_ld1(text_qkv_data, source_offset + p.inner_dim + head_offset));
        v = float(buffer_ld1(text_qkv_data, source_offset + p.inner_dim * 2 + head_offset));
    }

    q_values[d] = q;
    k_values[d] = k;
    q_squares[d] = q * q;
    k_squares[d] = k * k;
    barrier();

    for (int stride = p.head_dim / 2; stride > 0; stride /= 2)
    {
        if (d < stride)
        {
            q_squares[d] += q_squares[d + stride];
            k_squares[d] += k_squares[d + stride];
        }
        barrier();
    }

    const float q_scale = inversesqrt(q_squares[0] / float(p.head_dim) + p.epsilon);
    const float k_scale = inversesqrt(k_squares[0] / float(p.head_dim) + p.epsilon);
    q = q_values[d] * q_scale * float(buffer_ld1(norm_data, norm_q_offset + d));
    k = k_values[d] * k_scale * float(buffer_ld1(norm_data, norm_k_offset + d));

    const int axis_width = p.rope_pairs * 2;
    if (d < axis_width * 3)
    {
        const int axis = d / axis_width;
        const int axis_d = d - axis * axis_width;
        const int pair = axis_d / 2;
        const int partner_d = (d & 1) == 0 ? d + 1 : d - 1;
        const float q_partner = q_values[partner_d] * q_scale *
            float(buffer_ld1(norm_data, norm_q_offset + partner_d));
        const float k_partner = k_values[partner_d] * k_scale *
            float(buffer_ld1(norm_data, norm_k_offset + partner_d));
        const float angle = float(position[axis]) * float(buffer_ld1(rope_data, pair));
        const float cosine = cos(angle);
        const float sine = sin(angle);
        if ((d & 1) == 0)
        {
            q = q * cosine - q_partner * sine;
            k = k * cosine - k_partner * sine;
        }
        else
        {
            q = q * cosine + q_partner * sine;
            k = k * cosine + k_partner * sine;
        }
    }

    const int output_offset = head * p.output_cstep + token * p.head_dim + d;
    if (p.official_bf16_attention != 0)
    {
        q = round_bf16(q);
        k = round_bf16(k);
        v = round_bf16(v);
    }
    buffer_st1(query_data, output_offset, afp(q));
    buffer_st1(key_data, output_offset, afp(k));
    buffer_st1(value_data, output_offset, afp(v));
}
)glsl";

// Split-QKV variant: video/text streams each provide separate q, k and v
// buffers ([tokens, inner_dim]) so no single activation exceeds the Vulkan
// driver buffer-size limit at production token counts. The push-constant
// block is unchanged; p.qkv_w is passed as inner_dim by the caller.
const char prepare_qkv_split_shader[] = R"glsl(
#version 450

layout(binding = 0) readonly buffer video_q_blob { sfp video_q_data[]; };
layout(binding = 1) readonly buffer video_k_blob { sfp video_k_data[]; };
layout(binding = 2) readonly buffer video_v_blob { sfp video_v_data[]; };
layout(binding = 3) readonly buffer text_q_blob { sfp text_q_data[]; };
layout(binding = 4) readonly buffer text_k_blob { sfp text_k_data[]; };
layout(binding = 5) readonly buffer text_v_blob { sfp text_v_data[]; };
layout(binding = 6) readonly buffer norm_blob { sfp norm_data[]; };
layout(binding = 7) readonly buffer rope_blob { sfp rope_data[]; };
layout(binding = 8) readonly buffer index_blob { float index_data[]; };
layout(binding = 9) readonly buffer position_blob { float position_data[]; };
layout(binding = 10) writeonly buffer query_blob { sfp query_data[]; };
layout(binding = 11) writeonly buffer key_blob { sfp key_data[]; };
layout(binding = 12) writeonly buffer value_blob { sfp value_data[]; };

layout(push_constant) uniform parameter
{
    int qkv_w;
    int inner_dim;
    int head_dim;
    int text_offset;
    int text_length;
    int plan_offset;
    int video_length;
    int sequence_length;
    int output_cstep;
    int rope_pairs;
    float epsilon;
    int official_bf16_attention;
} p;

float round_bf16(float value)
{
    uint bits = floatBitsToUint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u)
        return value;
    bits += 0x00007fffu + ((bits >> 16) & 1u);
    return uintBitsToFloat(bits & 0xffff0000u);
}

shared float q_values[256];
shared float k_values[256];
shared float q_squares[256];
shared float k_squares[256];

void main()
{
    const int d = int(gl_LocalInvocationID.x);
    const int token = int(gl_GlobalInvocationID.y);
    const int head = int(gl_GlobalInvocationID.z);
    if (d >= p.head_dim || token >= p.sequence_length)
        return;

    int source_offset;
    int norm_q_offset;
    int norm_k_offset;
    ivec3 position;
    if (token < p.video_length)
    {
        const int plan_index = p.plan_offset + token;
        const int source_token = int(index_data[plan_index]);
        source_offset = source_token * p.qkv_w;
        norm_q_offset = 0;
        norm_k_offset = p.head_dim;
        position = ivec3(
            int(position_data[plan_index * 3]) + p.text_length,
            int(position_data[plan_index * 3 + 1]),
            int(position_data[plan_index * 3 + 2]));
    }
    else
    {
        const int text_token = token - p.video_length;
        source_offset = (p.text_offset + text_token) * p.qkv_w;
        norm_q_offset = p.head_dim * 2;
        norm_k_offset = p.head_dim * 3;
        position = ivec3(text_token);
    }

    const int head_offset = head * p.head_dim + d;
    float q;
    float k;
    float v;
    if (token < p.video_length)
    {
        q = float(buffer_ld1(video_q_data, source_offset + head_offset));
        k = float(buffer_ld1(video_k_data, source_offset + head_offset));
        v = float(buffer_ld1(video_v_data, source_offset + head_offset));
    }
    else
    {
        q = float(buffer_ld1(text_q_data, source_offset + head_offset));
        k = float(buffer_ld1(text_k_data, source_offset + head_offset));
        v = float(buffer_ld1(text_v_data, source_offset + head_offset));
    }

    q_values[d] = q;
    k_values[d] = k;
    q_squares[d] = q * q;
    k_squares[d] = k * k;
    barrier();

    for (int stride = p.head_dim / 2; stride > 0; stride /= 2)
    {
        if (d < stride)
        {
            q_squares[d] += q_squares[d + stride];
            k_squares[d] += k_squares[d + stride];
        }
        barrier();
    }

    const float q_scale = inversesqrt(q_squares[0] / float(p.head_dim) + p.epsilon);
    const float k_scale = inversesqrt(k_squares[0] / float(p.head_dim) + p.epsilon);
    q = q_values[d] * q_scale * float(buffer_ld1(norm_data, norm_q_offset + d));
    k = k_values[d] * k_scale * float(buffer_ld1(norm_data, norm_k_offset + d));

    const int axis_width = p.rope_pairs * 2;
    if (d < axis_width * 3)
    {
        const int axis = d / axis_width;
        const int axis_d = d - axis * axis_width;
        const int pair = axis_d / 2;
        const int partner_d = (d & 1) == 0 ? d + 1 : d - 1;
        const float q_partner = q_values[partner_d] * q_scale *
            float(buffer_ld1(norm_data, norm_q_offset + partner_d));
        const float k_partner = k_values[partner_d] * k_scale *
            float(buffer_ld1(norm_data, norm_k_offset + partner_d));
        const float angle = float(position[axis]) * float(buffer_ld1(rope_data, pair));
        const float cosine = cos(angle);
        const float sine = sin(angle);
        if ((d & 1) == 0)
        {
            q = q * cosine - q_partner * sine;
            k = k * cosine - k_partner * sine;
        }
        else
        {
            q = q * cosine + q_partner * sine;
            k = k * cosine + k_partner * sine;
        }
    }

    const int output_offset = head * p.output_cstep + token * p.head_dim + d;
    if (p.official_bf16_attention != 0)
    {
        q = round_bf16(q);
        k = round_bf16(k);
        v = round_bf16(v);
    }
    buffer_st1(query_data, output_offset, afp(q));
    buffer_st1(key_data, output_offset, afp(k));
    buffer_st1(value_data, output_offset, afp(v));
}
)glsl";

const char zero_shader[] = R"glsl(
#version 450

layout(binding = 0) writeonly buffer output_blob { sfp output_data[]; };

layout(push_constant) uniform parameter
{
    int count;
} p;

void main()
{
    const int i = int(gl_GlobalInvocationID.x);
    if (i < p.count)
        buffer_st1(output_data, i, afp(0.f));
}
)glsl";

const char scatter_shader[] = R"glsl(
#version 450

layout(binding = 0) readonly buffer context_blob { sfp context_data[]; };
layout(binding = 1) readonly buffer index_blob { float index_data[]; };
layout(binding = 2) buffer video_output_blob { sfp video_output_data[]; };
layout(binding = 3) buffer text_output_blob { sfp text_output_data[]; };

layout(push_constant) uniform parameter
{
    int inner_dim;
    int head_dim;
    int text_offset;
    int plan_offset;
    int video_length;
    int sequence_length;
    int context_cstep;
    float text_pool_scale;
    int official_bf16_attention;
} p;

float round_bf16(float value)
{
    uint bits = floatBitsToUint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u)
        return value;
    bits += 0x00007fffu + ((bits >> 16) & 1u);
    return uintBitsToFloat(bits & 0xffff0000u);
}

void main()
{
    const int d = int(gl_GlobalInvocationID.x);
    const int token = int(gl_GlobalInvocationID.y);
    const int head = int(gl_GlobalInvocationID.z);
    if (d >= p.head_dim || token >= p.sequence_length)
        return;

    const int context_offset = head * p.context_cstep + token * p.head_dim + d;
    float value = float(buffer_ld1(context_data, context_offset));
    if (p.official_bf16_attention != 0)
        value = round_bf16(value);
    if (token < p.video_length)
    {
        const int source_token = int(index_data[p.plan_offset + token]);
        const int output_offset = source_token * p.inner_dim + head * p.head_dim + d;
        buffer_st1(video_output_data, output_offset, afp(value));
    }
    else
    {
        const int text_token = token - p.video_length;
        const int output_offset = (p.text_offset + text_token) * p.inner_dim + head * p.head_dim + d;
        const float previous = float(buffer_ld1(text_output_data, output_offset));
        buffer_st1(text_output_data, output_offset, afp(previous + value * p.text_pool_scale));
    }
}
)glsl";

int create_shader_pipeline(const char* source, const ncnn::Option& opt,
                           const ncnn::VulkanDevice* vkdev,
                           int local_x, int local_y, int local_z,
                           ncnn::Pipeline*& pipeline)
{
    std::vector<uint32_t> spirv;
    int ret = ncnn::compile_spirv_module(source, opt, spirv);
    if (ret != 0)
        return ret;

    pipeline = new ncnn::Pipeline(vkdev);
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
#endif

int ceil_div_extent(int extent, int window)
{
    return (extent + window - 1) / window;
}

int python_round_positive(double value)
{
    // nearbyint follows the default round-to-nearest, ties-to-even mode used by
    // Python round(). Window scale values are positive.
    return static_cast<int>(std::nearbyint(value));
}

void append_slice(int sample_index, int sample_offset, int t, int h, int w,
                  int t0, int t1, int h0, int h1, int w0, int w1,
                  AdaptiveWindowPlan& plan)
{
    WindowSlice slice;
    slice.sample_index = sample_index;
    slice.t0 = t0;
    slice.t1 = t1;
    slice.h0 = h0;
    slice.h1 = h1;
    slice.w0 = w0;
    slice.w1 = w1;
    slice.partition_offset = static_cast<int>(plan.video_indices.size());
    slice.token_count = (t1 - t0) * (h1 - h0) * (w1 - w0);
    plan.windows.push_back(slice);

    for (int ti = t0; ti < t1; ti++)
    {
        for (int hi = h0; hi < h1; hi++)
        {
            for (int wi = w0; wi < w1; wi++)
            {
                plan.video_indices.push_back(sample_offset + (ti * h + hi) * w + wi);
                plan.local_positions.push_back({ti - t0, hi - h0, wi - w0});
            }
        }
    }
}

int read_shape_value(const ncnn::Mat& values, int row, int column, bool int32_values)
{
    if (values.dims == 1)
    {
        return int32_values ? static_cast<const int*>(values.data)[column]
                            : static_cast<int>(static_cast<const float*>(values.data)[column]);
    }

    return int32_values ? values.row<const int>(row)[column]
                        : static_cast<int>(values.row(row)[column]);
}

int copy_vector_to_mat(const std::vector<float>& src, int expected, ncnn::Mat& dst)
{
    if (static_cast<int>(src.size()) != expected)
        return -1;

    dst.create(expected);
    if (dst.empty())
        return -100;
    std::memcpy(dst.data, src.data(), src.size() * sizeof(float));
    return 0;
}

float round_bf16(float value)
{
    unsigned int bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x00007fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void rms_norm(const float* input, const float* weight, int size, float epsilon, float* output)
{
    float square_sum = 0.f;
    for (int i = 0; i < size; i++)
        square_sum += input[i] * input[i];

    const float inv_rms = 1.f / std::sqrt(square_sum / size + epsilon);
    for (int i = 0; i < size; i++)
        output[i] = input[i] * inv_rms * weight[i];
}

void apply_mm_rope(float* values, int head_dim, const float* freqs, int pairs_per_axis,
                   const std::array<int, 3>& position)
{
    const int axis_width = pairs_per_axis * 2;
    const int rotary_width = axis_width * 3;
    if (rotary_width > head_dim)
        return;

    for (int axis = 0; axis < 3; axis++)
    {
        float* axis_values = values + axis * axis_width;
        for (int pair = 0; pair < pairs_per_axis; pair++)
        {
            const float angle = position[axis] * freqs[pair];
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            const float x0 = axis_values[pair * 2];
            const float x1 = axis_values[pair * 2 + 1];
            axis_values[pair * 2] = x0 * cosine - x1 * sine;
            axis_values[pair * 2 + 1] = x0 * sine + x1 * cosine;
        }
    }
}

struct LayerPipelineGuard
{
    LayerPipelineGuard(ncnn::Layer* layer, const ncnn::Option& option)
        : layer(layer), option(option), created(false)
    {
    }

    ~LayerPipelineGuard()
    {
        if (created)
            layer->destroy_pipeline(option);
    }

    ncnn::Layer* layer;
    const ncnn::Option& option;
    bool created;
};

} // namespace

int build_adaptive_window_plan(const std::vector<std::array<int, 3> >& video_shapes,
                               const std::array<int, 3>& target_windows,
                               bool shifted,
                               AdaptiveWindowPlan& plan)
{
    plan = AdaptiveWindowPlan();
    plan.sample_window_count.reserve(video_shapes.size());

    int total_tokens = 0;
    for (size_t sample_index = 0; sample_index < video_shapes.size(); sample_index++)
    {
        const int t = video_shapes[sample_index][0];
        const int h = video_shapes[sample_index][1];
        const int w = video_shapes[sample_index][2];
        if (t <= 0 || h <= 0 || w <= 0 || target_windows[0] <= 0 ||
            target_windows[1] <= 0 || target_windows[2] <= 0)
            return -1;

        const double scale = std::sqrt(3600.0 / static_cast<double>(h * w));
        const int resized_h = python_round_positive(h * scale);
        const int resized_w = python_round_positive(w * scale);
        const int wt = ceil_div_extent(std::min(t, 30), target_windows[0]);
        const int wh = ceil_div_extent(resized_h, target_windows[1]);
        const int ww = ceil_div_extent(resized_w, target_windows[2]);
        if (wt <= 0 || wh <= 0 || ww <= 0)
            return -1;

        const double st = shifted && wt < t ? 0.5 : 0.0;
        const double sh = shifted && wh < h ? 0.5 : 0.0;
        const double sw = shifted && ww < w ? 0.5 : 0.0;

        const int nt = st > 0.0 ? static_cast<int>(std::ceil((t - st) / wt)) + 1 : 1;
        const int nh = sh > 0.0 ? static_cast<int>(std::ceil((h - sh) / wh)) + 1 : 1;
        const int nw = sw > 0.0 ? static_cast<int>(std::ceil((w - sw) / ww)) + 1 : 1;

        // The non-shifted path uses the ordinary ceil(extent/window) counts.
        const int normal_nt = ceil_div_extent(t, wt);
        const int normal_nh = ceil_div_extent(h, wh);
        const int normal_nw = ceil_div_extent(w, ww);
        const int loop_nt = shifted ? nt : normal_nt;
        const int loop_nh = shifted ? nh : normal_nh;
        const int loop_nw = shifted ? nw : normal_nw;

        const int first_window = static_cast<int>(plan.windows.size());
        for (int iw = 0; iw < loop_nw; iw++)
        {
            const int w0 = shifted ? std::max(static_cast<int>((iw - sw) * ww), 0) : iw * ww;
            const int w1 = shifted ? std::min(static_cast<int>((iw - sw + 1.0) * ww), w)
                                   : std::min((iw + 1) * ww, w);
            if (w1 <= w0)
                continue;

            for (int ih = 0; ih < loop_nh; ih++)
            {
                const int h0 = shifted ? std::max(static_cast<int>((ih - sh) * wh), 0) : ih * wh;
                const int h1 = shifted ? std::min(static_cast<int>((ih - sh + 1.0) * wh), h)
                                       : std::min((ih + 1) * wh, h);
                if (h1 <= h0)
                    continue;

                for (int it = 0; it < loop_nt; it++)
                {
                    const int t0 = shifted ? std::max(static_cast<int>((it - st) * wt), 0) : it * wt;
                    const int t1 = shifted ? std::min(static_cast<int>((it - st + 1.0) * wt), t)
                                           : std::min((it + 1) * wt, t);
                    if (t1 <= t0)
                        continue;
                    append_slice(static_cast<int>(sample_index), total_tokens, t, h, w,
                                 t0, t1, h0, h1, w0, w1, plan);
                }
            }
        }

        plan.sample_window_count.push_back(static_cast<int>(plan.windows.size()) - first_window);
        total_tokens += t * h * w;
    }

    if (static_cast<int>(plan.video_indices.size()) != total_tokens)
        return -2;

    plan.inverse_video_indices.assign(total_tokens, -1);
    for (size_t partition_index = 0; partition_index < plan.video_indices.size(); partition_index++)
    {
        const int source_index = plan.video_indices[partition_index];
        if (source_index < 0 || source_index >= total_tokens ||
            plan.inverse_video_indices[source_index] != -1)
            return -2;
        plan.inverse_video_indices[source_index] = static_cast<int>(partition_index);
    }
    return 0;
}

AdaptiveWindowAttention::AdaptiveWindowAttention()
    : AdaptiveWindowAttention(0)
{
}

AdaptiveWindowAttention::AdaptiveWindowAttention(AdaptiveWindowRuntimeState* new_runtime_state)
    : heads(20), head_dim(128), target_windows({4, 3, 3}), shifted(false),
      epsilon(1e-5f), rope_pairs_per_axis(21), shared_norm(false),
      shape_values_are_int32(false), official_bf16_attention(false),
      runtime_state(new_runtime_state)
#if NCNN_VULKAN
      , sdpa_vulkan(0), pipeline_prepare_qkv(0), pipeline_prepare_qkv_split(0), pipeline_zero(0), pipeline_scatter(0)
#endif
{
    one_blob_only = false;
    support_inplace = false;
#if NCNN_VULKAN
    support_vulkan = true;
    support_vulkan_packing = false;
#endif
}

AdaptiveWindowAttention::~AdaptiveWindowAttention()
{
}

void AdaptiveWindowAttention::configure(int new_heads, int new_head_dim,
                                        const std::array<int, 3>& new_target_windows,
                                        bool new_shifted, float new_epsilon,
                                        int new_rope_pairs_per_axis,
                                        bool new_shape_values_are_int32,
                                        bool new_official_bf16_attention)
{
    heads = new_heads;
    head_dim = new_head_dim;
    target_windows = new_target_windows;
    shifted = new_shifted;
    epsilon = new_epsilon;
    rope_pairs_per_axis = new_rope_pairs_per_axis;
    shape_values_are_int32 = new_shape_values_are_int32;
    official_bf16_attention = new_official_bf16_attention;
}

int AdaptiveWindowAttention::load_param(const ncnn::ParamDict& pd)
{
    heads = pd.get(0, 20);
    head_dim = pd.get(1, 128);
    target_windows[0] = pd.get(2, 4);
    target_windows[1] = pd.get(3, 3);
    target_windows[2] = pd.get(4, 3);
    shifted = pd.get(5, 0) != 0;
    epsilon = pd.get(6, 1e-5f);
    rope_pairs_per_axis = pd.get(7, 21);
    shared_norm = pd.get(8, 0) != 0;
    shape_values_are_int32 = pd.get(9, 0) != 0;
    official_bf16_attention = pd.get(10, 0) != 0;
    return 0;
}

int AdaptiveWindowAttention::load_model(const ncnn::ModelBin& mb)
{
    video_norm_q = mb.load(head_dim, 1);
    video_norm_k = mb.load(head_dim, 1);
    if (video_norm_q.empty() || video_norm_k.empty())
        return -100;

    if (shared_norm)
    {
        text_norm_q = video_norm_q;
        text_norm_k = video_norm_k;
    }
    else
    {
        text_norm_q = mb.load(head_dim, 1);
        text_norm_k = mb.load(head_dim, 1);
        if (text_norm_q.empty() || text_norm_k.empty())
            return -100;
    }

    rope_freqs = mb.load(rope_pairs_per_axis, 1);
    return rope_freqs.empty() ? -100 : 0;
}

int AdaptiveWindowAttention::set_weights(const std::vector<float>& new_video_norm_q,
                                         const std::vector<float>& new_video_norm_k,
                                         const std::vector<float>& new_text_norm_q,
                                         const std::vector<float>& new_text_norm_k,
                                         const std::vector<float>& new_rope_freqs)
{
    if (copy_vector_to_mat(new_video_norm_q, head_dim, video_norm_q) != 0 ||
        copy_vector_to_mat(new_video_norm_k, head_dim, video_norm_k) != 0 ||
        copy_vector_to_mat(new_text_norm_q, head_dim, text_norm_q) != 0 ||
        copy_vector_to_mat(new_text_norm_k, head_dim, text_norm_k) != 0 ||
        copy_vector_to_mat(new_rope_freqs, rope_pairs_per_axis, rope_freqs) != 0)
        return -1;
    return 0;
}

void AdaptiveWindowAttention::set_runtime_shapes(
    const std::vector<std::array<int, 3> >& video_shapes,
    const std::vector<int>& text_lengths)
{
    owned_runtime_state.video_shapes = video_shapes;
    owned_runtime_state.text_lengths = text_lengths;
    runtime_state = &owned_runtime_state;
}

int AdaptiveWindowAttention::forward(const std::vector<ncnn::Mat>& bottom_blobs,
                                     std::vector<ncnn::Mat>& top_blobs,
                                     const ncnn::Option& opt) const
{
    const bool split_qkv = bottom_blobs.size() == 8;
    if ((!split_qkv && bottom_blobs.size() != 4) || top_blobs.size() != 2 || heads <= 0 || head_dim <= 0 ||
        rope_pairs_per_axis * 6 > head_dim || video_norm_q.empty() || video_norm_k.empty() ||
        text_norm_q.empty() || text_norm_k.empty() || rope_freqs.empty())
        return -1;

    const ncnn::Mat& video_qkv = bottom_blobs[0];
    const ncnn::Mat& video_k_mat = split_qkv ? bottom_blobs[1] : bottom_blobs[0];
    const ncnn::Mat& video_v_mat = split_qkv ? bottom_blobs[2] : bottom_blobs[0];
    const ncnn::Mat& text_qkv = bottom_blobs[split_qkv ? 3 : 1];
    const ncnn::Mat& text_k_mat = split_qkv ? bottom_blobs[4] : bottom_blobs[1];
    const ncnn::Mat& text_v_mat = split_qkv ? bottom_blobs[5] : bottom_blobs[1];
    const ncnn::Mat& video_shape_blob = bottom_blobs[split_qkv ? 6 : 2];
    const ncnn::Mat& text_length_blob = bottom_blobs[split_qkv ? 7 : 3];
    const int inner_dim = heads * head_dim;
    if (split_qkv)
    {
        for (int part = 0; part < 6; part++)
        {
            if (bottom_blobs[part].dims != 2 || bottom_blobs[part].w != inner_dim ||
                bottom_blobs[part].elempack != 1 ||
                bottom_blobs[part].elemsize != sizeof(float))
                return -1;
        }
        if (video_shape_blob.w != 3)
            return -1;
    }
    else if (video_qkv.dims != 2 || text_qkv.dims != 2 || video_qkv.w != inner_dim * 3 ||
             text_qkv.w != inner_dim * 3 || video_shape_blob.w != 3)
        return -1;

    const int batch = video_shape_blob.dims == 1 ? 1 : video_shape_blob.h;
    const int text_batch = text_length_blob.dims == 1 ? text_length_blob.w : text_length_blob.h;
    if (batch <= 0 || text_batch != batch)
        return -1;

    std::vector<std::array<int, 3> > video_shapes(batch);
    std::vector<int> text_lengths(batch);
    int expected_video_tokens = 0;
    int expected_text_tokens = 0;
    for (int sample = 0; sample < batch; sample++)
    {
        for (int axis = 0; axis < 3; axis++)
            video_shapes[sample][axis] = read_shape_value(video_shape_blob, sample, axis, shape_values_are_int32);
        text_lengths[sample] = read_shape_value(text_length_blob, sample, 0, shape_values_are_int32);
        if (text_lengths[sample] <= 0)
            return -1;
        expected_video_tokens += video_shapes[sample][0] * video_shapes[sample][1] * video_shapes[sample][2];
        expected_text_tokens += text_lengths[sample];
    }
    if (split_qkv)
    {
        for (int part = 0; part < 3; part++)
        {
            if (bottom_blobs[part].h != expected_video_tokens ||
                bottom_blobs[part + 3].h != expected_text_tokens)
                return -1;
        }
    }
    else if (expected_video_tokens != video_qkv.h ||
             expected_text_tokens != text_qkv.h)
    {
        return -1;
    }

    AdaptiveWindowPlan plan;
    int ret = build_adaptive_window_plan(video_shapes, target_windows, shifted, plan);
    if (ret != 0)
        return ret;

    ncnn::Mat& video_output = top_blobs[0];
    ncnn::Mat& text_output = top_blobs[1];
    video_output.create(inner_dim, expected_video_tokens, 4u, opt.blob_allocator);
    text_output.create(inner_dim, expected_text_tokens, 4u, opt.blob_allocator);
    if (video_output.empty() || text_output.empty())
        return -100;
    video_output.fill(0.f);
    text_output.fill(0.f);

    std::unique_ptr<ncnn::Layer> sdpa(ncnn::create_layer_cpu("SDPA"));
    if (!sdpa)
        return -1;
    ncnn::ParamDict sdpa_params;
    sdpa_params.set(5, 0);
    sdpa_params.set(6, 0.f);
    sdpa_params.set(7, 0);
    ret = sdpa->load_param(sdpa_params);
    if (ret != 0)
        return ret;
    LayerPipelineGuard sdpa_pipeline(sdpa.get(), opt);
    ret = sdpa->create_pipeline(opt);
    if (ret != 0)
        return ret;
    sdpa_pipeline.created = true;

    std::vector<int> text_offsets(batch, 0);
    for (int sample = 1; sample < batch; sample++)
        text_offsets[sample] = text_offsets[sample - 1] + text_lengths[sample - 1];

    const float* video_q_weight = video_norm_q;
    const float* video_k_weight = video_norm_k;
    const float* text_q_weight = text_norm_q;
    const float* text_k_weight = text_norm_k;
    const float* frequencies = rope_freqs;

    for (size_t window_index = 0; window_index < plan.windows.size(); window_index++)
    {
        const WindowSlice& window = plan.windows[window_index];
        const int sample = window.sample_index;
        const int text_length = text_lengths[sample];
        const int sequence_length = window.token_count + text_length;

        ncnn::Mat query(head_dim, sequence_length, heads, 4u, opt.workspace_allocator);
        ncnn::Mat key(head_dim, sequence_length, heads, 4u, opt.workspace_allocator);
        ncnn::Mat value(head_dim, sequence_length, heads, 4u, opt.workspace_allocator);
        if (query.empty() || key.empty() || value.empty())
            return -100;

        for (int local_token = 0; local_token < window.token_count; local_token++)
        {
            const int partition_token = window.partition_offset + local_token;
            const int source_token = plan.video_indices[partition_token];
            const float* source = video_qkv.row(source_token);
            const float* source_k = split_qkv ? video_k_mat.row(source_token) : source + inner_dim;
            const float* source_v = split_qkv ? video_v_mat.row(source_token) : source + inner_dim * 2;
            for (int head = 0; head < heads; head++)
            {
                float* q = query.channel(head).row(local_token);
                float* k = key.channel(head).row(local_token);
                float* v = value.channel(head).row(local_token);
                rms_norm(source + head * head_dim, video_q_weight, head_dim, epsilon, q);
                rms_norm(source_k + head * head_dim, video_k_weight, head_dim, epsilon, k);
                std::memcpy(v, source_v + head * head_dim, head_dim * sizeof(float));
                std::array<int, 3> position = plan.local_positions[partition_token];
                position[0] += text_length;
                apply_mm_rope(q, head_dim, frequencies, rope_pairs_per_axis, position);
                apply_mm_rope(k, head_dim, frequencies, rope_pairs_per_axis, position);
            }
        }

        for (int text_token = 0; text_token < text_length; text_token++)
        {
            const int source_token = text_offsets[sample] + text_token;
            const int target_token = window.token_count + text_token;
            const float* source = text_qkv.row(source_token);
            const float* source_k = split_qkv ? text_k_mat.row(source_token) : source + inner_dim;
            const float* source_v = split_qkv ? text_v_mat.row(source_token) : source + inner_dim * 2;
            const std::array<int, 3> position = {text_token, text_token, text_token};
            for (int head = 0; head < heads; head++)
            {
                float* q = query.channel(head).row(target_token);
                float* k = key.channel(head).row(target_token);
                float* v = value.channel(head).row(target_token);
                rms_norm(source + head * head_dim, text_q_weight, head_dim, epsilon, q);
                rms_norm(source_k + head * head_dim, text_k_weight, head_dim, epsilon, k);
                std::memcpy(v, source_v + head * head_dim, head_dim * sizeof(float));
                apply_mm_rope(q, head_dim, frequencies, rope_pairs_per_axis, position);
                apply_mm_rope(k, head_dim, frequencies, rope_pairs_per_axis, position);
            }
        }

        std::vector<ncnn::Mat> sdpa_inputs(3);
        if (official_bf16_attention)
        {
            for (size_t i = 0; i < query.total(); i++)
                query[i] = round_bf16(query[i]);
            for (size_t i = 0; i < key.total(); i++)
                key[i] = round_bf16(key[i]);
            for (size_t i = 0; i < value.total(); i++)
                value[i] = round_bf16(value[i]);
        }
        sdpa_inputs[0] = query;
        sdpa_inputs[1] = key;
        sdpa_inputs[2] = value;
        std::vector<ncnn::Mat> sdpa_outputs(1);
        ret = sdpa->forward(sdpa_inputs, sdpa_outputs, opt);
        if (ret != 0)
            return ret;
        ncnn::Mat& context = sdpa_outputs[0];
        if (official_bf16_attention)
        {
            for (size_t i = 0; i < context.total(); i++)
                static_cast<float*>(context.data)[i] =
                    round_bf16(static_cast<float*>(context.data)[i]);
        }

        for (int local_token = 0; local_token < window.token_count; local_token++)
        {
            const int source_token = plan.video_indices[window.partition_offset + local_token];
            float* output = video_output.row(source_token);
            for (int head = 0; head < heads; head++)
                std::memcpy(output + head * head_dim, context.channel(head).row(local_token),
                            head_dim * sizeof(float));
        }

        const float pool_scale = 1.f / plan.sample_window_count[sample];
        for (int text_token = 0; text_token < text_length; text_token++)
        {
            float* output = text_output.row(text_offsets[sample] + text_token);
            const int context_token = window.token_count + text_token;
            for (int head = 0; head < heads; head++)
            {
                const float* input = context.channel(head).row(context_token);
                for (int d = 0; d < head_dim; d++)
                    output[head * head_dim + d] += input[d] * pool_scale;
            }
        }
    }

    return 0;
}

#if NCNN_VULKAN
int AdaptiveWindowAttention::create_pipeline(const ncnn::Option& opt)
{
    if (!opt.use_vulkan_compute)
        return 0;
    if (!vkdev || head_dim <= 0 || head_dim > 256 || (head_dim & (head_dim - 1)) != 0)
        return -1;

    sdpa_vulkan = ncnn::create_layer_vulkan("SDPA");
    if (!sdpa_vulkan)
        return -1;
    sdpa_vulkan->vkdev = vkdev;
    ncnn::ParamDict sdpa_params;
    sdpa_params.set(5, 0);
    sdpa_params.set(6, 0.f);
    sdpa_params.set(7, 0);
    int ret = sdpa_vulkan->load_param(sdpa_params);
    if (ret != 0)
        return ret;
    ret = sdpa_vulkan->create_pipeline(opt);
    if (ret != 0)
        return ret;

    ret = create_shader_pipeline(prepare_qkv_shader, opt, vkdev, head_dim, 1, 1,
                                 pipeline_prepare_qkv);
    if (ret != 0)
        return ret;
    ret = create_shader_pipeline(prepare_qkv_split_shader, opt, vkdev, head_dim, 1, 1,
                                 pipeline_prepare_qkv_split);
    if (ret != 0)
        return ret;
    ret = create_shader_pipeline(zero_shader, opt, vkdev, 256, 1, 1, pipeline_zero);
    if (ret != 0)
        return ret;
    return create_shader_pipeline(scatter_shader, opt, vkdev, head_dim, 1, 1,
                                  pipeline_scatter);
}

int AdaptiveWindowAttention::destroy_pipeline(const ncnn::Option& opt)
{
    norm_weights_gpu.release();
    rope_freqs_gpu.release();

    delete pipeline_prepare_qkv;
    delete pipeline_prepare_qkv_split;
    delete pipeline_zero;
    delete pipeline_scatter;
    pipeline_prepare_qkv = 0;
    pipeline_prepare_qkv_split = 0;
    pipeline_zero = 0;
    pipeline_scatter = 0;

    if (sdpa_vulkan)
    {
        sdpa_vulkan->destroy_pipeline(opt);
        delete sdpa_vulkan;
        sdpa_vulkan = 0;
    }
    return 0;
}

int AdaptiveWindowAttention::upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt)
{
    if (video_norm_q.empty() || video_norm_k.empty() || text_norm_q.empty() ||
        text_norm_k.empty() || rope_freqs.empty())
        return -1;

    ncnn::Mat norm_weights(head_dim * 4);
    if (norm_weights.empty())
        return -100;
    std::memcpy(norm_weights.data, video_norm_q.data, head_dim * sizeof(float));
    std::memcpy(norm_weights.row(0) + head_dim, video_norm_k.data, head_dim * sizeof(float));
    std::memcpy(norm_weights.row(0) + head_dim * 2, text_norm_q.data, head_dim * sizeof(float));
    std::memcpy(norm_weights.row(0) + head_dim * 3, text_norm_k.data, head_dim * sizeof(float));
    cmd.record_upload(norm_weights, norm_weights_gpu, opt);
    cmd.record_upload(rope_freqs, rope_freqs_gpu, opt);
    return norm_weights_gpu.empty() || rope_freqs_gpu.empty() ? -100 : 0;
}

int AdaptiveWindowAttention::forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                                     std::vector<ncnn::VkMat>& top_blobs,
                                     ncnn::VkCompute& cmd,
                                     const ncnn::Option& opt) const
{
    const bool split_qkv = bottom_blobs.size() == 8;
    if ((!split_qkv && bottom_blobs.size() != 4) || top_blobs.size() != 2 || !runtime_state ||
        runtime_state->video_shapes.empty() ||
        runtime_state->video_shapes.size() != runtime_state->text_lengths.size() ||
        !sdpa_vulkan || !pipeline_prepare_qkv || !pipeline_prepare_qkv_split ||
        !pipeline_zero || !pipeline_scatter ||
        norm_weights_gpu.empty() || rope_freqs_gpu.empty())
        return -1;

    const ncnn::VkMat& video_qkv = bottom_blobs[0];
    const ncnn::VkMat& text_qkv = bottom_blobs[split_qkv ? 3 : 1];
    const int inner_dim = heads * head_dim;
    if (split_qkv)
    {
        for (int part = 0; part < 6; part++)
        {
            if (bottom_blobs[part].dims != 2 || bottom_blobs[part].w != inner_dim ||
                bottom_blobs[part].elempack != 1 ||
                bottom_blobs[part].elembits() != video_qkv.elembits())
                return -1;
        }
    }
    else if (video_qkv.dims != 2 || text_qkv.dims != 2 || video_qkv.w != inner_dim * 3 ||
             text_qkv.w != inner_dim * 3 || video_qkv.elembits() != text_qkv.elembits())
        return -1;

    int total_video_tokens = 0;
    int total_text_tokens = 0;
    for (size_t sample = 0; sample < runtime_state->video_shapes.size(); sample++)
    {
        const std::array<int, 3>& shape = runtime_state->video_shapes[sample];
        if (shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 ||
            runtime_state->text_lengths[sample] <= 0)
            return -1;
        total_video_tokens += shape[0] * shape[1] * shape[2];
        total_text_tokens += runtime_state->text_lengths[sample];
    }
    if (split_qkv)
    {
        for (int part = 0; part < 3; part++)
        {
            if (bottom_blobs[part].h != total_video_tokens ||
                bottom_blobs[part + 3].h != total_text_tokens)
                return -1;
        }
    }
    else if (video_qkv.h * video_qkv.elempack != total_video_tokens ||
             text_qkv.h * text_qkv.elempack != total_text_tokens)
    {
        return -1;
    }

    AdaptiveWindowPlan plan;
    int ret = build_adaptive_window_plan(runtime_state->video_shapes, target_windows,
                                         shifted, plan);
    if (ret != 0)
        return ret;

    std::vector<float> plan_indices(plan.video_indices.begin(), plan.video_indices.end());
    std::vector<float> plan_positions(plan.local_positions.size() * 3);
    for (size_t i = 0; i < plan.local_positions.size(); i++)
    {
        plan_positions[i * 3] = static_cast<float>(plan.local_positions[i][0]);
        plan_positions[i * 3 + 1] = static_cast<float>(plan.local_positions[i][1]);
        plan_positions[i * 3 + 2] = static_cast<float>(plan.local_positions[i][2]);
    }
    ncnn::Mat plan_indices_cpu(static_cast<int>(plan_indices.size()), plan_indices.data(), 4u);
    ncnn::Mat plan_positions_cpu(static_cast<int>(plan_positions.size()), plan_positions.data(), 4u);
    ncnn::Option metadata_opt = opt;
    metadata_opt.blob_vkallocator = opt.workspace_vkallocator;
    ncnn::VkMat plan_indices_gpu;
    ncnn::VkMat plan_positions_gpu;
    cmd.record_clone(plan_indices_cpu, plan_indices_gpu, metadata_opt);
    cmd.record_clone(plan_positions_cpu, plan_positions_gpu, metadata_opt);
    if (plan_indices_gpu.empty() || plan_positions_gpu.empty())
        return -100;

    const size_t scalar_elemsize = static_cast<size_t>(video_qkv.elembits() / 8);
    ncnn::VkMat& video_output = top_blobs[0];
    ncnn::VkMat& text_output = top_blobs[1];
    video_output.create(inner_dim, total_video_tokens, scalar_elemsize, 1,
                        opt.blob_vkallocator);
    text_output.create(inner_dim, total_text_tokens, scalar_elemsize, 1,
                       opt.blob_vkallocator);
    if (video_output.empty() || text_output.empty())
        return -100;

    {
        std::vector<ncnn::VkMat> bindings(1);
        bindings[0] = video_output;
        std::vector<ncnn::vk_constant_type> constants(1);
        constants[0].i = inner_dim * total_video_tokens;
        ncnn::VkMat dispatcher;
        dispatcher.w = constants[0].i;
        dispatcher.h = 1;
        dispatcher.c = 1;
        cmd.record_pipeline(pipeline_zero, bindings, constants, dispatcher);

        bindings[0] = text_output;
        constants[0].i = inner_dim * total_text_tokens;
        dispatcher.w = constants[0].i;
        cmd.record_pipeline(pipeline_zero, bindings, constants, dispatcher);
    }

    std::vector<int> text_offsets(runtime_state->text_lengths.size(), 0);
    for (size_t sample = 1; sample < runtime_state->text_lengths.size(); sample++)
        text_offsets[sample] = text_offsets[sample - 1] + runtime_state->text_lengths[sample - 1];

    for (size_t window_index = 0; window_index < plan.windows.size(); window_index++)
    {
        const WindowSlice& window = plan.windows[window_index];
        const int sample = window.sample_index;
        const int text_length = runtime_state->text_lengths[sample];
        const int sequence_length = window.token_count + text_length;

        ncnn::VkMat query(head_dim, sequence_length, heads, scalar_elemsize, 1,
                          opt.workspace_vkallocator);
        ncnn::VkMat key(head_dim, sequence_length, heads, scalar_elemsize, 1,
                        opt.workspace_vkallocator);
        ncnn::VkMat value(head_dim, sequence_length, heads, scalar_elemsize, 1,
                          opt.workspace_vkallocator);
        if (query.empty() || key.empty() || value.empty())
            return -100;

        std::vector<ncnn::VkMat> prepare_bindings(split_qkv ? 13 : 9);
        if (split_qkv)
        {
            prepare_bindings[0] = bottom_blobs[0];
            prepare_bindings[1] = bottom_blobs[1];
            prepare_bindings[2] = bottom_blobs[2];
            prepare_bindings[3] = bottom_blobs[3];
            prepare_bindings[4] = bottom_blobs[4];
            prepare_bindings[5] = bottom_blobs[5];
            prepare_bindings[6] = norm_weights_gpu;
            prepare_bindings[7] = rope_freqs_gpu;
            prepare_bindings[8] = plan_indices_gpu;
            prepare_bindings[9] = plan_positions_gpu;
            prepare_bindings[10] = query;
            prepare_bindings[11] = key;
            prepare_bindings[12] = value;
        }
        else
        {
            prepare_bindings[0] = video_qkv;
            prepare_bindings[1] = text_qkv;
            prepare_bindings[2] = norm_weights_gpu;
            prepare_bindings[3] = rope_freqs_gpu;
            prepare_bindings[4] = plan_indices_gpu;
            prepare_bindings[5] = plan_positions_gpu;
            prepare_bindings[6] = query;
            prepare_bindings[7] = key;
            prepare_bindings[8] = value;
        }
        std::vector<ncnn::vk_constant_type> prepare_constants(12);
        prepare_constants[0].i = split_qkv ? inner_dim : inner_dim * 3;
        prepare_constants[1].i = inner_dim;
        prepare_constants[2].i = head_dim;
        prepare_constants[3].i = text_offsets[sample];
        prepare_constants[4].i = text_length;
        prepare_constants[5].i = window.partition_offset;
        prepare_constants[6].i = window.token_count;
        prepare_constants[7].i = sequence_length;
        prepare_constants[8].i = static_cast<int>(query.cstep);
        prepare_constants[9].i = rope_pairs_per_axis;
        prepare_constants[10].f = epsilon;
        prepare_constants[11].i = official_bf16_attention ? 1 : 0;
        ncnn::VkMat prepare_dispatcher;
        prepare_dispatcher.w = head_dim;
        prepare_dispatcher.h = sequence_length;
        prepare_dispatcher.c = heads;
        cmd.record_pipeline(split_qkv ? pipeline_prepare_qkv_split : pipeline_prepare_qkv,
                            prepare_bindings, prepare_constants, prepare_dispatcher);

        std::vector<ncnn::VkMat> sdpa_inputs(3);
        sdpa_inputs[0] = query;
        sdpa_inputs[1] = key;
        sdpa_inputs[2] = value;
        std::vector<ncnn::VkMat> sdpa_outputs(1);
        ret = sdpa_vulkan->forward(sdpa_inputs, sdpa_outputs, cmd, opt);
        if (ret != 0)
            return ret;

        const ncnn::VkMat& context = sdpa_outputs[0];
        std::vector<ncnn::VkMat> scatter_bindings(4);
        scatter_bindings[0] = context;
        scatter_bindings[1] = plan_indices_gpu;
        scatter_bindings[2] = video_output;
        scatter_bindings[3] = text_output;
        std::vector<ncnn::vk_constant_type> scatter_constants(9);
        scatter_constants[0].i = inner_dim;
        scatter_constants[1].i = head_dim;
        scatter_constants[2].i = text_offsets[sample];
        scatter_constants[3].i = window.partition_offset;
        scatter_constants[4].i = window.token_count;
        scatter_constants[5].i = sequence_length;
        scatter_constants[6].i = static_cast<int>(context.cstep);
        scatter_constants[7].f = 1.f / plan.sample_window_count[sample];
        scatter_constants[8].i = official_bf16_attention ? 1 : 0;
        ncnn::VkMat scatter_dispatcher;
        scatter_dispatcher.w = head_dim;
        scatter_dispatcher.h = sequence_length;
        scatter_dispatcher.c = heads;
        cmd.record_pipeline(pipeline_scatter, scatter_bindings, scatter_constants,
                            scatter_dispatcher);
    }
    return 0;
}
#endif

ncnn::Layer* adaptive_window_attention_layer_creator(void* userdata)
{
    return new AdaptiveWindowAttention(
        static_cast<AdaptiveWindowRuntimeState*>(userdata));
}

} // namespace seedvr2
