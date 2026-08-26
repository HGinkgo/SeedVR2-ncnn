#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "awa/awa_layers.h"

namespace
{

void require(bool condition, const char* message)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

std::uint64_t fnv1a(const ncnn::Mat& values)
{
    std::uint64_t hash = 1469598103934665603ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    for (int channel = 0; channel < values.c; channel++)
        for (int row = 0; row < values.h; row++)
            for (int feature = 0; feature < values.w; feature++)
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, values.channel(channel).row(row) + feature, sizeof(bits));
        for (int byte = 0; byte < 4; byte++)
        {
            hash ^= (bits >> (byte * 8)) & 0xffU;
            hash *= prime;
        }
    }
    return hash;
}

void run_case(bool shifted)
{
    constexpr int source_tokens = 2 * 19 * 23;
    constexpr int text_tokens = 5;
    constexpr int heads = 2;
    constexpr int head_dim = 3;

    ncnn::Mat video(head_dim, heads, 3, source_tokens, 4u, 1, 1);
    ncnn::Mat text(head_dim, heads, 3, text_tokens, 4u, 1, 1);
    for (int token = 0; token < source_tokens; token++)
    {
        float* token_data = static_cast<float*>(video.channel(token).data);
        for (int qkv = 0; qkv < 3 * heads; qkv++)
        {
            for (int feature = 0; feature < head_dim; feature++)
                token_data[qkv * head_dim + feature] =
                    token * 1000.f + qkv * 100.f + feature;
        }
    }
    for (int token = 0; token < text_tokens; token++)
    {
        float* token_data = static_cast<float*>(text.channel(token).data);
        for (int qkv = 0; qkv < 3 * heads; qkv++)
        {
            for (int feature = 0; feature < head_dim; feature++)
                token_data[qkv * head_dim + feature] =
                    100000.f + token * 1000.f + qkv * 100.f + feature;
        }
    }

    ncnn::ParamDict params;
    params.set(0, 2);
    params.set(1, 19);
    params.set(2, 23);
    params.set(3, 4);
    params.set(4, 3);
    params.set(5, 3);
    params.set(6, text_tokens);
    params.set(7, shifted ? 1 : 0);

    SeedVR2AWAPack pack;
    require(pack.load_param(params) == 0, "pack parameters load");
    std::vector<ncnn::Mat> packed_outputs;
    ncnn::Option option;
    require(pack.forward({video, text}, packed_outputs, option) == 0, "pack forward");
    require(packed_outputs.size() == 2, "pack output count");
    const ncnn::Mat& packed = packed_outputs[0];
    const ncnn::Mat& cu_seqlens = packed_outputs[1];
    require(packed.w == head_dim && packed.h == 3 * heads && packed.c == 894,
            "packed shape");
    const std::uint64_t expected_hash = shifted ? 0xcec5dfa15e6d23f0ULL : 0xcc6a100597e1721cULL;
    const std::uint64_t actual_hash = fnv1a(packed);
    require(actual_hash == expected_hash, "packed sequence matches PyTorch reference");
    const int expected_cu[] = {0, 214, 428, 661, 894};
    const int expected_cu_unshifted[] = {0, 423, 846, 870, 894};
    const int* expected = shifted ? expected_cu : expected_cu_unshifted;
    require(cu_seqlens.w == 5, "cu_seqlens shape");
    for (int i = 0; i < 5; i++)
        require(static_cast<int>(cu_seqlens[i]) == expected[i], "cu_seqlens value");

    require(packed.channel(0).row(0)[0] == static_cast<const float*>(video.channel(0).data)[0],
            "first video token is packed");
    require(packed.channel(expected[1] - text_tokens).row(0)[0] ==
                static_cast<const float*>(text.channel(0).data)[0],
            "text sequence is appended to each window");

    ncnn::Mat video_batched(head_dim, heads, 3, static_cast<size_t>(4u), 1, source_tokens);
    ncnn::Mat text_batched(head_dim, heads, 3, static_cast<size_t>(4u), 1, text_tokens);
    for (int token = 0; token < source_tokens; token++)
        for (int qkv = 0; qkv < 3; qkv++)
            std::memcpy(video_batched.batch(token).channel(qkv).data,
                        video.channel(token).depth(qkv).data,
                        static_cast<size_t>(heads * head_dim) * sizeof(float));
    for (int token = 0; token < text_tokens; token++)
        for (int qkv = 0; qkv < 3; qkv++)
            std::memcpy(text_batched.batch(token).channel(qkv).data,
                        text.channel(token).depth(qkv).data,
                        static_cast<size_t>(heads * head_dim) * sizeof(float));

    std::vector<ncnn::Mat> batched_pack_outputs;
    require(pack.forward({video_batched, text_batched}, batched_pack_outputs, option) == 0,
            "pack accepts token-batch QKV inputs");
    require(batched_pack_outputs.size() == 2 && fnv1a(batched_pack_outputs[0]) == expected_hash,
            "token-batch pack sequence matches channel-token layout");

    SeedVR2AWAUnpack unpack;
    require(unpack.load_param(params) == 0, "unpack parameters load");
    ncnn::Mat attended(head_dim, heads, packed.c);
    for (int token = 0; token < packed.c; token++)
    {
        for (int feature = 0; feature < head_dim; feature++)
            attended.channel(token).row(0)[feature] = packed.channel(token).row(0)[feature];
        for (int feature = 0; feature < head_dim; feature++)
            attended.channel(token).row(1)[feature] = packed.channel(token).row(1)[feature];
    }
    std::vector<ncnn::Mat> unpacked_outputs;
    require(unpack.forward({attended}, unpacked_outputs, option) == 0, "unpack forward");
    require(unpacked_outputs.size() == 2, "unpack output count");
    const ncnn::Mat& unpacked_video = unpacked_outputs[0];
    const ncnn::Mat& unpacked_text = unpacked_outputs[1];
    require(unpacked_video.w == head_dim && unpacked_video.h == heads && unpacked_video.d == 23 &&
                unpacked_video.c == 19 && unpacked_video.n == 2,
            "unpacked video shape");
    require(unpacked_text.w == head_dim && unpacked_text.h == heads && unpacked_text.c == text_tokens,
            "unpacked text shape");
    for (int token = 0; token < source_tokens; token++)
        for (int head = 0; head < heads; head++)
            for (int feature = 0; feature < head_dim; feature++)
                require(static_cast<const float*>(unpacked_video.batch(token / (19 * 23)).channel((token / 23) % 19).depth(token % 23).data)[head * head_dim + feature] ==
                            static_cast<const float*>(video.channel(token).data)[head * head_dim + feature],
                        "video inverse mapping");
    for (int token = 0; token < text_tokens; token++)
        for (int head = 0; head < heads; head++)
            for (int feature = 0; feature < head_dim; feature++)
                require(static_cast<const float*>(unpacked_text.channel(token).data)[head * head_dim + feature] ==
                            static_cast<const float*>(text.channel(token).data)[head * head_dim + feature],
                        "text aggregation");
}

} // namespace

int main()
{
    run_case(false);
    run_case(true);
    std::puts("seedvr2-awa-test: ok");
    return 0;
}
