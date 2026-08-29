#include "video/ffmpeg_video_reader.h"

#include <climits>
#include <cstdint>
#include <vector>

#if defined(SEEDVR2_HAS_FFMPEG)
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}
#endif

namespace seedvr2
{

#if defined(SEEDVR2_HAS_FFMPEG)
namespace
{

std::string ffmpeg_error(const char* operation, int code)
{
    char message[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(code, message, sizeof(message));
    return std::string(operation) + " failed: " + message;
}

int positive_dimension(int value)
{
    return value > 0 ? value : 0;
}

} // namespace

class FfmpegVideoReader::Impl final
{
public:
    ~Impl()
    {
        if (sws_context)
            sws_freeContext(sws_context);
        av_frame_free(&decoded_frame);
        av_packet_free(&packet);
        avcodec_free_context(&codec_context);
        avformat_close_input(&format_context);
    }

    bool initialize(const std::filesystem::path& path, std::string& error)
    {
        error.clear();
        int result = avformat_open_input(&format_context, path.string().c_str(), nullptr, nullptr);
        if (result < 0)
        {
            error = ffmpeg_error("avformat_open_input", result);
            return false;
        }
        result = avformat_find_stream_info(format_context, nullptr);
        if (result < 0)
        {
            error = ffmpeg_error("avformat_find_stream_info", result);
            return false;
        }

        video_stream_index = av_find_best_stream(format_context, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (video_stream_index < 0)
        {
            error = ffmpeg_error("av_find_best_stream", video_stream_index);
            return false;
        }
        AVStream* stream = format_context->streams[video_stream_index];
        if (!stream || !stream->codecpar)
        {
            error = "FFmpeg video stream has no codec parameters";
            return false;
        }

        codec_context = avcodec_alloc_context3(nullptr);
        if (!codec_context)
        {
            error = "avcodec_alloc_context3 failed";
            return false;
        }
        result = avcodec_parameters_to_context(codec_context, stream->codecpar);
        if (result < 0)
        {
            error = ffmpeg_error("avcodec_parameters_to_context", result);
            return false;
        }
        const AVCodec* decoder = avcodec_find_decoder(codec_context->codec_id);
        if (!decoder)
        {
            error = "avcodec_find_decoder failed for the input video codec";
            return false;
        }
        result = avcodec_open2(codec_context, decoder, nullptr);
        if (result < 0)
        {
            error = ffmpeg_error("avcodec_open2", result);
            return false;
        }

        info.width = positive_dimension(codec_context->width);
        info.height = positive_dimension(codec_context->height);
        if (info.width <= 0 || info.height <= 0)
        {
            error = "FFmpeg video stream has invalid dimensions";
            return false;
        }
        info.frame_count = stream->nb_frames > 0 && stream->nb_frames <= INT_MAX
                               ? static_cast<int>(stream->nb_frames)
                               : 0;
        const AVRational frame_rate = stream->avg_frame_rate.num > 0 && stream->avg_frame_rate.den > 0
                                          ? stream->avg_frame_rate
                                          : stream->r_frame_rate;
        info.fps_num = frame_rate.num > 0 ? frame_rate.num : 0;
        info.fps_den = frame_rate.den > 0 ? frame_rate.den : 1;

        decoded_frame = av_frame_alloc();
        packet = av_packet_alloc();
        if (!decoded_frame || !packet)
        {
            error = "FFmpeg frame or packet allocation failed";
            return false;
        }
        sws_context = sws_getContext(info.width, info.height, codec_context->pix_fmt, info.width, info.height,
                                     AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_context)
        {
            error = "sws_getContext failed for RGB24 conversion";
            return false;
        }
        rgb_pixels.resize(static_cast<std::size_t>(info.width) * static_cast<std::size_t>(info.height) * 3u);
        return true;
    }

    bool convert_frame(RgbImage& output, std::string& error)
    {
        if (!decoded_frame || decoded_frame->width != info.width || decoded_frame->height != info.height)
        {
            error = "FFmpeg decoder returned a frame with invalid dimensions";
            return false;
        }
        sws_context = sws_getCachedContext(sws_context, info.width, info.height,
                                            static_cast<AVPixelFormat>(decoded_frame->format), info.width,
                                            info.height, AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws_context)
        {
            error = "sws_getCachedContext failed for RGB24 conversion";
            return false;
        }
        std::uint8_t* destination[] = {rgb_pixels.data()};
        int destination_stride[] = {info.width * 3};
        const int rows = sws_scale(sws_context, decoded_frame->data, decoded_frame->linesize, 0, info.height,
                                   destination, destination_stride);
        if (rows != info.height)
        {
            error = "sws_scale failed for RGB24 conversion";
            return false;
        }
        output.width = info.width;
        output.height = info.height;
        output.pixels = rgb_pixels;
        return true;
    }

    bool read_next(RgbImage& output, std::string& error)
    {
        output = RgbImage();
        error.clear();
        if (finished)
            return false;

        for (;;)
        {
            const int receive_result = avcodec_receive_frame(codec_context, decoded_frame);
            if (receive_result == 0)
                return convert_frame(output, error);
            if (receive_result != AVERROR(EAGAIN) && receive_result != AVERROR_EOF)
            {
                error = ffmpeg_error("avcodec_receive_frame", receive_result);
                return false;
            }

            if (input_eof)
            {
                if (!flush_sent)
                {
                    const int flush_result = avcodec_send_packet(codec_context, nullptr);
                    if (flush_result < 0 && flush_result != AVERROR_EOF)
                    {
                        error = ffmpeg_error("avcodec_send_packet(flush)", flush_result);
                        return false;
                    }
                    flush_sent = true;
                    continue;
                }
                finished = true;
                return false;
            }

            const int read_result = av_read_frame(format_context, packet);
            if (read_result == AVERROR_EOF)
            {
                input_eof = true;
                continue;
            }
            if (read_result < 0)
            {
                error = ffmpeg_error("av_read_frame", read_result);
                return false;
            }
            if (packet->stream_index != video_stream_index)
            {
                av_packet_unref(packet);
                continue;
            }
            const int send_result = avcodec_send_packet(codec_context, packet);
            av_packet_unref(packet);
            if (send_result < 0 && send_result != AVERROR(EAGAIN))
            {
                error = ffmpeg_error("avcodec_send_packet", send_result);
                return false;
            }
        }
    }

    VideoInfo info;
    AVFormatContext* format_context = nullptr;
    AVCodecContext* codec_context = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* decoded_frame = nullptr;
    SwsContext* sws_context = nullptr;
    std::vector<std::uint8_t> rgb_pixels;
    int video_stream_index = -1;
    bool input_eof = false;
    bool flush_sent = false;
    bool finished = false;
};
#else

class FfmpegVideoReader::Impl final
{
public:
    VideoInfo info;
};

#endif

FfmpegVideoReader::FfmpegVideoReader() = default;

FfmpegVideoReader::~FfmpegVideoReader() = default;

FfmpegVideoReader::FfmpegVideoReader(FfmpegVideoReader&&) noexcept = default;

FfmpegVideoReader& FfmpegVideoReader::operator=(FfmpegVideoReader&&) noexcept = default;

bool FfmpegVideoReader::open(const std::filesystem::path& path, FfmpegVideoReader& reader, std::string& error)
{
    reader.impl_.reset();
    error.clear();
#if defined(SEEDVR2_HAS_FFMPEG)
    auto implementation = std::make_unique<Impl>();
    if (!implementation->initialize(path, error))
        return false;
    reader.impl_ = std::move(implementation);
    return true;
#else
    (void)path;
    error = "compressed video input requires a build with SEEDVR2_ENABLE_FFMPEG=ON";
    return false;
#endif
}

const VideoInfo& FfmpegVideoReader::info() const
{
    static const VideoInfo empty_info;
    return impl_ ? impl_->info : empty_info;
}

bool FfmpegVideoReader::read_next(RgbImage& frame, std::string& error)
{
#if defined(SEEDVR2_HAS_FFMPEG)
    if (!impl_)
    {
        frame = RgbImage();
        error = "FFmpeg video reader is not open";
        return false;
    }
    return impl_->read_next(frame, error);
#else
    frame = RgbImage();
    error = "compressed video input requires a build with SEEDVR2_ENABLE_FFMPEG=ON";
    return false;
#endif
}

} // namespace seedvr2
