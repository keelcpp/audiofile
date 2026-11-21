#include "audio_internal.h"

namespace keel::audiofile {

bool is_au_bytes(ArrayRef<u8> bytes) noexcept {
    if (bytes.size() < 4) return false;
    return mem_eq(bytes.data(), ".snd", 4);
}

namespace {
namespace au_detail {

struct AuEncodingSpec {
    u16 bits_per_sample = 0;
    usize bytes_per_sample = 0;
    AudioSampleType native_type = AudioSampleType::F64;
    StringRef label{};
    bool supported = false;
};

[[nodiscard]] AuEncodingSpec describe_au_encoding(u32 encoding) noexcept {
    switch (encoding) {
        case 1:
            return {8, 1, AudioSampleType::I16, "Mu-law", true};
        case 2:
            return {8, 1, AudioSampleType::I8, "Linear 8-bit PCM", true};
        case 3:
            return {16, 2, AudioSampleType::I16, "Linear 16-bit PCM", true};
        case 4:
            return {24, 3, AudioSampleType::I32, "Linear 24-bit PCM", true};
        case 5:
            return {32, 4, AudioSampleType::I32, "Linear 32-bit PCM", true};
        case 6:
            return {32, 4, AudioSampleType::F32, "IEEE 32-bit Float", true};
        case 7:
            return {64, 8, AudioSampleType::F64, "IEEE 64-bit Float", true};
        case 27:
            return {8, 1, AudioSampleType::I16, "A-law", true};
        default:
            break;
    }
    return {0, 2, AudioSampleType::F64, {}, false};
}

[[nodiscard]] String parse_au_annotation(const u8* data, u32 data_offset, Allocator* alloc) {
    if (data_offset <= 24u) {
        return String(alloc);
    }
    const u8* ptr = data + 24;
    usize annot_len = static_cast<usize>(data_offset - 24u);
    for (usize i = 0; i < annot_len; ++i) {
        if (ptr[i] == 0u) {
            annot_len = i;
            break;
        }
    }
    return audio_detail::make_trimmed_string(ptr, annot_len, alloc);
}

} // namespace au_detail
} // namespace

Result<AudioInfo> peek_au(ArrayRef<u8> bytes, Allocator* alloc) {
    if (bytes.size() < 4) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }
    if (!is_au_bytes(bytes)) [[unlikely]] {
        return ErrorCode::InvalidData;
    }
    if (bytes.size() < 24) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }

    Allocator* dst_alloc = alloc ? alloc : default_allocator();
    const u8* d = bytes.data();
    const usize len = bytes.size();

    const u32 data_offset = audio_detail::read_u32_be(d + 4);
    const u32 data_size = audio_detail::read_u32_be(d + 8);
    const u32 encoding = audio_detail::read_u32_be(d + 12);
    const u32 sample_rate = audio_detail::read_u32_be(d + 16);
    const u32 num_channels = audio_detail::read_u32_be(d + 20);

    if (data_offset < 24u || num_channels == 0u || num_channels > 0xFFFFu || sample_rate == 0u) [[unlikely]] {
        return ErrorCode::InvalidData;
    }
    if (data_offset > len) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }

    const au_detail::AuEncodingSpec spec = au_detail::describe_au_encoding(encoding);

    AudioInfo info;
    info.format = AudioFormat::Au;
    info.channels = num_channels;
    info.sample_rate = static_cast<f64>(sample_rate);
    info.bits_per_sample = spec.bits_per_sample;
    if (spec.supported) {
        info.compression_method = String(spec.label, dst_alloc);
    } else {
        info.compression_method = audio_detail::format_label_u32("Encoding ", encoding, dst_alloc);
    }
    info.metadata.comment = au_detail::parse_au_annotation(d, data_offset, dst_alloc);

    const u64 actual_data_bytes = (data_size != 0xFFFFFFFFu && data_size > 0u)
                                      ? static_cast<u64>(data_size)
                                      : static_cast<u64>(len - static_cast<usize>(data_offset));
    const u64 block_align = static_cast<u64>(num_channels) * static_cast<u64>(spec.bytes_per_sample);
    if (block_align > 0ULL) {
        info.total_frames = actual_data_bytes / block_align;
        info.duration_sec = static_cast<f64>(info.total_frames) / info.sample_rate;
        if (info.duration_sec > 0.0 && len > 0) {
            info.bit_rate_kbps = static_cast<u32>((static_cast<f64>(len) * 8.0) / (info.duration_sec * 1000.0));
        }
    }

    return info;
}

Result<AudioBuffer> decode_au(ArrayRef<u8> bytes, const AudioDecodeOptions& opts) {
    if (bytes.size() < 4) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }
    if (!is_au_bytes(bytes)) [[unlikely]] {
        return ErrorCode::InvalidData;
    }
    if (bytes.size() < 24) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }

    Allocator* dst_alloc = opts.allocator ? opts.allocator : default_allocator();
    const u8* d = bytes.data();
    const usize len = bytes.size();

    const u32 data_offset = audio_detail::read_u32_be(d + 4);
    const u32 data_size = audio_detail::read_u32_be(d + 8);
    const u32 encoding = audio_detail::read_u32_be(d + 12);
    const u32 sample_rate = audio_detail::read_u32_be(d + 16);
    const u32 num_channels_u32 = audio_detail::read_u32_be(d + 20);

    if (data_offset < 24u || num_channels_u32 == 0u || num_channels_u32 > 0xFFFFu || sample_rate == 0u) [[unlikely]] {
        return ErrorCode::InvalidData;
    }
    if (data_offset > len) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }

    const au_detail::AuEncodingSpec spec = au_detail::describe_au_encoding(encoding);
    if (!spec.supported) [[unlikely]] {
        return ErrorCode::NotSupported;
    }

    const u16 num_channels = static_cast<u16>(num_channels_u32);
    const usize block_align = static_cast<usize>(num_channels) * spec.bytes_per_sample;
    const usize remaining = len - static_cast<usize>(data_offset);

    const u64 declared_bytes =
        (data_size != 0xFFFFFFFFu) ? static_cast<u64>(data_size) : static_cast<u64>(remaining);
    const u64 available_frames = declared_bytes / static_cast<u64>(block_align);

    KEEL_TRY_ASSIGN(const audio_detail::FrameWindow win, audio_detail::compute_frame_window(available_frames, opts));

    const AudioSampleType out_type = opts.native_sample_type ? spec.native_type : AudioSampleType::F64;

    AudioBuffer res;
    res.sample_rate = static_cast<f64>(sample_rate);
    res.bits_per_sample = spec.bits_per_sample;
    res.channels = num_channels;
    res.frames = win.count;
    res.total_source_frames = available_frames;
    res.sample_type = out_type;
    res.layout = opts.layout;
    res.metadata.comment = au_detail::parse_au_annotation(d, data_offset, dst_alloc);

    if (win.count == 0) {
        res.data = Array<u8>(dst_alloc);
        return res;
    }

    const u64 end_frame = win.start + win.count;
    if (end_frame > static_cast<u64>(remaining) / static_cast<u64>(block_align)) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }

    KEEL_TRY_ASSIGN(res.data, audio_detail::alloc_sample_buffer(win.count, num_channels, out_type, dst_alloc));

    const u8* raw_samples = d + static_cast<usize>(data_offset) + static_cast<usize>(win.start) * block_align;
    const u64 nframes = win.count;
    const usize bps_bytes = spec.bytes_per_sample;

    if (!opts.native_sample_type) {
        f64* dst = res.samples_f64().data();
        for (u64 t = 0; t < nframes; ++t) {
            const u8* frame = raw_samples + static_cast<usize>(t) * block_align;
            for (u16 ch = 0; ch < num_channels; ++ch) {
                const u8* s = frame + static_cast<usize>(ch) * bps_bytes;
                f64 val = 0.0;
                switch (encoding) {
                    case 1:
                        val = static_cast<f64>(audio_detail::decode_mu_law(*s)) / 32768.0;
                        break;
                    case 2:
                        val = static_cast<f64>(static_cast<i8>(*s)) / 128.0;
                        break;
                    case 3:
                        val = static_cast<f64>(audio_detail::read_i16_be(s)) / 32768.0;
                        break;
                    case 4:
                        val = static_cast<f64>(audio_detail::read_i24_be(s)) / 8388608.0;
                        break;
                    case 5:
                        val = static_cast<f64>(audio_detail::read_i32_be(s)) / 2147483648.0;
                        break;
                    case 6:
                        val = static_cast<f64>(audio_detail::read_f32_be(s));
                        break;
                    case 7:
                        val = audio_detail::read_f64_be(s);
                        break;
                    case 27:
                        val = static_cast<f64>(audio_detail::decode_a_law(*s)) / 32768.0;
                        break;
                    default:
                        break;
                }
                dst[audio_detail::sample_index(t, ch, nframes, num_channels, opts.layout)] = val;
            }
        }
    } else if (out_type == AudioSampleType::I8) {
        i8* dst = res.samples_i8().data();
        for (u64 t = 0; t < nframes; ++t) {
            const u8* frame = raw_samples + static_cast<usize>(t) * block_align;
            for (u16 ch = 0; ch < num_channels; ++ch) {
                dst[audio_detail::sample_index(t, ch, nframes, num_channels, opts.layout)] = static_cast<i8>(frame[ch]);
            }
        }
    } else if (out_type == AudioSampleType::I16) {
        i16* dst = res.samples_i16().data();
        for (u64 t = 0; t < nframes; ++t) {
            const u8* frame = raw_samples + static_cast<usize>(t) * block_align;
            for (u16 ch = 0; ch < num_channels; ++ch) {
                i16 val = 0;
                if (encoding == 1u) {
                    val = audio_detail::decode_mu_law(frame[ch]);
                } else if (encoding == 27u) {
                    val = audio_detail::decode_a_law(frame[ch]);
                } else {
                    val = audio_detail::read_i16_be(frame + static_cast<usize>(ch) * 2);
                }
                dst[audio_detail::sample_index(t, ch, nframes, num_channels, opts.layout)] = val;
            }
        }
    } else if (out_type == AudioSampleType::I32) {
        i32* dst = res.samples_i32().data();
        for (u64 t = 0; t < nframes; ++t) {
            const u8* frame = raw_samples + static_cast<usize>(t) * block_align;
            for (u16 ch = 0; ch < num_channels; ++ch) {
                const i32 val = (encoding == 4u)
                                    ? audio_detail::read_i24_be(frame + static_cast<usize>(ch) * 3)
                                    : audio_detail::read_i32_be(frame + static_cast<usize>(ch) * 4);
                dst[audio_detail::sample_index(t, ch, nframes, num_channels, opts.layout)] = val;
            }
        }
    } else if (out_type == AudioSampleType::F32) {
        f32* dst = res.samples_f32().data();
        for (u64 t = 0; t < nframes; ++t) {
            const u8* frame = raw_samples + static_cast<usize>(t) * block_align;
            for (u16 ch = 0; ch < num_channels; ++ch) {
                dst[audio_detail::sample_index(t, ch, nframes, num_channels, opts.layout)] =
                    audio_detail::read_f32_be(frame + static_cast<usize>(ch) * 4);
            }
        }
    } else {
        f64* dst = res.samples_f64().data();
        for (u64 t = 0; t < nframes; ++t) {
            const u8* frame = raw_samples + static_cast<usize>(t) * block_align;
            for (u16 ch = 0; ch < num_channels; ++ch) {
                dst[audio_detail::sample_index(t, ch, nframes, num_channels, opts.layout)] =
                    audio_detail::read_f64_be(frame + static_cast<usize>(ch) * 8);
            }
        }
    }

    return res;
}

Result<Array<u8>> encode_au(const AudioView& view, const AudioEncodeOptions& opts) {
    const ErrorCode v_ec = audio_detail::validate_encode_view(view);
    if (v_ec != ErrorCode::Ok) [[unlikely]] {
        return v_ec;
    }

    u16 bps = 16;
    bool use_float = false;
    const ErrorCode f_ec = audio_detail::resolve_encode_format(view, opts, bps, use_float);
    if (f_ec != ErrorCode::Ok) [[unlikely]] {
        return f_ec;
    }

    const f64 sr_rounded = std::round(view.sample_rate);
    if (!(sr_rounded >= 1.0 && sr_rounded <= static_cast<f64>(0xFFFFFFFFu))) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }
    const u32 sample_rate_u32 = static_cast<u32>(sr_rounded);

    u32 encoding = 3;
    usize bytes_per_sample = 2;
    if (use_float) {
        if (bps == 32) {
            encoding = 6;
            bytes_per_sample = 4;
        } else {
            encoding = 7;
            bytes_per_sample = 8;
        }
    } else if (bps == 8) {
        encoding = 2;
        bytes_per_sample = 1;
    } else if (bps == 16) {
        encoding = 3;
        bytes_per_sample = 2;
    } else if (bps == 24) {
        encoding = 4;
        bytes_per_sample = 3;
    } else {
        encoding = 5;
        bytes_per_sample = 4;
    }

    if (opts.comment.size() > 0xFFFFFFD0u) [[unlikely]] {
        return ErrorCode::OutOfRange;
    }
    usize header_size = 24u + opts.comment.size();
    if ((header_size & 7u) != 0u) {
        header_size += 8u - (header_size & 7u);
    }

    const u64 block_align = static_cast<u64>(view.channels) * static_cast<u64>(bytes_per_sample);
    if (view.frames > 0xFFFFFFFFULL / block_align) [[unlikely]] {
        return ErrorCode::OutOfRange;
    }
    const u64 data_size_u64 = view.frames * block_align;
    if (data_size_u64 > static_cast<u64>(~usize{0}) - static_cast<u64>(header_size)) [[unlikely]] {
        return ErrorCode::OutOfRange;
    }

    Allocator* dst_alloc = opts.allocator ? opts.allocator : default_allocator();
    Array<u8> out = Array<u8>::with_capacity(header_size + static_cast<usize>(data_size_u64), dst_alloc);

    audio_detail::write_fourcc(out, ".snd");
    audio_detail::write_u32_be(out, static_cast<u32>(header_size));
    audio_detail::write_u32_be(out, static_cast<u32>(data_size_u64));
    audio_detail::write_u32_be(out, encoding);
    audio_detail::write_u32_be(out, sample_rate_u32);
    audio_detail::write_u32_be(out, static_cast<u32>(view.channels));

    if (!opts.comment.empty()) {
        out.append(ArrayRef<u8>(reinterpret_cast<const u8*>(opts.comment.data()), opts.comment.size()));
    }
    while (out.size() < header_size) {
        out.push_back(0u);
    }

    for (u64 t = 0; t < view.frames; ++t) {
        for (u16 ch = 0; ch < view.channels; ++ch) {
            const usize idx = audio_detail::sample_index(t, ch, view.frames, view.channels, view.layout);
            if (use_float) {
                if (bps == 32) {
                    audio_detail::write_f32_be(out, audio_detail::view_sample_f32(view, idx));
                } else {
                    audio_detail::write_f64_be(out, audio_detail::view_sample_f64(view, idx));
                }
            } else if (bps == 8) {
                out.push_back(static_cast<u8>(audio_detail::view_sample_i8_pcm(view, idx)));
            } else if (bps == 16) {
                audio_detail::write_u16_be(out, static_cast<u16>(audio_detail::view_sample_i16_pcm(view, idx)));
            } else if (bps == 24) {
                audio_detail::write_i24_be(out, audio_detail::view_sample_i24_pcm(view, idx));
            } else {
                audio_detail::write_u32_be(out, static_cast<u32>(audio_detail::view_sample_i32_pcm(view, idx)));
            }
        }
    }

    return out;
}

} // namespace keel::audiofile
