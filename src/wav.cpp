#include "audio_internal.h"

#include "keel/arena.h"

namespace keel::audiofile {

bool is_wav_bytes(ArrayRef<u8> bytes) noexcept {
    if (bytes.size() < 12) return false;
    const u8* d = bytes.data();
    return d[0] == 'R' && d[1] == 'I' && d[2] == 'F' && d[3] == 'F' && d[8] == 'W' && d[9] == 'A' && d[10] == 'V' &&
           d[11] == 'E';
}

namespace {
namespace wav_detail {

struct RiffChunk {
    usize offset = 0;
    u32 size = 0;
    bool found = false;
};

struct WavChunks {
    RiffChunk fmt{};
    RiffChunk data{};
    RiffChunk list_info{};
};

WavChunks scan_wav_chunks(const u8* data, usize len) noexcept {
    WavChunks out{};
    usize pos = 12;
    while (pos + 8 <= len) {
        const u8* hdr = data + pos;
        const u32 csize = audio_detail::read_u32_le(hdr + 4);
        const usize payload = pos + 8;

        if (!out.fmt.found && mem_eq(hdr, "fmt ", 4)) {
            out.fmt = {payload, csize, true};
        } else if (!out.data.found && mem_eq(hdr, "data", 4)) {
            out.data = {payload, csize, true};
        } else if (!out.list_info.found && mem_eq(hdr, "LIST", 4)) {
            if (csize >= 4 && csize <= len - payload && mem_eq(data + payload, "INFO", 4)) {
                out.list_info = {payload, csize, true};
            }
        }

        usize advance = static_cast<usize>(csize);
        if ((advance & 1u) != 0u) {
            if (advance == ~usize{0}) break;
            ++advance;
        }
        if (advance > len - payload) {
            break;
        }
        pos = payload + advance;
    }
    return out;
}

void parse_list_info(const u8* data, usize len, const RiffChunk& list_chunk, AudioMetadata& meta, Allocator* alloc) {
    if (!list_chunk.found || list_chunk.size < 4 || list_chunk.size > len - list_chunk.offset) {
        return;
    }
    const usize end = list_chunk.offset + static_cast<usize>(list_chunk.size);
    if (!mem_eq(data + list_chunk.offset, "INFO", 4)) {
        return;
    }

    usize pos = list_chunk.offset + 4;
    while (pos + 8 <= end) {
        const u8* tag = data + pos;
        const u32 sub_size = audio_detail::read_u32_le(data + pos + 4);
        pos += 8;
        if (sub_size > end - pos) {
            break;
        }

        if (mem_eq(tag, "INAM", 4)) {
            meta.title = audio_detail::make_trimmed_string(data + pos, sub_size, alloc);
        } else if (mem_eq(tag, "IART", 4)) {
            meta.artist = audio_detail::make_trimmed_string(data + pos, sub_size, alloc);
        } else if (mem_eq(tag, "ICMT", 4) || mem_eq(tag, "ICRD", 4)) {
            if (meta.comment.empty() || mem_eq(tag, "ICMT", 4)) {
                meta.comment = audio_detail::make_trimmed_string(data + pos, sub_size, alloc);
            }
        }

        usize step = static_cast<usize>(sub_size);
        if ((step & 1u) != 0u && step < end - pos) {
            ++step;
        }
        pos += step;
    }
}

[[nodiscard]] bool is_supported_wav_encoding(u16 format_tag, u16 bps) noexcept {
    if (format_tag == 1) {
        return bps == 8 || bps == 16 || bps == 24 || bps == 32;
    }
    if (format_tag == 3) {
        return bps == 32 || bps == 64;
    }
    if (format_tag == 6 || format_tag == 7) {
        return bps == 8;
    }
    return false;
}

[[nodiscard]] AudioSampleType wav_native_sample_type(u16 format_tag, u16 bps) noexcept {
    if (format_tag == 3) {
        return (bps == 64) ? AudioSampleType::F64 : AudioSampleType::F32;
    }
    if (format_tag == 6 || format_tag == 7) {
        return AudioSampleType::I16;
    }
    if (bps == 8) return AudioSampleType::U8;
    if (bps == 16) return AudioSampleType::I16;
    return AudioSampleType::I32;
}

} // namespace wav_detail
} // namespace

Result<AudioInfo> peek_wav(ArrayRef<u8> bytes, Allocator* alloc) {
    if (bytes.size() < 12) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }
    if (!is_wav_bytes(bytes)) [[unlikely]] {
        return ErrorCode::InvalidData;
    }

    Allocator* dst_alloc = alloc ? alloc : default_allocator();
    const u8* d = bytes.data();
    const usize len = bytes.size();
    const wav_detail::WavChunks chunks = wav_detail::scan_wav_chunks(d, len);

    if (!chunks.fmt.found || chunks.fmt.size < 16) [[unlikely]] {
        return ErrorCode::InvalidData;
    }
    if (chunks.fmt.size > len - chunks.fmt.offset) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }

    const u8* f = d + chunks.fmt.offset;
    u16 format_tag = audio_detail::read_u16_le(f);
    const u16 num_channels = audio_detail::read_u16_le(f + 2);
    const u32 sample_rate = audio_detail::read_u32_le(f + 4);
    const u32 byte_rate = audio_detail::read_u32_le(f + 8);
    u16 block_align = audio_detail::read_u16_le(f + 12);
    const u16 bits_per_sample = audio_detail::read_u16_le(f + 14);

    if (format_tag == 0xFFFEu && chunks.fmt.size >= 40) {
        format_tag = audio_detail::read_u16_le(f + 24);
    }
    if (num_channels == 0 || sample_rate == 0) [[unlikely]] {
        return ErrorCode::InvalidData;
    }

    AudioInfo info;
    info.format = AudioFormat::Wav;
    info.channels = num_channels;
    info.sample_rate = static_cast<f64>(sample_rate);
    info.bits_per_sample = bits_per_sample;
    info.bit_rate_kbps = static_cast<u32>((static_cast<u64>(byte_rate) * 8ULL) / 1000ULL);

    switch (format_tag) {
        case 1:
            info.compression_method = String("Uncompressed", dst_alloc);
            break;
        case 3:
            info.compression_method = String("IEEE Float", dst_alloc);
            break;
        case 6:
            info.compression_method = String("A-law", dst_alloc);
            break;
        case 7:
            info.compression_method = String("Mu-law", dst_alloc);
            break;
        default:
            info.compression_method = audio_detail::format_label_u32("Format ", format_tag, dst_alloc);
            break;
    }

    if (block_align == 0 && bits_per_sample > 0) {
        block_align = static_cast<u16>(num_channels * ((bits_per_sample + 7u) / 8u));
    }
    if (chunks.data.found && block_align > 0) {
        info.total_frames = static_cast<u64>(chunks.data.size) / static_cast<u64>(block_align);
        info.duration_sec = static_cast<f64>(info.total_frames) / info.sample_rate;
        if (info.bit_rate_kbps == 0 && info.duration_sec > 0.0) {
            info.bit_rate_kbps = static_cast<u32>((static_cast<f64>(len) * 8.0) / (info.duration_sec * 1000.0));
        }
    }

    wav_detail::parse_list_info(d, len, chunks.list_info, info.metadata, dst_alloc);
    return info;
}

Result<AudioBuffer> decode_wav(ArrayRef<u8> bytes, const AudioDecodeOptions& opts) {
    if (bytes.size() < 12) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }
    if (!is_wav_bytes(bytes)) [[unlikely]] {
        return ErrorCode::InvalidData;
    }

    Allocator* dst_alloc = opts.allocator ? opts.allocator : default_allocator();
    const u8* d = bytes.data();
    const usize len = bytes.size();
    const wav_detail::WavChunks chunks = wav_detail::scan_wav_chunks(d, len);

    if (!chunks.fmt.found || chunks.fmt.size < 16) [[unlikely]] {
        return ErrorCode::InvalidData;
    }
    if (chunks.fmt.size > len - chunks.fmt.offset) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }
    if (!chunks.data.found) [[unlikely]] {
        return ErrorCode::InvalidData;
    }

    const u8* f = d + chunks.fmt.offset;
    u16 format_tag = audio_detail::read_u16_le(f);
    const u16 num_channels = audio_detail::read_u16_le(f + 2);
    const u32 sample_rate = audio_detail::read_u32_le(f + 4);
    u16 block_align = audio_detail::read_u16_le(f + 12);
    const u16 bits_per_sample = audio_detail::read_u16_le(f + 14);

    if (format_tag == 0xFFFEu && chunks.fmt.size >= 40) {
        format_tag = audio_detail::read_u16_le(f + 24);
    }
    if (num_channels == 0 || sample_rate == 0) [[unlikely]] {
        return ErrorCode::InvalidData;
    }
    if (!wav_detail::is_supported_wav_encoding(format_tag, bits_per_sample)) [[unlikely]] {
        return ErrorCode::NotSupported;
    }

    const usize bytes_per_sample = static_cast<usize>((bits_per_sample + 7u) / 8u);
    const usize min_block_align = static_cast<usize>(num_channels) * bytes_per_sample;
    const usize frame_stride = (block_align >= min_block_align) ? static_cast<usize>(block_align) : min_block_align;

    const u64 available_frames = static_cast<u64>(chunks.data.size) / static_cast<u64>(frame_stride);
    KEEL_TRY_ASSIGN(const audio_detail::FrameWindow win, audio_detail::compute_frame_window(available_frames, opts));

    const AudioSampleType out_type =
        opts.native_sample_type ? wav_detail::wav_native_sample_type(format_tag, bits_per_sample) : AudioSampleType::F64;

    AudioBuffer res;
    res.sample_rate = static_cast<f64>(sample_rate);
    res.bits_per_sample = bits_per_sample;
    res.channels = num_channels;
    res.frames = win.count;
    res.total_source_frames = available_frames;
    res.sample_type = out_type;
    res.layout = opts.layout;
    wav_detail::parse_list_info(d, len, chunks.list_info, res.metadata, dst_alloc);

    if (win.count == 0) {
        res.data = Array<u8>(dst_alloc);
        return res;
    }

    if (chunks.data.offset > len) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }
    const u64 end_frame = win.start + win.count;
    if (end_frame > static_cast<u64>(len - chunks.data.offset) / static_cast<u64>(frame_stride)) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }

    KEEL_TRY_ASSIGN(res.data, audio_detail::alloc_sample_buffer(win.count, num_channels, out_type, dst_alloc));

    const u8* raw_samples = d + chunks.data.offset + static_cast<usize>(win.start) * frame_stride;
    const u64 nframes = win.count;

    if (!opts.native_sample_type) {
        f64* dst = res.samples_f64().data();
        for (u64 t = 0; t < nframes; ++t) {
            const u8* frame = raw_samples + static_cast<usize>(t) * frame_stride;
            for (u16 ch = 0; ch < num_channels; ++ch) {
                const u8* s = frame + static_cast<usize>(ch) * bytes_per_sample;
                f64 val = 0.0;
                if (format_tag == 1) {
                    if (bits_per_sample == 8) {
                        val = (static_cast<f64>(*s) - 128.0) / 128.0;
                    } else if (bits_per_sample == 16) {
                        val = static_cast<f64>(audio_detail::read_i16_le(s)) / 32768.0;
                    } else if (bits_per_sample == 24) {
                        val = static_cast<f64>(audio_detail::read_i24_le(s)) / 8388608.0;
                    } else if (bits_per_sample == 32) {
                        val = static_cast<f64>(audio_detail::read_i32_le(s)) / 2147483648.0;
                    }
                } else if (format_tag == 3) {
                    if (bits_per_sample == 32) {
                        val = static_cast<f64>(audio_detail::read_f32_le(s));
                    } else {
                        val = audio_detail::read_f64_le(s);
                    }
                } else if (format_tag == 6) {
                    val = static_cast<f64>(audio_detail::decode_a_law(*s)) / 32768.0;
                } else if (format_tag == 7) {
                    val = static_cast<f64>(audio_detail::decode_mu_law(*s)) / 32768.0;
                }
                dst[audio_detail::sample_index(t, ch, nframes, num_channels, opts.layout)] = val;
            }
        }
    } else if (out_type == AudioSampleType::F32) {
        f32* dst = res.samples_f32().data();
        for (u64 t = 0; t < nframes; ++t) {
            const u8* frame = raw_samples + static_cast<usize>(t) * frame_stride;
            for (u16 ch = 0; ch < num_channels; ++ch) {
                dst[audio_detail::sample_index(t, ch, nframes, num_channels, opts.layout)] =
                    audio_detail::read_f32_le(frame + static_cast<usize>(ch) * 4);
            }
        }
    } else if (out_type == AudioSampleType::F64) {
        f64* dst = res.samples_f64().data();
        for (u64 t = 0; t < nframes; ++t) {
            const u8* frame = raw_samples + static_cast<usize>(t) * frame_stride;
            for (u16 ch = 0; ch < num_channels; ++ch) {
                dst[audio_detail::sample_index(t, ch, nframes, num_channels, opts.layout)] =
                    audio_detail::read_f64_le(frame + static_cast<usize>(ch) * 8);
            }
        }
    } else if (out_type == AudioSampleType::U8) {
        u8* dst = res.samples_u8().data();
        for (u64 t = 0; t < nframes; ++t) {
            const u8* frame = raw_samples + static_cast<usize>(t) * frame_stride;
            for (u16 ch = 0; ch < num_channels; ++ch) {
                dst[audio_detail::sample_index(t, ch, nframes, num_channels, opts.layout)] = frame[ch];
            }
        }
    } else if (out_type == AudioSampleType::I16) {
        i16* dst = res.samples_i16().data();
        for (u64 t = 0; t < nframes; ++t) {
            const u8* frame = raw_samples + static_cast<usize>(t) * frame_stride;
            for (u16 ch = 0; ch < num_channels; ++ch) {
                i16 val = 0;
                if (format_tag == 6) {
                    val = audio_detail::decode_a_law(frame[ch]);
                } else if (format_tag == 7) {
                    val = audio_detail::decode_mu_law(frame[ch]);
                } else {
                    val = audio_detail::read_i16_le(frame + static_cast<usize>(ch) * 2);
                }
                dst[audio_detail::sample_index(t, ch, nframes, num_channels, opts.layout)] = val;
            }
        }
    } else {
        i32* dst = res.samples_i32().data();
        for (u64 t = 0; t < nframes; ++t) {
            const u8* frame = raw_samples + static_cast<usize>(t) * frame_stride;
            for (u16 ch = 0; ch < num_channels; ++ch) {
                const i32 val = (bits_per_sample == 24)
                                    ? audio_detail::read_i24_le(frame + static_cast<usize>(ch) * 3)
                                    : audio_detail::read_i32_le(frame + static_cast<usize>(ch) * 4);
                dst[audio_detail::sample_index(t, ch, nframes, num_channels, opts.layout)] = val;
            }
        }
    }

    return res;
}

Result<Array<u8>> encode_wav(const AudioView& view, const AudioEncodeOptions& opts) {
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

    const u16 format_tag = use_float ? u16{3} : u16{1};
    const usize bytes_per_sample = static_cast<usize>(bps / 8u);
    const usize block_align = static_cast<usize>(view.channels) * bytes_per_sample;
    if (block_align > 0xFFFFu) [[unlikely]] {
        return ErrorCode::OutOfRange;
    }

    const f64 sr_rounded = std::round(view.sample_rate);
    if (!(sr_rounded >= 1.0 && sr_rounded <= static_cast<f64>(0xFFFFFFFFu))) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }
    const u32 sample_rate_u32 = static_cast<u32>(sr_rounded);
    const u64 byte_rate_u64 = static_cast<u64>(sample_rate_u32) * static_cast<u64>(block_align);
    const u32 byte_rate = (byte_rate_u64 > 0xFFFFFFFFULL) ? 0xFFFFFFFFu : static_cast<u32>(byte_rate_u64);

    if (view.frames > 0xFFFFFFFFULL / static_cast<u64>(block_align)) [[unlikely]] {
        return ErrorCode::OutOfRange;
    }
    const u64 data_size_u64 = view.frames * static_cast<u64>(block_align);
    const u64 data_padded = data_size_u64 + (data_size_u64 & 1ULL);

    Allocator* dst_alloc = opts.allocator ? opts.allocator : default_allocator();
    ArenaTemp scratch = get_scratch(dst_alloc);
    ArenaArray<u8> list_info(scratch.arena());

    if (!opts.title.empty() || !opts.artist.empty() || !opts.comment.empty()) {
        auto append_subchunk = [&](const char (&id)[5], StringRef text) -> ErrorCode {
            if (text.empty()) return ErrorCode::Ok;
            if (text.size() >= 0xFFFFFFFFu) [[unlikely]] {
                return ErrorCode::OutOfRange;
            }
            audio_detail::write_fourcc(list_info, id);
            const u32 sz = static_cast<u32>(text.size() + 1u);
            audio_detail::write_u32_le(list_info, sz);
            list_info.append(ArrayRef<u8>(reinterpret_cast<const u8*>(text.data()), text.size()));
            list_info.push_back(0u);
            if ((sz & 1u) != 0u) {
                list_info.push_back(0u);
            }
            return ErrorCode::Ok;
        };

        audio_detail::write_fourcc(list_info, "INFO");
        if (append_subchunk("INAM", opts.title) != ErrorCode::Ok ||
            append_subchunk("IART", opts.artist) != ErrorCode::Ok ||
            append_subchunk("ICMT", opts.comment) != ErrorCode::Ok) [[unlikely]] {
            return ErrorCode::OutOfRange;
        }
    }

    const u64 list_total = list_info.empty() ? 0ULL : (8ULL + static_cast<u64>(list_info.size()));
    const u64 riff_size_u64 = 4ULL + (8ULL + 16ULL) + (8ULL + data_padded) + list_total;
    if (riff_size_u64 > 0xFFFFFFFFULL) [[unlikely]] {
        return ErrorCode::OutOfRange;
    }

    Array<u8> out = Array<u8>::with_capacity(static_cast<usize>(riff_size_u64 + 8ULL), dst_alloc);

    audio_detail::write_fourcc(out, "RIFF");
    audio_detail::write_u32_le(out, static_cast<u32>(riff_size_u64));
    audio_detail::write_fourcc(out, "WAVE");

    audio_detail::write_fourcc(out, "fmt ");
    audio_detail::write_u32_le(out, 16u);
    audio_detail::write_u16_le(out, format_tag);
    audio_detail::write_u16_le(out, view.channels);
    audio_detail::write_u32_le(out, sample_rate_u32);
    audio_detail::write_u32_le(out, byte_rate);
    audio_detail::write_u16_le(out, static_cast<u16>(block_align));
    audio_detail::write_u16_le(out, bps);

    audio_detail::write_fourcc(out, "data");
    audio_detail::write_u32_le(out, static_cast<u32>(data_size_u64));

    for (u64 t = 0; t < view.frames; ++t) {
        for (u16 ch = 0; ch < view.channels; ++ch) {
            const usize idx = audio_detail::sample_index(t, ch, view.frames, view.channels, view.layout);
            if (use_float) {
                if (bps == 32) {
                    audio_detail::write_f32_le(out, audio_detail::view_sample_f32(view, idx));
                } else {
                    audio_detail::write_f64_le(out, audio_detail::view_sample_f64(view, idx));
                }
            } else if (bps == 8) {
                out.push_back(audio_detail::view_sample_u8_pcm(view, idx));
            } else if (bps == 16) {
                audio_detail::write_u16_le(out, static_cast<u16>(audio_detail::view_sample_i16_pcm(view, idx)));
            } else if (bps == 24) {
                audio_detail::write_i24_le(out, audio_detail::view_sample_i24_pcm(view, idx));
            } else {
                audio_detail::write_u32_le(out, static_cast<u32>(audio_detail::view_sample_i32_pcm(view, idx)));
            }
        }
    }
    if ((data_size_u64 & 1ULL) != 0ULL) {
        out.push_back(0u);
    }

    if (!list_info.empty()) {
        audio_detail::write_fourcc(out, "LIST");
        audio_detail::write_u32_le(out, static_cast<u32>(list_info.size()));
        out.append(ArrayRef<u8>(list_info.data(), list_info.size()));
    }

    return out;
}

} // namespace keel::audiofile
