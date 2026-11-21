#include "audio_internal.h"

#include "keel/arena.h"

namespace keel::audiofile {

bool is_aiff_bytes(ArrayRef<u8> bytes) noexcept {
    if (bytes.size() < 12) return false;
    const u8* d = bytes.data();
    return mem_eq(d, "FORM", 4) && (mem_eq(d + 8, "AIFF", 4) || mem_eq(d + 8, "AIFC", 4));
}

namespace {
namespace aiff_detail {

enum class AiffEncoding : u8 {
    PcmBe = 0,
    PcmLe,
    PcmU8,
    Float32Be,
    Float64Be,
    MuLaw,
    ALaw,
    Unsupported,
};

struct IffChunk {
    usize offset = 0;
    u32 size = 0;
    bool found = false;
};

struct AiffChunks {
    IffChunk comm{};
    IffChunk ssnd{};
    IffChunk name{};
    IffChunk auth{};
    IffChunk anno{};
    bool has_explicit_anno = false;
};

[[nodiscard]] f64 read_extended80_be(const u8* p) noexcept {
    u16 exp = audio_detail::read_u16_be(p);
    const u64 mant = audio_detail::read_u64_be(p + 2);
    if (exp == 0u && mant == 0ULL) {
        return 0.0;
    }
    const int sign = ((exp & 0x8000u) != 0u) ? -1 : 1;
    exp &= 0x7FFFu;
    if (exp == 0x7FFFu) {
        return 0.0;
    }
    const int unbiased_exp = static_cast<int>(exp) - 16383;
    return static_cast<f64>(sign) * std::ldexp(static_cast<f64>(mant), unbiased_exp - 63);
}

template <typename ByteBuf> void write_extended80_be(ByteBuf& buf, f64 val) {
    if (!(val > 0.0) || !std::isfinite(val)) {
        for (int i = 0; i < 10; ++i) {
            buf.push_back(0u);
        }
        return;
    }
    int exp = 0;
    const f64 mant = std::frexp(val, &exp);
    const u16 biased_exp = static_cast<u16>(exp + 16382);
    const u64 mant_int = static_cast<u64>(std::ldexp(mant, 64));
    audio_detail::write_u16_be(buf, biased_exp);
    audio_detail::write_u64_be(buf, mant_int);
}

[[nodiscard]] AiffChunks scan_aiff_chunks(const u8* data, usize len) noexcept {
    AiffChunks out{};
    usize pos = 12;
    while (pos + 8 <= len) {
        const u8* hdr = data + pos;
        const u32 csize = audio_detail::read_u32_be(hdr + 4);
        const usize payload = pos + 8;

        if (!out.comm.found && mem_eq(hdr, "COMM", 4)) {
            out.comm = {payload, csize, true};
        } else if (!out.ssnd.found && mem_eq(hdr, "SSND", 4)) {
            out.ssnd = {payload, csize, true};
        } else if (!out.name.found && mem_eq(hdr, "NAME", 4)) {
            out.name = {payload, csize, true};
        } else if (!out.auth.found && mem_eq(hdr, "AUTH", 4)) {
            out.auth = {payload, csize, true};
        } else if (mem_eq(hdr, "ANNO", 4)) {
            if (!out.has_explicit_anno) {
                out.anno = {payload, csize, true};
                out.has_explicit_anno = true;
            }
        } else if (!out.anno.found && mem_eq(hdr, "(c) ", 4)) {
            out.anno = {payload, csize, true};
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

void parse_aiff_metadata(const u8* data, usize len, const AiffChunks& chunks, AudioMetadata& meta, Allocator* alloc) {
    if (chunks.name.found && chunks.name.size <= len - chunks.name.offset) {
        meta.title = audio_detail::make_trimmed_string(data + chunks.name.offset, chunks.name.size, alloc);
    }
    if (chunks.auth.found && chunks.auth.size <= len - chunks.auth.offset) {
        meta.artist = audio_detail::make_trimmed_string(data + chunks.auth.offset, chunks.auth.size, alloc);
    }
    if (chunks.anno.found && chunks.anno.size <= len - chunks.anno.offset) {
        meta.comment = audio_detail::make_trimmed_string(data + chunks.anno.offset, chunks.anno.size, alloc);
    }
}

struct ParsedComm {
    u16 channels = 0;
    u32 num_frames = 0;
    u16 bits_per_sample = 0;
    usize bytes_per_sample = 0;
    f64 sample_rate = 0.0;
    AiffEncoding encoding = AiffEncoding::PcmBe;
    String compression_method;
};

[[nodiscard]] Result<ParsedComm> parse_comm_chunk(const u8* data, usize len, bool is_aifc, const IffChunk& comm,
                                                  Allocator* alloc) {
    if (!comm.found || comm.size < 18u) [[unlikely]] {
        return ErrorCode::InvalidData;
    }
    if (comm.size > len - comm.offset) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }
    if (is_aifc && comm.size < 22u) [[unlikely]] {
        return ErrorCode::InvalidData;
    }

    const u8* c = data + comm.offset;
    ParsedComm out;
    out.channels = audio_detail::read_u16_be(c);
    out.num_frames = audio_detail::read_u32_be(c + 2);
    out.bits_per_sample = audio_detail::read_u16_be(c + 6);
    out.sample_rate = read_extended80_be(c + 8);

    if (out.channels == 0u || !(out.sample_rate > 0.0)) [[unlikely]] {
        return ErrorCode::InvalidData;
    }

    if (!is_aifc) {
        out.encoding = AiffEncoding::PcmBe;
        out.bytes_per_sample = static_cast<usize>((out.bits_per_sample + 7u) / 8u);
        out.compression_method = String("Uncompressed", alloc);
        return out;
    }

    const u8* fcc = c + 18;
    if (mem_eq(fcc, "NONE", 4) || mem_eq(fcc, "twos", 4)) {
        out.encoding = AiffEncoding::PcmBe;
        out.bytes_per_sample = static_cast<usize>((out.bits_per_sample + 7u) / 8u);
        out.compression_method = String("Uncompressed", alloc);
    } else if (mem_eq(fcc, "sowt", 4)) {
        out.encoding = AiffEncoding::PcmLe;
        out.bytes_per_sample = static_cast<usize>((out.bits_per_sample + 7u) / 8u);
        out.compression_method = String("Little-Endian PCM", alloc);
    } else if (mem_eq(fcc, "raw ", 4)) {
        out.encoding = AiffEncoding::PcmU8;
        out.bits_per_sample = 8;
        out.bytes_per_sample = 1;
        out.compression_method = String("Unsigned 8-bit PCM", alloc);
    } else if (mem_eq(fcc, "fl32", 4) || mem_eq(fcc, "FL32", 4)) {
        out.encoding = AiffEncoding::Float32Be;
        out.bits_per_sample = 32;
        out.bytes_per_sample = 4;
        out.compression_method = String("IEEE 32-bit Float", alloc);
    } else if (mem_eq(fcc, "fl64", 4) || mem_eq(fcc, "FL64", 4)) {
        out.encoding = AiffEncoding::Float64Be;
        out.bits_per_sample = 64;
        out.bytes_per_sample = 8;
        out.compression_method = String("IEEE 64-bit Float", alloc);
    } else if (mem_eq(fcc, "ulaw", 4) || mem_eq(fcc, "ULAW", 4)) {
        out.encoding = AiffEncoding::MuLaw;
        out.bits_per_sample = 8;
        out.bytes_per_sample = 1;
        out.compression_method = String("Mu-law", alloc);
    } else if (mem_eq(fcc, "alaw", 4) || mem_eq(fcc, "ALAW", 4)) {
        out.encoding = AiffEncoding::ALaw;
        out.bits_per_sample = 8;
        out.bytes_per_sample = 1;
        out.compression_method = String("A-law", alloc);
    } else {
        out.encoding = AiffEncoding::Unsupported;
        out.bytes_per_sample = static_cast<usize>((out.bits_per_sample + 7u) / 8u);
        String label = String::with_capacity(9, alloc);
        label.append("AIFC ");
        label.append(StringRef(reinterpret_cast<const char*>(fcc), 4));
        out.compression_method = keel::move(label);
    }

    return out;
}

[[nodiscard]] bool is_valid_aiff_sample_depth(AiffEncoding enc, u16 bps) noexcept {
    switch (enc) {
        case AiffEncoding::PcmBe:
        case AiffEncoding::PcmLe:
            return bps == 8 || bps == 16 || bps == 24 || bps == 32;
        case AiffEncoding::PcmU8:
        case AiffEncoding::MuLaw:
        case AiffEncoding::ALaw:
            return bps == 8;
        case AiffEncoding::Float32Be:
            return bps == 32;
        case AiffEncoding::Float64Be:
            return bps == 64;
        case AiffEncoding::Unsupported:
            break;
    }
    return false;
}

[[nodiscard]] AudioSampleType aiff_native_sample_type(AiffEncoding enc, u16 bps) noexcept {
    switch (enc) {
        case AiffEncoding::PcmU8:
            return AudioSampleType::U8;
        case AiffEncoding::MuLaw:
        case AiffEncoding::ALaw:
            return AudioSampleType::I16;
        case AiffEncoding::Float32Be:
            return AudioSampleType::F32;
        case AiffEncoding::Float64Be:
            return AudioSampleType::F64;
        case AiffEncoding::PcmBe:
        case AiffEncoding::PcmLe:
            if (bps == 8) return AudioSampleType::I8;
            if (bps == 16) return AudioSampleType::I16;
            return AudioSampleType::I32;
        case AiffEncoding::Unsupported:
            break;
    }
    return AudioSampleType::F64;
}

} // namespace aiff_detail
} // namespace

Result<AudioInfo> peek_aiff(ArrayRef<u8> bytes, Allocator* alloc) {
    if (bytes.size() < 12) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }
    if (!is_aiff_bytes(bytes)) [[unlikely]] {
        return ErrorCode::InvalidData;
    }

    Allocator* dst_alloc = alloc ? alloc : default_allocator();
    const u8* d = bytes.data();
    const usize len = bytes.size();
    const bool is_aifc = mem_eq(d + 8, "AIFC", 4);
    const aiff_detail::AiffChunks chunks = aiff_detail::scan_aiff_chunks(d, len);

    KEEL_TRY_ASSIGN(aiff_detail::ParsedComm comm, aiff_detail::parse_comm_chunk(d, len, is_aifc, chunks.comm, dst_alloc));

    AudioInfo info;
    info.format = AudioFormat::Aiff;
    info.compression_method = keel::move(comm.compression_method);
    info.channels = comm.channels;
    info.sample_rate = comm.sample_rate;
    info.bits_per_sample = comm.bits_per_sample;
    info.total_frames = comm.num_frames;
    info.duration_sec = static_cast<f64>(comm.num_frames) / comm.sample_rate;
    if (info.duration_sec > 0.0 && len > 0) {
        info.bit_rate_kbps = static_cast<u32>((static_cast<f64>(len) * 8.0) / (info.duration_sec * 1000.0));
    }

    aiff_detail::parse_aiff_metadata(d, len, chunks, info.metadata, dst_alloc);
    return info;
}

Result<AudioBuffer> decode_aiff(ArrayRef<u8> bytes, const AudioDecodeOptions& opts) {
    if (bytes.size() < 12) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }
    if (!is_aiff_bytes(bytes)) [[unlikely]] {
        return ErrorCode::InvalidData;
    }

    Allocator* dst_alloc = opts.allocator ? opts.allocator : default_allocator();
    const u8* d = bytes.data();
    const usize len = bytes.size();
    const bool is_aifc = mem_eq(d + 8, "AIFC", 4);
    const aiff_detail::AiffChunks chunks = aiff_detail::scan_aiff_chunks(d, len);

    KEEL_TRY_ASSIGN(const aiff_detail::ParsedComm comm,
                    aiff_detail::parse_comm_chunk(d, len, is_aifc, chunks.comm, dst_alloc));

    if (comm.encoding == aiff_detail::AiffEncoding::Unsupported ||
        !aiff_detail::is_valid_aiff_sample_depth(comm.encoding, comm.bits_per_sample)) [[unlikely]] {
        return ErrorCode::NotSupported;
    }

    const u64 available_frames = comm.num_frames;
    if (available_frames > 0 && (!chunks.ssnd.found || chunks.ssnd.size < 8u)) [[unlikely]] {
        return ErrorCode::InvalidData;
    }

    KEEL_TRY_ASSIGN(const audio_detail::FrameWindow win, audio_detail::compute_frame_window(available_frames, opts));

    const AudioSampleType out_type =
        opts.native_sample_type ? aiff_detail::aiff_native_sample_type(comm.encoding, comm.bits_per_sample)
                                : AudioSampleType::F64;

    AudioBuffer res;
    res.sample_rate = comm.sample_rate;
    res.bits_per_sample = comm.bits_per_sample;
    res.channels = comm.channels;
    res.frames = win.count;
    res.total_source_frames = available_frames;
    res.sample_type = out_type;
    res.layout = opts.layout;
    aiff_detail::parse_aiff_metadata(d, len, chunks, res.metadata, dst_alloc);

    if (win.count == 0) {
        res.data = Array<u8>(dst_alloc);
        return res;
    }

    if (chunks.ssnd.offset > len || len - chunks.ssnd.offset < 8u) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }
    const u32 ssnd_offset = audio_detail::read_u32_be(d + chunks.ssnd.offset);
    if (static_cast<usize>(ssnd_offset) > len - (chunks.ssnd.offset + 8u) ||
        static_cast<u64>(ssnd_offset) + 8ULL > static_cast<u64>(chunks.ssnd.size)) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }

    const usize data_start = chunks.ssnd.offset + 8u + static_cast<usize>(ssnd_offset);
    const usize ssnd_avail_in_chunk = static_cast<usize>(chunks.ssnd.size) - (8u + static_cast<usize>(ssnd_offset));
    const usize max_data_bytes = keel::min(len - data_start, ssnd_avail_in_chunk);

    const usize block_align = static_cast<usize>(comm.channels) * comm.bytes_per_sample;
    const u64 end_frame = win.start + win.count;
    if (end_frame > static_cast<u64>(max_data_bytes) / static_cast<u64>(block_align)) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }

    KEEL_TRY_ASSIGN(res.data, audio_detail::alloc_sample_buffer(win.count, comm.channels, out_type, dst_alloc));

    const u8* raw_samples = d + data_start + static_cast<usize>(win.start) * block_align;
    const u64 nframes = win.count;
    const u16 nch = comm.channels;
    const usize bps_bytes = comm.bytes_per_sample;
    const aiff_detail::AiffEncoding enc = comm.encoding;

    if (!opts.native_sample_type) {
        f64* dst = res.samples_f64().data();
        for (u64 t = 0; t < nframes; ++t) {
            const u8* frame = raw_samples + static_cast<usize>(t) * block_align;
            for (u16 ch = 0; ch < nch; ++ch) {
                const u8* s = frame + static_cast<usize>(ch) * bps_bytes;
                f64 val = 0.0;
                switch (enc) {
                    case aiff_detail::AiffEncoding::PcmBe:
                        if (comm.bits_per_sample == 8) {
                            val = static_cast<f64>(static_cast<i8>(*s)) / 128.0;
                        } else if (comm.bits_per_sample == 16) {
                            val = static_cast<f64>(audio_detail::read_i16_be(s)) / 32768.0;
                        } else if (comm.bits_per_sample == 24) {
                            val = static_cast<f64>(audio_detail::read_i24_be(s)) / 8388608.0;
                        } else {
                            val = static_cast<f64>(audio_detail::read_i32_be(s)) / 2147483648.0;
                        }
                        break;
                    case aiff_detail::AiffEncoding::PcmLe:
                        if (comm.bits_per_sample == 8) {
                            val = static_cast<f64>(static_cast<i8>(*s)) / 128.0;
                        } else if (comm.bits_per_sample == 16) {
                            val = static_cast<f64>(audio_detail::read_i16_le(s)) / 32768.0;
                        } else if (comm.bits_per_sample == 24) {
                            val = static_cast<f64>(audio_detail::read_i24_le(s)) / 8388608.0;
                        } else {
                            val = static_cast<f64>(audio_detail::read_i32_le(s)) / 2147483648.0;
                        }
                        break;
                    case aiff_detail::AiffEncoding::PcmU8:
                        val = (static_cast<f64>(*s) - 128.0) / 128.0;
                        break;
                    case aiff_detail::AiffEncoding::Float32Be:
                        val = static_cast<f64>(audio_detail::read_f32_be(s));
                        break;
                    case aiff_detail::AiffEncoding::Float64Be:
                        val = audio_detail::read_f64_be(s);
                        break;
                    case aiff_detail::AiffEncoding::MuLaw:
                        val = static_cast<f64>(audio_detail::decode_mu_law(*s)) / 32768.0;
                        break;
                    case aiff_detail::AiffEncoding::ALaw:
                        val = static_cast<f64>(audio_detail::decode_a_law(*s)) / 32768.0;
                        break;
                    case aiff_detail::AiffEncoding::Unsupported:
                        break;
                }
                dst[audio_detail::sample_index(t, ch, nframes, nch, opts.layout)] = val;
            }
        }
    } else if (out_type == AudioSampleType::U8) {
        u8* dst = res.samples_u8().data();
        for (u64 t = 0; t < nframes; ++t) {
            const u8* frame = raw_samples + static_cast<usize>(t) * block_align;
            for (u16 ch = 0; ch < nch; ++ch) {
                dst[audio_detail::sample_index(t, ch, nframes, nch, opts.layout)] = frame[ch];
            }
        }
    } else if (out_type == AudioSampleType::I8) {
        i8* dst = res.samples_i8().data();
        for (u64 t = 0; t < nframes; ++t) {
            const u8* frame = raw_samples + static_cast<usize>(t) * block_align;
            for (u16 ch = 0; ch < nch; ++ch) {
                dst[audio_detail::sample_index(t, ch, nframes, nch, opts.layout)] = static_cast<i8>(frame[ch]);
            }
        }
    } else if (out_type == AudioSampleType::I16) {
        i16* dst = res.samples_i16().data();
        for (u64 t = 0; t < nframes; ++t) {
            const u8* frame = raw_samples + static_cast<usize>(t) * block_align;
            for (u16 ch = 0; ch < nch; ++ch) {
                i16 val = 0;
                if (enc == aiff_detail::AiffEncoding::MuLaw) {
                    val = audio_detail::decode_mu_law(frame[ch]);
                } else if (enc == aiff_detail::AiffEncoding::ALaw) {
                    val = audio_detail::decode_a_law(frame[ch]);
                } else if (enc == aiff_detail::AiffEncoding::PcmLe) {
                    val = audio_detail::read_i16_le(frame + static_cast<usize>(ch) * 2);
                } else {
                    val = audio_detail::read_i16_be(frame + static_cast<usize>(ch) * 2);
                }
                dst[audio_detail::sample_index(t, ch, nframes, nch, opts.layout)] = val;
            }
        }
    } else if (out_type == AudioSampleType::I32) {
        i32* dst = res.samples_i32().data();
        for (u64 t = 0; t < nframes; ++t) {
            const u8* frame = raw_samples + static_cast<usize>(t) * block_align;
            for (u16 ch = 0; ch < nch; ++ch) {
                const u8* s = frame + static_cast<usize>(ch) * bps_bytes;
                i32 val = 0;
                if (enc == aiff_detail::AiffEncoding::PcmLe) {
                    val = (comm.bits_per_sample == 24) ? audio_detail::read_i24_le(s) : audio_detail::read_i32_le(s);
                } else {
                    val = (comm.bits_per_sample == 24) ? audio_detail::read_i24_be(s) : audio_detail::read_i32_be(s);
                }
                dst[audio_detail::sample_index(t, ch, nframes, nch, opts.layout)] = val;
            }
        }
    } else if (out_type == AudioSampleType::F32) {
        f32* dst = res.samples_f32().data();
        for (u64 t = 0; t < nframes; ++t) {
            const u8* frame = raw_samples + static_cast<usize>(t) * block_align;
            for (u16 ch = 0; ch < nch; ++ch) {
                dst[audio_detail::sample_index(t, ch, nframes, nch, opts.layout)] =
                    audio_detail::read_f32_be(frame + static_cast<usize>(ch) * 4);
            }
        }
    } else {
        f64* dst = res.samples_f64().data();
        for (u64 t = 0; t < nframes; ++t) {
            const u8* frame = raw_samples + static_cast<usize>(t) * block_align;
            for (u16 ch = 0; ch < nch; ++ch) {
                dst[audio_detail::sample_index(t, ch, nframes, nch, opts.layout)] =
                    audio_detail::read_f64_be(frame + static_cast<usize>(ch) * 8);
            }
        }
    }

    return res;
}

Result<Array<u8>> encode_aiff(const AudioView& view, const AudioEncodeOptions& opts) {
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
    if (view.frames > 0xFFFFFFFFULL) [[unlikely]] {
        return ErrorCode::OutOfRange;
    }

    const usize bytes_per_sample = static_cast<usize>(bps / 8u);
    const u64 block_align = static_cast<u64>(view.channels) * static_cast<u64>(bytes_per_sample);
    if (view.frames > (0xFFFFFFFFULL - 8ULL) / block_align) [[unlikely]] {
        return ErrorCode::OutOfRange;
    }

    const u64 data_size_u64 = view.frames * block_align;
    const u32 ssnd_size = static_cast<u32>(8ULL + data_size_u64);
    const u64 ssnd_padded = static_cast<u64>(ssnd_size) + (static_cast<u64>(ssnd_size) & 1ULL);

    Allocator* dst_alloc = opts.allocator ? opts.allocator : default_allocator();
    ArenaTemp scratch = get_scratch(dst_alloc);
    ArenaArray<u8> meta(scratch.arena());

    auto append_meta_chunk = [&](const char (&id)[5], StringRef text) -> ErrorCode {
        if (text.empty()) return ErrorCode::Ok;
        if (text.size() > 0xFFFFFFFFu) [[unlikely]] {
            return ErrorCode::OutOfRange;
        }
        audio_detail::write_fourcc(meta, id);
        audio_detail::write_u32_be(meta, static_cast<u32>(text.size()));
        meta.append(ArrayRef<u8>(reinterpret_cast<const u8*>(text.data()), text.size()));
        if ((text.size() & 1u) != 0u) {
            meta.push_back(0u);
        }
        return ErrorCode::Ok;
    };

    if (append_meta_chunk("NAME", opts.title) != ErrorCode::Ok ||
        append_meta_chunk("AUTH", opts.artist) != ErrorCode::Ok ||
        append_meta_chunk("ANNO", opts.comment) != ErrorCode::Ok) [[unlikely]] {
        return ErrorCode::OutOfRange;
    }

    const u32 comm_size = use_float ? 24u : 18u;
    const u64 fver_total = use_float ? 12ULL : 0ULL;
    const u64 form_size_u64 = 4ULL + fver_total + (8ULL + static_cast<u64>(comm_size)) + (8ULL + ssnd_padded) +
                              static_cast<u64>(meta.size());
    if (form_size_u64 > 0xFFFFFFFFULL) [[unlikely]] {
        return ErrorCode::OutOfRange;
    }

    Array<u8> out = Array<u8>::with_capacity(static_cast<usize>(form_size_u64 + 8ULL), dst_alloc);

    audio_detail::write_fourcc(out, "FORM");
    audio_detail::write_u32_be(out, static_cast<u32>(form_size_u64));
    if (use_float) {
        audio_detail::write_fourcc(out, "AIFC");
        audio_detail::write_fourcc(out, "FVER");
        audio_detail::write_u32_be(out, 4u);
        audio_detail::write_u32_be(out, 0xA2805140u);
    } else {
        audio_detail::write_fourcc(out, "AIFF");
    }

    audio_detail::write_fourcc(out, "COMM");
    audio_detail::write_u32_be(out, comm_size);
    audio_detail::write_u16_be(out, view.channels);
    audio_detail::write_u32_be(out, static_cast<u32>(view.frames));
    audio_detail::write_u16_be(out, bps);
    aiff_detail::write_extended80_be(out, view.sample_rate);
    if (use_float) {
        if (bps == 32) {
            audio_detail::write_fourcc(out, "fl32");
        } else {
            audio_detail::write_fourcc(out, "fl64");
        }
        out.push_back(0u); // empty Pascal compression name
        out.push_back(0u); // even-alignment pad byte
    }

    audio_detail::write_fourcc(out, "SSND");
    audio_detail::write_u32_be(out, ssnd_size);
    audio_detail::write_u32_be(out, 0u); // offset
    audio_detail::write_u32_be(out, 0u); // blockSize

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
    if ((ssnd_size & 1u) != 0u) {
        out.push_back(0u);
    }

    if (!meta.empty()) {
        out.append(ArrayRef<u8>(meta.data(), meta.size()));
    }

    return out;
}

} // namespace keel::audiofile
