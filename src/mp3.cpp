#include "audio_internal.h"

#include "keel/arena.h"

namespace keel::audiofile {

bool is_mp3_bytes(ArrayRef<u8> bytes) noexcept {
    if (bytes.size() < 4) return false;
    const u8* d = bytes.data();
    if (d[0] == 'I' && d[1] == 'D' && d[2] == '3') return true;
    if (d[0] == 0xFFu && (d[1] & 0xE0u) == 0xE0u) {
        const u8 version = as<u8>((d[1] >> 3) & 0x03u);
        const u8 layer = as<u8>((d[1] >> 1) & 0x03u);
        return version != 1u && layer == 1u;
    }
    return false;
}

namespace {
namespace mp3_detail {

constexpr f64 kPi = 3.14159265358979323846;

// Bitrate lookup table [version][layer][bitrate_idx] (in kbps)
// version: 0 = MPEG-2.5, 1 = reserved, 2 = MPEG-2, 3 = MPEG-1
// layer:   0 = reserved, 1 = Layer III, 2 = Layer II, 3 = Layer I
constexpr u16 kBitrates[4][4][16] = {
    // MPEG-2.5
    {
        {},
        {0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0},
        {0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0},
        {0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 0},
    },
    // Reserved
    {},
    // MPEG-2
    {
        {},
        {0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0},
        {0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0},
        {0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 0},
    },
    // MPEG-1
    {
        {},
        {0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0},
        {0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0},
        {0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 0},
    },
};

constexpr u32 kSampleRates[4][4] = {
    {11025, 12000, 8000, 0},  // MPEG-2.5
    {0, 0, 0, 0},             // Reserved
    {22050, 24000, 16000, 0}, // MPEG-2
    {44100, 48000, 32000, 0}, // MPEG-1
};

[[nodiscard]] u32 read_syncsafe_u32(const u8* p) noexcept {
    return (as<u32>(p[0] & 0x7Fu) << 21) | (as<u32>(p[1] & 0x7Fu) << 14) |
           (as<u32>(p[2] & 0x7Fu) << 7) | as<u32>(p[3] & 0x7Fu);
}

void append_utf8_codepoint(String& out, u32 cp) {
    if (cp <= 0x7Fu) {
        out.push_back(as<char>(cp));
    } else if (cp <= 0x7FFu) {
        out.push_back(as<char>(0xC0u | (cp >> 6)));
        out.push_back(as<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        out.push_back(as<char>(0xE0u | (cp >> 12)));
        out.push_back(as<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(as<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0x10FFFFu) {
        out.push_back(as<char>(0xF0u | (cp >> 18)));
        out.push_back(as<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(as<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(as<char>(0x80u | (cp & 0x3Fu)));
    }
}

[[nodiscard]] String decode_id3_text(const u8* ptr, usize len, u8 encoding, Allocator* alloc) {
    if (encoding == 1u || encoding == 2u) {
        bool big_endian = (encoding == 2u);
        usize pos = 0;
        if (encoding == 1u && len >= 2) {
            if (ptr[0] == 0xFEu && ptr[1] == 0xFFu) {
                big_endian = true;
                pos = 2;
            } else if (ptr[0] == 0xFFu && ptr[1] == 0xFEu) {
                big_endian = false;
                pos = 2;
            }
        }
        String out(alloc);
        while (pos + 2 <= len) {
            const u16 u = big_endian ? audio_detail::read_u16_be(ptr + pos) : audio_detail::read_u16_le(ptr + pos);
            pos += 2;
            if (u == 0u) break;
            if (u >= 0xD800u && u <= 0xDBFFu && pos + 2 <= len) {
                const u16 lo = big_endian ? audio_detail::read_u16_be(ptr + pos) : audio_detail::read_u16_le(ptr + pos);
                if (lo >= 0xDC00u && lo <= 0xDFFFu) {
                    pos += 2;
                    const u32 cp = 0x10000u + ((as<u32>(u - 0xD800u) << 10) | as<u32>(lo - 0xDC00u));
                    append_utf8_codepoint(out, cp);
                    continue;
                }
            }
            append_utf8_codepoint(out, u);
        }
        while (!out.empty() && (out.back() == ' ' || out.back() == '\0')) {
            out.pop_back();
        }
        return out;
    }
    return audio_detail::make_trimmed_string(ptr, len, alloc);
}

[[nodiscard]] String decode_id3_comment(const u8* ptr, usize len, u8 encoding, Allocator* alloc) {
    if (len <= 3) {
        return String(alloc);
    }
    const u8* body = ptr + 3;
    usize body_len = len - 3;

    if (encoding == 1u || encoding == 2u) {
        // Skip UTF-16 null-terminated short description if present
        usize sep = 0;
        bool found_sep = false;
        while (sep + 2 <= body_len) {
            if (body[sep] == 0u && body[sep + 1] == 0u) {
                found_sep = true;
                break;
            }
            sep += 2;
        }
        if (found_sep && sep + 2 <= body_len) {
            return decode_id3_text(body + sep + 2, body_len - (sep + 2), encoding, alloc);
        }
        return decode_id3_text(body, body_len, encoding, alloc);
    }

    // Trim trailing nulls/spaces first so we can distinguish an internal description '\0' from trailing padding
    while (body_len > 0 && (body[body_len - 1] == 0u || body[body_len - 1] == as<u8>(' '))) {
        --body_len;
    }
    for (usize i = 0; i < body_len; ++i) {
        if (body[i] == 0u) {
            return audio_detail::make_trimmed_string(body + i + 1, body_len - (i + 1), alloc);
        }
    }
    return audio_detail::make_trimmed_string(body, body_len, alloc);
}

struct Id3ParseResult {
    usize audio_start = 0;
    usize audio_end = 0;
    bool truncated_id3v2 = false;
    AudioMetadata metadata;
};

[[nodiscard]] Id3ParseResult parse_id3_tags(const u8* data, usize len, Allocator* alloc) {
    Id3ParseResult res;
    res.audio_start = 0;
    res.audio_end = len;
    res.metadata = AudioMetadata{String(alloc), String(alloc), String(alloc)};

    if (len >= 10 && data[0] == 'I' && data[1] == 'D' && data[2] == '3') {
        const u8 major = data[3];
        const u8 flags = data[5];
        const u32 tag_size = read_syncsafe_u32(data + 6);
        const usize footer_bytes = ((flags & 0x10u) != 0u) ? 10u : 0u;
        const usize full_header = 10u + as<usize>(tag_size) + footer_bytes;
        if (full_header > len) {
            res.truncated_id3v2 = true;
            res.audio_start = len;
        } else {
            res.audio_start = full_header;
        }

        const usize tag_end = keel::min(10u + as<usize>(tag_size), len);
        usize pos = 10;

        // Skip optional ID3v2.3/v2.4 extended header
        if ((flags & 0x40u) != 0u && major >= 3u && pos + 4 <= tag_end) {
            const u32 ext_size = (major == 4u) ? read_syncsafe_u32(data + pos) : audio_detail::read_u32_be(data + pos);
            if (major == 4u) {
                pos = (ext_size >= 6u && ext_size <= tag_end - pos) ? (pos + ext_size) : tag_end;
            } else {
                pos = (ext_size <= tag_end - (pos + 4u)) ? (pos + 4u + ext_size) : tag_end;
            }
        }

        if (major == 2u) {
            while (pos + 6 <= tag_end) {
                if (data[pos] == 0u) break;
                const u8* fid = data + pos;
                const u32 fsize = (as<u32>(data[pos + 3]) << 16) | (as<u32>(data[pos + 4]) << 8) |
                                  as<u32>(data[pos + 5]);
                pos += 6;
                if (fsize > tag_end - pos) break;
                if (fsize > 1u) {
                    const u8 enc = data[pos];
                    if (mem_eq(fid, "TT2", 3)) {
                        res.metadata.title = decode_id3_text(data + pos + 1, fsize - 1u, enc, alloc);
                    } else if (mem_eq(fid, "TP1", 3)) {
                        res.metadata.artist = decode_id3_text(data + pos + 1, fsize - 1u, enc, alloc);
                    } else if (mem_eq(fid, "COM", 3)) {
                        res.metadata.comment = decode_id3_comment(data + pos + 1, fsize - 1u, enc, alloc);
                    }
                }
                pos += fsize;
            }
        } else {
            while (pos + 10 <= tag_end) {
                if (data[pos] == 0u) break;
                const u8* fid = data + pos;
                const u32 fsize =
                    (major == 4u) ? read_syncsafe_u32(data + pos + 4) : audio_detail::read_u32_be(data + pos + 4);
                pos += 10;
                if (fsize > tag_end - pos) break;
                if (fsize > 1u) {
                    const u8 enc = data[pos];
                    if (mem_eq(fid, "TIT2", 4)) {
                        res.metadata.title = decode_id3_text(data + pos + 1, fsize - 1u, enc, alloc);
                    } else if (mem_eq(fid, "TPE1", 4)) {
                        res.metadata.artist = decode_id3_text(data + pos + 1, fsize - 1u, enc, alloc);
                    } else if (mem_eq(fid, "COMM", 4)) {
                        res.metadata.comment = decode_id3_comment(data + pos + 1, fsize - 1u, enc, alloc);
                    }
                }
                pos += fsize;
            }
        }
    }

    // Check trailing 128-byte ID3v1 / ID3v1.1 footer
    if (len >= 128 && len - 128 >= res.audio_start && mem_eq(data + (len - 128), "TAG", 3)) {
        const u8* tag = data + (len - 128);
        res.audio_end = len - 128;
        if (res.metadata.title.empty()) {
            res.metadata.title = audio_detail::make_trimmed_string(tag + 3, 30, alloc);
        }
        if (res.metadata.artist.empty()) {
            res.metadata.artist = audio_detail::make_trimmed_string(tag + 33, 30, alloc);
        }
        if (res.metadata.comment.empty()) {
            const usize comment_len = (tag[125] == 0u && tag[126] != 0u) ? 28u : 30u;
            res.metadata.comment = audio_detail::make_trimmed_string(tag + 97, comment_len, alloc);
        }
    }

    return res;
}

class Mp3BitReader {
public:
    Mp3BitReader(const u8* data, usize len) noexcept : m_data(data), m_len(len) {}

    [[nodiscard]] u32 read_bits(u32 n) noexcept {
        if (n == 0) return 0;
        while (m_bits_left < n) {
            if (m_pos >= m_len) {
                const u32 res = (m_bits_left == 0) ? 0u : as<u32>(m_bit_buf << (n - m_bits_left));
                m_bits_left = 0;
                return res;
            }
            m_bit_buf = (m_bit_buf << 8) | as<u64>(m_data[m_pos++]);
            m_bits_left += 8;
        }
        m_bits_left -= n;
        const u64 mask = (n == 32) ? 0xFFFFFFFFULL : ((u64{1} << n) - 1ULL);
        return as<u32>((m_bit_buf >> m_bits_left) & mask);
    }

private:
    const u8* m_data = nullptr;
    usize m_len = 0;
    usize m_pos = 0;
    u64 m_bit_buf = 0;
    u32 m_bits_left = 0;
};

struct FrameHeader {
    u8 version = 0; // 0=2.5, 2=2, 3=1
    u8 layer = 0;   // 1=Layer III
    bool error_protection = false;
    u16 bitrate = 0;
    u32 sample_rate = 0;
    bool padding = false;
    u8 channel_mode = 0; // 0=Stereo, 1=Joint, 2=Dual, 3=Mono
    u8 mode_extension = 0;
    u16 frame_size = 0;
    u16 samples_per_frame = 0;
    u16 num_channels = 0;
};

[[nodiscard]] bool parse_frame_header(const u8* p, usize len, FrameHeader& hdr) noexcept {
    if (len < 4) return false;
    if (p[0] != 0xFFu || (p[1] & 0xE0u) != 0xE0u) return false;

    hdr.version = as<u8>((p[1] >> 3) & 0x03u);
    hdr.layer = as<u8>((p[1] >> 1) & 0x03u);
    hdr.error_protection = (p[1] & 0x01u) == 0u;

    if (hdr.version == 1u || hdr.layer != 1u) return false;

    const u8 bitrate_idx = as<u8>((p[2] >> 4) & 0x0Fu);
    const u8 sr_idx = as<u8>((p[2] >> 2) & 0x03u);
    hdr.padding = ((p[2] >> 1) & 0x01u) != 0u;

    if (bitrate_idx == 0u || bitrate_idx == 15u || sr_idx == 3u) return false;

    hdr.bitrate = kBitrates[hdr.version][hdr.layer][bitrate_idx];
    hdr.sample_rate = kSampleRates[hdr.version][sr_idx];
    if (hdr.bitrate == 0u || hdr.sample_rate == 0u) return false;

    hdr.channel_mode = as<u8>((p[3] >> 6) & 0x03u);
    hdr.mode_extension = as<u8>((p[3] >> 4) & 0x03u);
    hdr.num_channels = (hdr.channel_mode == 3u) ? u16{1} : u16{2};

    if (hdr.version == 3u) {
        hdr.samples_per_frame = 1152u;
        hdr.frame_size =
            as<u16>((144000u * as<u32>(hdr.bitrate)) / hdr.sample_rate + (hdr.padding ? 1u : 0u));
    } else {
        hdr.samples_per_frame = 576u;
        hdr.frame_size =
            as<u16>((72000u * as<u32>(hdr.bitrate)) / hdr.sample_rate + (hdr.padding ? 1u : 0u));
    }
    return hdr.frame_size >= 4u;
}

struct VbrHeaderInfo {
    bool is_vbr_tag_frame = false;
    bool has_frames = false;
    bool has_bytes = false;
    u32 total_frames = 0;
    u32 total_bytes = 0;
};

[[nodiscard]] VbrHeaderInfo parse_vbr_header(const u8* frame_ptr, usize avail_bytes, const FrameHeader& hdr) noexcept {
    VbrHeaderInfo vbr{};
    const usize frame_len = keel::min(avail_bytes, as<usize>(hdr.frame_size));
    const usize crc_bytes = hdr.error_protection ? 2u : 0u;
    const usize side_info_size =
        (hdr.version == 3u) ? ((hdr.num_channels == 1u) ? 17u : 32u) : ((hdr.num_channels == 1u) ? 9u : 17u);
    const usize xing_off = 4u + crc_bytes + side_info_size;

    if (xing_off + 8u <= frame_len) {
        const u8* xp = frame_ptr + xing_off;
        if (mem_eq(xp, "Xing", 4) || mem_eq(xp, "Info", 4)) {
            vbr.is_vbr_tag_frame = true;
            const u32 flags = audio_detail::read_u32_be(xp + 4);
            usize cur = xing_off + 8u;
            if ((flags & 0x01u) != 0u && cur + 4u <= frame_len) {
                vbr.total_frames = audio_detail::read_u32_be(frame_ptr + cur);
                vbr.has_frames = true;
                cur += 4u;
            }
            if ((flags & 0x02u) != 0u && cur + 4u <= frame_len) {
                vbr.total_bytes = audio_detail::read_u32_be(frame_ptr + cur);
                vbr.has_bytes = true;
            }
            return vbr;
        }
    }

    if (36u + 18u <= frame_len && mem_eq(frame_ptr + 36u, "VBRI", 4)) {
        vbr.is_vbr_tag_frame = true;
        vbr.total_bytes = audio_detail::read_u32_be(frame_ptr + 46u);
        vbr.total_frames = audio_detail::read_u32_be(frame_ptr + 50u);
        vbr.has_bytes = true;
        vbr.has_frames = true;
    }

    return vbr;
}

[[nodiscard]] f64 dequant_power_4bit(i32 is) noexcept {
    // Exact precomputed lookup table for |is|^(4/3), is in [-8, 7]
    static constexpr f64 kPow43[9] = {
        0.0,
        1.0,
        2.5198420997897464,
        4.3267487109222245,
        6.3496042078727974,
        8.549879733383484,
        10.902723556992836,
        13.390518279406722,
        16.0,
    };
    if (is == 0) return 0.0;
    const i32 mag = (is < 0) ? -is : is;
    const f64 val = (mag <= 8) ? kPow43[mag] : std::pow(as<f64>(mag), 4.0 / 3.0);
    return (is < 0) ? -val : val;
}

} // namespace mp3_detail
} // namespace

Result<AudioInfo> peek_mp3(ArrayRef<u8> bytes, Allocator* alloc) {
    if (bytes.size() < 4) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }

    Allocator* dst_alloc = alloc ? alloc : default_allocator();
    const u8* d = bytes.data();
    const usize len = bytes.size();

    mp3_detail::Id3ParseResult id3 = mp3_detail::parse_id3_tags(d, len, dst_alloc);
    if (id3.truncated_id3v2) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }

    usize pos = id3.audio_start;
    mp3_detail::FrameHeader first_hdr{};
    bool found = false;

    while (pos + 4 <= id3.audio_end) {
        if (mp3_detail::parse_frame_header(d + pos, id3.audio_end - pos, first_hdr)) {
            found = true;
            break;
        }
        ++pos;
    }

    if (!found) [[unlikely]] {
        return ErrorCode::InvalidData;
    }

    AudioInfo info;
    info.format = AudioFormat::Mp3;
    info.compression_method = String("MPEG Layer III", dst_alloc);
    info.channels = first_hdr.num_channels;
    info.sample_rate = as<f64>(first_hdr.sample_rate);
    info.bits_per_sample = 16;
    info.bit_rate_kbps = first_hdr.bitrate;
    info.metadata = keel::move(id3.metadata);

    const mp3_detail::VbrHeaderInfo vbr = mp3_detail::parse_vbr_header(d + pos, id3.audio_end - pos, first_hdr);
    if (vbr.has_frames) {
        info.total_frames = as<u64>(vbr.total_frames) * as<u64>(first_hdr.samples_per_frame);
        info.duration_sec = as<f64>(info.total_frames) / info.sample_rate;
        if (info.duration_sec > 0.0) {
            const u64 stream_bytes =
                vbr.has_bytes ? as<u64>(vbr.total_bytes) : as<u64>(id3.audio_end - pos);
            info.bit_rate_kbps =
                as<u32>(std::lround((as<f64>(stream_bytes) * 8.0) / (info.duration_sec * 1000.0)));
        }
    } else if (info.bit_rate_kbps > 0 && id3.audio_end > pos) {
        const usize audio_bytes = id3.audio_end - pos;
        info.duration_sec =
            as<f64>(as<u64>(audio_bytes) * 8ULL) / (as<f64>(info.bit_rate_kbps) * 1000.0);
        info.total_frames = as<u64>(info.duration_sec * info.sample_rate);
    }

    return info;
}

Result<AudioBuffer> decode_mp3(ArrayRef<u8> bytes, const AudioDecodeOptions& opts) {
    if (opts.end_frame != ~u64{0} && opts.end_frame < opts.start_frame) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }
    if (bytes.size() < 4) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }

    Allocator* dst_alloc = opts.allocator ? opts.allocator : default_allocator();
    const u8* d = bytes.data();
    const usize len = bytes.size();

    mp3_detail::Id3ParseResult id3 = mp3_detail::parse_id3_tags(d, len, dst_alloc);
    if (id3.truncated_id3v2) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }

    usize pos = id3.audio_start;
    mp3_detail::FrameHeader first_hdr{};
    bool found = false;

    while (pos + 4 <= id3.audio_end) {
        if (mp3_detail::parse_frame_header(d + pos, id3.audio_end - pos, first_hdr)) {
            found = true;
            break;
        }
        ++pos;
    }

    if (!found) [[unlikely]] {
        return ErrorCode::InvalidData;
    }
    if (pos + first_hdr.frame_size > id3.audio_end) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }

    const u16 num_channels = first_hdr.num_channels;
    const f64 sample_rate = as<f64>(first_hdr.sample_rate);

    const mp3_detail::VbrHeaderInfo vbr = mp3_detail::parse_vbr_header(d + pos, id3.audio_end - pos, first_hdr);
    if (vbr.is_vbr_tag_frame) {
        pos += first_hdr.frame_size;
    }

    ArenaTemp scratch = get_scratch(dst_alloc);
    ArenaArray<f64> channel_pcm[2];
    for (u16 ch = 0; ch < num_channels; ++ch) {
        channel_pcm[ch] = ArenaArray<f64>(scratch.arena());
    }

    f64 cos_poly[32][64]{};
    for (i32 i = 0; i < 32; ++i) {
        for (i32 k = 0; k < 64; ++k) {
            cos_poly[i][k] = std::cos((mp3_detail::kPi / 64.0) * as<f64>(2 * i + 1) * as<f64>(k - 16));
        }
    }

    f64 synth_win[512]{};
    for (i32 i = 0; i < 512; ++i) {
        const f64 angle = (2.0 * mp3_detail::kPi * (as<f64>(i) + 0.5)) / 1024.0;
        synth_win[i] = -std::sin(angle) * std::cos(angle * 0.5);
    }

    u64 stream_cursor = 0;
    f64 frame_samples[2][1152]{};

    while (pos + 4 <= id3.audio_end) {
        mp3_detail::FrameHeader hdr{};
        if (!mp3_detail::parse_frame_header(d + pos, id3.audio_end - pos, hdr)) {
            ++pos;
            continue;
        }
        if (pos + hdr.frame_size > id3.audio_end) {
            if (stream_cursor == 0 && !vbr.is_vbr_tag_frame) [[unlikely]] {
                return ErrorCode::EndOfFile;
            }
            break;
        }

        const u64 frame_start = stream_cursor;
        const u64 frame_end = stream_cursor + as<u64>(hdr.samples_per_frame);
        const u64 win_lo = keel::max(frame_start, opts.start_frame);
        const u64 win_hi = keel::min(frame_end, opts.end_frame);

        if (win_hi > win_lo) {
            const u64 add_count = win_hi - win_lo;
            if (opts.max_frames > 0 &&
                as<u64>(channel_pcm[0].size()) + add_count > opts.max_frames) [[unlikely]] {
                return ErrorCode::OutOfRange;
            }

            const usize header_bytes = hdr.error_protection ? 6u : 4u;
            const u8* frame_data = d + pos + header_bytes;
            const usize frame_data_len = (hdr.frame_size >= header_bytes) ? (hdr.frame_size - header_bytes) : 0u;

            const i32 num_granules = (hdr.version == 3u) ? 2 : 1;
            const u16 frame_ch = hdr.num_channels;
            mp3_detail::Mp3BitReader br(frame_data, frame_data_len);

            for (i32 gr = 0; gr < num_granules; ++gr) {
                const usize gr_offset = as<usize>(gr) * 576u;
                for (u16 ch = 0; ch < frame_ch; ++ch) {
                    f64 xr[32][18]{};
                    for (i32 sb = 0; sb < 32; ++sb) {
                        for (i32 ss = 0; ss < 18; ++ss) {
                            i32 is = as<i32>(br.read_bits(4));
                            if (is > 7) is -= 16;
                            xr[sb][ss] = mp3_detail::dequant_power_4bit(is) * 0.05;
                        }
                    }

                    const u16 dst_ch = (ch < num_channels) ? ch : u16{0};
                    for (i32 ss = 0; ss < 18; ++ss) {
                        f64 u[64]{};
                        for (i32 k = 0; k < 64; ++k) {
                            f64 acc = 0.0;
                            for (i32 sb = 0; sb < 32; ++sb) {
                                acc += xr[sb][ss] * cos_poly[sb][k];
                            }
                            u[k] = acc;
                        }

                        const usize ss_offset = gr_offset + as<usize>(ss) * 32u;
                        for (i32 j = 0; j < 32; ++j) {
                            f64 sample = 0.0;
                            for (i32 k = 0; k < 16; ++k) {
                                const i32 idx = k * 32 + j;
                                sample += u[idx & 63] * synth_win[idx];
                            }
                            if (sample < -1.0) sample = -1.0;
                            if (sample > 1.0) sample = 1.0;
                            frame_samples[dst_ch][ss_offset + as<usize>(j)] = sample;
                        }
                    }

                    if (frame_ch == 1u && num_channels == 2u) {
                        mem_copy(frame_samples[1] + gr_offset, frame_samples[0] + gr_offset, 576u * sizeof(f64));
                    }
                }
            }

            const usize local_off = as<usize>(win_lo - frame_start);
            const usize local_cnt = as<usize>(add_count);
            for (u16 ch = 0; ch < num_channels; ++ch) {
                channel_pcm[ch].append(ArrayRef<f64>(frame_samples[ch] + local_off, local_cnt));
            }
        }

        stream_cursor = frame_end;
        pos += hdr.frame_size;
    }

    const u64 out_frames = as<u64>(channel_pcm[0].size());
    if (opts.max_frames > 0 && out_frames > opts.max_frames) [[unlikely]] {
        return ErrorCode::OutOfRange;
    }

    const AudioSampleType out_type = opts.native_sample_type ? AudioSampleType::I16 : AudioSampleType::F64;

    AudioBuffer res;
    res.sample_rate = sample_rate;
    res.bits_per_sample = 16;
    res.channels = num_channels;
    res.frames = out_frames;
    res.total_source_frames = stream_cursor;
    res.sample_type = out_type;
    res.layout = opts.layout;
    res.metadata = keel::move(id3.metadata);

    if (out_frames == 0) {
        res.data = Array<u8>(dst_alloc);
        return res;
    }

    KEEL_TRY_ASSIGN(res.data, audio_detail::alloc_sample_buffer(out_frames, num_channels, out_type, dst_alloc));

    if (!opts.native_sample_type) {
        f64* dst = res.samples_f64().data();
        for (u16 ch = 0; ch < num_channels; ++ch) {
            const f64* src = channel_pcm[ch].data();
            for (u64 t = 0; t < out_frames; ++t) {
                dst[audio_detail::sample_index(t, ch, out_frames, num_channels, opts.layout)] =
                    src[as<usize>(t)];
            }
        }
    } else {
        i16* dst = res.samples_i16().data();
        for (u16 ch = 0; ch < num_channels; ++ch) {
            const f64* src = channel_pcm[ch].data();
            for (u64 t = 0; t < out_frames; ++t) {
                const f64 s = audio_detail::clamp_unit(src[as<usize>(t)]);
                dst[audio_detail::sample_index(t, ch, out_frames, num_channels, opts.layout)] =
                    as<i16>(std::lround(s * 32767.0));
            }
        }
    }

    return res;
}

} // namespace keel::audiofile
