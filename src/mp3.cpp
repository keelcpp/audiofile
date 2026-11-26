#include "audio_internal.h"

#include "keel/arena.h"

// Full ISO/IEC 11172-3 (MPEG-1) & ISO/IEC 13818-3 (MPEG-2 / MPEG-2.5) Layer I / II / III decoder.
// Standard specification tables (Annex B Huffman trees, scalefactor bands, polyphase window)
// and scalar synthesis pipeline adapted from the CC0 1.0 Public Domain minimp3 implementation
// (https://github.com/lieff/minimp3) into pure C++20 keel.

namespace keel::audiofile {

bool is_mp3_bytes(ArrayRef<u8> bytes) noexcept {
    if (bytes.size() < 4) return false;
    const u8* d = bytes.data();
    if (d[0] == 'I' && d[1] == 'D' && d[2] == '3') return true;
    if (d[0] == 0xFFu && (d[1] & 0xE0u) == 0xE0u) {
        const u8 version = as<u8>((d[1] >> 3) & 0x03u);
        const u8 layer = as<u8>((d[1] >> 1) & 0x03u);
        const u8 bitrate_idx = as<u8>((d[2] >> 4) & 0x0Fu);
        const u8 sr_idx = as<u8>((d[2] >> 2) & 0x03u);
        return version != 1u && layer != 0u && bitrate_idx != 15u && sr_idx != 3u;
    }
    return false;
}

namespace {
namespace mp3_detail {

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

struct FrameHeader {
    u8 version = 0; // 0=2.5, 2=2, 3=1
    u8 layer = 0;   // 1=Layer III, 2=Layer II, 3=Layer I
    bool error_protection = false;
    bool is_free_format = false;
    u16 bitrate = 0;
    u32 sample_rate = 0;
    bool padding = false;
    u8 channel_mode = 0; // 0=Stereo, 1=Joint, 2=Dual, 3=Mono
    u8 mode_extension = 0;
    u16 frame_size = 0;
    u16 samples_per_frame = 0;
    u16 num_channels = 0;
};

[[nodiscard]] bool hdr_valid_raw(const u8* p) noexcept {
    if (p[0] != 0xFFu || (p[1] & 0xE0u) != 0xE0u) return false;
    const u8 version = as<u8>((p[1] >> 3) & 0x03u);
    const u8 layer = as<u8>((p[1] >> 1) & 0x03u);
    const u8 br_idx = as<u8>((p[2] >> 4) & 0x0Fu);
    const u8 sr_idx = as<u8>((p[2] >> 2) & 0x03u);
    return version != 1u && layer != 0u && br_idx != 15u && sr_idx != 3u;
}

[[nodiscard]] bool hdr_compare_raw(const u8* h1, const u8* h2) noexcept {
    return hdr_valid_raw(h2) && (((h1[1] ^ h2[1]) & 0xFEu) == 0u) && (((h1[2] ^ h2[2]) & 0x0Cu) == 0u) &&
           (((h1[2] & 0xF0u) == 0u) == ((h2[2] & 0xF0u) == 0u));
}

[[nodiscard]] bool parse_frame_header(const u8* p, usize len, FrameHeader& hdr,
                                      u16 free_format_base = 0u) noexcept {
    if (len < 4) return false;
    if (p[0] != 0xFFu || (p[1] & 0xE0u) != 0xE0u) return false;

    hdr.version = as<u8>((p[1] >> 3) & 0x03u);
    hdr.layer = as<u8>((p[1] >> 1) & 0x03u);
    hdr.error_protection = (p[1] & 0x01u) == 0u;

    if (hdr.version == 1u || hdr.layer == 0u) return false;

    const u8 bitrate_idx = as<u8>((p[2] >> 4) & 0x0Fu);
    const u8 sr_idx = as<u8>((p[2] >> 2) & 0x03u);
    hdr.padding = ((p[2] >> 1) & 0x01u) != 0u;

    if (bitrate_idx == 15u || sr_idx == 3u) return false;

    hdr.sample_rate = kSampleRates[hdr.version][sr_idx];
    if (hdr.sample_rate == 0u) return false;

    hdr.channel_mode = as<u8>((p[3] >> 6) & 0x03u);
    hdr.mode_extension = as<u8>((p[3] >> 4) & 0x03u);
    hdr.num_channels = (hdr.channel_mode == 3u) ? u16{1} : u16{2};

    if (hdr.layer == 3u) {
        hdr.samples_per_frame = 384u;
    } else if (hdr.layer == 2u) {
        hdr.samples_per_frame = 1152u;
    } else {
        hdr.samples_per_frame = (hdr.version == 3u) ? u16{1152} : u16{576};
    }

    const u32 pad_bytes = hdr.padding ? ((hdr.layer == 3u) ? 4u : 1u) : 0u;
    if (bitrate_idx == 0u) {
        if (free_format_base < 4u) return false;
        hdr.is_free_format = true;
        hdr.bitrate = as<u16>((as<u64>(free_format_base) * as<u64>(hdr.sample_rate)) /
                              (as<u64>(hdr.samples_per_frame) * 125ULL));
        hdr.frame_size = as<u16>(as<u32>(free_format_base) + pad_bytes);
        return hdr.frame_size >= 4u && hdr.frame_size <= 2308u;
    }

    hdr.is_free_format = false;
    hdr.bitrate = kBitrates[hdr.version][hdr.layer][bitrate_idx];
    if (hdr.bitrate == 0u) return false;

    u32 frame_bytes = (as<u32>(hdr.samples_per_frame) * as<u32>(hdr.bitrate) * 125u) / hdr.sample_rate;
    if (hdr.layer == 3u) {
        frame_bytes &= ~3u;
    }
    hdr.frame_size = as<u16>(frame_bytes + pad_bytes);
    return hdr.frame_size >= 4u;
}

[[nodiscard]] bool find_first_frame(const u8* d, usize start, usize end, FrameHeader& out_hdr, usize& out_pos,
                                    u16& out_free_base) noexcept {
    constexpr usize kMaxFreeFormatFrameSize = 2304u;
    out_free_base = 0u;
    usize pos = start;
    while (pos + 4u <= end) {
        const u8* p = d + pos;
        if (!hdr_valid_raw(p)) {
            ++pos;
            continue;
        }
        const u8 br_idx = as<u8>((p[2] >> 4) & 0x0Fu);
        if (br_idx != 0u) {
            if (parse_frame_header(p, end - pos, out_hdr, 0u)) {
                out_pos = pos;
                return true;
            }
            ++pos;
            continue;
        }
        // Free-format bitrate (bitrate_idx == 0): probe for consecutive matching headers to derive frame size.
        const u8 layer = as<u8>((p[1] >> 1) & 0x03u);
        const u32 pad0 = ((p[2] & 0x02u) != 0u) ? ((layer == 3u) ? 4u : 1u) : 0u;
        const usize avail = end - pos;
        const usize max_k = keel::min<usize>(kMaxFreeFormatFrameSize, avail);
        for (usize k = 4u + pad0; k + 4u <= max_k; ++k) {
            if (!hdr_compare_raw(p, p + k)) continue;
            const usize base = k - pad0;
            if (base < 4u || base > kMaxFreeFormatFrameSize) continue;
            const u32 pad1 = ((p[k + 2u] & 0x02u) != 0u) ? ((layer == 3u) ? 4u : 1u) : 0u;
            const usize next_off = k + base + pad1;
            const bool confirmed = (next_off == avail) || (next_off + 4u <= avail && hdr_compare_raw(p, p + next_off));
            if (confirmed && parse_frame_header(p, avail, out_hdr, as<u16>(base))) {
                out_free_base = as<u16>(base);
                out_pos = pos;
                return true;
            }
        }
        ++pos;
    }
    return false;
}

struct VbrHeaderInfo {
    bool is_vbr_tag_frame = false;
    bool has_frames = false;
    bool has_bytes = false;
    bool has_gapless = false;
    u32 total_frames = 0;
    u32 total_bytes = 0;
    u32 enc_delay = 0;
    u32 enc_padding = 0;
};

[[nodiscard]] VbrHeaderInfo parse_vbr_header(const u8* frame_ptr, usize avail_bytes, const FrameHeader& hdr) noexcept {
    VbrHeaderInfo vbr{};
    if (hdr.layer != 1u) return vbr;
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
                cur += 4u;
            }
            if ((flags & 0x04u) != 0u && cur + 100u <= frame_len) {
                cur += 100u; // TOC
            }
            if ((flags & 0x08u) != 0u && cur + 4u <= frame_len) {
                cur += 4u; // Quality indicator
            }
            // LAME / Lavc / Lavf Info tag extension (encoder delay + padding at +21..+23)
            if (cur + 24u <= frame_len) {
                const u8* lp = frame_ptr + cur;
                if (mem_eq(lp, "LAME", 4) || mem_eq(lp, "Lavc", 4) || mem_eq(lp, "Lavf", 4) || mem_eq(lp, "GOGO", 4)) {
                    const u8 b0 = lp[21];
                    const u8 b1 = lp[22];
                    const u8 b2 = lp[23];
                    const u32 delay = (as<u32>(b0) << 4) | (as<u32>(b1) >> 4);
                    const u32 pad = ((as<u32>(b1) & 0x0Fu) << 8) | as<u32>(b2);
                    if (delay > 0u || pad > 0u) {
                        vbr.has_gapless = true;
                        vbr.enc_delay = delay;
                        vbr.enc_padding = pad;
                    }
                }
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

// ============================================================================
// ISO/IEC 11172-3 & 13818-3 Layer III Core Decoder
// ============================================================================

constexpr i32 kMaxBitReservoirBytes = 511;
constexpr i32 kMaxL3PayloadBytes = 2304;
constexpr i32 kMaindataPadBytes = 64;
constexpr u8 kShortBlockType = 2;
constexpr u8 kStopBlockType = 3;
constexpr i32 kBitsDequantizerOut = -1;
constexpr i32 kMaxScf = 255 + kBitsDequantizerOut * 4 - 210;
constexpr i32 kMaxScfi = (kMaxScf + 3) & ~3;

[[nodiscard]] inline bool hdr_is_mono(const u8* h) noexcept { return (h[3] & 0xC0u) == 0xC0u; }
[[nodiscard]] inline bool hdr_is_ms_stereo(const u8* h) noexcept { return (h[3] & 0xE0u) == 0x60u; }
[[nodiscard]] inline bool hdr_is_crc(const u8* h) noexcept { return (h[1] & 0x01u) == 0u; }
[[nodiscard]] inline bool hdr_test_mpeg1(const u8* h) noexcept { return (h[1] & 0x08u) != 0u; }
[[nodiscard]] inline bool hdr_test_i_stereo(const u8* h) noexcept { return (h[3] & 0x10u) != 0u; }
[[nodiscard]] inline bool hdr_test_ms_stereo(const u8* h) noexcept { return (h[3] & 0x20u) != 0u; }
[[nodiscard]] inline i32 hdr_get_sample_rate_idx(const u8* h) noexcept { return as<i32>((h[2] >> 2) & 3u); }
[[nodiscard]] inline i32 hdr_get_my_sample_rate(const u8* h) noexcept {
    return hdr_get_sample_rate_idx(h) + as<i32>(((h[1] >> 3) & 1u) + ((h[1] >> 4) & 1u)) * 3;
}

struct BitStream {
    const u8* buf = nullptr;
    i32 pos = 0;
    i32 limit = 0;
};

void bs_init(BitStream& bs, const u8* data, i32 bytes) noexcept {
    bs.buf = data;
    bs.pos = 0;
    bs.limit = bytes * 8;
}

[[nodiscard]] u32 bs_get_bits(BitStream& bs, i32 n) noexcept {
    if (n <= 0) return 0u;
    const u32 s = as<u32>(bs.pos & 7);
    i32 shl = n + as<i32>(s);
    const u8* p = bs.buf + (bs.pos >> 3);
    bs.pos += n;
    if (bs.pos > bs.limit) return 0u;
    u32 cache = 0u;
    u32 next = as<u32>(*p++) & (255u >> s);
    while ((shl -= 8) > 0) {
        cache |= next << shl;
        next = as<u32>(*p++);
    }
    return cache | (next >> (-shl));
}

struct L3GranuleInfo {
    const u8* sfbtab = nullptr;
    u16 part_23_length = 0;
    u16 big_values = 0;
    u16 scalefac_compress = 0;
    u8 global_gain = 0;
    u8 block_type = 0;
    u8 mixed_block_flag = 0;
    u8 n_long_sfb = 0;
    u8 n_short_sfb = 0;
    u8 table_select[3]{};
    u8 region_count[3]{};
    u8 subblock_gain[3]{};
    u8 preflag = 0;
    u8 scalefac_scale = 0;
    u8 count1_table = 0;
    u8 scfsi = 0;
};

struct L3DecoderState {
    f32 mdct_overlap[2][9 * 32]{};
    f32 qmf_state[15 * 2 * 32]{};
    i32 reserv = 0;
    u8 header[4]{};
    u8 reserv_buf[kMaxBitReservoirBytes]{};

    void reset() noexcept {
        mem_zero(mdct_overlap, sizeof(mdct_overlap));
        mem_zero(qmf_state, sizeof(qmf_state));
        reserv = 0;
        mem_zero(header, sizeof(header));
        mem_zero(reserv_buf, sizeof(reserv_buf));
    }
};

struct L3Scratch {
    BitStream bs{};
    u8 maindata[kMaxBitReservoirBytes + kMaxL3PayloadBytes + kMaindataPadBytes]{};
    L3GranuleInfo gr_info[4]{};
    f32 grbuf[2][576]{};
    f32 scf[40]{};
    f32 syn[18 + 15][2 * 32]{};
    u8 ist_pos[2][39]{};
};

[[nodiscard]] i32 l3_read_side_info(BitStream& bs, L3GranuleInfo* gr, const u8* hdr) noexcept {
    static constexpr u8 kScfLong[8][23] = {
        {6, 6, 6, 6, 6, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 38, 46, 52, 60, 68, 58, 54, 0},
        {12, 12, 12, 12, 12, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 76, 90, 2, 2, 2, 2, 2, 0},
        {6, 6, 6, 6, 6, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 38, 46, 52, 60, 68, 58, 54, 0},
        {6, 6, 6, 6, 6, 6, 8, 10, 12, 14, 16, 18, 22, 26, 32, 38, 46, 54, 62, 70, 76, 36, 0},
        {6, 6, 6, 6, 6, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 38, 46, 52, 60, 68, 58, 54, 0},
        {4, 4, 4, 4, 4, 4, 6, 6, 8, 8, 10, 12, 16, 20, 24, 28, 34, 42, 50, 54, 76, 158, 0},
        {4, 4, 4, 4, 4, 4, 6, 6, 6, 8, 10, 12, 16, 18, 22, 28, 34, 40, 46, 54, 54, 192, 0},
        {4, 4, 4, 4, 4, 4, 6, 6, 8, 10, 12, 16, 20, 24, 30, 38, 46, 56, 68, 84, 102, 26, 0},
    };
    static constexpr u8 kScfShort[8][40] = {
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14, 14, 18, 18, 18, 24, 24, 24, 30, 30, 30, 40, 40, 40, 18, 18, 18, 0},
        {8, 8, 8, 8, 8, 8, 8, 8, 8, 12, 12, 12, 16, 16, 16, 20, 20, 20, 24, 24, 24, 28, 28, 28, 36, 36, 36, 2, 2, 2, 2, 2, 2, 2, 2, 2, 26, 26, 26, 0},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 8, 8, 8, 10, 10, 10, 14, 14, 14, 18, 18, 18, 26, 26, 26, 32, 32, 32, 42, 42, 42, 18, 18, 18, 0},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14, 14, 18, 18, 18, 24, 24, 24, 32, 32, 32, 44, 44, 44, 12, 12, 12, 0},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14, 14, 18, 18, 18, 24, 24, 24, 30, 30, 30, 40, 40, 40, 18, 18, 18, 0},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14, 14, 18, 18, 18, 22, 22, 22, 30, 30, 30, 56, 56, 56, 0},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 10, 10, 10, 12, 12, 12, 14, 14, 14, 16, 16, 16, 20, 20, 20, 26, 26, 26, 66, 66, 66, 0},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 8, 8, 8, 12, 12, 12, 16, 16, 16, 20, 20, 20, 26, 26, 26, 34, 34, 34, 42, 42, 42, 12, 12, 12, 0},
    };
    static constexpr u8 kScfMixed[8][40] = {
        {6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14, 14, 18, 18, 18, 24, 24, 24, 30, 30, 30, 40, 40, 40, 18, 18, 18, 0},
        {12, 12, 12, 4, 4, 4, 8, 8, 8, 12, 12, 12, 16, 16, 16, 20, 20, 20, 24, 24, 24, 28, 28, 28, 36, 36, 36, 2, 2, 2, 2, 2, 2, 2, 2, 2, 26, 26, 26, 0},
        {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 10, 10, 10, 14, 14, 14, 18, 18, 18, 26, 26, 26, 32, 32, 32, 42, 42, 42, 18, 18, 18, 0},
        {6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14, 14, 18, 18, 18, 24, 24, 24, 32, 32, 32, 44, 44, 44, 12, 12, 12, 0},
        {6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14, 14, 18, 18, 18, 24, 24, 24, 30, 30, 30, 40, 40, 40, 18, 18, 18, 0},
        {4, 4, 4, 4, 4, 4, 6, 6, 4, 4, 4, 6, 6, 6, 8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14, 14, 18, 18, 18, 22, 22, 22, 30, 30, 30, 56, 56, 56, 0},
        {4, 4, 4, 4, 4, 4, 6, 6, 4, 4, 4, 6, 6, 6, 6, 6, 6, 10, 10, 10, 12, 12, 12, 14, 14, 14, 16, 16, 16, 20, 20, 20, 26, 26, 26, 66, 66, 66, 0},
        {4, 4, 4, 4, 4, 4, 6, 6, 4, 4, 4, 6, 6, 6, 8, 8, 8, 12, 12, 12, 16, 16, 16, 20, 20, 20, 26, 26, 26, 34, 34, 34, 42, 42, 42, 12, 12, 12, 0},
    };

    u32 tables = 0;
    u32 scfsi = 0;
    i32 main_data_begin = 0;
    i32 part_23_sum = 0;
    i32 gr_count = hdr_is_mono(hdr) ? 1 : 2;
    i32 sr_idx = hdr_get_my_sample_rate(hdr);
    sr_idx -= (sr_idx != 0) ? 1 : 0;
    if (sr_idx < 0 || sr_idx >= 8) return -1;

    if (hdr_test_mpeg1(hdr)) {
        gr_count *= 2;
        main_data_begin = as<i32>(bs_get_bits(bs, 9));
        scfsi = bs_get_bits(bs, 7 + gr_count);
    } else {
        main_data_begin = as<i32>(bs_get_bits(bs, 8 + gr_count) >> gr_count);
    }

    do {
        if (hdr_is_mono(hdr)) {
            scfsi <<= 4;
        }
        gr->part_23_length = as<u16>(bs_get_bits(bs, 12));
        part_23_sum += as<i32>(gr->part_23_length);
        gr->big_values = as<u16>(bs_get_bits(bs, 9));
        if (gr->big_values > 288u) {
            return -1;
        }
        gr->global_gain = as<u8>(bs_get_bits(bs, 8));
        gr->scalefac_compress = as<u16>(bs_get_bits(bs, hdr_test_mpeg1(hdr) ? 4 : 9));
        gr->sfbtab = kScfLong[sr_idx];
        gr->n_long_sfb = 22u;
        gr->n_short_sfb = 0u;
        if (bs_get_bits(bs, 1) != 0u) {
            gr->block_type = as<u8>(bs_get_bits(bs, 2));
            if (gr->block_type == 0u) {
                return -1;
            }
            gr->mixed_block_flag = as<u8>(bs_get_bits(bs, 1));
            gr->region_count[0] = 7u;
            gr->region_count[1] = 255u;
            gr->region_count[2] = 255u;
            if (gr->block_type == kShortBlockType) {
                scfsi &= 0x0F0Fu;
                if (gr->mixed_block_flag == 0u) {
                    gr->region_count[0] = 8u;
                    gr->sfbtab = kScfShort[sr_idx];
                    gr->n_long_sfb = 0u;
                    gr->n_short_sfb = 39u;
                } else {
                    gr->sfbtab = kScfMixed[sr_idx];
                    gr->n_long_sfb = hdr_test_mpeg1(hdr) ? u8{8} : u8{6};
                    gr->n_short_sfb = 30u;
                }
            }
            tables = bs_get_bits(bs, 10) << 5;
            gr->subblock_gain[0] = as<u8>(bs_get_bits(bs, 3));
            gr->subblock_gain[1] = as<u8>(bs_get_bits(bs, 3));
            gr->subblock_gain[2] = as<u8>(bs_get_bits(bs, 3));
        } else {
            gr->block_type = 0u;
            gr->mixed_block_flag = 0u;
            tables = bs_get_bits(bs, 15);
            gr->region_count[0] = as<u8>(bs_get_bits(bs, 4));
            gr->region_count[1] = as<u8>(bs_get_bits(bs, 3));
            gr->region_count[2] = 255u;
            gr->subblock_gain[0] = 0u;
            gr->subblock_gain[1] = 0u;
            gr->subblock_gain[2] = 0u;
        }
        gr->table_select[0] = as<u8>((tables >> 10) & 31u);
        gr->table_select[1] = as<u8>((tables >> 5) & 31u);
        gr->table_select[2] = as<u8>(tables & 31u);
        gr->preflag = hdr_test_mpeg1(hdr) ? as<u8>(bs_get_bits(bs, 1)) : as<u8>(gr->scalefac_compress >= 500u ? 1u : 0u);
        gr->scalefac_scale = as<u8>(bs_get_bits(bs, 1));
        gr->count1_table = as<u8>(bs_get_bits(bs, 1));
        gr->scfsi = as<u8>((scfsi >> 12) & 15u);
        scfsi <<= 4;
        ++gr;
    } while (--gr_count > 0);

    if (part_23_sum + bs.pos > bs.limit + main_data_begin * 8) {
        return -1;
    }
    return main_data_begin;
}

void l3_read_scalefactors(u8* scf, u8* ist_pos, const u8* scf_size, const u8* scf_count, BitStream& bitbuf,
                          i32 scfsi) noexcept {
    for (i32 i = 0; i < 4 && scf_count[i] != 0u; ++i, scfsi *= 2) {
        const i32 cnt = as<i32>(scf_count[i]);
        if ((scfsi & 8) != 0) {
            mem_copy(scf, ist_pos, as<usize>(cnt));
        } else {
            const i32 bits = as<i32>(scf_size[i]);
            if (bits == 0) {
                mem_zero(scf, as<usize>(cnt));
                mem_zero(ist_pos, as<usize>(cnt));
            } else {
                const i32 max_scf = (scfsi < 0) ? ((1 << bits) - 1) : -1;
                for (i32 k = 0; k < cnt; ++k) {
                    const i32 s = as<i32>(bs_get_bits(bitbuf, bits));
                    ist_pos[k] = (s == max_scf) ? u8{0xFF} : as<u8>(s);
                    scf[k] = as<u8>(s);
                }
            }
        }
        ist_pos += cnt;
        scf += cnt;
    }
    scf[0] = scf[1] = scf[2] = 0u;
}

[[nodiscard]] f32 l3_ldexp_q2(f32 y, i32 exp_q2) noexcept {
    static constexpr f32 kExpFrac[4] = {
        9.31322575e-10f, 7.83145814e-10f, 6.58544508e-10f, 5.53767716e-10f,
    };
    if (exp_q2 < 0) exp_q2 = 0;
    do {
        const i32 e = keel::min(30 * 4, exp_q2);
        y *= kExpFrac[e & 3] * as<f32>(1 << (30 - (e >> 2)));
        exp_q2 -= e;
    } while (exp_q2 > 0);
    return y;
}

void l3_decode_scalefactors(const u8* hdr, u8* ist_pos, BitStream& bs, const L3GranuleInfo& gr, f32* scf,
                            i32 ch) noexcept {
    static constexpr u8 kScfPartitions[3][28] = {
        {6, 5, 5, 5, 6, 5, 5, 5, 6, 5, 7, 3, 11, 10, 0, 0, 7, 7, 7, 0, 6, 6, 6, 3, 8, 8, 5, 0},
        {8, 9, 6, 12, 6, 9, 9, 9, 6, 9, 12, 6, 15, 18, 0, 0, 6, 15, 12, 0, 6, 12, 9, 6, 6, 18, 9, 0},
        {9, 9, 6, 12, 9, 9, 9, 9, 9, 9, 12, 6, 18, 18, 0, 0, 12, 12, 12, 0, 12, 9, 9, 6, 15, 12, 9, 0},
    };
    const i32 part_row = (gr.n_short_sfb != 0u ? 1 : 0) + (gr.n_long_sfb == 0u ? 1 : 0);
    const u8* scf_partition = kScfPartitions[part_row];
    u8 scf_size[4]{};
    u8 iscf[40]{};
    const i32 scf_shift = as<i32>(gr.scalefac_scale) + 1;
    i32 scfsi = as<i32>(gr.scfsi);

    if (hdr_test_mpeg1(hdr)) {
        static constexpr u8 kScfcDecode[16] = {0, 1, 2, 3, 12, 5, 6, 7, 9, 10, 11, 13, 14, 15, 18, 19};
        const i32 part = as<i32>(kScfcDecode[gr.scalefac_compress & 15u]);
        scf_size[1] = scf_size[0] = as<u8>(part >> 2);
        scf_size[3] = scf_size[2] = as<u8>(part & 3);
    } else {
        static constexpr u8 kMod[6 * 4] = {
            5, 5, 4, 4, 5, 5, 4, 1, 4, 3, 1, 1, 5, 6, 6, 1, 4, 4, 4, 1, 4, 3, 1, 1,
        };
        const i32 ist = (hdr_test_i_stereo(hdr) && ch != 0) ? 1 : 0;
        i32 sfc = as<i32>(gr.scalefac_compress >> ist);
        i32 k = ist * 12;
        for (; sfc >= 0 && k < 24; k += 4) {
            i32 modprod = 1;
            for (i32 i = 3; i >= 0; --i) {
                scf_size[i] = as<u8>((sfc / modprod) % as<i32>(kMod[k + i]));
                modprod *= as<i32>(kMod[k + i]);
            }
            sfc -= modprod;
        }
        scf_partition += k;
        scfsi = -16;
    }

    l3_read_scalefactors(iscf, ist_pos, scf_size, scf_partition, bs, scfsi);

    if (gr.n_short_sfb != 0u) {
        const i32 sh = 3 - scf_shift;
        for (i32 i = 0; i < as<i32>(gr.n_short_sfb) && (as<i32>(gr.n_long_sfb) + i + 2) < 40; i += 3) {
            iscf[gr.n_long_sfb + i + 0] = as<u8>(iscf[gr.n_long_sfb + i + 0] + (gr.subblock_gain[0] << sh));
            iscf[gr.n_long_sfb + i + 1] = as<u8>(iscf[gr.n_long_sfb + i + 1] + (gr.subblock_gain[1] << sh));
            iscf[gr.n_long_sfb + i + 2] = as<u8>(iscf[gr.n_long_sfb + i + 2] + (gr.subblock_gain[2] << sh));
        }
    } else if (gr.preflag != 0u) {
        static constexpr u8 kPreamp[10] = {1, 1, 1, 1, 2, 2, 3, 3, 3, 2};
        for (i32 i = 0; i < 10; ++i) {
            iscf[11 + i] = as<u8>(iscf[11 + i] + kPreamp[i]);
        }
    }

    const i32 gain_exp =
        as<i32>(gr.global_gain) + kBitsDequantizerOut * 4 - 210 - (hdr_is_ms_stereo(hdr) ? 2 : 0);
    const f32 gain = l3_ldexp_q2(as<f32>(1 << (kMaxScfi / 4)), kMaxScfi - gain_exp);
    const i32 total_sfb = keel::min(as<i32>(gr.n_long_sfb) + as<i32>(gr.n_short_sfb), 39);
    for (i32 i = 0; i < total_sfb; ++i) {
        scf[i] = l3_ldexp_q2(gain, as<i32>(iscf[i]) << scf_shift);
    }
}

constexpr f32 kPow43[129 + 16] = {
    0, -1, -2.519842f, -4.326749f, -6.349604f, -8.549880f, -10.902724f, -13.390518f, -16.000000f, -18.720754f,
    -21.544347f, -24.463781f, -27.473142f, -30.567351f, -33.741992f, -36.993181f,
    0, 1, 2.519842f, 4.326749f, 6.349604f, 8.549880f, 10.902724f, 13.390518f, 16.000000f, 18.720754f, 21.544347f,
    24.463781f, 27.473142f, 30.567351f, 33.741992f, 36.993181f, 40.317474f, 43.711787f, 47.173345f, 50.699631f,
    54.288352f, 57.937408f, 61.644865f, 65.408941f, 69.227979f, 73.100443f, 77.024898f, 81.000000f, 85.024491f,
    89.097188f, 93.216975f, 97.382800f, 101.593667f, 105.848633f, 110.146801f, 114.487321f, 118.869381f, 123.292209f,
    127.755065f, 132.257246f, 136.798076f, 141.376907f, 145.993119f, 150.646117f, 155.335327f, 160.060199f, 164.820202f,
    169.614826f, 174.443577f, 179.305980f, 184.201575f, 189.129918f, 194.090580f, 199.083145f, 204.107210f, 209.162385f,
    214.248292f, 219.364564f, 224.510845f, 229.686789f, 234.892058f, 240.126328f, 245.389280f, 250.680604f, 256.000000f,
    261.347174f, 266.721841f, 272.123723f, 277.552547f, 283.008049f, 288.489971f, 293.998060f, 299.532071f, 305.091761f,
    310.676898f, 316.287249f, 321.922592f, 327.582707f, 333.267377f, 338.976394f, 344.709550f, 350.466646f, 356.247482f,
    362.051866f, 367.879608f, 373.730522f, 379.604427f, 385.501143f, 391.420496f, 397.362314f, 403.326427f, 409.312672f,
    415.320884f, 421.350905f, 427.402579f, 433.475750f, 439.570269f, 445.685987f, 451.822757f, 457.980436f, 464.158883f,
    470.357960f, 476.577530f, 482.817459f, 489.077615f, 495.357868f, 501.658090f, 507.978156f, 514.317941f, 520.677324f,
    527.056184f, 533.454404f, 539.871867f, 546.308458f, 552.764065f, 559.238575f, 565.731879f, 572.243870f, 578.774440f,
    585.323483f, 591.890898f, 598.476581f, 605.080431f, 611.702349f, 618.342238f, 625.000000f, 631.675540f, 638.368763f,
    645.079578f,
};

[[nodiscard]] f32 l3_pow_43(i32 x) noexcept {
    if (x < 0) x = 0;
    if (x > 8206) x = 8206;
    if (x < 129) {
        return kPow43[16 + x];
    }
    i32 mult = 256;
    if (x < 1024) {
        mult = 16;
        x <<= 3;
    }
    const i32 sign = (2 * x) & 64;
    const f32 frac = as<f32>((x & 63) - sign) / as<f32>((x & ~63) + sign);
    return kPow43[16 + ((x + sign) >> 6)] * (1.0f + frac * ((4.0f / 3.0f) + frac * (2.0f / 9.0f))) * as<f32>(mult);
}

void l3_huffman(f32* dst, BitStream& bs, const L3GranuleInfo& gr_info, const f32* scf, i32 layer3gr_limit,
                const u8* buf_end) noexcept {
    static constexpr i16 kTabs[] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        785, 785, 785, 785, 784, 784, 784, 784, 513, 513, 513, 513, 513, 513, 513, 513, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
        -255, 1313, 1298, 1282, 785, 785, 785, 785, 784, 784, 784, 784, 769, 769, 769, 769, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 290, 288,
        -255, 1313, 1298, 1282, 769, 769, 769, 769, 529, 529, 529, 529, 529, 529, 529, 529, 528, 528, 528, 528, 528, 528, 528, 528, 512, 512, 512, 512, 512, 512, 512, 512, 290, 288,
        -253, -318, -351, -367, 785, 785, 785, 785, 784, 784, 784, 784, 769, 769, 769, 769, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 819, 818, 547, 547, 275, 275, 275, 275, 561, 560, 515, 546, 289, 274, 288, 258,
        -254, -287, 1329, 1299, 1314, 1312, 1057, 1057, 1042, 1042, 1026, 1026, 784, 784, 784, 784, 529, 529, 529, 529, 529, 529, 529, 529, 769, 769, 769, 769, 768, 768, 768, 768, 563, 560, 306, 306, 291, 259,
        -252, -413, -477, -542, 1298, -575, 1041, 1041, 784, 784, 784, 784, 769, 769, 769, 769, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, -383, -399, 1107, 1092, 1106, 1061, 849, 849, 789, 789, 1104, 1091, 773, 773, 1076, 1075, 341, 340, 325, 309, 834, 804, 577, 577, 532, 532, 516, 516, 832, 818, 803, 816, 561, 561, 531, 531, 515, 546, 289, 289, 288, 258,
        -252, -429, -493, -559, 1057, 1057, 1042, 1042, 529, 529, 529, 529, 529, 529, 529, 529, 784, 784, 784, 784, 769, 769, 769, 769, 512, 512, 512, 512, 512, 512, 512, 512, -382, 1077, -415, 1106, 1061, 1104, 849, 849, 789, 789, 1091, 1076, 1029, 1075, 834, 834, 597, 581, 340, 340, 339, 324, 804, 833, 532, 532, 832, 772, 818, 803, 817, 787, 816, 771, 290, 290, 290, 290, 288, 258,
        -253, -349, -414, -447, -463, 1329, 1299, -479, 1314, 1312, 1057, 1057, 1042, 1042, 1026, 1026, 785, 785, 785, 785, 784, 784, 784, 784, 769, 769, 769, 769, 768, 768, 768, 768, -319, 851, 821, -335, 836, 850, 805, 849, 341, 340, 325, 336, 533, 533, 579, 579, 564, 564, 773, 832, 578, 548, 563, 516, 321, 276, 306, 291, 304, 259,
        -251, -572, -733, -830, -863, -879, 1041, 1041, 784, 784, 784, 784, 769, 769, 769, 769, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, -511, -527, -543, 1396, 1351, 1381, 1366, 1395, 1335, 1380, -559, 1334, 1138, 1138, 1063, 1063, 1350, 1392, 1031, 1031, 1062, 1062, 1364, 1363, 1120, 1120, 1333, 1348, 881, 881, 881, 881, 375, 374, 359, 373, 343, 358, 341, 325, 791, 791, 1123, 1122, -703, 1105, 1045, -719, 865, 865, 790, 790, 774, 774, 1104, 1029, 338, 293, 323, 308, -799, -815, 833, 788, 772, 818, 803, 816, 322, 292, 307, 320, 561, 531, 515, 546, 289, 274, 288, 258,
        -251, -525, -605, -685, -765, -831, -846, 1298, 1057, 1057, 1312, 1282, 785, 785, 785, 785, 784, 784, 784, 784, 769, 769, 769, 769, 512, 512, 512, 512, 512, 512, 512, 512, 1399, 1398, 1383, 1367, 1382, 1396, 1351, -511, 1381, 1366, 1139, 1139, 1079, 1079, 1124, 1124, 1364, 1349, 1363, 1333, 882, 882, 882, 882, 807, 807, 807, 807, 1094, 1094, 1136, 1136, 373, 341, 535, 535, 881, 775, 867, 822, 774, -591, 324, 338, -671, 849, 550, 550, 866, 864, 609, 609, 293, 336, 534, 534, 789, 835, 773, -751, 834, 804, 308, 307, 833, 788, 832, 772, 562, 562, 547, 547, 305, 275, 560, 515, 290, 290,
        -252, -397, -477, -557, -622, -653, -719, -735, -750, 1329, 1299, 1314, 1057, 1057, 1042, 1042, 1312, 1282, 1024, 1024, 785, 785, 785, 785, 784, 784, 784, 784, 769, 769, 769, 769, -383, 1127, 1141, 1111, 1126, 1140, 1095, 1110, 869, 869, 883, 883, 1079, 1109, 882, 882, 375, 374, 807, 868, 838, 881, 791, -463, 867, 822, 368, 263, 852, 837, 836, -543, 610, 610, 550, 550, 352, 336, 534, 534, 865, 774, 851, 821, 850, 805, 593, 533, 579, 564, 773, 832, 578, 578, 548, 548, 577, 577, 307, 276, 306, 291, 516, 560, 259, 259,
        -250, -2107, -2507, -2764, -2909, -2974, -3007, -3023, 1041, 1041, 1040, 1040, 769, 769, 769, 769, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, -767, -1052, -1213, -1277, -1358, -1405, -1469, -1535, -1550, -1582, -1614, -1647, -1662, -1694, -1726, -1759, -1774, -1807, -1822, -1854, -1886, 1565, -1919, -1935, -1951, -1967, 1731, 1730, 1580, 1717, -1983, 1729, 1564, -1999, 1548, -2015, -2031, 1715, 1595, -2047, 1714, -2063, 1610, -2079, 1609, -2095, 1323, 1323, 1457, 1457, 1307, 1307, 1712, 1547, 1641, 1700, 1699, 1594, 1685, 1625, 1442, 1442, 1322, 1322, -780, -973, -910, 1279, 1278, 1277, 1262, 1276, 1261, 1275, 1215, 1260, 1229, -959, 974, 974, 989, 989, -943, 735, 478, 478, 495, 463, 506, 414, -1039, 1003, 958, 1017, 927, 942, 987, 957, 431, 476, 1272, 1167, 1228, -1183, 1256, -1199, 895, 895, 941, 941, 1242, 1227, 1212, 1135, 1014, 1014, 490, 489, 503, 487, 910, 1013, 985, 925, 863, 894, 970, 955, 1012, 847, -1343, 831, 755, 755, 984, 909, 428, 366, 754, 559, -1391, 752, 486, 457, 924, 997, 698, 698, 983, 893, 740, 740, 908, 877, 739, 739, 667, 667, 953, 938, 497, 287, 271, 271, 683, 606, 590, 712, 726, 574, 302, 302, 738, 736, 481, 286, 526, 725, 605, 711, 636, 724, 696, 651, 589, 681, 666, 710, 364, 467, 573, 695, 466, 466, 301, 465, 379, 379, 709, 604, 665, 679, 316, 316, 634, 633, 436, 436, 464, 269, 424, 394, 452, 332, 438, 363, 347, 408, 393, 448, 331, 422, 362, 407, 392, 421, 346, 406, 391, 376, 375, 359, 1441, 1306, -2367, 1290, -2383, 1337, -2399, -2415, 1426, 1321, -2431, 1411, 1336, -2447, -2463, -2479, 1169, 1169, 1049, 1049, 1424, 1289, 1412, 1352, 1319, -2495, 1154, 1154, 1064, 1064, 1153, 1153, 416, 390, 360, 404, 403, 389, 344, 374, 373, 343, 358, 372, 327, 357, 342, 311, 356, 326, 1395, 1394, 1137, 1137, 1047, 1047, 1365, 1392, 1287, 1379, 1334, 1364, 1349, 1378, 1318, 1363, 792, 792, 792, 792, 1152, 1152, 1032, 1032, 1121, 1121, 1046, 1046, 1120, 1120, 1030, 1030, -2895, 1106, 1061, 1104, 849, 849, 789, 789, 1091, 1076, 1029, 1090, 1060, 1075, 833, 833, 309, 324, 532, 532, 832, 772, 818, 803, 561, 561, 531, 560, 515, 546, 289, 274, 288, 258,
        -250, -1179, -1579, -1836, -1996, -2124, -2253, -2333, -2413, -2477, -2542, -2574, -2607, -2622, -2655, 1314, 1313, 1298, 1312, 1282, 785, 785, 785, 785, 1040, 1040, 1025, 1025, 768, 768, 768, 768, -766, -798, -830, -862, -895, -911, -927, -943, -959, -975, -991, -1007, -1023, -1039, -1055, -1070, 1724, 1647, -1103, -1119, 1631, 1767, 1662, 1738, 1708, 1723, -1135, 1780, 1615, 1779, 1599, 1677, 1646, 1778, 1583, -1151, 1777, 1567, 1737, 1692, 1765, 1722, 1707, 1630, 1751, 1661, 1764, 1614, 1736, 1676, 1763, 1750, 1645, 1598, 1721, 1691, 1762, 1706, 1582, 1761, 1566, -1167, 1749, 1629, 767, 766, 751, 765, 494, 494, 735, 764, 719, 749, 734, 763, 447, 447, 748, 718, 477, 506, 431, 491, 446, 476, 461, 505, 415, 430, 475, 445, 504, 399, 460, 489, 414, 503, 383, 474, 429, 459, 502, 502, 746, 752, 488, 398, 501, 473, 413, 472, 486, 271, 480, 270, -1439, -1455, 1357, -1471, -1487, -1503, 1341, 1325, -1519, 1489, 1463, 1403, 1309, -1535, 1372, 1448, 1418, 1476, 1356, 1462, 1387, -1551, 1475, 1340, 1447, 1402, 1386, -1567, 1068, 1068, 1474, 1461, 455, 380, 468, 440, 395, 425, 410, 454, 364, 467, 466, 464, 453, 269, 409, 448, 268, 432, 1371, 1473, 1432, 1417, 1308, 1460, 1355, 1446, 1459, 1431, 1083, 1083, 1401, 1416, 1458, 1445, 1067, 1067, 1370, 1457, 1051, 1051, 1291, 1430, 1385, 1444, 1354, 1415, 1400, 1443, 1082, 1082, 1173, 1113, 1186, 1066, 1185, 1050, -1967, 1158, 1128, 1172, 1097, 1171, 1081, -1983, 1157, 1112, 416, 266, 375, 400, 1170, 1142, 1127, 1065, 793, 793, 1169, 1033, 1156, 1096, 1141, 1111, 1155, 1080, 1126, 1140, 898, 898, 808, 808, 897, 897, 792, 792, 1095, 1152, 1032, 1125, 1110, 1139, 1079, 1124, 882, 807, 838, 881, 853, 791, -2319, 867, 368, 263, 822, 852, 837, 866, 806, 865, -2399, 851, 352, 262, 534, 534, 821, 836, 594, 594, 549, 549, 593, 593, 533, 533, 848, 773, 579, 579, 564, 578, 548, 563, 276, 276, 577, 576, 306, 291, 516, 560, 305, 305, 275, 259,
        -251, -892, -2058, -2620, -2828, -2957, -3023, -3039, 1041, 1041, 1040, 1040, 769, 769, 769, 769, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, -511, -527, -543, -559, 1530, -575, -591, 1528, 1527, 1407, 1526, 1391, 1023, 1023, 1023, 1023, 1525, 1375, 1268, 1268, 1103, 1103, 1087, 1087, 1039, 1039, 1523, -604, 815, 815, 815, 815, 510, 495, 509, 479, 508, 463, 507, 447, 431, 505, 415, 399, -734, -782, 1262, -815, 1259, 1244, -831, 1258, 1228, -847, -863, 1196, -879, 1253, 987, 987, 748, -767, 493, 493, 462, 477, 414, 414, 686, 669, 478, 446, 461, 445, 474, 429, 487, 458, 412, 471, 1266, 1264, 1009, 1009, 799, 799, -1019, -1276, -1452, -1581, -1677, -1757, -1821, -1886, -1933, -1997, 1257, 1257, 1483, 1468, 1512, 1422, 1497, 1406, 1467, 1496, 1421, 1510, 1134, 1134, 1225, 1225, 1466, 1451, 1374, 1405, 1252, 1252, 1358, 1480, 1164, 1164, 1251, 1251, 1238, 1238, 1389, 1465, -1407, 1054, 1101, -1423, 1207, -1439, 830, 830, 1248, 1038, 1237, 1117, 1223, 1148, 1236, 1208, 411, 426, 395, 410, 379, 269, 1193, 1222, 1132, 1235, 1221, 1116, 976, 976, 1192, 1162, 1177, 1220, 1131, 1191, 963, 963, -1647, 961, 780, -1663, 558, 558, 994, 993, 437, 408, 393, 407, 829, 978, 813, 797, 947, -1743, 721, 721, 377, 392, 844, 950, 828, 890, 706, 706, 812, 859, 796, 960, 948, 843, 934, 874, 571, 571, -1919, 690, 555, 689, 421, 346, 539, 539, 944, 779, 918, 873, 932, 842, 903, 888, 570, 570, 931, 917, 674, 674, -2575, 1562, -2591, 1609, -2607, 1654, 1322, 1322, 1441, 1441, 1696, 1546, 1683, 1593, 1669, 1624, 1426, 1426, 1321, 1321, 1639, 1680, 1425, 1425, 1305, 1305, 1545, 1668, 1608, 1623, 1667, 1592, 1638, 1666, 1320, 1320, 1652, 1607, 1409, 1409, 1304, 1304, 1288, 1288, 1664, 1637, 1395, 1395, 1335, 1335, 1622, 1636, 1394, 1394, 1319, 1319, 1606, 1621, 1392, 1392, 1137, 1137, 1137, 1137, 345, 390, 360, 375, 404, 373, 1047, -2751, -2767, -2783, 1062, 1121, 1046, -2799, 1077, -2815, 1106, 1061, 789, 789, 1105, 1104, 263, 355, 310, 340, 325, 354, 352, 262, 339, 324, 1091, 1076, 1029, 1090, 1060, 1075, 833, 833, 788, 788, 1088, 1028, 818, 818, 803, 803, 561, 561, 531, 531, 816, 771, 546, 546, 289, 274, 288, 258,
        -253, -317, -381, -446, -478, -509, 1279, 1279, -811, -1179, -1451, -1756, -1900, -2028, -2189, -2253, -2333, -2414, -2445, -2511, -2526, 1313, 1298, -2559, 1041, 1041, 1040, 1040, 1025, 1025, 1024, 1024, 1022, 1007, 1021, 991, 1020, 975, 1019, 959, 687, 687, 1018, 1017, 671, 671, 655, 655, 1016, 1015, 639, 639, 758, 758, 623, 623, 757, 607, 756, 591, 755, 575, 754, 559, 543, 543, 1009, 783, -575, -621, -685, -749, 496, -590, 750, 749, 734, 748, 974, 989, 1003, 958, 988, 973, 1002, 942, 987, 957, 972, 1001, 926, 986, 941, 971, 956, 1000, 910, 985, 925, 999, 894, 970, -1071, -1087, -1102, 1390, -1135, 1436, 1509, 1451, 1374, -1151, 1405, 1358, 1480, 1420, -1167, 1507, 1494, 1389, 1342, 1465, 1435, 1450, 1326, 1505, 1310, 1493, 1373, 1479, 1404, 1492, 1464, 1419, 428, 443, 472, 397, 736, 526, 464, 464, 486, 457, 442, 471, 484, 482, 1357, 1449, 1434, 1478, 1388, 1491, 1341, 1490, 1325, 1489, 1463, 1403, 1309, 1477, 1372, 1448, 1418, 1433, 1476, 1356, 1462, 1387, -1439, 1475, 1340, 1447, 1402, 1474, 1324, 1461, 1371, 1473, 269, 448, 1432, 1417, 1308, 1460, -1711, 1459, -1727, 1441, 1099, 1099, 1446, 1386, 1431, 1401, -1743, 1289, 1083, 1083, 1160, 1160, 1458, 1445, 1067, 1067, 1370, 1457, 1307, 1430, 1129, 1129, 1098, 1098, 268, 432, 267, 416, 266, 400, -1887, 1144, 1187, 1082, 1173, 1113, 1186, 1066, 1050, 1158, 1128, 1143, 1172, 1097, 1171, 1081, 420, 391, 1157, 1112, 1170, 1142, 1127, 1065, 1169, 1049, 1156, 1096, 1141, 1111, 1155, 1080, 1126, 1154, 1064, 1153, 1140, 1095, 1048, -2159, 1125, 1110, 1137, -2175, 823, 823, 1139, 1138, 807, 807, 384, 264, 368, 263, 868, 838, 853, 791, 867, 822, 852, 837, 866, 806, 865, 790, -2319, 851, 821, 836, 352, 262, 850, 805, 849, -2399, 533, 533, 835, 820, 336, 261, 578, 548, 563, 577, 532, 532, 832, 772, 562, 562, 547, 547, 305, 275, 560, 515, 290, 290, 288, 258,
    };
    static constexpr u8 kTab32[] = {
        130, 162, 193, 209, 44, 28, 76, 140, 9, 9, 9, 9, 9, 9, 9, 9, 190, 254, 222, 238, 126, 94, 157, 157, 109, 61,
        173, 205,
    };
    static constexpr u8 kTab33[] = {
        252, 236, 220, 204, 188, 172, 156, 140, 124, 108, 92, 76, 60, 44, 28, 12,
    };
    static constexpr i16 kTabIndex[32] = {
        0, 32, 64, 98, 0, 132, 180, 218, 292, 364, 426, 538, 648, 746, 0, 1126,
        1460, 1460, 1460, 1460, 1460, 1460, 1460, 1460, 1842, 1842, 1842, 1842, 1842, 1842, 1842, 1842,
    };
    static constexpr u8 kLinbits[32] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 2, 3, 4, 6, 8, 10, 13, 4, 5, 6, 7, 8, 9, 11, 13,
    };

    if (bs.pos > layer3gr_limit || bs.pos < 0 || bs.pos > bs.limit) {
        bs.pos = layer3gr_limit;
        return;
    }

    f32* dst_end = dst + 576;
    f32 one = 0.0f;
    i32 ireg = 0;
    i32 big_val_cnt = as<i32>(gr_info.big_values);
    const u8* sfb = gr_info.sfbtab;
    const u8* bs_next_ptr = bs.buf + (bs.pos / 8);
    if (bs_next_ptr + 4 > buf_end) {
        bs.pos = layer3gr_limit;
        return;
    }
    u32 bs_cache = ((((as<u32>(bs_next_ptr[0]) * 256u + as<u32>(bs_next_ptr[1])) * 256u +
                      as<u32>(bs_next_ptr[2])) * 256u + as<u32>(bs_next_ptr[3])) << as<u32>(bs.pos & 7));
    i32 bs_sh = (bs.pos & 7) - 8;
    bs_next_ptr += 4;

    auto peek_bits = [&](i32 n) -> u32 { return bs_cache >> (32 - n); };
    auto flush_bits = [&](i32 n) {
        bs_cache <<= n;
        bs_sh += n;
    };
    auto check_bits = [&]() {
        while (bs_sh >= 0) {
            const u32 byte_val = (bs_next_ptr < buf_end) ? as<u32>(*bs_next_ptr++) : 0u;
            bs_cache |= byte_val << bs_sh;
            bs_sh -= 8;
        }
    };
    auto bs_bit_pos = [&]() -> i32 {
        return as<i32>(bs_next_ptr - bs.buf) * 8 - 24 + bs_sh;
    };

    while (big_val_cnt > 0 && ireg < 3) {
        const i32 tab_num = as<i32>(gr_info.table_select[ireg]);
        i32 sfb_cnt = as<i32>(gr_info.region_count[ireg++]);
        const i16* codebook = kTabs + kTabIndex[tab_num];
        const i32 linbits = as<i32>(kLinbits[tab_num]);
        if (linbits > 0) {
            do {
                const i32 np = as<i32>(*sfb++) / 2;
                if (np <= 0) {
                    big_val_cnt = 0;
                    break;
                }
                i32 pairs_to_decode = keel::min(big_val_cnt, np);
                one = *scf++;
                do {
                    if (dst + 2 > dst_end) {
                        big_val_cnt = 0;
                        break;
                    }
                    i32 w = 5;
                    i32 leaf = codebook[peek_bits(w)];
                    while (leaf < 0) {
                        flush_bits(w);
                        w = leaf & 7;
                        leaf = codebook[as<i32>(peek_bits(w)) - (leaf >> 3)];
                    }
                    flush_bits(leaf >> 8);
                    for (i32 j = 0; j < 2; ++j, ++dst, leaf >>= 4) {
                        i32 lsb = leaf & 0x0F;
                        if (lsb == 15) {
                            lsb += as<i32>(peek_bits(linbits));
                            flush_bits(linbits);
                            check_bits();
                            *dst = one * l3_pow_43(lsb) * (as<i32>(bs_cache) < 0 ? -1.0f : 1.0f);
                        } else {
                            *dst = kPow43[16 + lsb - 16 * as<i32>(bs_cache >> 31)] * one;
                        }
                        if (lsb != 0) flush_bits(1);
                    }
                    check_bits();
                } while (--pairs_to_decode > 0);
                big_val_cnt -= np;
            } while (big_val_cnt > 0 && --sfb_cnt >= 0);
        } else {
            do {
                const i32 np = as<i32>(*sfb++) / 2;
                if (np <= 0) {
                    big_val_cnt = 0;
                    break;
                }
                i32 pairs_to_decode = keel::min(big_val_cnt, np);
                one = *scf++;
                do {
                    if (dst + 2 > dst_end) {
                        big_val_cnt = 0;
                        break;
                    }
                    i32 w = 5;
                    i32 leaf = codebook[peek_bits(w)];
                    while (leaf < 0) {
                        flush_bits(w);
                        w = leaf & 7;
                        leaf = codebook[as<i32>(peek_bits(w)) - (leaf >> 3)];
                    }
                    flush_bits(leaf >> 8);
                    for (i32 j = 0; j < 2; ++j, ++dst, leaf >>= 4) {
                        const i32 lsb = leaf & 0x0F;
                        *dst = kPow43[16 + lsb - 16 * as<i32>(bs_cache >> 31)] * one;
                        if (lsb != 0) flush_bits(1);
                    }
                    check_bits();
                } while (--pairs_to_decode > 0);
                big_val_cnt -= np;
            } while (big_val_cnt > 0 && --sfb_cnt >= 0);
        }
    }

    for (i32 np = 1 - big_val_cnt; dst + 4 <= dst_end; dst += 4) {
        const u8* codebook_count1 = (gr_info.count1_table != 0u) ? kTab33 : kTab32;
        i32 leaf = as<i32>(codebook_count1[peek_bits(4)]);
        if ((leaf & 8) == 0) {
            leaf = as<i32>(codebook_count1[(leaf >> 3) + as<i32>((bs_cache << 4) >> (32 - (leaf & 3)))]);
        }
        flush_bits(leaf & 7);
        if (bs_bit_pos() > layer3gr_limit) {
            break;
        }
        if (--np == 0) {
            np = as<i32>(*sfb++) / 2;
            if (np == 0) break;
            one = *scf++;
        }
        if ((leaf & 128) != 0) {
            dst[0] = (as<i32>(bs_cache) < 0) ? -one : one;
            flush_bits(1);
        }
        if ((leaf & 64) != 0) {
            dst[1] = (as<i32>(bs_cache) < 0) ? -one : one;
            flush_bits(1);
        }
        if (--np == 0) {
            np = as<i32>(*sfb++) / 2;
            if (np == 0) break;
            one = *scf++;
        }
        if ((leaf & 32) != 0) {
            dst[2] = (as<i32>(bs_cache) < 0) ? -one : one;
            flush_bits(1);
        }
        if ((leaf & 16) != 0) {
            dst[3] = (as<i32>(bs_cache) < 0) ? -one : one;
            flush_bits(1);
        }
        check_bits();
    }

    bs.pos = layer3gr_limit;
}

void l3_midside_stereo(f32* left, i32 n) noexcept {
    f32* right = left + 576;
    for (i32 i = 0; i < n; ++i) {
        const f32 a = left[i];
        const f32 b = right[i];
        left[i] = a + b;
        right[i] = a - b;
    }
}

void l3_intensity_stereo_band(f32* left, i32 n, f32 kl, f32 kr) noexcept {
    for (i32 i = 0; i < n; ++i) {
        left[i + 576] = left[i] * kr;
        left[i] = left[i] * kl;
    }
}

void l3_stereo_top_band(const f32* right, const u8* sfb, i32 nbands, i32 max_band[3]) noexcept {
    max_band[0] = max_band[1] = max_band[2] = -1;
    for (i32 i = 0; i < nbands; ++i) {
        for (i32 k = 0; k < as<i32>(sfb[i]); k += 2) {
            if (right[k] != 0.0f || right[k + 1] != 0.0f) {
                max_band[i % 3] = i;
                break;
            }
        }
        right += sfb[i];
    }
}

void l3_stereo_process(f32* left, const u8* ist_pos, const u8* sfb, const u8* hdr, i32 max_band[3],
                       i32 mpeg2_sh) noexcept {
    static constexpr f32 kPan[14] = {
        0.0f, 1.0f, 0.21132487f, 0.78867513f, 0.36602540f, 0.63397460f, 0.5f, 0.5f,
        0.63397460f, 0.36602540f, 0.78867513f, 0.21132487f, 1.0f, 0.0f,
    };
    const bool mpeg1 = hdr_test_mpeg1(hdr);
    const u32 max_pos = mpeg1 ? 7u : 64u;
    for (i32 i = 0; sfb[i] != 0u; ++i) {
        const u32 ipos = ist_pos[i];
        if (i > max_band[i % 3] && ipos < max_pos) {
            f32 kl = 1.0f;
            f32 kr = 1.0f;
            const f32 s = hdr_test_ms_stereo(hdr) ? 1.41421356f : 1.0f;
            if (mpeg1) {
                kl = kPan[2u * ipos];
                kr = kPan[2u * ipos + 1u];
            } else {
                kr = l3_ldexp_q2(1.0f, as<i32>(((ipos + 1u) >> 1) << mpeg2_sh));
                if ((ipos & 1u) != 0u) {
                    kl = kr;
                    kr = 1.0f;
                }
            }
            l3_intensity_stereo_band(left, as<i32>(sfb[i]), kl * s, kr * s);
        } else if (hdr_test_ms_stereo(hdr)) {
            l3_midside_stereo(left, as<i32>(sfb[i]));
        }
        left += sfb[i];
    }
}

void l3_intensity_stereo(f32* left, u8* ist_pos, const L3GranuleInfo* gr, const u8* hdr) noexcept {
    i32 max_band[3]{};
    const i32 n_sfb = as<i32>(gr->n_long_sfb) + as<i32>(gr->n_short_sfb);
    const i32 max_blocks = (gr->n_short_sfb != 0u) ? 3 : 1;

    l3_stereo_top_band(left + 576, gr->sfbtab, n_sfb, max_band);
    if (gr->n_long_sfb != 0u) {
        max_band[0] = max_band[1] = max_band[2] = keel::max(keel::max(max_band[0], max_band[1]), max_band[2]);
    }
    for (i32 i = 0; i < max_blocks; ++i) {
        const u8 default_pos = hdr_test_mpeg1(hdr) ? u8{3} : u8{0};
        const i32 itop = n_sfb - max_blocks + i;
        const i32 prev = itop - max_blocks;
        if (itop >= 0 && itop < 39 && prev >= 0 && prev < 39) {
            ist_pos[itop] = (max_band[i] >= prev) ? default_pos : ist_pos[prev];
        }
    }
    l3_stereo_process(left, ist_pos, gr->sfbtab, hdr, max_band, as<i32>(gr[1].scalefac_compress & 1u));
}

void l3_reorder(f32* grbuf, f32* scratch, const u8* sfb) noexcept {
    f32* src = grbuf;
    f32* dst = scratch;
    for (i32 len = as<i32>(*sfb); len != 0; sfb += 3, src += 2 * len, len = as<i32>(*sfb)) {
        for (i32 i = 0; i < len; ++i, ++src) {
            *dst++ = src[0 * len];
            *dst++ = src[1 * len];
            *dst++ = src[2 * len];
        }
    }
    mem_copy(grbuf, scratch, as<usize>(dst - scratch) * sizeof(f32));
}

void l3_antialias(f32* grbuf, i32 nbands) noexcept {
    static constexpr f32 kAa[2][8] = {
        {0.85749293f, 0.88174200f, 0.94962865f, 0.98331459f, 0.99551782f, 0.99916056f, 0.99989920f, 0.99999316f},
        {0.51449576f, 0.47173197f, 0.31337745f, 0.18191320f, 0.09457419f, 0.04096558f, 0.01419856f, 0.00369997f},
    };
    for (; nbands > 0; --nbands, grbuf += 18) {
        for (i32 i = 0; i < 8; ++i) {
            const f32 u = grbuf[18 + i];
            const f32 d = grbuf[17 - i];
            grbuf[18 + i] = u * kAa[0][i] - d * kAa[1][i];
            grbuf[17 - i] = u * kAa[1][i] + d * kAa[0][i];
        }
    }
}

void l3_dct3_9(f32* y) noexcept {
    f32 s0 = y[0], s2 = y[2], s4 = y[4], s6 = y[6], s8 = y[8];
    f32 t0 = s0 + s6 * 0.5f;
    s0 -= s6;
    f32 t4 = (s4 + s2) * 0.93969262f;
    f32 t2 = (s8 + s2) * 0.76604444f;
    s6 = (s4 - s8) * 0.17364818f;
    s4 += s8 - s2;

    s2 = s0 - s4 * 0.5f;
    y[4] = s4 + s0;
    s8 = t0 - t2 + s6;
    s0 = t0 - t4 + t2;
    s4 = t0 + t4 - s6;

    f32 s1 = y[1], s3 = y[3], s5 = y[5], s7 = y[7];
    s3 *= 0.86602540f;
    t0 = (s5 + s1) * 0.98480775f;
    t4 = (s5 - s7) * 0.34202014f;
    t2 = (s1 + s7) * 0.64278761f;
    s1 = (s1 - s5 - s7) * 0.86602540f;

    s5 = t0 - s3 - t2;
    s7 = t4 - s3 - t0;
    s3 = t4 + s3 - t2;

    y[0] = s4 - s7;
    y[1] = s2 + s1;
    y[2] = s0 - s3;
    y[3] = s8 + s5;
    y[5] = s8 - s5;
    y[6] = s0 + s3;
    y[7] = s2 - s1;
    y[8] = s4 + s7;
}

void l3_imdct36(f32* grbuf, f32* overlap, const f32* window, i32 nbands) noexcept {
    static constexpr f32 kTwid9[18] = {
        0.73727734f, 0.79335334f, 0.84339145f, 0.88701083f, 0.92387953f, 0.95371695f, 0.97629601f, 0.99144486f,
        0.99904822f, 0.67559021f, 0.60876143f, 0.53729961f, 0.46174861f, 0.38268343f, 0.30070580f, 0.21643961f,
        0.13052619f, 0.04361938f,
    };
    for (i32 j = 0; j < nbands; ++j, grbuf += 18, overlap += 9) {
        f32 co[9], si[9];
        co[0] = -grbuf[0];
        si[0] = grbuf[17];
        for (i32 i = 0; i < 4; ++i) {
            si[8 - 2 * i] = grbuf[4 * i + 1] - grbuf[4 * i + 2];
            co[1 + 2 * i] = grbuf[4 * i + 1] + grbuf[4 * i + 2];
            si[7 - 2 * i] = grbuf[4 * i + 4] - grbuf[4 * i + 3];
            co[2 + 2 * i] = -(grbuf[4 * i + 3] + grbuf[4 * i + 4]);
        }
        l3_dct3_9(co);
        l3_dct3_9(si);
        si[1] = -si[1];
        si[3] = -si[3];
        si[5] = -si[5];
        si[7] = -si[7];

        for (i32 i = 0; i < 9; ++i) {
            const f32 ovl = overlap[i];
            const f32 sum = co[i] * kTwid9[9 + i] + si[i] * kTwid9[i];
            overlap[i] = co[i] * kTwid9[i] - si[i] * kTwid9[9 + i];
            grbuf[i] = ovl * window[i] - sum * window[9 + i];
            grbuf[17 - i] = ovl * window[9 + i] + sum * window[i];
        }
    }
}

void l3_idct3(f32 x0, f32 x1, f32 x2, f32* dst) noexcept {
    const f32 m1 = x1 * 0.86602540f;
    const f32 a1 = x0 - x2 * 0.5f;
    dst[1] = x0 + x2;
    dst[0] = a1 + m1;
    dst[2] = a1 - m1;
}

void l3_imdct12(const f32* x, f32* dst, f32* overlap) noexcept {
    static constexpr f32 kTwid3[6] = {
        0.79335334f, 0.92387953f, 0.99144486f, 0.60876143f, 0.38268343f, 0.13052619f,
    };
    f32 co[3], si[3];
    l3_idct3(-x[0], x[6] + x[3], x[12] + x[9], co);
    l3_idct3(x[15], x[12] - x[9], x[6] - x[3], si);
    si[1] = -si[1];

    for (i32 i = 0; i < 3; ++i) {
        const f32 ovl = overlap[i];
        const f32 sum = co[i] * kTwid3[3 + i] + si[i] * kTwid3[i];
        overlap[i] = co[i] * kTwid3[i] - si[i] * kTwid3[3 + i];
        dst[i] = ovl * kTwid3[2 - i] - sum * kTwid3[5 - i];
        dst[5 - i] = ovl * kTwid3[5 - i] + sum * kTwid3[2 - i];
    }
}

void l3_imdct_short(f32* grbuf, f32* overlap, i32 nbands) noexcept {
    for (; nbands > 0; --nbands, overlap += 9, grbuf += 18) {
        f32 tmp[18];
        mem_copy(tmp, grbuf, sizeof(tmp));
        mem_copy(grbuf, overlap, 6 * sizeof(f32));
        l3_imdct12(tmp, grbuf + 6, overlap + 6);
        l3_imdct12(tmp + 1, grbuf + 12, overlap + 6);
        l3_imdct12(tmp + 2, overlap, overlap + 6);
    }
}

void l3_change_sign(f32* grbuf) noexcept {
    grbuf += 18;
    for (i32 b = 0; b < 32; b += 2, grbuf += 36) {
        for (i32 i = 1; i < 18; i += 2) {
            grbuf[i] = -grbuf[i];
        }
    }
}

void l3_imdct_gr(f32* grbuf, f32* overlap, u32 block_type, i32 n_long_bands) noexcept {
    static constexpr f32 kMdctWindow[2][18] = {
        {0.99904822f, 0.99144486f, 0.97629601f, 0.95371695f, 0.92387953f, 0.88701083f, 0.84339145f, 0.79335334f,
         0.73727734f, 0.04361938f, 0.13052619f, 0.21643961f, 0.30070580f, 0.38268343f, 0.46174861f, 0.53729961f,
         0.60876143f, 0.67559021f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.99144486f, 0.92387953f, 0.79335334f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
         0.13052619f, 0.38268343f, 0.60876143f},
    };
    if (n_long_bands > 0) {
        l3_imdct36(grbuf, overlap, kMdctWindow[0], n_long_bands);
        grbuf += 18 * n_long_bands;
        overlap += 9 * n_long_bands;
    }
    if (block_type == kShortBlockType) {
        l3_imdct_short(grbuf, overlap, 32 - n_long_bands);
    } else {
        l3_imdct36(grbuf, overlap, kMdctWindow[block_type == kStopBlockType ? 1 : 0], 32 - n_long_bands);
    }
}

void l3_save_reservoir(L3DecoderState& h, const L3Scratch& s) noexcept {
    i32 pos = (s.bs.pos + 7) / 8;
    i32 remains = (s.bs.limit / 8) - pos;
    if (remains > kMaxBitReservoirBytes) {
        pos += remains - kMaxBitReservoirBytes;
        remains = kMaxBitReservoirBytes;
    }
    if (remains > 0 && pos >= 0 && pos + remains <= kMaxBitReservoirBytes + kMaxL3PayloadBytes) {
        mem_move(h.reserv_buf, s.maindata + pos, as<usize>(remains));
        h.reserv = remains;
    } else {
        h.reserv = 0;
    }
}

[[nodiscard]] bool l3_restore_reservoir(L3DecoderState& h, const BitStream& bs, L3Scratch& s,
                                        i32 main_data_begin) noexcept {
    const i32 frame_bytes = (bs.limit - bs.pos) / 8;
    if (frame_bytes < 0 || frame_bytes > kMaxL3PayloadBytes) {
        h.reserv = 0;
        return false;
    }
    const i32 bytes_have = keel::min(h.reserv, main_data_begin);
    if (bytes_have > 0) {
        mem_copy(s.maindata, h.reserv_buf + keel::max(0, h.reserv - main_data_begin), as<usize>(bytes_have));
    }
    if (frame_bytes > 0) {
        mem_copy(s.maindata + bytes_have, bs.buf + (bs.pos / 8), as<usize>(frame_bytes));
    }
    mem_zero(s.maindata + bytes_have + frame_bytes, as<usize>(kMaindataPadBytes));
    bs_init(s.bs, s.maindata, bytes_have + frame_bytes);
    return h.reserv >= main_data_begin;
}

void l3_decode_granule(L3DecoderState& h, L3Scratch& s, L3GranuleInfo* gr_info, i32 nch) noexcept {
    const u8* buf_end = s.maindata + sizeof(s.maindata);
    for (i32 ch = 0; ch < nch; ++ch) {
        const i32 layer3gr_limit = s.bs.pos + as<i32>(gr_info[ch].part_23_length);
        l3_decode_scalefactors(h.header, s.ist_pos[ch], s.bs, gr_info[ch], s.scf, ch);
        l3_huffman(s.grbuf[ch], s.bs, gr_info[ch], s.scf, layer3gr_limit, buf_end);
    }

    if (nch == 2) {
        if (hdr_test_i_stereo(h.header)) {
            l3_intensity_stereo(s.grbuf[0], s.ist_pos[1], gr_info, h.header);
        } else if (hdr_is_ms_stereo(h.header)) {
            l3_midside_stereo(s.grbuf[0], 576);
        }
    }

    for (i32 ch = 0; ch < nch; ++ch, ++gr_info) {
        i32 aa_bands = 31;
        const i32 n_long_bands = (gr_info->mixed_block_flag != 0u ? 2 : 0)
                                 << (hdr_get_my_sample_rate(h.header) == 2 ? 1 : 0);
        if (gr_info->n_short_sfb != 0u) {
            aa_bands = n_long_bands - 1;
            l3_reorder(s.grbuf[ch] + n_long_bands * 18, s.syn[0], gr_info->sfbtab + gr_info->n_long_sfb);
        }
        l3_antialias(s.grbuf[ch], aa_bands);
        l3_imdct_gr(s.grbuf[ch], h.mdct_overlap[ch], gr_info->block_type, n_long_bands);
        l3_change_sign(s.grbuf[ch]);
    }
}

void mp3d_dct_ii(f32* grbuf, i32 n) noexcept {
    static constexpr f32 kSec[24] = {
        10.19000816f, 0.50060302f, 0.50241929f, 3.40760851f, 0.50547093f, 0.52249861f, 2.05778098f, 0.51544732f,
        0.56694406f, 1.48416460f, 0.53104258f, 0.64682180f, 1.16943991f, 0.55310392f, 0.78815460f, 0.97256821f,
        0.58293498f, 1.06067765f, 0.83934963f, 0.62250412f, 1.72244716f, 0.74453628f, 0.67480832f, 5.10114861f,
    };
    for (i32 k = 0; k < n; ++k) {
        f32 t[4][8];
        f32* y = grbuf + k;
        f32* x = t[0];
        for (i32 i = 0; i < 8; ++i, ++x) {
            const f32 x0 = y[i * 18];
            const f32 x1 = y[(15 - i) * 18];
            const f32 x2 = y[(16 + i) * 18];
            const f32 x3 = y[(31 - i) * 18];
            const f32 t0 = x0 + x3;
            const f32 t1 = x1 + x2;
            const f32 t2 = (x1 - x2) * kSec[3 * i + 0];
            const f32 t3 = (x0 - x3) * kSec[3 * i + 1];
            x[0] = t0 + t1;
            x[8] = (t0 - t1) * kSec[3 * i + 2];
            x[16] = t3 + t2;
            x[24] = (t3 - t2) * kSec[3 * i + 2];
        }
        x = t[0];
        for (i32 i = 0; i < 4; ++i, x += 8) {
            f32 x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4], x5 = x[5], x6 = x[6], x7 = x[7];
            f32 xt = x0 - x7;
            x0 += x7;
            x7 = x1 - x6;
            x1 += x6;
            x6 = x2 - x5;
            x2 += x5;
            x5 = x3 - x4;
            x3 += x4;
            x4 = x0 - x3;
            x0 += x3;
            x3 = x1 - x2;
            x1 += x2;
            x[0] = x0 + x1;
            x[4] = (x0 - x1) * 0.70710677f;
            x5 = x5 + x6;
            x6 = (x6 + x7) * 0.70710677f;
            x7 = x7 + xt;
            x3 = (x3 + x4) * 0.70710677f;
            x5 -= x7 * 0.198912367f;
            x7 += x5 * 0.382683432f;
            x5 -= x7 * 0.198912367f;
            x0 = xt - x6;
            xt += x6;
            x[1] = (xt + x7) * 0.50979561f;
            x[2] = (x4 + x3) * 0.54119611f;
            x[3] = (x0 - x5) * 0.60134488f;
            x[5] = (x0 + x5) * 0.89997619f;
            x[6] = (x4 - x3) * 1.30656302f;
            x[7] = (xt - x7) * 2.56291556f;
        }
        for (i32 i = 0; i < 7; ++i, y += 4 * 18) {
            y[0 * 18] = t[0][i];
            y[1 * 18] = t[2][i] + t[3][i] + t[3][i + 1];
            y[2 * 18] = t[1][i] + t[1][i + 1];
            y[3 * 18] = t[2][i + 1] + t[3][i] + t[3][i + 1];
        }
        y[0 * 18] = t[0][7];
        y[1 * 18] = t[2][7] + t[3][7];
        y[2 * 18] = t[1][7];
        y[3 * 18] = t[3][7];
    }
}

[[nodiscard]] inline f32 scale_pcm_f32(f32 sample) noexcept {
    return sample * (1.0f / 32768.0f);
}

void mp3d_synth_pair(f32* pcm, i32 nch, const f32* z) noexcept {
    f32 a = (z[14 * 64] - z[0]) * 29.0f;
    a += (z[1 * 64] + z[13 * 64]) * 213.0f;
    a += (z[12 * 64] - z[2 * 64]) * 459.0f;
    a += (z[3 * 64] + z[11 * 64]) * 2037.0f;
    a += (z[10 * 64] - z[4 * 64]) * 5153.0f;
    a += (z[5 * 64] + z[9 * 64]) * 6574.0f;
    a += (z[8 * 64] - z[6 * 64]) * 37489.0f;
    a += z[7 * 64] * 75038.0f;
    pcm[0] = scale_pcm_f32(a);

    z += 2;
    a = z[14 * 64] * 104.0f;
    a += z[12 * 64] * 1567.0f;
    a += z[10 * 64] * 9727.0f;
    a += z[8 * 64] * 64019.0f;
    a += z[6 * 64] * -9975.0f;
    a += z[4 * 64] * -45.0f;
    a += z[2 * 64] * 146.0f;
    a += z[0 * 64] * -5.0f;
    pcm[16 * nch] = scale_pcm_f32(a);
}

void mp3d_synth(const f32* xl, f32* dstl, i32 nch, f32* lins) noexcept {
    const f32* xr = xl + 576 * (nch - 1);
    f32* dstr = dstl + (nch - 1);

    static constexpr f32 kWin[] = {
        -1, 26, -31, 208, 218, 401, -519, 2063, 2000, 4788, -5517, 7134, 5959, 35640, -39336, 74992,
        -1, 24, -35, 202, 222, 347, -581, 2080, 1952, 4425, -5879, 7640, 5288, 33791, -41176, 74856,
        -1, 21, -38, 196, 225, 294, -645, 2087, 1893, 4063, -6237, 8092, 4561, 31947, -43006, 74630,
        -1, 19, -41, 190, 227, 244, -711, 2085, 1822, 3705, -6589, 8492, 3776, 30112, -44821, 74313,
        -1, 17, -45, 183, 228, 197, -779, 2075, 1739, 3351, -6935, 8840, 2935, 28289, -46617, 73908,
        -1, 16, -49, 176, 228, 153, -848, 2057, 1644, 3004, -7271, 9139, 2037, 26482, -48390, 73415,
        -2, 14, -53, 169, 227, 111, -919, 2032, 1535, 2663, -7597, 9389, 1082, 24694, -50137, 72835,
        -2, 13, -58, 161, 224, 72, -991, 2001, 1414, 2330, -7910, 9592, 70, 22929, -51853, 72169,
        -2, 11, -63, 154, 221, 36, -1064, 1962, 1280, 2006, -8209, 9750, -998, 21189, -53534, 71420,
        -2, 10, -68, 147, 215, 2, -1137, 1919, 1131, 1692, -8491, 9863, -2122, 19478, -55178, 70590,
        -3, 9, -73, 139, 208, -29, -1210, 1870, 970, 1388, -8755, 9935, -3300, 17799, -56778, 69679,
        -3, 8, -79, 132, 200, -57, -1283, 1817, 794, 1095, -8998, 9966, -4533, 16155, -58333, 68692,
        -4, 7, -85, 125, 189, -83, -1356, 1759, 605, 814, -9219, 9959, -5818, 14548, -59838, 67629,
        -4, 7, -91, 117, 177, -106, -1428, 1698, 402, 545, -9416, 9916, -7154, 12980, -61289, 66494,
        -5, 6, -97, 111, 163, -127, -1498, 1634, 185, 288, -9585, 9838, -8540, 11455, -62684, 65290,
    };

    f32* zlin = lins + 15 * 64;
    const f32* w = kWin;

    zlin[4 * 15] = xl[18 * 16];
    zlin[4 * 15 + 1] = xr[18 * 16];
    zlin[4 * 15 + 2] = xl[0];
    zlin[4 * 15 + 3] = xr[0];

    zlin[4 * 31] = xl[1 + 18 * 16];
    zlin[4 * 31 + 1] = xr[1 + 18 * 16];
    zlin[4 * 31 + 2] = xl[1];
    zlin[4 * 31 + 3] = xr[1];

    mp3d_synth_pair(dstr, nch, lins + 4 * 15 + 1);
    mp3d_synth_pair(dstr + 32 * nch, nch, lins + 4 * 15 + 64 + 1);
    mp3d_synth_pair(dstl, nch, lins + 4 * 15);
    mp3d_synth_pair(dstl + 32 * nch, nch, lins + 4 * 15 + 64);

    for (i32 i = 14; i >= 0; --i) {
        f32 a[4]{};
        f32 b[4]{};

        zlin[4 * i] = xl[18 * (31 - i)];
        zlin[4 * i + 1] = xr[18 * (31 - i)];
        zlin[4 * i + 2] = xl[1 + 18 * (31 - i)];
        zlin[4 * i + 3] = xr[1 + 18 * (31 - i)];
        zlin[4 * (i + 16)] = xl[1 + 18 * (1 + i)];
        zlin[4 * (i + 16) + 1] = xr[1 + 18 * (1 + i)];
        zlin[4 * (i - 16) + 2] = xl[18 * (1 + i)];
        zlin[4 * (i - 16) + 3] = xr[18 * (1 + i)];

        auto step_pos = [&](i32 k, bool first) {
            const f32 w0 = *w++;
            const f32 w1 = *w++;
            const f32* vz = &zlin[4 * i - k * 64];
            const f32* vy = &zlin[4 * i - (15 - k) * 64];
            for (i32 j = 0; j < 4; ++j) {
                if (first) {
                    b[j] = vz[j] * w1 + vy[j] * w0;
                    a[j] = vz[j] * w0 - vy[j] * w1;
                } else {
                    b[j] += vz[j] * w1 + vy[j] * w0;
                    a[j] += vz[j] * w0 - vy[j] * w1;
                }
            }
        };
        auto step_neg = [&](i32 k) {
            const f32 w0 = *w++;
            const f32 w1 = *w++;
            const f32* vz = &zlin[4 * i - k * 64];
            const f32* vy = &zlin[4 * i - (15 - k) * 64];
            for (i32 j = 0; j < 4; ++j) {
                b[j] += vz[j] * w1 + vy[j] * w0;
                a[j] += vy[j] * w1 - vz[j] * w0;
            }
        };

        step_pos(0, true);
        step_neg(1);
        step_pos(2, false);
        step_neg(3);
        step_pos(4, false);
        step_neg(5);
        step_pos(6, false);
        step_neg(7);

        dstr[(15 - i) * nch] = scale_pcm_f32(a[1]);
        dstr[(17 + i) * nch] = scale_pcm_f32(b[1]);
        dstl[(15 - i) * nch] = scale_pcm_f32(a[0]);
        dstl[(17 + i) * nch] = scale_pcm_f32(b[0]);
        dstr[(47 - i) * nch] = scale_pcm_f32(a[3]);
        dstr[(49 + i) * nch] = scale_pcm_f32(b[3]);
        dstl[(47 - i) * nch] = scale_pcm_f32(a[2]);
        dstl[(49 + i) * nch] = scale_pcm_f32(b[2]);
    }
}

void mp3d_synth_granule(f32* qmf_state, f32* grbuf, i32 nbands, i32 nch, f32* pcm, f32* lins) noexcept {
    for (i32 i = 0; i < nch; ++i) {
        mp3d_dct_ii(grbuf + 576 * i, nbands);
    }
    mem_copy(lins, qmf_state, sizeof(f32) * 15 * 64);
    mem_zero(lins + 15 * 64, sizeof(f32) * 64 * as<usize>(nbands));

    for (i32 i = 0; i < nbands; i += 2) {
        mp3d_synth(grbuf + i, pcm + 32 * nch * i, nch, lins + i * 64);
    }
    if (nch == 1) {
        for (i32 i = 0; i < 15 * 64; i += 2) {
            qmf_state[i] = lins[nbands * 64 + i];
        }
    } else {
        mem_copy(qmf_state, lins + nbands * 64, sizeof(f32) * 15 * 64);
    }
}

// ============================================================================
// ISO/IEC 11172-3 & 13818-3 Layer I / II Core Decoder (CC0 minimp3 port)
// ============================================================================

struct L12ScaleInfo {
    f32 scf[3 * 64]{};
    u8 total_bands = 0;
    u8 stereo_bands = 0;
    u8 bitalloc[64]{};
    u8 scfcod[64]{};
};

struct L12SubbandAlloc {
    u8 tab_offset = 0;
    u8 code_tab_width = 0;
    u8 band_count = 0;
};

[[nodiscard]] const L12SubbandAlloc* l12_subband_alloc_table(const u8* hdr_bytes, const FrameHeader& hdr,
                                                             L12ScaleInfo& sci) noexcept {
    static constexpr L12SubbandAlloc kAllocL1[] = {{76, 4, 32}};
    static constexpr L12SubbandAlloc kAllocL2M2[] = {{60, 4, 4}, {44, 3, 7}, {44, 2, 19}};
    static constexpr L12SubbandAlloc kAllocL2M1[] = {{0, 4, 3}, {16, 4, 8}, {32, 3, 12}, {40, 2, 7}};
    static constexpr L12SubbandAlloc kAllocL2M1Low[] = {{44, 4, 2}, {44, 3, 10}};

    const i32 mode = as<i32>(hdr.channel_mode);
    const i32 stereo_bands =
        (mode == 3) ? 0 : ((mode == 1) ? ((as<i32>(hdr.mode_extension) << 2) + 4) : 32);

    const L12SubbandAlloc* alloc = kAllocL1;
    i32 nbands = 32;
    if (hdr.layer == 3u) {
        alloc = kAllocL1;
        nbands = 32;
    } else if (!hdr_test_mpeg1(hdr_bytes)) {
        alloc = kAllocL2M2;
        nbands = 30;
    } else {
        const i32 sr_idx = hdr_get_sample_rate_idx(hdr_bytes);
        u32 kbps = as<u32>(hdr.bitrate) >> ((mode != 3) ? 1u : 0u);
        if (hdr.is_free_format || kbps == 0u) {
            kbps = 192u;
        }
        alloc = kAllocL2M1;
        nbands = 27;
        if (kbps < 56u) {
            alloc = kAllocL2M1Low;
            nbands = (sr_idx == 2) ? 12 : 8;
        } else if (kbps >= 96u && sr_idx != 1) {
            nbands = 30;
        }
    }

    sci.total_bands = as<u8>(nbands);
    sci.stereo_bands = as<u8>(keel::min(stereo_bands, nbands));
    return alloc;
}

void l12_read_scalefactors(BitStream& bs, const u8* pba, const u8* scfcod, i32 bands, f32* scf) noexcept {
    static constexpr f32 kDeqL12[18 * 3] = {
        9.53674316e-07f / 3.0f,     7.56931807e-07f / 3.0f,     6.00777173e-07f / 3.0f,
        9.53674316e-07f / 7.0f,     7.56931807e-07f / 7.0f,     6.00777173e-07f / 7.0f,
        9.53674316e-07f / 15.0f,    7.56931807e-07f / 15.0f,    6.00777173e-07f / 15.0f,
        9.53674316e-07f / 31.0f,    7.56931807e-07f / 31.0f,    6.00777173e-07f / 31.0f,
        9.53674316e-07f / 63.0f,    7.56931807e-07f / 63.0f,    6.00777173e-07f / 63.0f,
        9.53674316e-07f / 127.0f,   7.56931807e-07f / 127.0f,   6.00777173e-07f / 127.0f,
        9.53674316e-07f / 255.0f,   7.56931807e-07f / 255.0f,   6.00777173e-07f / 255.0f,
        9.53674316e-07f / 511.0f,   7.56931807e-07f / 511.0f,   6.00777173e-07f / 511.0f,
        9.53674316e-07f / 1023.0f,  7.56931807e-07f / 1023.0f,  6.00777173e-07f / 1023.0f,
        9.53674316e-07f / 2047.0f,  7.56931807e-07f / 2047.0f,  6.00777173e-07f / 2047.0f,
        9.53674316e-07f / 4095.0f,  7.56931807e-07f / 4095.0f,  6.00777173e-07f / 4095.0f,
        9.53674316e-07f / 8191.0f,  7.56931807e-07f / 8191.0f,  6.00777173e-07f / 8191.0f,
        9.53674316e-07f / 16383.0f, 7.56931807e-07f / 16383.0f, 6.00777173e-07f / 16383.0f,
        9.53674316e-07f / 32767.0f, 7.56931807e-07f / 32767.0f, 6.00777173e-07f / 32767.0f,
        9.53674316e-07f / 65535.0f, 7.56931807e-07f / 65535.0f, 6.00777173e-07f / 65535.0f,
        9.53674316e-07f / 3.0f,     7.56931807e-07f / 3.0f,     6.00777173e-07f / 3.0f,
        9.53674316e-07f / 5.0f,     7.56931807e-07f / 5.0f,     6.00777173e-07f / 5.0f,
        9.53674316e-07f / 9.0f,     7.56931807e-07f / 9.0f,     6.00777173e-07f / 9.0f,
    };

    for (i32 i = 0; i < bands; ++i) {
        f32 s = 0.0f;
        const i32 ba = as<i32>(pba[i]);
        const i32 mask = (ba >= 2 && ba <= 19) ? (4 + ((19 >> scfcod[i]) & 3)) : 0;
        for (i32 m = 4; m != 0; m >>= 1) {
            if ((mask & m) != 0) {
                const u32 b = bs_get_bits(bs, 6);
                const usize idx = as<usize>(ba * 3 - 6) + as<usize>(b % 3u);
                const u32 shift = b / 3u;
                const u32 pow2 = (shift <= 21u) ? (u32{1} << (21u - shift)) : 0u;
                s = kDeqL12[idx] * as<f32>(pow2);
            }
            *scf++ = s;
        }
    }
}

void l12_read_scale_info(const u8* hdr_bytes, const FrameHeader& hdr, BitStream& bs, L12ScaleInfo& sci) noexcept {
    static constexpr u8 kBitallocCodeTab[92] = {
        0, 17, 3,  4,  5,  6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        0, 17, 18, 3,  19, 4, 5, 6, 7, 8,  9,  10, 11, 12, 13, 16,
        0, 17, 18, 3,  19, 4, 5, 16,
        0, 17, 18, 16,
        0, 17, 18, 19, 4,  5, 6, 7, 8, 9,  10, 11, 12, 13, 14, 15,
        0, 17, 18, 3,  19, 4, 5, 6, 7, 8,  9,  10, 11, 12, 13, 14,
        0, 2,  3,  4,  5,  6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    };

    const L12SubbandAlloc* subband_alloc = l12_subband_alloc_table(hdr_bytes, hdr, sci);
    i32 k = 0;
    i32 ba_bits = 0;
    const u8* ba_code_tab = kBitallocCodeTab;

    for (i32 i = 0; i < as<i32>(sci.total_bands); ++i) {
        if (i == k) {
            k += as<i32>(subband_alloc->band_count);
            ba_bits = as<i32>(subband_alloc->code_tab_width);
            ba_code_tab = kBitallocCodeTab + subband_alloc->tab_offset;
            ++subband_alloc;
        }
        u8 ba = ba_code_tab[bs_get_bits(bs, ba_bits)];
        sci.bitalloc[2 * i] = ba;
        if (i < as<i32>(sci.stereo_bands)) {
            ba = ba_code_tab[bs_get_bits(bs, ba_bits)];
        }
        sci.bitalloc[2 * i + 1] = (sci.stereo_bands != 0u) ? ba : u8{0};
    }

    for (i32 i = 0; i < 2 * as<i32>(sci.total_bands); ++i) {
        sci.scfcod[i] =
            (sci.bitalloc[i] != 0u) ? ((hdr.layer == 3u) ? u8{2} : as<u8>(bs_get_bits(bs, 2))) : u8{6};
    }

    l12_read_scalefactors(bs, sci.bitalloc, sci.scfcod, as<i32>(sci.total_bands) * 2, sci.scf);

    for (i32 i = as<i32>(sci.stereo_bands); i < as<i32>(sci.total_bands); ++i) {
        sci.bitalloc[2 * i + 1] = 0u;
    }
}

[[nodiscard]] i32 l12_dequantize_granule(f32* grbuf, BitStream& bs, const L12ScaleInfo& sci, i32 group_size) noexcept {
    i32 choff = 576;
    for (i32 j = 0; j < 4; ++j) {
        f32* dst = grbuf + group_size * j;
        for (i32 i = 0; i < 2 * as<i32>(sci.total_bands); ++i) {
            const i32 ba = as<i32>(sci.bitalloc[i]);
            if (ba != 0) {
                if (ba < 17) {
                    const i32 half = (1 << (ba - 1)) - 1;
                    for (i32 k = 0; k < group_size; ++k) {
                        dst[k] = as<f32>(as<i32>(bs_get_bits(bs, ba)) - half);
                    }
                } else {
                    const u32 mod = (2u << (ba - 17)) + 1u; // 3, 5, 9
                    u32 code = bs_get_bits(bs, as<i32>(mod + 2u - (mod >> 3))); // 5, 7, 10
                    for (i32 k = 0; k < group_size; ++k, code /= mod) {
                        dst[k] = as<f32>(as<i32>(code % mod) - as<i32>(mod / 2u));
                    }
                }
            }
            dst += choff;
            choff = 18 - choff;
        }
    }
    return group_size * 4;
}

void l12_apply_scf_384(const L12ScaleInfo& sci, const f32* scf, f32* dst) noexcept {
    const usize tail_bands = as<usize>(sci.total_bands) - as<usize>(sci.stereo_bands);
    if (tail_bands > 0) {
        mem_copy(dst + 576 + as<usize>(sci.stereo_bands) * 18u, dst + as<usize>(sci.stereo_bands) * 18u,
                 tail_bands * 18u * sizeof(f32));
    }
    for (i32 i = 0; i < as<i32>(sci.total_bands); ++i, dst += 18, scf += 6) {
        for (i32 k = 0; k < 12; ++k) {
            dst[k + 0] *= scf[0];
            dst[k + 576] *= scf[3];
        }
    }
}

[[nodiscard]] u32 decode_l12_frame(L3DecoderState& dec, L3Scratch& scratch, const u8* frame_ptr,
                                   const FrameHeader& hdr, f32* pcm_interleaved) noexcept {
    mem_copy(dec.header, frame_ptr, 4);

    BitStream bs_frame{};
    bs_init(bs_frame, frame_ptr + 4, as<i32>(hdr.frame_size) - 4);
    if (hdr.error_protection) {
        (void)bs_get_bits(bs_frame, 16);
    }

    L12ScaleInfo sci{};
    l12_read_scale_info(frame_ptr, hdr, bs_frame, sci);

    const i32 nch = as<i32>(hdr.num_channels);
    const i32 mpeg_layer = 4 - as<i32>(hdr.layer); // 1 for Layer I, 2 for Layer II
    const i32 group_size = mpeg_layer | 1;         // 1 for Layer I, 3 for Layer II
    f32* pcm = pcm_interleaved;

    mem_zero(scratch.grbuf, sizeof(scratch.grbuf));
    i32 acc = 0;
    for (i32 igr = 0; igr < 3; ++igr) {
        acc += l12_dequantize_granule(scratch.grbuf[0] + acc, bs_frame, sci, group_size);
        if (acc == 12) {
            acc = 0;
            l12_apply_scf_384(sci, sci.scf + igr, scratch.grbuf[0]);
            mp3d_synth_granule(dec.qmf_state, scratch.grbuf[0], 12, nch, pcm, scratch.syn[0]);
            mem_zero(scratch.grbuf, sizeof(scratch.grbuf));
            pcm += 384 * nch;
        }
        if (bs_frame.pos > bs_frame.limit) {
            dec.reserv = 0;
            return 0u;
        }
    }
    return as<u32>(hdr.samples_per_frame);
}

// Decodes one MPEG-1/2/2.5 Layer III frame into interleaved f32 PCM (`pcm` holds up to 1152*2 samples).
// Returns number of PCM frames produced (1152 or 576), or 0 if side-info/reservoir failed.
[[nodiscard]] u32 decode_l3_frame(L3DecoderState& dec, L3Scratch& scratch, const u8* frame_ptr,
                                  const FrameHeader& hdr, f32* pcm_interleaved) noexcept {
    mem_zero(scratch.ist_pos, sizeof(scratch.ist_pos));
    mem_copy(dec.header, frame_ptr, 4);

    BitStream bs_frame{};
    bs_init(bs_frame, frame_ptr + 4, as<i32>(hdr.frame_size) - 4);
    if (hdr_is_crc(frame_ptr)) {
        (void)bs_get_bits(bs_frame, 16);
    }

    const i32 main_data_begin = l3_read_side_info(bs_frame, scratch.gr_info, frame_ptr);
    if (main_data_begin < 0 || bs_frame.pos > bs_frame.limit) {
        dec.reserv = 0;
        return 0u;
    }

    const i32 nch = as<i32>(hdr.num_channels);
    const i32 num_gr = hdr_test_mpeg1(frame_ptr) ? 2 : 1;
    const bool ok = l3_restore_reservoir(dec, bs_frame, scratch, main_data_begin);
    if (ok) {
        for (i32 igr = 0; igr < num_gr; ++igr) {
            mem_zero(scratch.grbuf, sizeof(scratch.grbuf));
            l3_decode_granule(dec, scratch, scratch.gr_info + igr * nch, nch);
            mp3d_synth_granule(dec.qmf_state, scratch.grbuf[0], 18, nch, pcm_interleaved + igr * 576 * nch,
                               scratch.syn[0]);
        }
    }
    l3_save_reservoir(dec, scratch);
    return ok ? as<u32>(hdr.samples_per_frame) : 0u;
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
    u16 free_format_base = 0u;
    if (!mp3_detail::find_first_frame(d, id3.audio_start, id3.audio_end, first_hdr, pos, free_format_base)) [[unlikely]] {
        return ErrorCode::InvalidData;
    }

    const char* layer_str = (first_hdr.layer == 3u)
                                ? "MPEG Layer I"
                                : ((first_hdr.layer == 2u) ? "MPEG Layer II" : "MPEG Layer III");
    AudioInfo info;
    info.format = AudioFormat::Mp3;
    info.compression_method = String(layer_str, dst_alloc);
    info.channels = first_hdr.num_channels;
    info.sample_rate = as<f64>(first_hdr.sample_rate);
    info.bits_per_sample = 16;
    info.bit_rate_kbps = first_hdr.bitrate;
    info.metadata = keel::move(id3.metadata);

    const mp3_detail::VbrHeaderInfo vbr = mp3_detail::parse_vbr_header(d + pos, id3.audio_end - pos, first_hdr);
    if (vbr.has_frames) {
        u64 total = as<u64>(vbr.total_frames) * as<u64>(first_hdr.samples_per_frame);
        if (vbr.has_gapless) {
            const u64 trim = as<u64>(vbr.enc_delay) + as<u64>(vbr.enc_padding);
            total = (total > trim) ? (total - trim) : 0ULL;
        }
        info.total_frames = total;
        info.duration_sec = as<f64>(info.total_frames) / info.sample_rate;
        if (info.duration_sec > 0.0) {
            const u64 stream_bytes =
                vbr.has_bytes ? as<u64>(vbr.total_bytes) : as<u64>(id3.audio_end - pos);
            info.bit_rate_kbps = audio_detail::safe_f64_to_u32(
                keel::round((as<f64>(stream_bytes) * 8.0) / (info.duration_sec * 1000.0)));
        }
    } else if (info.bit_rate_kbps > 0 && id3.audio_end > pos) {
        const usize audio_bytes = id3.audio_end - pos;
        info.duration_sec =
            as<f64>(as<u64>(audio_bytes) * 8ULL) / (as<f64>(info.bit_rate_kbps) * 1000.0);
        info.total_frames = audio_detail::safe_f64_to_u64(info.duration_sec * info.sample_rate);
    }

    return info;
}

Result<AudioBuffer> decode_mp3(ArrayRef<u8> bytes, const AudioDecodeOptions& opts) {
    if (opts.end_frame != Limits<u64>::kMax && opts.end_frame < opts.start_frame) [[unlikely]] {
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
    u16 free_format_base = 0u;
    if (!mp3_detail::find_first_frame(d, id3.audio_start, id3.audio_end, first_hdr, pos, free_format_base)) [[unlikely]] {
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

    // Standard ISO Layer III decoder delay (MDCT + polyphase filterbank = 529 samples).
    // When a LAME/Lavc gapless tag is present, trim (enc_delay + 529) at start and
    // max(0, enc_padding - 529) at end to match FFmpeg mp3float & MATLAB audioread length and phase.
    constexpr u64 kL3DecoderDelay = 529ULL;
    const u64 trim_start = vbr.has_gapless ? (as<u64>(vbr.enc_delay) + kL3DecoderDelay) : 0ULL;
    const u64 trim_end =
        (vbr.has_gapless && as<u64>(vbr.enc_padding) > kL3DecoderDelay)
            ? (as<u64>(vbr.enc_padding) - kL3DecoderDelay)
            : 0ULL;

    const u64 raw_start =
        (opts.start_frame <= Limits<u64>::kMax - trim_start) ? (opts.start_frame + trim_start) : Limits<u64>::kMax;
    const u64 raw_end =
        (opts.end_frame == Limits<u64>::kMax)
            ? Limits<u64>::kMax
            : ((opts.end_frame <= Limits<u64>::kMax - trim_start) ? (opts.end_frame + trim_start) : Limits<u64>::kMax);

    Array<f32> channel_pcm[2];
    for (u16 ch = 0; ch < num_channels; ++ch) {
        channel_pcm[ch] = Array<f32>(dst_alloc);
    }

    ArenaTemp scratch_arena = get_scratch(dst_alloc);
    mp3_detail::L3DecoderState* dec = scratch_arena.arena().create<mp3_detail::L3DecoderState>();
    mp3_detail::L3Scratch* scratch = scratch_arena.arena().create<mp3_detail::L3Scratch>();
    dec->reset();

    f32 frame_interleaved[1152 * 2]{};
    u64 stream_cursor = 0;

    while (pos + 4 <= id3.audio_end) {
        mp3_detail::FrameHeader hdr{};
        if (!mp3_detail::parse_frame_header(d + pos, id3.audio_end - pos, hdr, free_format_base)) {
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
        const u64 win_lo = keel::max(frame_start, raw_start);
        const u64 win_hi = keel::min(frame_end, raw_end);

        // Decode frames up to raw_end so bit reservoir, IMDCT overlap, and QMF state stay exact.
        if (frame_start < raw_end) {
            mem_zero(frame_interleaved, sizeof(frame_interleaved));
            const u32 decoded_samples =
                (hdr.layer == 1u)
                    ? mp3_detail::decode_l3_frame(*dec, *scratch, d + pos, hdr, frame_interleaved)
                    : mp3_detail::decode_l12_frame(*dec, *scratch, d + pos, hdr, frame_interleaved);
            if (win_hi > win_lo) {
                const u64 add_count = win_hi - win_lo;
                if (opts.max_frames > 0 &&
                    as<u64>(channel_pcm[0].size()) + add_count > opts.max_frames + trim_end) [[unlikely]] {
                    return ErrorCode::OutOfRange;
                }

                const usize local_off = as<usize>(win_lo - frame_start);
                const usize local_cnt = as<usize>(add_count);
                const u16 fch = hdr.num_channels;
                for (u16 ch = 0; ch < num_channels; ++ch) {
                    const u16 src_ch = (ch < fch) ? ch : u16{0};
                    channel_pcm[ch].reserve(channel_pcm[ch].size() + local_cnt);
                    for (usize k = 0; k < local_cnt; ++k) {
                        const usize idx = (local_off + k) * as<usize>(fch) + as<usize>(src_ch);
                        const f32 s = (decoded_samples > 0u) ? frame_interleaved[idx] : 0.0f;
                        channel_pcm[ch].push_back(s);
                    }
                }
            }
        }

        stream_cursor = frame_end;
        pos += hdr.frame_size;
    }

    const u64 logical_total =
        (stream_cursor > trim_start + trim_end) ? (stream_cursor - trim_start - trim_end) : 0ULL;
    const u64 max_valid_out = (logical_total > opts.start_frame) ? (logical_total - opts.start_frame) : 0ULL;
    if (as<u64>(channel_pcm[0].size()) > max_valid_out) {
        for (u16 ch = 0; ch < num_channels; ++ch) {
            channel_pcm[ch].resize(as<usize>(max_valid_out));
        }
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
    res.total_source_frames = logical_total;
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
            const f32* src = channel_pcm[ch].data();
            for (u64 t = 0; t < out_frames; ++t) {
                dst[audio_detail::sample_index(t, ch, out_frames, num_channels, opts.layout)] =
                    as<f64>(src[as<usize>(t)]);
            }
        }
    } else {
        i16* dst = res.samples_i16().data();
        for (u16 ch = 0; ch < num_channels; ++ch) {
            const f32* src = channel_pcm[ch].data();
            for (u64 t = 0; t < out_frames; ++t) {
                const f64 s = audio_detail::clamp_unit(as<f64>(src[as<usize>(t)]));
                dst[audio_detail::sample_index(t, ch, out_frames, num_channels, opts.layout)] =
                    as<i16>(keel::round_i64(s * 32767.0));
            }
        }
    }

    return res;
}

} // namespace keel::audiofile
