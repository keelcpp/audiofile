#include "audio_internal.h"

#include "keel/arena.h"

namespace keel::audiofile {

bool is_flac_bytes(ArrayRef<u8> bytes) noexcept {
    if (bytes.size() < 4) return false;
    const u8* d = bytes.data();
    return d[0] == 'f' && d[1] == 'L' && d[2] == 'a' && d[3] == 'C';
}

namespace {
namespace flac_detail {

class FlacBitReader {
public:
    FlacBitReader(const u8* data, usize len) noexcept : m_data(data), m_len(len) {}

    [[nodiscard]] bool eof() const noexcept { return m_pos >= m_len && m_bits_left == 0; }
    [[nodiscard]] bool overrun() const noexcept { return m_overrun; }

    [[nodiscard]] u32 read_bits(u32 n) noexcept {
        if (n == 0) return 0;
        KEEL_ASSERT(n <= 32);
        while (m_bits_left < n) {
            if (m_pos >= m_len) [[unlikely]] {
                m_overrun = true;
                if (m_bits_left == 0) return 0;
                const u32 res = as<u32>(m_bit_buf << (n - m_bits_left));
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

    [[nodiscard]] u64 read_bits64(u32 n) noexcept {
        if (n <= 32) return read_bits(n);
        const u64 high = read_bits(n - 32);
        const u64 low = read_bits(32);
        return (high << 32) | low;
    }

    [[nodiscard]] i64 read_signed_bits(u32 n) noexcept {
        if (n == 0) return 0;
        const u64 u = read_bits64(n);
        if (n >= 64) return as<i64>(u);
        const u64 sign_bit = u64{1} << (n - 1);
        if ((u & sign_bit) != 0ULL) {
            return as<i64>(u | (~((u64{1} << n) - 1ULL)));
        }
        return as<i64>(u);
    }

    [[nodiscard]] u32 read_unary() noexcept {
        u32 count = 0;
        while (true) {
            if (m_bits_left == 0) {
                if (m_pos >= m_len) [[unlikely]] {
                    m_overrun = true;
                    return count;
                }
                m_bit_buf = (m_bit_buf << 8) | as<u64>(m_data[m_pos++]);
                m_bits_left += 8;
            }
            --m_bits_left;
            if (((m_bit_buf >> m_bits_left) & 1ULL) != 0ULL) {
                break;
            }
            ++count;
        }
        return count;
    }

    [[nodiscard]] i64 read_rice_signed(u32 k) noexcept {
        const u64 msbs = read_unary();
        const u64 lsbs = (k > 0) ? read_bits64(k) : 0ULL;
        const u64 uval = (msbs << k) | lsbs;
        if ((uval & 1ULL) != 0ULL) {
            return -as<i64>((uval >> 1) + 1ULL);
        }
        return as<i64>(uval >> 1);
    }

    [[nodiscard]] u64 read_utf8_coded() noexcept {
        const u32 v0 = read_bits(8);
        if ((v0 & 0x80u) == 0u) return v0;
        i32 cont_bytes = 0;
        u64 val = 0;
        if ((v0 & 0xE0u) == 0xC0u) {
            cont_bytes = 1;
            val = v0 & 0x1Fu;
        } else if ((v0 & 0xF0u) == 0xE0u) {
            cont_bytes = 2;
            val = v0 & 0x0Fu;
        } else if ((v0 & 0xF8u) == 0xF0u) {
            cont_bytes = 3;
            val = v0 & 0x07u;
        } else if ((v0 & 0xFCu) == 0xF8u) {
            cont_bytes = 4;
            val = v0 & 0x03u;
        } else if ((v0 & 0xFEu) == 0xFCu) {
            cont_bytes = 5;
            val = v0 & 0x01u;
        } else if (v0 == 0xFEu) {
            cont_bytes = 6;
            val = 0;
        } else {
            return 0;
        }
        for (i32 i = 0; i < cont_bytes; ++i) {
            val = (val << 6) | (read_bits(8) & 0x3Fu);
        }
        return val;
    }

    void align_to_byte() noexcept { m_bits_left &= ~7u; }
    [[nodiscard]] usize bits_consumed() const noexcept { return m_pos * 8u - as<usize>(m_bits_left); }
    [[nodiscard]] usize byte_pos() const noexcept { return m_pos - as<usize>(m_bits_left >> 3); }

private:
    const u8* m_data = nullptr;
    usize m_len = 0;
    usize m_pos = 0;
    u64 m_bit_buf = 0;
    u32 m_bits_left = 0;
    bool m_overrun = false;
};

struct FlacStreamHeader {
    u32 min_block_size = 0;
    u32 max_block_size = 0;
    u32 min_frame_size = 0;
    u32 max_frame_size = 0;
    u32 sample_rate = 0;
    u32 channels = 0;
    u32 bits_per_sample = 0;
    u64 total_frames = 0;
    usize audio_offset = 0;
    AudioMetadata metadata;
};

[[nodiscard]] Result<FlacStreamHeader> parse_flac_metadata(ArrayRef<u8> bytes, Allocator* alloc) {
    if (bytes.size() < 4) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }
    if (!is_flac_bytes(bytes)) [[unlikely]] {
        return ErrorCode::InvalidData;
    }
    if (bytes.size() < 42) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }

    const u8* d = bytes.data();
    const usize len = bytes.size();
    FlacStreamHeader st;
    bool has_streaminfo = false;
    bool is_last = false;
    usize pos = 4;

    while (!is_last) {
        if (pos + 4 > len) [[unlikely]] {
            return ErrorCode::EndOfFile;
        }
        is_last = (d[pos] & 0x80u) != 0u;
        const u8 block_type = as<u8>(d[pos] & 0x7Fu);
        if (block_type == 127u) [[unlikely]] {
            return ErrorCode::InvalidData;
        }
        const u32 block_size = (as<u32>(d[pos + 1]) << 16) | (as<u32>(d[pos + 2]) << 8) |
                               as<u32>(d[pos + 3]);
        pos += 4;
        if (block_size > len - pos) [[unlikely]] {
            return ErrorCode::EndOfFile;
        }

        if (block_type == 0) {
            if (block_size < 34) [[unlikely]] {
                return ErrorCode::InvalidData;
            }
            FlacBitReader br(d + pos, block_size);
            st.min_block_size = br.read_bits(16);
            st.max_block_size = br.read_bits(16);
            st.min_frame_size = br.read_bits(24);
            st.max_frame_size = br.read_bits(24);
            st.sample_rate = br.read_bits(20);
            st.channels = br.read_bits(3) + 1u;
            st.bits_per_sample = br.read_bits(5) + 1u;
            st.total_frames = br.read_bits64(36);
            has_streaminfo = true;
        } else if (block_type == 4 && block_size >= 8) {
            const usize block_end = pos + as<usize>(block_size);
            usize cpos = pos;
            const u32 vendor_len = audio_detail::read_u32_le(d + cpos);
            cpos += 4;
            if (vendor_len <= block_end - cpos) {
                cpos += vendor_len;
                if (cpos + 4 <= block_end) {
                    const u32 num_comments = audio_detail::read_u32_le(d + cpos);
                    cpos += 4;
                    for (u32 i = 0; i < num_comments && cpos + 4 <= block_end; ++i) {
                        const u32 clen = audio_detail::read_u32_le(d + cpos);
                        cpos += 4;
                        if (clen > block_end - cpos) break;
                        const StringRef entry(reinterpret_cast<const char*>(d + cpos), clen);
                        cpos += clen;

                        const StringSplit sp = entry.split_once('=');
                        if (sp.found) {
                            if (sp.first.equals_ignore_ascii_case("TITLE")) {
                                st.metadata.title = String(sp.second, alloc);
                            } else if (sp.first.equals_ignore_ascii_case("ARTIST")) {
                                st.metadata.artist = String(sp.second, alloc);
                            } else if (sp.first.equals_ignore_ascii_case("COMMENT") ||
                                       sp.first.equals_ignore_ascii_case("DESCRIPTION")) {
                                if (st.metadata.comment.empty() || sp.first.equals_ignore_ascii_case("COMMENT")) {
                                    st.metadata.comment = String(sp.second, alloc);
                                }
                            }
                        }
                    }
                }
            }
        }

        pos += block_size;
    }

    if (!has_streaminfo || st.sample_rate == 0 || st.channels == 0 || st.channels > 8 || st.bits_per_sample < 4 ||
        st.bits_per_sample > 32) [[unlikely]] {
        return ErrorCode::InvalidData;
    }

    st.audio_offset = pos;
    return st;
}

[[nodiscard]] ErrorCode decode_rice_residual(FlacBitReader& br, u32 predictor_order, u32 block_size,
                                             i64* samples) noexcept {
    const u32 coding_method = br.read_bits(2);
    if (coding_method > 1u) [[unlikely]] {
        return ErrorCode::InvalidData;
    }
    const u32 partition_order = br.read_bits(4);
    const usize num_partitions = usize{1} << partition_order;
    if ((as<usize>(block_size) % num_partitions) != 0u) [[unlikely]] {
        return ErrorCode::InvalidData;
    }
    const usize part_samples = as<usize>(block_size) / num_partitions;
    if (part_samples < as<usize>(predictor_order)) [[unlikely]] {
        return ErrorCode::InvalidData;
    }

    const u32 rice_param_bits = (coding_method == 0u) ? 4u : 5u;
    const u32 rice_escape = (coding_method == 0u) ? 0x0Fu : 0x1Fu;

    for (usize p = 0; p < num_partitions; ++p) {
        const usize p_start = (p == 0) ? as<usize>(predictor_order) : (p * part_samples);
        const usize p_end = (p + 1) * part_samples;
        const u32 k = br.read_bits(rice_param_bits);
        if (br.overrun()) [[unlikely]] {
            return ErrorCode::EndOfFile;
        }
        if (k == rice_escape) {
            const u32 escape_bits = br.read_bits(5);
            for (usize i = p_start; i < p_end; ++i) {
                samples[i] = br.read_signed_bits(escape_bits);
            }
        } else {
            for (usize i = p_start; i < p_end; ++i) {
                samples[i] = br.read_rice_signed(k);
            }
        }
        if (br.overrun()) [[unlikely]] {
            return ErrorCode::EndOfFile;
        }
    }
    return ErrorCode::Ok;
}

[[nodiscard]] u8 flac_crc8(const u8* data, usize len) noexcept {
    u8 crc = 0;
    for (usize i = 0; i < len; ++i) {
        crc = as<u8>(crc ^ data[i]);
        for (i32 b = 0; b < 8; ++b) {
            crc = ((crc & 0x80u) != 0u) ? as<u8>((crc << 1) ^ 0x07u) : as<u8>(crc << 1);
        }
    }
    return crc;
}

[[nodiscard]] u16 flac_crc16(const u8* data, usize len) noexcept {
    u16 crc = 0;
    for (usize i = 0; i < len; ++i) {
        crc = as<u16>(crc ^ (as<u16>(data[i]) << 8));
        for (i32 b = 0; b < 8; ++b) {
            crc = ((crc & 0x8000u) != 0u) ? as<u16>((crc << 1) ^ 0x8005u) : as<u16>(crc << 1);
        }
    }
    return crc;
}

[[nodiscard]] AudioSampleType flac_native_sample_type(u32 bps) noexcept {
    if (bps <= 8) return AudioSampleType::I8;
    if (bps <= 16) return AudioSampleType::I16;
    return AudioSampleType::I32;
}

} // namespace flac_detail
} // namespace

Result<AudioInfo> peek_flac(ArrayRef<u8> bytes, Allocator* alloc) {
    Allocator* dst_alloc = alloc ? alloc : default_allocator();
    KEEL_TRY_ASSIGN(flac_detail::FlacStreamHeader st, flac_detail::parse_flac_metadata(bytes, dst_alloc));

    AudioInfo info;
    info.format = AudioFormat::Flac;
    info.compression_method = String("FLAC Lossless", dst_alloc);
    info.channels = st.channels;
    info.sample_rate = as<f64>(st.sample_rate);
    info.bits_per_sample = as<u16>(st.bits_per_sample);
    info.total_frames = st.total_frames;
    info.duration_sec = as<f64>(st.total_frames) / info.sample_rate;
    if (info.duration_sec > 0.0 && !bytes.empty()) {
        info.bit_rate_kbps = audio_detail::safe_f64_to_u32((as<f64>(bytes.size()) * 8.0) / (info.duration_sec * 1000.0));
    }
    info.metadata = keel::move(st.metadata);
    return info;
}

Result<AudioBuffer> decode_flac(ArrayRef<u8> bytes, const AudioDecodeOptions& opts) {
    if (opts.end_frame != Limits<u64>::kMax && opts.end_frame < opts.start_frame) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }

    Allocator* dst_alloc = opts.allocator ? opts.allocator : default_allocator();
    KEEL_TRY_ASSIGN(flac_detail::FlacStreamHeader st, flac_detail::parse_flac_metadata(bytes, dst_alloc));

    const u16 num_channels = as<u16>(st.channels);
    const u64 total_expected = st.total_frames;

    if (total_expected > 0 && st.audio_offset >= bytes.size()) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }

    ArenaTemp scratch = get_scratch(dst_alloc);
    Array<i32> channel_buffers[8];
    ArenaArray<i64> subframe_buffers[8];
    const usize init_block = (st.max_block_size > 0 && st.max_block_size <= 65536u) ? st.max_block_size : 4096u;
    for (u16 ch = 0; ch < num_channels; ++ch) {
        channel_buffers[ch] = Array<i32>(dst_alloc);
        subframe_buffers[ch] = ArenaArray<i64>(scratch.arena(), init_block);
    }

    const u8* audio_data = bytes.data() + st.audio_offset;
    const usize audio_len = bytes.size() - st.audio_offset;
    flac_detail::FlacBitReader br(audio_data, audio_len);
    u64 stream_cursor = 0;

    while (!br.eof()) {
        u32 sync = br.read_bits(14);
        if (br.overrun()) {
            if (stream_cursor == 0 && total_expected > 0) [[unlikely]] {
                return ErrorCode::EndOfFile;
            }
            break;
        }
        if (sync != 0x3FFEu) {
            while (!br.eof() && sync != 0x3FFEu) {
                sync = ((sync << 1) | br.read_bits(1)) & 0x3FFFu;
            }
            if (sync != 0x3FFEu) {
                if (stream_cursor == 0 && total_expected > 0) [[unlikely]] {
                    return ErrorCode::InvalidData;
                }
                break;
            }
        }

        const usize sync_start_bits = br.bits_consumed() - 14u;
        if ((sync_start_bits & 7u) != 0u) [[unlikely]] {
            return ErrorCode::InvalidData;
        }
        const usize frame_start_byte = sync_start_bits >> 3;

        const u32 reserved1 = br.read_bits(1);
        const u32 blocking_strategy = br.read_bits(1);
        const u32 block_size_code = br.read_bits(4);
        const u32 sample_rate_code = br.read_bits(4);
        const u32 chan_assign = br.read_bits(4);
        const u32 sample_size_code = br.read_bits(3);
        const u32 reserved2 = br.read_bits(1);
        (void)blocking_strategy;

        if (br.overrun()) [[unlikely]] {
            return ErrorCode::EndOfFile;
        }
        if (reserved1 != 0u || reserved2 != 0u || block_size_code == 0u || sample_rate_code == 15u ||
            sample_size_code == 3u || chan_assign > 10u) [[unlikely]] {
            return ErrorCode::InvalidData;
        }

        u32 frame_bps = st.bits_per_sample;
        switch (sample_size_code) {
            case 1u: frame_bps = 8u; break;
            case 2u: frame_bps = 12u; break;
            case 4u: frame_bps = 16u; break;
            case 5u: frame_bps = 20u; break;
            case 6u: frame_bps = 24u; break;
            case 7u: frame_bps = 32u; break;
            default: break;
        }

        const u32 frame_channels = (chan_assign <= 7u) ? (chan_assign + 1u) : 2u;
        if (frame_channels != st.channels) [[unlikely]] {
            return ErrorCode::InvalidData;
        }

        (void)br.read_utf8_coded();

        u32 block_size = 0;
        if (block_size_code == 1u) {
            block_size = 192u;
        } else if (block_size_code >= 2u && block_size_code <= 5u) {
            block_size = 576u << (block_size_code - 2u);
        } else if (block_size_code == 6u) {
            block_size = br.read_bits(8) + 1u;
        } else if (block_size_code == 7u) {
            block_size = br.read_bits(16) + 1u;
        } else {
            block_size = 256u << (block_size_code - 8u);
        }

        if (sample_rate_code == 12u) {
            (void)br.read_bits(8);
        } else if (sample_rate_code == 13u || sample_rate_code == 14u) {
            (void)br.read_bits(16);
        }

        const usize header_end_byte = br.byte_pos();
        const u32 hdr_crc = br.read_bits(8); // Header CRC-8 (RFC 9639 poly 0x07)
        if (br.overrun()) [[unlikely]] {
            return ErrorCode::EndOfFile;
        }
        if (header_end_byte < frame_start_byte || header_end_byte > audio_len) [[unlikely]] {
            return ErrorCode::InvalidData;
        }
        const u8 calc_crc8 = flac_detail::flac_crc8(audio_data + frame_start_byte, header_end_byte - frame_start_byte);
        if (as<u8>(hdr_crc) != calc_crc8) [[unlikely]] {
            return ErrorCode::InvalidData;
        }
        if (block_size == 0u || block_size > 65536u) [[unlikely]] {
            return ErrorCode::InvalidData;
        }

        for (u16 ch = 0; ch < num_channels; ++ch) {
            subframe_buffers[ch].resize(block_size);
            i64* samples = subframe_buffers[ch].data();
            mem_zero(samples, as<usize>(block_size) * sizeof(i64));

            const u32 zero_bit = br.read_bits(1);
            const u32 subframe_type = br.read_bits(6);
            const u32 wasted_flag = br.read_bits(1);
            if (br.overrun()) [[unlikely]] {
                return ErrorCode::EndOfFile;
            }
            if (zero_bit != 0u) [[unlikely]] {
                return ErrorCode::InvalidData;
            }

            u32 wasted_bits = 0;
            if (wasted_flag != 0u) {
                wasted_bits = br.read_unary() + 1u;
                if (br.overrun()) [[unlikely]] {
                    return ErrorCode::EndOfFile;
                }
            }

            u32 subframe_bps = frame_bps;
            if ((chan_assign == 8u && ch == 1) || (chan_assign == 9u && ch == 0) || (chan_assign == 10u && ch == 1)) {
                ++subframe_bps;
            }
            if (wasted_bits >= subframe_bps) [[unlikely]] {
                return ErrorCode::InvalidData;
            }
            subframe_bps -= wasted_bits;

            if (subframe_type == 0u) {
                const i64 val = br.read_signed_bits(subframe_bps);
                if (br.overrun()) [[unlikely]] {
                    return ErrorCode::EndOfFile;
                }
                for (u32 i = 0; i < block_size; ++i) {
                    samples[i] = val;
                }
            } else if (subframe_type == 1u) {
                for (u32 i = 0; i < block_size; ++i) {
                    samples[i] = br.read_signed_bits(subframe_bps);
                }
                if (br.overrun()) [[unlikely]] {
                    return ErrorCode::EndOfFile;
                }
            } else if ((subframe_type & 0x38u) == 0x08u) {
                const u32 order = subframe_type & 0x07u;
                if (order > 4u || order > block_size) [[unlikely]] {
                    return ErrorCode::InvalidData;
                }
                for (u32 i = 0; i < order; ++i) {
                    samples[i] = br.read_signed_bits(subframe_bps);
                }
                if (br.overrun()) [[unlikely]] {
                    return ErrorCode::EndOfFile;
                }
                const ErrorCode res_ec = flac_detail::decode_rice_residual(br, order, block_size, samples);
                if (res_ec != ErrorCode::Ok) [[unlikely]] {
                    return res_ec;
                }
                for (u32 i = order; i < block_size; ++i) {
                    u64 pred = 0;
                    if (order == 1u) {
                        pred = as<u64>(samples[i - 1]);
                    } else if (order == 2u) {
                        pred = 2ULL * as<u64>(samples[i - 1]) - as<u64>(samples[i - 2]);
                    } else if (order == 3u) {
                        pred = 3ULL * as<u64>(samples[i - 1]) - 3ULL * as<u64>(samples[i - 2]) +
                               as<u64>(samples[i - 3]);
                    } else if (order == 4u) {
                        pred = 4ULL * as<u64>(samples[i - 1]) - 6ULL * as<u64>(samples[i - 2]) +
                               4ULL * as<u64>(samples[i - 3]) - as<u64>(samples[i - 4]);
                    }
                    samples[i] = as<i64>(as<u64>(samples[i]) + pred);
                }
            } else if (subframe_type >= 0x20u) {
                const u32 order = (subframe_type & 0x1Fu) + 1u;
                if (order > block_size) [[unlikely]] {
                    return ErrorCode::InvalidData;
                }
                for (u32 i = 0; i < order; ++i) {
                    samples[i] = br.read_signed_bits(subframe_bps);
                }
                const u32 qlp_prec_raw = br.read_bits(4);
                if (qlp_prec_raw == 0x0Fu) [[unlikely]] {
                    return ErrorCode::InvalidData;
                }
                const u32 qlp_precision = qlp_prec_raw + 1u;
                const i32 qlp_shift = as<i32>(br.read_signed_bits(5));
                if (qlp_shift < 0) [[unlikely]] {
                    return ErrorCode::InvalidData;
                }
                i64 qlp_coeffs[32]{};
                for (u32 i = 0; i < order; ++i) {
                    qlp_coeffs[i] = br.read_signed_bits(qlp_precision);
                }
                if (br.overrun()) [[unlikely]] {
                    return ErrorCode::EndOfFile;
                }
                const ErrorCode res_ec = flac_detail::decode_rice_residual(br, order, block_size, samples);
                if (res_ec != ErrorCode::Ok) [[unlikely]] {
                    return res_ec;
                }
                for (u32 i = order; i < block_size; ++i) {
                    u64 sum_u = 0;
                    for (u32 j = 0; j < order; ++j) {
                        sum_u += as<u64>(qlp_coeffs[j]) * as<u64>(samples[i - 1u - j]);
                    }
                    const i64 shifted = as<i64>(sum_u) >> qlp_shift;
                    samples[i] = as<i64>(as<u64>(samples[i]) + as<u64>(shifted));
                }
            } else {
                return ErrorCode::InvalidData;
            }

            if (wasted_bits > 0u) {
                for (u32 i = 0; i < block_size; ++i) {
                    samples[i] = as<i64>(as<u64>(samples[i]) << wasted_bits);
                }
            }
        }

        br.align_to_byte();
        const usize footer_pos_byte = br.byte_pos();
        const u32 footer_crc = br.read_bits(16); // Frame footer CRC-16 (RFC 9639 poly 0x8005)
        if (br.overrun()) [[unlikely]] {
            return ErrorCode::EndOfFile;
        }
        if (footer_pos_byte < frame_start_byte || footer_pos_byte > audio_len) [[unlikely]] {
            return ErrorCode::InvalidData;
        }
        const u16 calc_crc16 =
            flac_detail::flac_crc16(audio_data + frame_start_byte, footer_pos_byte - frame_start_byte);
        if (as<u16>(footer_crc) != calc_crc16) [[unlikely]] {
            return ErrorCode::InvalidData;
        }

        if (num_channels == 2) {
            i64* l = subframe_buffers[0].data();
            i64* r = subframe_buffers[1].data();
            if (chan_assign == 8u) {
                for (u32 i = 0; i < block_size; ++i) {
                    r[i] = as<i64>(as<u64>(l[i]) - as<u64>(r[i]));
                }
            } else if (chan_assign == 9u) {
                for (u32 i = 0; i < block_size; ++i) {
                    l[i] = as<i64>(as<u64>(r[i]) + as<u64>(l[i]));
                }
            } else if (chan_assign == 10u) {
                for (u32 i = 0; i < block_size; ++i) {
                    const i64 side = r[i];
                    const u64 mid_u = (as<u64>(l[i]) << 1) | as<u64>(side & 1LL);
                    l[i] = as<i64>(mid_u + as<u64>(side)) >> 1;
                    r[i] = as<i64>(mid_u - as<u64>(side)) >> 1;
                }
            }
        }

        u64 valid_in_block = block_size;
        if (total_expected > 0 && stream_cursor + valid_in_block > total_expected) {
            valid_in_block = total_expected - stream_cursor;
        }

        const u64 block_start = stream_cursor;
        const u64 block_end = stream_cursor + valid_in_block;
        const u64 win_lo = keel::max(block_start, opts.start_frame);
        const u64 win_hi = keel::min(block_end, opts.end_frame);
        if (win_hi > win_lo) {
            const u64 add_count = win_hi - win_lo;
            if (opts.max_frames > 0 &&
                as<u64>(channel_buffers[0].size()) + add_count > opts.max_frames) [[unlikely]] {
                return ErrorCode::OutOfRange;
            }
            const usize local_off = as<usize>(win_lo - block_start);
            const usize local_cnt = as<usize>(add_count);
            for (u16 ch = 0; ch < num_channels; ++ch) {
                const i64* src = subframe_buffers[ch].data() + local_off;
                for (usize k = 0; k < local_cnt; ++k) {
                    channel_buffers[ch].push_back(as<i32>(src[k]));
                }
            }
        }

        stream_cursor = block_end;
        if (total_expected > 0 && stream_cursor >= total_expected) {
            break;
        }
        if (total_expected > 0 && opts.end_frame != Limits<u64>::kMax && stream_cursor >= opts.end_frame) {
            stream_cursor = total_expected;
            break;
        }
    }

    const u64 available_frames = stream_cursor;
    const u64 out_frames = as<u64>(channel_buffers[0].size());
    if (opts.max_frames > 0 && out_frames > opts.max_frames) [[unlikely]] {
        return ErrorCode::OutOfRange;
    }

    const AudioSampleType out_type =
        opts.native_sample_type ? flac_detail::flac_native_sample_type(st.bits_per_sample) : AudioSampleType::F64;

    AudioBuffer res;
    res.sample_rate = as<f64>(st.sample_rate);
    res.bits_per_sample = as<u16>(st.bits_per_sample);
    res.channels = num_channels;
    res.frames = out_frames;
    res.total_source_frames = available_frames;
    res.sample_type = out_type;
    res.layout = opts.layout;
    res.metadata = keel::move(st.metadata);

    if (out_frames == 0) {
        res.data = Array<u8>(dst_alloc);
        return res;
    }

    KEEL_TRY_ASSIGN(res.data, audio_detail::alloc_sample_buffer(out_frames, num_channels, out_type, dst_alloc));

    if (!opts.native_sample_type) {
        const f64 scale = 1.0 / as<f64>(u64{1} << (st.bits_per_sample - 1u));
        f64* dst = res.samples_f64().data();
        for (u16 ch = 0; ch < num_channels; ++ch) {
            const i32* src = channel_buffers[ch].data();
            for (u64 t = 0; t < out_frames; ++t) {
                dst[audio_detail::sample_index(t, ch, out_frames, num_channels, opts.layout)] =
                    as<f64>(src[as<usize>(t)]) * scale;
            }
        }
    } else if (out_type == AudioSampleType::I8) {
        i8* dst = res.samples_i8().data();
        for (u16 ch = 0; ch < num_channels; ++ch) {
            const i32* src = channel_buffers[ch].data();
            for (u64 t = 0; t < out_frames; ++t) {
                dst[audio_detail::sample_index(t, ch, out_frames, num_channels, opts.layout)] =
                    as<i8>(src[as<usize>(t)]);
            }
        }
    } else if (out_type == AudioSampleType::I16) {
        i16* dst = res.samples_i16().data();
        for (u16 ch = 0; ch < num_channels; ++ch) {
            const i32* src = channel_buffers[ch].data();
            for (u64 t = 0; t < out_frames; ++t) {
                dst[audio_detail::sample_index(t, ch, out_frames, num_channels, opts.layout)] =
                    as<i16>(src[as<usize>(t)]);
            }
        }
    } else {
        i32* dst = res.samples_i32().data();
        for (u16 ch = 0; ch < num_channels; ++ch) {
            const i32* src = channel_buffers[ch].data();
            for (u64 t = 0; t < out_frames; ++t) {
                dst[audio_detail::sample_index(t, ch, out_frames, num_channels, opts.layout)] =
                    src[as<usize>(t)];
            }
        }
    }

    return res;
}

} // namespace keel::audiofile
