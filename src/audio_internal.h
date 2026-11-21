#pragma once

#include "keel/audiofile.h"
#include "keel/midi.h"
#include "keel/number.h"

#include <cmath>

namespace keel::audiofile::audio_detail {

[[nodiscard]] inline u16 read_u16_le(const u8* p) noexcept {
    return static_cast<u16>(static_cast<u16>(p[0]) | (static_cast<u16>(p[1]) << 8));
}

[[nodiscard]] inline u32 read_u32_le(const u8* p) noexcept {
    return static_cast<u32>(p[0]) | (static_cast<u32>(p[1]) << 8) | (static_cast<u32>(p[2]) << 16) |
           (static_cast<u32>(p[3]) << 24);
}

[[nodiscard]] inline u64 read_u64_le(const u8* p) noexcept {
    return static_cast<u64>(read_u32_le(p)) | (static_cast<u64>(read_u32_le(p + 4)) << 32);
}

[[nodiscard]] inline i16 read_i16_le(const u8* p) noexcept { return static_cast<i16>(read_u16_le(p)); }

[[nodiscard]] inline i32 read_i32_le(const u8* p) noexcept { return static_cast<i32>(read_u32_le(p)); }

[[nodiscard]] inline i32 read_i24_le(const u8* p) noexcept {
    u32 val = static_cast<u32>(p[0]) | (static_cast<u32>(p[1]) << 8) | (static_cast<u32>(p[2]) << 16);
    if ((val & 0x00800000u) != 0u) {
        val |= 0xFF000000u;
    }
    return static_cast<i32>(val);
}

[[nodiscard]] inline f32 read_f32_le(const u8* p) noexcept {
    const u32 u = read_u32_le(p);
    f32 f = 0.0f;
    mem_copy(&f, &u, sizeof(f32));
    return f;
}

[[nodiscard]] inline f64 read_f64_le(const u8* p) noexcept {
    const u64 u = read_u64_le(p);
    f64 d = 0.0;
    mem_copy(&d, &u, sizeof(f64));
    return d;
}

[[nodiscard]] inline u16 read_u16_be(const u8* p) noexcept {
    return static_cast<u16>((static_cast<u16>(p[0]) << 8) | static_cast<u16>(p[1]));
}

[[nodiscard]] inline u32 read_u32_be(const u8* p) noexcept {
    return (static_cast<u32>(p[0]) << 24) | (static_cast<u32>(p[1]) << 16) | (static_cast<u32>(p[2]) << 8) |
           static_cast<u32>(p[3]);
}

[[nodiscard]] inline u64 read_u64_be(const u8* p) noexcept {
    return (static_cast<u64>(read_u32_be(p)) << 32) | static_cast<u64>(read_u32_be(p + 4));
}

[[nodiscard]] inline i16 read_i16_be(const u8* p) noexcept { return static_cast<i16>(read_u16_be(p)); }

[[nodiscard]] inline i32 read_i32_be(const u8* p) noexcept { return static_cast<i32>(read_u32_be(p)); }

[[nodiscard]] inline i32 read_i24_be(const u8* p) noexcept {
    u32 val = (static_cast<u32>(p[0]) << 16) | (static_cast<u32>(p[1]) << 8) | static_cast<u32>(p[2]);
    if ((val & 0x00800000u) != 0u) {
        val |= 0xFF000000u;
    }
    return static_cast<i32>(val);
}

[[nodiscard]] inline f32 read_f32_be(const u8* p) noexcept {
    const u32 u = read_u32_be(p);
    f32 f = 0.0f;
    mem_copy(&f, &u, sizeof(f32));
    return f;
}

[[nodiscard]] inline f64 read_f64_be(const u8* p) noexcept {
    const u64 u = read_u64_be(p);
    f64 d = 0.0;
    mem_copy(&d, &u, sizeof(f64));
    return d;
}

template <typename ByteBuf> inline void write_fourcc(ByteBuf& buf, const char (&id)[5]) {
    buf.append(ArrayRef<u8>(reinterpret_cast<const u8*>(id), 4));
}

template <typename ByteBuf> inline void write_u16_le(ByteBuf& buf, u16 v) {
    buf.push_back(static_cast<u8>(v & 0xFFu));
    buf.push_back(static_cast<u8>((v >> 8) & 0xFFu));
}

template <typename ByteBuf> inline void write_u32_le(ByteBuf& buf, u32 v) {
    buf.push_back(static_cast<u8>(v & 0xFFu));
    buf.push_back(static_cast<u8>((v >> 8) & 0xFFu));
    buf.push_back(static_cast<u8>((v >> 16) & 0xFFu));
    buf.push_back(static_cast<u8>((v >> 24) & 0xFFu));
}

template <typename ByteBuf> inline void write_u64_le(ByteBuf& buf, u64 v) {
    write_u32_le(buf, static_cast<u32>(v & 0xFFFFFFFFu));
    write_u32_le(buf, static_cast<u32>(v >> 32));
}

template <typename ByteBuf> inline void write_i24_le(ByteBuf& buf, i32 v) {
    const u32 u = static_cast<u32>(v);
    buf.push_back(static_cast<u8>(u & 0xFFu));
    buf.push_back(static_cast<u8>((u >> 8) & 0xFFu));
    buf.push_back(static_cast<u8>((u >> 16) & 0xFFu));
}

template <typename ByteBuf> inline void write_f32_le(ByteBuf& buf, f32 f) {
    u32 u = 0;
    mem_copy(&u, &f, sizeof(u32));
    write_u32_le(buf, u);
}

template <typename ByteBuf> inline void write_f64_le(ByteBuf& buf, f64 d) {
    u64 u = 0;
    mem_copy(&u, &d, sizeof(u64));
    write_u64_le(buf, u);
}

template <typename ByteBuf> inline void write_u16_be(ByteBuf& buf, u16 v) {
    buf.push_back(static_cast<u8>((v >> 8) & 0xFFu));
    buf.push_back(static_cast<u8>(v & 0xFFu));
}

template <typename ByteBuf> inline void write_u32_be(ByteBuf& buf, u32 v) {
    buf.push_back(static_cast<u8>((v >> 24) & 0xFFu));
    buf.push_back(static_cast<u8>((v >> 16) & 0xFFu));
    buf.push_back(static_cast<u8>((v >> 8) & 0xFFu));
    buf.push_back(static_cast<u8>(v & 0xFFu));
}

template <typename ByteBuf> inline void write_u64_be(ByteBuf& buf, u64 v) {
    write_u32_be(buf, static_cast<u32>(v >> 32));
    write_u32_be(buf, static_cast<u32>(v & 0xFFFFFFFFu));
}

template <typename ByteBuf> inline void write_i24_be(ByteBuf& buf, i32 v) {
    const u32 u = static_cast<u32>(v);
    buf.push_back(static_cast<u8>((u >> 16) & 0xFFu));
    buf.push_back(static_cast<u8>((u >> 8) & 0xFFu));
    buf.push_back(static_cast<u8>(u & 0xFFu));
}

template <typename ByteBuf> inline void write_f32_be(ByteBuf& buf, f32 f) {
    u32 u = 0;
    mem_copy(&u, &f, sizeof(u32));
    write_u32_be(buf, u);
}

template <typename ByteBuf> inline void write_f64_be(ByteBuf& buf, f64 d) {
    u64 u = 0;
    mem_copy(&u, &d, sizeof(u64));
    write_u64_be(buf, u);
}

// ITU-T G.711 Mu-law and A-law decoders
[[nodiscard]] inline i16 decode_mu_law(u8 mu) noexcept {
    mu = static_cast<u8>(~mu);
    const int sign = ((mu & 0x80u) != 0u) ? -1 : 1;
    const int exponent = static_cast<int>((mu >> 4) & 0x07u);
    const int mantissa = static_cast<int>(mu & 0x0Fu);
    const int sample = ((mantissa << 3) + 0x84) << exponent;
    return static_cast<i16>(sign * (sample - 0x84));
}

[[nodiscard]] inline i16 decode_a_law(u8 a) noexcept {
    a ^= 0x55u;
    const int sign = ((a & 0x80u) != 0u) ? -1 : 1;
    const int exponent = static_cast<int>((a >> 4) & 0x07u);
    const int mantissa = static_cast<int>(a & 0x0Fu);
    const int sample = (exponent == 0) ? ((mantissa << 4) + 8) : (((mantissa << 4) + 0x108) << (exponent - 1));
    return static_cast<i16>(sign * sample);
}

[[nodiscard]] inline String make_trimmed_string(const u8* ptr, usize len, Allocator* alloc) {
    while (len > 0 && (ptr[len - 1] == 0u || ptr[len - 1] == static_cast<u8>(' '))) {
        --len;
    }
    return String(StringRef(reinterpret_cast<const char*>(ptr), len), alloc);
}

[[nodiscard]] inline String format_label_u32(StringRef prefix, u32 val, Allocator* alloc) {
    char num_buf[16];
    const usize num_len = format_u64(num_buf, sizeof(num_buf), val);
    String res = String::with_capacity(prefix.size() + num_len, alloc);
    res.append(prefix);
    res.append(StringRef(num_buf, num_len));
    return res;
}

struct FrameWindow {
    u64 start = 0;
    u64 count = 0;
};

[[nodiscard]] inline Result<FrameWindow> compute_frame_window(u64 available_frames,
                                                              const AudioDecodeOptions& opts) noexcept {
    if (opts.end_frame != ~u64{0} && opts.end_frame < opts.start_frame) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }
    const u64 s0 = keel::min(opts.start_frame, available_frames);
    const u64 s1 = (opts.end_frame == ~u64{0}) ? available_frames : keel::min(opts.end_frame, available_frames);
    const u64 count = (s1 > s0) ? (s1 - s0) : 0;
    if (opts.max_frames > 0 && count > opts.max_frames) [[unlikely]] {
        return ErrorCode::OutOfRange;
    }
    return FrameWindow{s0, count};
}

[[nodiscard]] inline Result<Array<u8>> alloc_sample_buffer(u64 frames, u16 channels, AudioSampleType sample_type,
                                                           Allocator* alloc) {
    Allocator* dst_alloc = alloc ? alloc : default_allocator();
    const usize elem_size = audio_sample_size(sample_type);
    if (elem_size == 0) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }
    if (frames == 0 || channels == 0) {
        return Array<u8>(dst_alloc);
    }

    const u64 max_usize = static_cast<u64>(~usize{0});
    if (frames > max_usize / static_cast<u64>(channels)) [[unlikely]] {
        return ErrorCode::OutOfRange;
    }
    const usize sample_count = static_cast<usize>(frames) * static_cast<usize>(channels);
    if (sample_count > (~usize{0}) / elem_size) [[unlikely]] {
        return ErrorCode::OutOfRange;
    }
    const usize total_bytes = sample_count * elem_size;

    Array<u8> out(total_bytes, u8{0}, dst_alloc);
    if (elem_size > 1 && (reinterpret_cast<usize>(out.data()) & (elem_size - 1)) != 0u) {
        // Align bump allocators (such as Arena at an unaligned offset) before allocating typed sample storage
        out = Array<u8>(dst_alloc);
        void* pad = dst_alloc->alloc_bytes(elem_size, elem_size);
        dst_alloc->free_bytes(pad, elem_size, elem_size);
        out = Array<u8>(total_bytes, u8{0}, dst_alloc);
    }
    return out;
}

[[nodiscard]] constexpr usize sample_index(u64 frame, u16 ch, u64 total_frames, u16 total_channels,
                                           AudioChannelLayout layout) noexcept {
    if (layout == AudioChannelLayout::Planar) {
        return static_cast<usize>(ch) * static_cast<usize>(total_frames) + static_cast<usize>(frame);
    }
    return static_cast<usize>(frame) * static_cast<usize>(total_channels) + static_cast<usize>(ch);
}

[[nodiscard]] inline ErrorCode validate_encode_view(const AudioView& view) noexcept {
    if (view.channels == 0 || view.frames == 0) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }
    if (!std::isfinite(view.sample_rate) || !(view.sample_rate > 0.0)) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }
    const usize elem_size = audio_sample_size(view.sample_type);
    if (elem_size == 0) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }
    const u64 max_usize = static_cast<u64>(~usize{0});
    if (view.frames > max_usize / static_cast<u64>(view.channels)) [[unlikely]] {
        return ErrorCode::OutOfRange;
    }
    const usize total_samples = static_cast<usize>(view.frames) * static_cast<usize>(view.channels);
    if (total_samples > (~usize{0}) / elem_size) [[unlikely]] {
        return ErrorCode::OutOfRange;
    }
    const usize needed_bytes = total_samples * elem_size;
    if (view.data.size() < needed_bytes) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }
    return ErrorCode::Ok;
}

[[nodiscard]] inline ErrorCode resolve_encode_format(const AudioView& view, const AudioEncodeOptions& opts,
                                                     u16& out_bps, bool& out_float) noexcept {
    const u16 bps = opts.bits_per_sample;
    if (bps != 8 && bps != 16 && bps != 24 && bps != 32 && bps != 64) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }
    if (opts.float_samples && bps != 32 && bps != 64) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }
    out_bps = bps;
    out_float = (bps == 64) || opts.float_samples || (bps == 32 && view.sample_type == AudioSampleType::F32);
    return ErrorCode::Ok;
}

[[nodiscard]] inline f64 view_sample_f64(const AudioView& view, usize idx) noexcept {
    const u8* base = view.data.data();
    switch (view.sample_type) {
        case AudioSampleType::U8:
            return (static_cast<f64>(base[idx]) - 128.0) / 128.0;
        case AudioSampleType::I8: {
            const i8 v = static_cast<i8>(base[idx]);
            return static_cast<f64>(v) / 128.0;
        }
        case AudioSampleType::I16: {
            i16 v = 0;
            mem_copy(&v, base + idx * sizeof(i16), sizeof(i16));
            return static_cast<f64>(v) / 32768.0;
        }
        case AudioSampleType::I32: {
            i32 v = 0;
            mem_copy(&v, base + idx * sizeof(i32), sizeof(i32));
            return static_cast<f64>(v) / 2147483648.0;
        }
        case AudioSampleType::F32: {
            f32 v = 0.0f;
            mem_copy(&v, base + idx * sizeof(f32), sizeof(f32));
            return static_cast<f64>(v);
        }
        case AudioSampleType::F64: {
            f64 v = 0.0;
            mem_copy(&v, base + idx * sizeof(f64), sizeof(f64));
            return v;
        }
    }
    return 0.0;
}

[[nodiscard]] inline f32 view_sample_f32(const AudioView& view, usize idx) noexcept {
    if (view.sample_type == AudioSampleType::F32) {
        f32 v = 0.0f;
        mem_copy(&v, view.data.data() + idx * sizeof(f32), sizeof(f32));
        return v;
    }
    return static_cast<f32>(view_sample_f64(view, idx));
}

[[nodiscard]] inline f64 clamp_unit(f64 s) noexcept {
    if (!(s >= -1.0)) {
        return std::isnan(s) ? 0.0 : -1.0;
    }
    return (s > 1.0) ? 1.0 : s;
}

[[nodiscard]] inline u8 view_sample_u8_pcm(const AudioView& view, usize idx) noexcept {
    if (view.sample_type == AudioSampleType::U8) {
        return view.data[idx];
    }
    if (view.sample_type == AudioSampleType::I8) {
        const i8 v = static_cast<i8>(view.data[idx]);
        return static_cast<u8>(static_cast<i32>(v) + 128);
    }
    const f64 s = clamp_unit(view_sample_f64(view, idx));
    return static_cast<u8>(std::lround((s + 1.0) * 127.5));
}

[[nodiscard]] inline i8 view_sample_i8_pcm(const AudioView& view, usize idx) noexcept {
    if (view.sample_type == AudioSampleType::I8) {
        return static_cast<i8>(view.data[idx]);
    }
    if (view.sample_type == AudioSampleType::U8) {
        return static_cast<i8>(static_cast<i32>(view.data[idx]) - 128);
    }
    const f64 s = clamp_unit(view_sample_f64(view, idx));
    return static_cast<i8>(std::lround(s * 127.0));
}

[[nodiscard]] inline i16 view_sample_i16_pcm(const AudioView& view, usize idx) noexcept {
    if (view.sample_type == AudioSampleType::I16) {
        i16 v = 0;
        mem_copy(&v, view.data.data() + idx * sizeof(i16), sizeof(i16));
        return v;
    }
    const f64 s = clamp_unit(view_sample_f64(view, idx));
    return static_cast<i16>(std::lround(s * 32767.0));
}

[[nodiscard]] inline i32 view_sample_i24_pcm(const AudioView& view, usize idx) noexcept {
    const f64 s = clamp_unit(view_sample_f64(view, idx));
    return static_cast<i32>(std::llround(s * 8388607.0));
}

[[nodiscard]] inline i32 view_sample_i32_pcm(const AudioView& view, usize idx) noexcept {
    if (view.sample_type == AudioSampleType::I32) {
        i32 v = 0;
        mem_copy(&v, view.data.data() + idx * sizeof(i32), sizeof(i32));
        return v;
    }
    const f64 s = clamp_unit(view_sample_f64(view, idx));
    return static_cast<i32>(std::llround(s * 2147483647.0));
}

} // namespace keel::audiofile::audio_detail
