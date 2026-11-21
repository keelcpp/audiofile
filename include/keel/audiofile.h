#pragma once

#include "keel/array.h"
#include "keel/result.h"
#include "keel/string.h"

namespace keel::audiofile {

inline constexpr u64 kDefaultMaxAudioFrames = 512ULL * 1024 * 1024;

enum class AudioFormat : u8 {
    Unknown = 0,
    Wav,
    Flac,
    Mp3,
    Aiff,
    Au,
    Midi,
};

enum class AudioSampleType : u8 {
    U8 = 0,
    I8,
    I16,
    I32,
    F32,
    F64,
};

enum class AudioChannelLayout : u8 {
    Interleaved = 0, // [frame * channels + ch]
    Planar,          // [ch * frames + frame]
};

[[nodiscard]] constexpr usize audio_sample_size(AudioSampleType type) noexcept {
    switch (type) {
        case AudioSampleType::U8:
        case AudioSampleType::I8:
            return 1;
        case AudioSampleType::I16:
            return 2;
        case AudioSampleType::I32:
        case AudioSampleType::F32:
            return 4;
        case AudioSampleType::F64:
            return 8;
    }
    return 0;
}

[[nodiscard]] constexpr StringRef audio_format_name(AudioFormat fmt) noexcept {
    switch (fmt) {
        case AudioFormat::Wav:
            return "wav";
        case AudioFormat::Flac:
            return "flac";
        case AudioFormat::Mp3:
            return "mp3";
        case AudioFormat::Aiff:
            return "aiff";
        case AudioFormat::Au:
            return "au";
        case AudioFormat::Midi:
            return "midi";
        case AudioFormat::Unknown:
            break;
    }
    return "unknown";
}

struct AudioMetadata {
    String title;
    String artist;
    String comment;

    [[nodiscard]] AudioMetadata clone(Allocator* alloc = nullptr) const;
};

struct AudioInfo {
    AudioFormat format = AudioFormat::Unknown;
    String compression_method;
    u32 channels = 0;
    f64 sample_rate = 0.0;
    u64 total_frames = 0;
    f64 duration_sec = 0.0;
    u16 bits_per_sample = 0;
    u32 bit_rate_kbps = 0;
    AudioMetadata metadata;

    [[nodiscard]] AudioInfo clone(Allocator* alloc = nullptr) const;
};

struct AudioDecodeOptions {
    u64 start_frame = 0;             // 0-based inclusive
    u64 end_frame = ~u64{0};         // 0-based exclusive, ~0 = end of stream
    bool native_sample_type = false; // false => normalized [-1, 1] F64; true => preserve native sample type
    AudioChannelLayout layout = AudioChannelLayout::Interleaved;
    u64 max_frames = kDefaultMaxAudioFrames;
    Allocator* allocator = nullptr;
};

struct AudioView {
    f64 sample_rate = 0.0;
    u16 channels = 0;
    u64 frames = 0;
    AudioSampleType sample_type = AudioSampleType::F64;
    AudioChannelLayout layout = AudioChannelLayout::Interleaved;
    ArrayRef<u8> data{};

    [[nodiscard]] static AudioView from_u8(ArrayRef<u8> samples, u64 frames, u16 channels, f64 sample_rate,
                                           AudioChannelLayout layout = AudioChannelLayout::Interleaved) noexcept {
        return AudioView{sample_rate, channels, frames, AudioSampleType::U8, layout, samples};
    }

    [[nodiscard]] static AudioView from_i8(ArrayRef<i8> samples, u64 frames, u16 channels, f64 sample_rate,
                                           AudioChannelLayout layout = AudioChannelLayout::Interleaved) noexcept {
        return AudioView{sample_rate, channels, frames, AudioSampleType::I8, layout,
                         ArrayRef<u8>(reinterpret_cast<const u8*>(samples.data()), samples.size())};
    }

    [[nodiscard]] static AudioView from_i16(ArrayRef<i16> samples, u64 frames, u16 channels, f64 sample_rate,
                                            AudioChannelLayout layout = AudioChannelLayout::Interleaved) noexcept {
        return AudioView{sample_rate, channels, frames, AudioSampleType::I16, layout,
                         ArrayRef<u8>(reinterpret_cast<const u8*>(samples.data()), samples.size() * sizeof(i16))};
    }

    [[nodiscard]] static AudioView from_i32(ArrayRef<i32> samples, u64 frames, u16 channels, f64 sample_rate,
                                            AudioChannelLayout layout = AudioChannelLayout::Interleaved) noexcept {
        return AudioView{sample_rate, channels, frames, AudioSampleType::I32, layout,
                         ArrayRef<u8>(reinterpret_cast<const u8*>(samples.data()), samples.size() * sizeof(i32))};
    }

    [[nodiscard]] static AudioView from_f32(ArrayRef<f32> samples, u64 frames, u16 channels, f64 sample_rate,
                                            AudioChannelLayout layout = AudioChannelLayout::Interleaved) noexcept {
        return AudioView{sample_rate, channels, frames, AudioSampleType::F32, layout,
                         ArrayRef<u8>(reinterpret_cast<const u8*>(samples.data()), samples.size() * sizeof(f32))};
    }

    [[nodiscard]] static AudioView from_f64(ArrayRef<f64> samples, u64 frames, u16 channels, f64 sample_rate,
                                            AudioChannelLayout layout = AudioChannelLayout::Interleaved) noexcept {
        return AudioView{sample_rate, channels, frames, AudioSampleType::F64, layout,
                         ArrayRef<u8>(reinterpret_cast<const u8*>(samples.data()), samples.size() * sizeof(f64))};
    }
};

struct AudioBuffer {
    f64 sample_rate = 0.0;
    u16 bits_per_sample = 16;
    u16 channels = 0;
    u64 frames = 0;
    u64 total_source_frames = 0;
    AudioSampleType sample_type = AudioSampleType::F64;
    AudioChannelLayout layout = AudioChannelLayout::Interleaved;
    Array<u8> data;
    AudioMetadata metadata;

    [[nodiscard]] usize sample_count() const noexcept {
        return static_cast<usize>(frames) * static_cast<usize>(channels);
    }

    [[nodiscard]] ArrayRef<u8> samples_u8() const noexcept {
        KEEL_ASSERT(sample_type == AudioSampleType::U8);
        return ArrayRef<u8>(data.data(), sample_count());
    }
    [[nodiscard]] MutArrayRef<u8> samples_u8() noexcept {
        KEEL_ASSERT(sample_type == AudioSampleType::U8);
        return MutArrayRef<u8>(data.data(), sample_count());
    }

    [[nodiscard]] ArrayRef<i8> samples_i8() const noexcept {
        KEEL_ASSERT(sample_type == AudioSampleType::I8);
        return ArrayRef<i8>(reinterpret_cast<const i8*>(data.data()), sample_count());
    }
    [[nodiscard]] MutArrayRef<i8> samples_i8() noexcept {
        KEEL_ASSERT(sample_type == AudioSampleType::I8);
        return MutArrayRef<i8>(reinterpret_cast<i8*>(data.data()), sample_count());
    }

    [[nodiscard]] ArrayRef<i16> samples_i16() const noexcept {
        KEEL_ASSERT(sample_type == AudioSampleType::I16);
        return ArrayRef<i16>(reinterpret_cast<const i16*>(data.data()), sample_count());
    }
    [[nodiscard]] MutArrayRef<i16> samples_i16() noexcept {
        KEEL_ASSERT(sample_type == AudioSampleType::I16);
        return MutArrayRef<i16>(reinterpret_cast<i16*>(data.data()), sample_count());
    }

    [[nodiscard]] ArrayRef<i32> samples_i32() const noexcept {
        KEEL_ASSERT(sample_type == AudioSampleType::I32);
        return ArrayRef<i32>(reinterpret_cast<const i32*>(data.data()), sample_count());
    }
    [[nodiscard]] MutArrayRef<i32> samples_i32() noexcept {
        KEEL_ASSERT(sample_type == AudioSampleType::I32);
        return MutArrayRef<i32>(reinterpret_cast<i32*>(data.data()), sample_count());
    }

    [[nodiscard]] ArrayRef<f32> samples_f32() const noexcept {
        KEEL_ASSERT(sample_type == AudioSampleType::F32);
        return ArrayRef<f32>(reinterpret_cast<const f32*>(data.data()), sample_count());
    }
    [[nodiscard]] MutArrayRef<f32> samples_f32() noexcept {
        KEEL_ASSERT(sample_type == AudioSampleType::F32);
        return MutArrayRef<f32>(reinterpret_cast<f32*>(data.data()), sample_count());
    }

    [[nodiscard]] ArrayRef<f64> samples_f64() const noexcept {
        KEEL_ASSERT(sample_type == AudioSampleType::F64);
        return ArrayRef<f64>(reinterpret_cast<const f64*>(data.data()), sample_count());
    }
    [[nodiscard]] MutArrayRef<f64> samples_f64() noexcept {
        KEEL_ASSERT(sample_type == AudioSampleType::F64);
        return MutArrayRef<f64>(reinterpret_cast<f64*>(data.data()), sample_count());
    }

    [[nodiscard]] AudioView as_view() const noexcept {
        return AudioView{sample_rate, channels, frames, sample_type, layout, data.as_ref()};
    }

    [[nodiscard]] AudioBuffer clone(Allocator* alloc = nullptr) const;
};

struct AudioEncodeOptions {
    u16 bits_per_sample = 16;
    bool float_samples = false; // Encode IEEE float when bits_per_sample == 32/64 (also implied when source is F32 & 32-bit, or 64-bit)
    StringRef title{};
    StringRef artist{};
    StringRef comment{};
    Allocator* allocator = nullptr;
};

// Format sniffing
[[nodiscard]] bool is_wav_bytes(ArrayRef<u8> bytes) noexcept;
[[nodiscard]] bool is_flac_bytes(ArrayRef<u8> bytes) noexcept;
[[nodiscard]] bool is_mp3_bytes(ArrayRef<u8> bytes) noexcept;
[[nodiscard]] bool is_aiff_bytes(ArrayRef<u8> bytes) noexcept;
[[nodiscard]] bool is_au_bytes(ArrayRef<u8> bytes) noexcept;
[[nodiscard]] AudioFormat sniff_audio_format(ArrayRef<u8> bytes) noexcept;

// Metadata peeking
[[nodiscard]] Result<AudioInfo> peek_audio(ArrayRef<u8> bytes, Allocator* alloc = nullptr);
[[nodiscard]] Result<AudioInfo> peek_wav(ArrayRef<u8> bytes, Allocator* alloc = nullptr);
[[nodiscard]] Result<AudioInfo> peek_flac(ArrayRef<u8> bytes, Allocator* alloc = nullptr);
[[nodiscard]] Result<AudioInfo> peek_mp3(ArrayRef<u8> bytes, Allocator* alloc = nullptr);
[[nodiscard]] Result<AudioInfo> peek_aiff(ArrayRef<u8> bytes, Allocator* alloc = nullptr);
[[nodiscard]] Result<AudioInfo> peek_au(ArrayRef<u8> bytes, Allocator* alloc = nullptr);
[[nodiscard]] Result<AudioInfo> peek_audio_file(StringRef path, Allocator* alloc = nullptr);

// Audio decoding
[[nodiscard]] Result<AudioBuffer> decode_audio(ArrayRef<u8> bytes, const AudioDecodeOptions& opts = {});
[[nodiscard]] Result<AudioBuffer> decode_wav(ArrayRef<u8> bytes, const AudioDecodeOptions& opts = {});
[[nodiscard]] Result<AudioBuffer> decode_flac(ArrayRef<u8> bytes, const AudioDecodeOptions& opts = {});
[[nodiscard]] Result<AudioBuffer> decode_mp3(ArrayRef<u8> bytes, const AudioDecodeOptions& opts = {});
[[nodiscard]] Result<AudioBuffer> decode_aiff(ArrayRef<u8> bytes, const AudioDecodeOptions& opts = {});
[[nodiscard]] Result<AudioBuffer> decode_au(ArrayRef<u8> bytes, const AudioDecodeOptions& opts = {});
[[nodiscard]] Result<AudioBuffer> read_audio(StringRef path, const AudioDecodeOptions& opts = {});

// Audio encoding
[[nodiscard]] Result<Array<u8>> encode_audio(const AudioView& view, AudioFormat fmt,
                                             const AudioEncodeOptions& opts = {});
[[nodiscard]] Result<Array<u8>> encode_wav(const AudioView& view, const AudioEncodeOptions& opts = {});
[[nodiscard]] Result<Array<u8>> encode_aiff(const AudioView& view, const AudioEncodeOptions& opts = {});
[[nodiscard]] Result<Array<u8>> encode_au(const AudioView& view, const AudioEncodeOptions& opts = {});
Error write_audio(StringRef path, const AudioView& view, const AudioEncodeOptions& opts = {});

} // namespace keel::audiofile
