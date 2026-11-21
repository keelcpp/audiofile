#include "audio_internal.h"

#include "keel/arena.h"
#include "keel/fs.h"

namespace keel::audiofile {

AudioMetadata AudioMetadata::clone(Allocator* alloc) const {
    AudioMetadata copy;
    copy.title = title.clone(alloc);
    copy.artist = artist.clone(alloc);
    copy.comment = comment.clone(alloc);
    return copy;
}

AudioInfo AudioInfo::clone(Allocator* alloc) const {
    AudioInfo copy;
    copy.format = format;
    copy.compression_method = compression_method.clone(alloc);
    copy.channels = channels;
    copy.sample_rate = sample_rate;
    copy.total_frames = total_frames;
    copy.duration_sec = duration_sec;
    copy.bits_per_sample = bits_per_sample;
    copy.bit_rate_kbps = bit_rate_kbps;
    copy.metadata = metadata.clone(alloc);
    return copy;
}

AudioBuffer AudioBuffer::clone(Allocator* alloc) const {
    Allocator* dst_alloc = alloc ? alloc : data.allocator();
    AudioBuffer copy;
    copy.sample_rate = sample_rate;
    copy.bits_per_sample = bits_per_sample;
    copy.channels = channels;
    copy.frames = frames;
    copy.total_source_frames = total_source_frames;
    copy.sample_type = sample_type;
    copy.layout = layout;
    if (!data.empty()) {
        auto aligned_res = audio_detail::alloc_sample_buffer(frames, channels, sample_type, dst_alloc);
        if (aligned_res.is_ok()) {
            copy.data = keel::move(aligned_res.value());
            mem_copy(copy.data.data(), data.data(), data.size());
        }
    } else {
        copy.data = Array<u8>(dst_alloc);
    }
    copy.metadata = metadata.clone(dst_alloc);
    return copy;
}

AudioFormat sniff_audio_format(ArrayRef<u8> bytes) noexcept {
    if (is_wav_bytes(bytes)) return AudioFormat::Wav;
    if (is_flac_bytes(bytes)) return AudioFormat::Flac;
    if (is_aiff_bytes(bytes)) return AudioFormat::Aiff;
    if (is_au_bytes(bytes)) return AudioFormat::Au;
    if (is_midi_bytes(bytes)) return AudioFormat::Midi;
    if (is_mp3_bytes(bytes)) return AudioFormat::Mp3;
    return AudioFormat::Unknown;
}

Result<AudioInfo> peek_audio(ArrayRef<u8> bytes, Allocator* alloc) {
    if (bytes.size() < 4) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }
    switch (sniff_audio_format(bytes)) {
        case AudioFormat::Wav:
            return peek_wav(bytes, alloc);
        case AudioFormat::Flac:
            return peek_flac(bytes, alloc);
        case AudioFormat::Mp3:
            return peek_mp3(bytes, alloc);
        case AudioFormat::Aiff:
            return peek_aiff(bytes, alloc);
        case AudioFormat::Au:
            return peek_au(bytes, alloc);
        case AudioFormat::Midi:
        case AudioFormat::Unknown:
            break;
    }
    return ErrorCode::NotSupported;
}

Result<AudioBuffer> decode_audio(ArrayRef<u8> bytes, const AudioDecodeOptions& opts) {
    if (bytes.size() < 4) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }
    switch (sniff_audio_format(bytes)) {
        case AudioFormat::Wav:
            return decode_wav(bytes, opts);
        case AudioFormat::Flac:
            return decode_flac(bytes, opts);
        case AudioFormat::Mp3:
            return decode_mp3(bytes, opts);
        case AudioFormat::Aiff:
            return decode_aiff(bytes, opts);
        case AudioFormat::Au:
            return decode_au(bytes, opts);
        case AudioFormat::Midi:
        case AudioFormat::Unknown:
            break;
    }
    return ErrorCode::NotSupported;
}

Result<Array<u8>> encode_audio(const AudioView& view, AudioFormat fmt, const AudioEncodeOptions& opts) {
    switch (fmt) {
        case AudioFormat::Wav:
            return encode_wav(view, opts);
        case AudioFormat::Aiff:
            return encode_aiff(view, opts);
        case AudioFormat::Au:
            return encode_au(view, opts);
        case AudioFormat::Flac:
        case AudioFormat::Mp3:
        case AudioFormat::Midi:
            return ErrorCode::NotSupported;
        case AudioFormat::Unknown:
            break;
    }
    return ErrorCode::InvalidArgument;
}

Result<AudioInfo> peek_audio_file(StringRef path, Allocator* alloc) {
    if (path.empty()) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }
    ArenaTemp scratch = get_scratch(alloc);
    KEEL_TRY_ASSIGN(Array<u8> file_bytes, fs_read_file(path, scratch.allocator()));
    return peek_audio(file_bytes, alloc);
}

Result<AudioBuffer> read_audio(StringRef path, const AudioDecodeOptions& opts) {
    if (path.empty()) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }
    ArenaTemp scratch = get_scratch(opts.allocator);
    KEEL_TRY_ASSIGN(Array<u8> file_bytes, fs_read_file(path, scratch.allocator()));
    return decode_audio(file_bytes, opts);
}

Error write_audio(StringRef path, const AudioView& view, const AudioEncodeOptions& opts) {
    if (path.empty()) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }
    const StringRef ext = path_extension(path);
    AudioFormat fmt = AudioFormat::Unknown;
    if (ext.empty() || ext.equals_ignore_ascii_case(".wav") || ext.equals_ignore_ascii_case(".wave")) {
        fmt = AudioFormat::Wav;
    } else if (ext.equals_ignore_ascii_case(".aiff") || ext.equals_ignore_ascii_case(".aif") ||
               ext.equals_ignore_ascii_case(".aifc")) {
        fmt = AudioFormat::Aiff;
    } else if (ext.equals_ignore_ascii_case(".au") || ext.equals_ignore_ascii_case(".snd")) {
        fmt = AudioFormat::Au;
    } else {
        return ErrorCode::NotSupported;
    }

    ArenaTemp scratch = get_scratch(opts.allocator);
    AudioEncodeOptions temp_opts = opts;
    if (!temp_opts.allocator) {
        temp_opts.allocator = scratch.allocator();
    }
    KEEL_TRY_ASSIGN(Array<u8> encoded, encode_audio(view, fmt, temp_opts));
    return fs_write_file(path, encoded);
}

} // namespace keel::audiofile
