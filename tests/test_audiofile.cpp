#include "keel/audiofile.h"
#include "keel/midi.h"

#include "keel/arena.h"
#include "keel/format.h"
#include "keel/fs.h"
#include "keel/math.h"
#include "keel/memory.h"

using namespace keel;
using namespace keel::audiofile;

namespace {

i32 g_tests_passed = 0;
i32 g_tests_failed = 0;

#define TEST_CHECK(cond)                                                                                               \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            keel::eprintln("FAIL: {} ({}:{})", #cond, __FILE__, __LINE__);                                             \
            ++g_tests_failed;                                                                                          \
        } else {                                                                                                       \
            ++g_tests_passed;                                                                                          \
        }                                                                                                              \
    } while (0)

[[nodiscard]] bool near_f64(f64 a, f64 b, f64 tol) noexcept { return keel::abs(a - b) <= tol; }
[[nodiscard]] bool near_f32(f32 a, f32 b, f32 tol) noexcept { return keel::abs(a - b) <= tol; }

// Helper MSB-first bitstream writer for constructing valid synthetic FLAC frames
class FlacBitWriter {
public:
    void write_bits(u64 val, u32 n) {
        for (i32 i = keel::as<i32>(n) - 1; i >= 0; --i) {
            m_bit_buf = keel::as<u8>((m_bit_buf << 1) | keel::as<u8>((val >> i) & 1ULL));
            if (++m_bit_count == 8u) {
                m_data.push_back(m_bit_buf);
                m_bit_buf = 0;
                m_bit_count = 0;
            }
        }
    }

    void write_signed_bits(i64 val, u32 n) {
        const u64 mask = (n >= 64u) ? Limits<u64>::kMax : ((u64{1} << n) - 1ULL);
        write_bits(keel::as<u64>(val) & mask, n);
    }

    void write_unary(u32 zeros) {
        for (u32 i = 0; i < zeros; ++i) {
            write_bits(0, 1);
        }
        write_bits(1, 1);
    }

    void write_rice_signed(i64 val, u32 k) {
        const u64 uval =
            (val >= 0) ? (keel::as<u64>(val) << 1) : ((keel::as<u64>(-val - 1) << 1) | 1ULL);
        const u32 msbs = keel::as<u32>(uval >> k);
        write_unary(msbs);
        if (k > 0u) {
            write_bits(uval & ((u64{1} << k) - 1ULL), k);
        }
    }

    void write_utf8_u32(u32 val) {
        if (val < 0x80u) {
            write_bits(val, 8);
        } else if (val < 0x800u) {
            write_bits(0xC0u | (val >> 6), 8);
            write_bits(0x80u | (val & 0x3Fu), 8);
        } else {
            write_bits(0xE0u | (val >> 12), 8);
            write_bits(0x80u | ((val >> 6) & 0x3Fu), 8);
            write_bits(0x80u | (val & 0x3Fu), 8);
        }
    }

    void write_bytes(const void* ptr, usize len) {
        align_to_byte();
        m_data.append(ArrayRef<u8>(static_cast<const u8*>(ptr), len));
    }

    void write_u32_le_bytes(u32 v) {
        align_to_byte();
        m_data.push_back(keel::as<u8>(v & 0xFFu));
        m_data.push_back(keel::as<u8>((v >> 8) & 0xFFu));
        m_data.push_back(keel::as<u8>((v >> 16) & 0xFFu));
        m_data.push_back(keel::as<u8>((v >> 24) & 0xFFu));
    }

    void align_to_byte() {
        if (m_bit_count > 0u) {
            write_bits(0, 8u - m_bit_count);
        }
    }

    [[nodiscard]] usize byte_offset() const noexcept { return m_data.size(); }

    void write_crc8_from(usize start_byte) {
        align_to_byte();
        u8 crc = 0;
        for (usize i = start_byte; i < m_data.size(); ++i) {
            crc = keel::as<u8>(crc ^ m_data[i]);
            for (i32 b = 0; b < 8; ++b) {
                crc = ((crc & 0x80u) != 0u) ? keel::as<u8>((crc << 1) ^ 0x07u) : keel::as<u8>(crc << 1);
            }
        }
        write_bits(crc, 8);
    }

    void write_crc16_from(usize start_byte) {
        align_to_byte();
        u16 crc = 0;
        for (usize i = start_byte; i < m_data.size(); ++i) {
            crc = keel::as<u16>(crc ^ (keel::as<u16>(m_data[i]) << 8));
            for (i32 b = 0; b < 8; ++b) {
                crc = ((crc & 0x8000u) != 0u) ? keel::as<u16>((crc << 1) ^ 0x8005u) : keel::as<u16>(crc << 1);
            }
        }
        write_bits(crc, 16);
    }

    [[nodiscard]] Array<u8> finish() {
        align_to_byte();
        return keel::move(m_data);
    }

private:
    Array<u8> m_data;
    u8 m_bit_buf = 0;
    u32 m_bit_count = 0;
};

void write_flac_stream_header(FlacBitWriter& bw, u32 block_size, u32 sample_rate, u32 channels, u32 bps,
                              u64 total_frames, bool has_vorbis_comment = false) {
    bw.write_bytes("fLaC", 4);

    // STREAMINFO block (type 0, length 34)
    bw.write_bits(has_vorbis_comment ? 0x00u : 0x80u, 8);
    bw.write_bits(34u, 24);
    bw.write_bits(block_size, 16);
    bw.write_bits(block_size, 16);
    bw.write_bits(0, 24);
    bw.write_bits(0, 24);
    bw.write_bits(sample_rate, 20);
    bw.write_bits(channels - 1u, 3);
    bw.write_bits(bps - 1u, 5);
    bw.write_bits(total_frames, 36);
    for (i32 i = 0; i < 16; ++i) {
        bw.write_bits(0, 8);
    }

    if (has_vorbis_comment) {
        const char* vendor = "keel-test";
        const char* c0 = "TITLE=Lossless Track";
        const char* c1 = "ARTIST=Keel Audio";
        const char* c2 = "COMMENT=Synthetic FLAC";
        const u32 vlen = keel::as<u32>(StringRef(vendor).size());
        const u32 l0 = keel::as<u32>(StringRef(c0).size());
        const u32 l1 = keel::as<u32>(StringRef(c1).size());
        const u32 l2 = keel::as<u32>(StringRef(c2).size());
        const u32 payload_len = 4u + vlen + 4u + (4u + l0) + (4u + l1) + (4u + l2);

        bw.write_bits(0x84u, 8); // is_last = 1, block_type = 4 (VORBIS_COMMENT)
        bw.write_bits(payload_len, 24);
        bw.write_u32_le_bytes(vlen);
        bw.write_bytes(vendor, vlen);
        bw.write_u32_le_bytes(3u);
        bw.write_u32_le_bytes(l0);
        bw.write_bytes(c0, l0);
        bw.write_u32_le_bytes(l1);
        bw.write_bytes(c1, l1);
        bw.write_u32_le_bytes(l2);
        bw.write_bytes(c2, l2);
    }
}

usize write_flac_frame_header(FlacBitWriter& bw, u32 frame_idx, u32 block_size, u32 chan_assign) {
    bw.align_to_byte();
    const usize frame_start = bw.byte_offset();
    bw.write_bits(0x3FFEu, 14);
    bw.write_bits(0, 1);           // reserved
    bw.write_bits(0, 1);           // fixed block size
    bw.write_bits(7, 4);           // 16-bit block_size - 1 at end of frame header
    bw.write_bits(0, 4);           // sample rate from STREAMINFO
    bw.write_bits(chan_assign, 4); // channel assignment
    bw.write_bits(0, 3);           // sample size from STREAMINFO
    bw.write_bits(0, 1);           // reserved
    bw.write_utf8_u32(frame_idx);
    bw.write_bits(block_size - 1u, 16);
    bw.write_crc8_from(frame_start);
    return frame_start;
}

// ============================================================================
// 1. Format Sniffing & Basic Utilities
// ============================================================================
void test_sniffing_and_helpers() {
    TEST_CHECK(audio_sample_size(AudioSampleType::U8) == 1);
    TEST_CHECK(audio_sample_size(AudioSampleType::I8) == 1);
    TEST_CHECK(audio_sample_size(AudioSampleType::I16) == 2);
    TEST_CHECK(audio_sample_size(AudioSampleType::I32) == 4);
    TEST_CHECK(audio_sample_size(AudioSampleType::F32) == 4);
    TEST_CHECK(audio_sample_size(AudioSampleType::F64) == 8);

    TEST_CHECK(audio_format_name(AudioFormat::Wav) == "wav");
    TEST_CHECK(audio_format_name(AudioFormat::Flac) == "flac");
    TEST_CHECK(audio_format_name(AudioFormat::Mp3) == "mp3");
    TEST_CHECK(audio_format_name(AudioFormat::Aiff) == "aiff");
    TEST_CHECK(audio_format_name(AudioFormat::Au) == "au");
    TEST_CHECK(audio_format_name(AudioFormat::Midi) == "midi");
    TEST_CHECK(audio_format_name(AudioFormat::Unknown) == "unknown");

    ArrayRef<u8> empty{};
    TEST_CHECK(!is_wav_bytes(empty));
    TEST_CHECK(!is_flac_bytes(empty));
    TEST_CHECK(!is_mp3_bytes(empty));
    TEST_CHECK(!is_aiff_bytes(empty));
    TEST_CHECK(!is_au_bytes(empty));
    TEST_CHECK(!is_midi_bytes(empty));
    TEST_CHECK(sniff_audio_format(empty) == AudioFormat::Unknown);

    const u8 wav_hdr[] = {'R', 'I', 'F', 'F', 0x24, 0, 0, 0, 'W', 'A', 'V', 'E'};
    const u8 flac_hdr[] = {'f', 'L', 'a', 'C', 0x80, 0, 0, 34};
    const u8 mp3_id3[] = {'I', 'D', '3', 3, 0, 0, 0, 0, 0, 0};
    const u8 mp3_sync[] = {0xFF, 0xFB, 0x90, 0x00};
    const u8 aiff_hdr[] = {'F', 'O', 'R', 'M', 0, 0, 0, 4, 'A', 'I', 'F', 'F'};
    const u8 aifc_hdr[] = {'F', 'O', 'R', 'M', 0, 0, 0, 4, 'A', 'I', 'F', 'C'};
    const u8 au_hdr[] = {'.', 's', 'n', 'd', 0, 0, 0, 24};
    const u8 midi_hdr[] = {'M', 'T', 'h', 'd', 0, 0, 0, 6};

    TEST_CHECK(sniff_audio_format(wav_hdr) == AudioFormat::Wav);
    TEST_CHECK(sniff_audio_format(flac_hdr) == AudioFormat::Flac);
    TEST_CHECK(sniff_audio_format(mp3_id3) == AudioFormat::Mp3);
    TEST_CHECK(sniff_audio_format(mp3_sync) == AudioFormat::Mp3);
    TEST_CHECK(sniff_audio_format(aiff_hdr) == AudioFormat::Aiff);
    TEST_CHECK(sniff_audio_format(aifc_hdr) == AudioFormat::Aiff);
    TEST_CHECK(sniff_audio_format(au_hdr) == AudioFormat::Au);
    TEST_CHECK(sniff_audio_format(midi_hdr) == AudioFormat::Midi);
}

// ============================================================================
// 2. WAV Codec Tests
// ============================================================================
void test_wav_codec() {
    const u64 frames = 500;
    const f64 fs = 44100.0;

    // Planar stereo F64 input -> encode_wav -> peek_wav -> decode_wav (Interleaved & Planar)
    Array<f64> planar_in(keel::as<usize>(frames * 2));
    for (u64 t = 0; t < frames; ++t) {
        planar_in[keel::as<usize>(0 * frames + t)] = 0.8 * keel::sin(2.0 * kPi * 440.0 * keel::as<f64>(t) / fs);
        planar_in[keel::as<usize>(1 * frames + t)] = -0.5 * keel::cos(2.0 * kPi * 880.0 * keel::as<f64>(t) / fs);
    }

    AudioView view_planar = AudioView::from_f64(planar_in, frames, 2, fs, AudioChannelLayout::Planar);
    auto wav16 = encode_wav(view_planar,
                            AudioEncodeOptions{
                                .bits_per_sample = 16,
                                .title = "Sine Wave",
                                .artist = "Keel DSP",
                                .comment = "LIST INFO Metadata",
                            });
    TEST_CHECK(wav16.is_ok());

    auto info = peek_audio(*wav16);
    TEST_CHECK(info.is_ok());
    TEST_CHECK(info->format == AudioFormat::Wav);
    TEST_CHECK(info->compression_method == "Uncompressed");
    TEST_CHECK(info->channels == 2);
    TEST_CHECK(info->sample_rate == 44100.0);
    TEST_CHECK(info->bits_per_sample == 16);
    TEST_CHECK(info->total_frames == frames);
    TEST_CHECK(near_f64(info->duration_sec, keel::as<f64>(frames) / fs, 1e-9));
    TEST_CHECK(info->metadata.title == "Sine Wave");
    TEST_CHECK(info->metadata.artist == "Keel DSP");
    TEST_CHECK(info->metadata.comment == "LIST INFO Metadata");

    // Decode Interleaved normalized F64
    auto dec_inter = decode_wav(*wav16, AudioDecodeOptions{.layout = AudioChannelLayout::Interleaved});
    TEST_CHECK(dec_inter.is_ok());
    TEST_CHECK(dec_inter->frames == frames && dec_inter->channels == 2);
    TEST_CHECK(dec_inter->sample_type == AudioSampleType::F64);
    TEST_CHECK(dec_inter->metadata.title == "Sine Wave");
    ArrayRef<f64> s_inter = dec_inter->samples_f64();
    for (u64 t = 0; t < frames; ++t) {
        TEST_CHECK(near_f64(s_inter[keel::as<usize>(t * 2 + 0)], planar_in[keel::as<usize>(t)], 1e-4));
        TEST_CHECK(near_f64(s_inter[keel::as<usize>(t * 2 + 1)], planar_in[keel::as<usize>(frames + t)], 1e-4));
    }

    // Decode Planar normalized F64 + partial range [100, 200)
    auto dec_slice = decode_wav(*wav16,
                                AudioDecodeOptions{
                                    .start_frame = 100,
                                    .end_frame = 200,
                                    .layout = AudioChannelLayout::Planar,
                                });
    TEST_CHECK(dec_slice.is_ok());
    TEST_CHECK(dec_slice->frames == 100 && dec_slice->total_source_frames == frames);
    ArrayRef<f64> s_slice = dec_slice->samples_f64();
    for (u64 t = 0; t < 100; ++t) {
        TEST_CHECK(near_f64(s_slice[keel::as<usize>(t)], planar_in[keel::as<usize>(100 + t)], 1e-4));
        TEST_CHECK(near_f64(s_slice[keel::as<usize>(100 + t)], planar_in[keel::as<usize>(frames + 100 + t)], 1e-4));
    }

    // All bit depths + native sample types (8-bit U8, 16-bit I16, 24-bit I32, 32-bit I32, 32-bit F32, 64-bit F64)
    const u64 n = 127; // odd frame count to test RIFF chunk word-alignment padding
    Array<f64> ramp(keel::as<usize>(n));
    for (u64 i = 0; i < n; ++i) {
        ramp[keel::as<usize>(i)] = -0.9 + 1.8 * (keel::as<f64>(i) / keel::as<f64>(n - 1));
    }
    AudioView mono_view = AudioView::from_f64(ramp, n, 1, 48000.0);

    // 8-bit PCM (native U8)
    auto b8 = encode_wav(mono_view, AudioEncodeOptions{.bits_per_sample = 8, .title = "OddPad"});
    TEST_CHECK(b8.is_ok());
    auto d8_nat = decode_wav(*b8, AudioDecodeOptions{.native_sample_type = true});
    auto d8_f64 = decode_wav(*b8);
    TEST_CHECK(d8_nat.is_ok() && d8_nat->sample_type == AudioSampleType::U8 && d8_nat->metadata.title == "OddPad");
    TEST_CHECK(d8_f64.is_ok());
    for (usize i = 0; i < keel::as<usize>(n); ++i) {
        TEST_CHECK(near_f64(d8_f64->samples_f64()[i], ramp[i], 0.02));
    }
    // Re-encode directly from U8 AudioView
    auto b8_re = encode_wav(d8_nat->as_view(), AudioEncodeOptions{.bits_per_sample = 8});
    auto d8_re = decode_wav(*b8_re, AudioDecodeOptions{.native_sample_type = true});
    TEST_CHECK(d8_re.is_ok() && ArrayRef<u8>(d8_re->samples_u8()) == ArrayRef<u8>(d8_nat->samples_u8()));

    // 24-bit PCM (native I32)
    auto b24 = encode_wav(mono_view, AudioEncodeOptions{.bits_per_sample = 24, .artist = "Studio"});
    TEST_CHECK(b24.is_ok());
    auto d24_nat = decode_wav(*b24, AudioDecodeOptions{.native_sample_type = true});
    auto d24_f64 = decode_wav(*b24);
    TEST_CHECK(d24_nat.is_ok() && d24_nat->sample_type == AudioSampleType::I32 && d24_nat->metadata.artist == "Studio");
    for (usize i = 0; i < keel::as<usize>(n); ++i) {
        TEST_CHECK(near_f64(d24_f64->samples_f64()[i], ramp[i], 1e-6));
    }

    // 32-bit integer PCM (native I32)
    auto b32i = encode_wav(mono_view, AudioEncodeOptions{.bits_per_sample = 32, .float_samples = false});
    TEST_CHECK(b32i.is_ok());
    auto d32i = decode_wav(*b32i, AudioDecodeOptions{.native_sample_type = true});
    TEST_CHECK(d32i.is_ok() && d32i->sample_type == AudioSampleType::I32);
    auto b32i_re = encode_wav(d32i->as_view(), AudioEncodeOptions{.bits_per_sample = 32});
    auto d32i_re = decode_wav(*b32i_re, AudioDecodeOptions{.native_sample_type = true});
    TEST_CHECK(d32i_re.is_ok() && ArrayRef<i32>(d32i_re->samples_i32()) == ArrayRef<i32>(d32i->samples_i32()));

    // 32-bit IEEE float (both via F32 source view and via float_samples=true)
    Array<f32> ramp_f32(keel::as<usize>(n));
    for (usize i = 0; i < keel::as<usize>(n); ++i) {
        ramp_f32[i] = keel::as<f32>(ramp[i]);
    }
    auto b32f = encode_wav(AudioView::from_f32(ramp_f32, n, 1, 48000.0), AudioEncodeOptions{.bits_per_sample = 32});
    TEST_CHECK(b32f.is_ok());
    TEST_CHECK(peek_wav(*b32f)->compression_method == "IEEE Float");
    auto d32f = decode_wav(*b32f, AudioDecodeOptions{.native_sample_type = true});
    TEST_CHECK(d32f.is_ok() && d32f->sample_type == AudioSampleType::F32);
    for (usize i = 0; i < keel::as<usize>(n); ++i) {
        TEST_CHECK(near_f32(d32f->samples_f32()[i], ramp_f32[i], 1e-7f));
    }

    // 64-bit IEEE float
    auto b64f = encode_wav(mono_view, AudioEncodeOptions{.bits_per_sample = 64, .float_samples = true});
    TEST_CHECK(b64f.is_ok());
    auto d64f = decode_wav(*b64f, AudioDecodeOptions{.native_sample_type = true});
    TEST_CHECK(d64f.is_ok() && d64f->sample_type == AudioSampleType::F64);
    for (usize i = 0; i < keel::as<usize>(n); ++i) {
        TEST_CHECK(d64f->samples_f64()[i] == ramp[i]);
    }

    // Synthetic Mu-law (tag 7) & WAVE_FORMAT_EXTENSIBLE (0xFFFE -> PCM 16-bit)
    const u8 mulaw_wav[] = {
        'R',  'I',  'F',  'F',  40,   0,    0,    0,    'W',  'A',  'V',  'E',
        'f',  'm',  't',  ' ',  16,   0,    0,    0,    7,    0,    1,    0,
        0x40, 0x1F, 0,    0,    0x40, 0x1F, 0,    0,    1,    0,    8,    0,
        'd',  'a',  't',  'a',  4,    0,    0,    0,    0xFF, 0x7F, 0x80, 0x00,
    };
    auto d_mulaw = decode_wav(mulaw_wav, AudioDecodeOptions{.native_sample_type = true});
    TEST_CHECK(d_mulaw.is_ok() && d_mulaw->sample_type == AudioSampleType::I16 && d_mulaw->frames == 4);
    TEST_CHECK(d_mulaw->samples_i16()[0] == 0 && d_mulaw->samples_i16()[1] == 0);
    TEST_CHECK(d_mulaw->samples_i16()[2] > 30000 && d_mulaw->samples_i16()[3] < -30000);

    const u8 ext_wav[] = {
        'R',  'I',  'F',  'F',  64,   0,    0,    0,    'W',  'A',  'V',  'E',
        'f',  'm',  't',  ' ',  40,   0,    0,    0,    0xFE, 0xFF, 1,    0,
        0x44, 0xAC, 0,    0,    0x88, 0x58, 1,    0,    2,    0,    16,   0,
        22,   0,    16,   0,    4,    0,    0,    0,    1,    0,    0,    0,
        0,    0,    0x10, 0,    0x80, 0,    0,    0xAA, 0,    0x38, 0x9B, 0x71,
        'd',  'a',  't',  'a',  4,    0,    0,    0,    0x00, 0x40, 0x00, 0xC0,
    };
    auto d_ext = decode_wav(ext_wav, AudioDecodeOptions{.native_sample_type = true});
    TEST_CHECK(d_ext.is_ok() && d_ext->frames == 2);
    TEST_CHECK(d_ext->samples_i16()[0] == 16384 && d_ext->samples_i16()[1] == -16384);
}

// ============================================================================
// 3. FLAC Codec Tests (Verbatim, Constant, Fixed LPC, Adaptive FIR LPC, Stereo, Metadata)
// ============================================================================
void test_flac_codec() {
    // (A) Verbatim + Vorbis Comments + Multi-frame + Frame Slicing
    {
        const u32 block_sz = 64;
        const u32 total = block_sz * 2;
        Array<i16> left(total), right(total);
        for (u32 i = 0; i < total; ++i) {
            left[i] = keel::as<i16>(keel::as<i32>(i) * 180 - 11000);
            right[i] = keel::as<i16>(9500 - keel::as<i32>(i) * 140);
        }

        FlacBitWriter bw;
        write_flac_stream_header(bw, block_sz, 44100, 2, 16, total, true);
        for (u32 f = 0; f < 2; ++f) {
            const usize fstart = write_flac_frame_header(bw, f, block_sz, 1u); // Independent L/R
            for (i32 ch = 0; ch < 2; ++ch) {
                bw.write_bits(0, 1);  // zero pad
                bw.write_bits(1u, 6); // Verbatim
                bw.write_bits(0, 1);  // no wasted bits
                const i16* src = (ch == 0) ? left.data() : right.data();
                for (u32 i = 0; i < block_sz; ++i) {
                    bw.write_signed_bits(src[f * block_sz + i], 16);
                }
            }
            bw.write_crc16_from(fstart);
        }
        Array<u8> flac_bytes = bw.finish();

        auto info = peek_audio(flac_bytes);
        TEST_CHECK(info.is_ok());
        TEST_CHECK(info->format == AudioFormat::Flac);
        TEST_CHECK(info->channels == 2 && info->sample_rate == 44100.0 && info->bits_per_sample == 16);
        TEST_CHECK(info->total_frames == total);
        TEST_CHECK(info->metadata.title == "Lossless Track");
        TEST_CHECK(info->metadata.artist == "Keel Audio");
        TEST_CHECK(info->metadata.comment == "Synthetic FLAC");

        auto dec = decode_flac(flac_bytes, AudioDecodeOptions{.native_sample_type = true});
        TEST_CHECK(dec.is_ok() && dec->sample_type == AudioSampleType::I16 && dec->frames == total);
        for (u32 i = 0; i < total; ++i) {
            TEST_CHECK(dec->samples_i16()[i * 2 + 0] == left[i]);
            TEST_CHECK(dec->samples_i16()[i * 2 + 1] == right[i]);
        }

        // Cross-frame partial slice [40, 90) in Planar normalized F64
        auto slice = decode_flac(flac_bytes,
                                 AudioDecodeOptions{
                                     .start_frame = 40,
                                     .end_frame = 90,
                                     .layout = AudioChannelLayout::Planar,
                                 });
        TEST_CHECK(slice.is_ok() && slice->frames == 50 && slice->total_source_frames == total);
        for (u32 i = 0; i < 50; ++i) {
            TEST_CHECK(near_f64(slice->samples_f64()[i], keel::as<f64>(left[40 + i]) / 32768.0, 1e-9));
            TEST_CHECK(near_f64(slice->samples_f64()[50 + i], keel::as<f64>(right[40 + i]) / 32768.0, 1e-9));
        }
    }

    // (B) Constant + Wasted Bits + Fixed LPC (order 2) + Mid-Side Stereo Decorrelation (chan_assign = 10)
    {
        const u32 n = 32;
        Array<i32> left(n), right(n);
        for (u32 i = 0; i < n; ++i) {
            // Smooth quadratic + small oscillation on Left, constant wasted-bits on Right -> encode via Mid-Side
            left[i] = keel::as<i32>(i * i * 4 - 200 + ((i & 1u) ? 3 : -3));
            right[i] = 512; // divisible by 2^9
        }

        FlacBitWriter bw;
        write_flac_stream_header(bw, n, 48000, 2, 16, n, false);
        const usize fstart = write_flac_frame_header(bw, 0, n, 10u); // Mid-Side stereo: ch0 = Mid (16-bit), ch1 = Side (17-bit)

        Array<i64> mid(n), side(n);
        for (u32 i = 0; i < n; ++i) {
            side[i] = keel::as<i64>(left[i]) - keel::as<i64>(right[i]);
            mid[i] = (keel::as<i64>(left[i]) + keel::as<i64>(right[i])) >> 1;
        }

        // Subframe 0 (Mid): Fixed LPC order 2 (subframe_type = 0x08 | 2 = 10) with Rice k=3
        bw.write_bits(0, 1);
        bw.write_bits(0x08u | 2u, 6);
        bw.write_bits(0, 1);
        bw.write_signed_bits(mid[0], 16);
        bw.write_signed_bits(mid[1], 16);
        bw.write_bits(0, 2); // coding_method = 0 (4-bit Rice param)
        bw.write_bits(0, 4); // partition_order = 0 (1 partition)
        bw.write_bits(3, 4); // Rice k = 3
        for (u32 i = 2; i < n; ++i) {
            const i64 pred = 2 * mid[i - 1] - mid[i - 2];
            bw.write_rice_signed(mid[i] - pred, 3u);
        }

        // Subframe 1 (Side, 17-bit): Verbatim or Fixed LPC order 1 with Rice escape partition
        bw.write_bits(0, 1);
        bw.write_bits(0x08u | 1u, 6); // Fixed LPC order 1
        bw.write_bits(0, 1);
        bw.write_signed_bits(side[0], 17);
        bw.write_bits(0, 2);    // coding_method = 0
        bw.write_bits(0, 4);    // partition_order = 0
        bw.write_bits(0x0Fu, 4); // Rice escape!
        bw.write_bits(12u, 5);   // raw 12-bit signed residuals
        for (u32 i = 1; i < n; ++i) {
            bw.write_signed_bits(side[i] - side[i - 1], 12u);
        }

        bw.write_crc16_from(fstart);
        Array<u8> flac_ms = bw.finish();

        auto dec = decode_flac(flac_ms, AudioDecodeOptions{.native_sample_type = true});
        TEST_CHECK(dec.is_ok() && dec->frames == n);
        for (u32 i = 0; i < n; ++i) {
            TEST_CHECK(dec->samples_i16()[i * 2 + 0] == keel::as<i16>(left[i]));
            TEST_CHECK(dec->samples_i16()[i * 2 + 1] == keel::as<i16>(right[i]));
        }
    }

    // (C) Adaptive FIR LPC (order 2) + Constant subframe with wasted bits + 24-bit native I32
    {
        const u32 n = 32;
        Array<i32> sig(n);
        sig[0] = 10000;
        sig[1] = 14000;
        // QLP coeffs: c0 = 12, c1 = -4, shift = 3 => pred = (12 * s[i-1] - 4 * s[i-2]) >> 3
        for (u32 i = 2; i < n; ++i) {
            const i64 pred = (12LL * sig[i - 1] - 4LL * sig[i - 2]) >> 3;
            const i64 res = keel::as<i64>((i % 5u) - 2) * 16; // divisible by 16 (4 wasted bits if desired)
            sig[i] = keel::as<i32>(pred + res);
        }

        FlacBitWriter bw;
        write_flac_stream_header(bw, n, 96000, 2, 24, n, false);
        const usize fstart =
            write_flac_frame_header(bw, 0, n, 8u); // Left-Side stereo: ch0 = Left (24-bit), ch1 = Side = L - R (25-bit)

        // Left: Adaptive FIR LPC order 2 (subframe_type = 0x20 | 1)
        bw.write_bits(0, 1);
        bw.write_bits(0x20u | 1u, 6);
        bw.write_bits(0, 1); // wasted = 0
        bw.write_signed_bits(sig[0], 24);
        bw.write_signed_bits(sig[1], 24);
        bw.write_bits(5u, 4);        // qlp_precision = 5 + 1 = 6 bits
        bw.write_signed_bits(3, 5);  // qlp_shift = 3
        bw.write_signed_bits(12, 6); // coeff[0]
        bw.write_signed_bits(-4, 6); // coeff[1]
        bw.write_bits(0, 2);
        bw.write_bits(0, 4);
        bw.write_bits(5u, 4); // Rice k = 5
        for (u32 i = 2; i < n; ++i) {
            const i64 pred = (12LL * sig[i - 1] - 4LL * sig[i - 2]) >> 3;
            bw.write_rice_signed(keel::as<i64>(sig[i]) - pred, 5u);
        }

        // Side = 256 (constant across all samples, with 8 wasted bits => 1 << 8 = 256 -> value = 1!)
        bw.write_bits(0, 1);
        bw.write_bits(0u, 6); // Constant subframe
        bw.write_bits(1, 1);  // wasted_flag = 1
        bw.write_unary(7u);   // wasted_bits = 7 + 1 = 8
        bw.write_signed_bits(1, 25u - 8u);

        bw.write_crc16_from(fstart);
        Array<u8> flac_lpc = bw.finish();

        auto dec = decode_flac(flac_lpc, AudioDecodeOptions{.native_sample_type = true});
        TEST_CHECK(dec.is_ok() && dec->sample_type == AudioSampleType::I32 && dec->frames == n);
        for (u32 i = 0; i < n; ++i) {
            TEST_CHECK(dec->samples_i32()[i * 2 + 0] == sig[i]);
            TEST_CHECK(dec->samples_i32()[i * 2 + 1] == sig[i] - 256);
        }

        // (D) RFC 9639 CRC-8 & CRC-16 corruption detection: 1-bit flip in header or footer must be rejected!
        Array<u8> bad_crc8 = flac_lpc.clone();
        bad_crc8[fstart + 7u] ^= 0x01u; // flip bit in header CRC-8 byte
        TEST_CHECK(decode_flac(bad_crc8).is_err());

        Array<u8> bad_crc16 = flac_lpc.clone();
        bad_crc16[bad_crc16.size() - 1u] ^= 0x01u; // flip bit in frame CRC-16 footer
        TEST_CHECK(decode_flac(bad_crc16).is_err());
    }
}

// ============================================================================
// 4. MP3 Codec Tests (MPEG-1/2 Layer III, ID3v2/ID3v1, Xing/Info Header, Slicing)
// ============================================================================
void append_id3v23_frame(Array<u8>& buf, const char (&id)[5], ArrayRef<u8> payload) {
    buf.append(ArrayRef<u8>(reinterpret_cast<const u8*>(id), 4));
    const u32 sz = keel::as<u32>(payload.size());
    buf.push_back(keel::as<u8>((sz >> 24) & 0xFFu));
    buf.push_back(keel::as<u8>((sz >> 16) & 0xFFu));
    buf.push_back(keel::as<u8>((sz >> 8) & 0xFFu));
    buf.push_back(keel::as<u8>(sz & 0xFFu));
    buf.push_back(0u);
    buf.push_back(0u);
    buf.append(payload);
}

void test_mp3_codec() {
    // Build ID3v2.3 header + Xing header frame + 2 MPEG-1 Layer III stereo frames + ID3v1 footer
    Array<u8> id3_frames;
    const u8 tit2_payload[] = {0, 'M', 'P', '3', ' ', 'T', 'i', 't', 'l', 'e', 0};
    const u8 comm_payload[] = {0, 'e', 'n', 'g', 0, 'V', 'B', 'R', ' ', 'T', 'a', 'g'};
    append_id3v23_frame(id3_frames, "TIT2", tit2_payload);
    append_id3v23_frame(id3_frames, "COMM", comm_payload);

    Array<u8> stream;
    stream.append(ArrayRef<u8>(reinterpret_cast<const u8*>("ID3"), 3));
    stream.push_back(3u); // v2.3
    stream.push_back(0u);
    stream.push_back(0u);
    const u32 tag_sz = keel::as<u32>(id3_frames.size());
    stream.push_back(keel::as<u8>((tag_sz >> 21) & 0x7Fu));
    stream.push_back(keel::as<u8>((tag_sz >> 14) & 0x7Fu));
    stream.push_back(keel::as<u8>((tag_sz >> 7) & 0x7Fu));
    stream.push_back(keel::as<u8>(tag_sz & 0x7Fu));
    stream.append(id3_frames);

    // MPEG-1 Layer III, 128 kbps, 44100 Hz, Joint Stereo => frame_size = 417 bytes, 1152 samples/frame
    const usize frame_size = 417;

    // Frame 0: Xing VBR metadata frame (side_info = 32 => Xing offset = 36)
    Array<u8> xing_frame(frame_size, u8{0});
    xing_frame[0] = 0xFFu;
    xing_frame[1] = 0xFBu;
    xing_frame[2] = 0x90u;
    xing_frame[3] = 0x40u;
    keel::mem_copy(xing_frame.data() + 36, "Xing", 4);
    xing_frame[40] = 0;
    xing_frame[41] = 0;
    xing_frame[42] = 0;
    xing_frame[43] = 0x03u; // frames + bytes present
    xing_frame[44] = 0;
    xing_frame[45] = 0;
    xing_frame[46] = 0;
    xing_frame[47] = 2u; // 2 audio frames => 2304 total samples
    const u32 total_vbr_bytes = keel::as<u32>(frame_size * 3);
    xing_frame[48] = keel::as<u8>((total_vbr_bytes >> 24) & 0xFFu);
    xing_frame[49] = keel::as<u8>((total_vbr_bytes >> 16) & 0xFFu);
    xing_frame[50] = keel::as<u8>((total_vbr_bytes >> 8) & 0xFFu);
    xing_frame[51] = keel::as<u8>(total_vbr_bytes & 0xFFu);
    stream.append(xing_frame);

    // Frames 1 & 2: Valid MPEG-1 Layer III 128 kbps 44100 Hz Joint-Stereo frames
    // Side info (32 bytes): main_data_begin=0, scfsi=0, gr0/gr1 ch0/ch1 have part2_3_length=64,
    // big_values=0, global_gain=200, count1_table=1 (Table B -> non-zero quadruples in main data).
    for (i32 f = 0; f < 2; ++f) {
        Array<u8> af(frame_size, u8{0});
        af[0] = 0xFFu;
        af[1] = 0xFBu;
        af[2] = 0x90u;
        af[3] = 0x40u;
        // Pack 32-byte MPEG-1 stereo side info MSB-first using FlacBitWriter
        FlacBitWriter sbw;
        sbw.write_bits(0, 9); // main_data_begin = 0
        sbw.write_bits(0, 3); // private_bits
        sbw.write_bits(0, 8); // scfsi[2][4]
        for (i32 gr_ch = 0; gr_ch < 4; ++gr_ch) {
            sbw.write_bits(64, 12);  // part2_3_length = 64 bits
            sbw.write_bits(0, 9);    // big_values = 0
            sbw.write_bits(160, 8);  // global_gain = 160 (moderate amplitude within [-1, 1])
            sbw.write_bits(0, 4);    // scalefac_compress = 0 (0 bits of scalefactors)
            sbw.write_bits(0, 1);    // window_switching_flag = 0
            sbw.write_bits(0, 15);   // table_select[0..2]
            sbw.write_bits(0, 4);    // region0_count
            sbw.write_bits(0, 3);    // region1_count
            sbw.write_bits(0, 1);    // preflag
            sbw.write_bits(0, 1);    // scalefac_scale
            sbw.write_bits(1, 1);    // count1table_select = 1 (Table B)
        }
        Array<u8> side = sbw.finish();
        keel::mem_copy(af.data() + 4, side.data(), side.size());
        // Main data starts at byte 36: Table B code 0x0 (4 ones) + alternating signs -> strong non-zero signal
        for (usize b = 36; b < 36 + 32; ++b) {
            af[b] = keel::as<u8>((b & 1u) ? 0x05u : 0x0Au);
        }
        stream.append(af);
    }

    // Trailing 128-byte ID3v1 tag (supplies fallback artist while preserving ID3v2 title/comment!)
    Array<u8> id3v1(128, u8{0});
    keel::mem_copy(id3v1.data(), "TAG", 3);
    keel::mem_copy(id3v1.data() + 3, "Ignored V1 Title", 16);
    keel::mem_copy(id3v1.data() + 33, "Fallback V1 Artist", 18);
    stream.append(id3v1);

    auto info = peek_audio(stream);
    TEST_CHECK(info.is_ok());
    TEST_CHECK(info->format == AudioFormat::Mp3);
    TEST_CHECK(info->channels == 2);
    TEST_CHECK(info->sample_rate == 44100.0);
    TEST_CHECK(info->total_frames == 2304);
    TEST_CHECK(info->metadata.title == "MP3 Title");
    TEST_CHECK(info->metadata.artist == "Fallback V1 Artist");
    TEST_CHECK(info->metadata.comment == "VBR Tag");

    auto dec_full = decode_mp3(stream);
    TEST_CHECK(dec_full.is_ok());
    TEST_CHECK(dec_full->frames == 2304 && dec_full->total_source_frames == 2304);
    TEST_CHECK(dec_full->channels == 2 && dec_full->sample_type == AudioSampleType::F64);
    bool has_nonzero = false;
    for (usize i = 0; i < dec_full->sample_count(); ++i) {
        const f64 v = dec_full->samples_f64()[i];
        TEST_CHECK(v >= -1.0 && v <= 1.0);
        if (keel::abs(v) > 1e-4) has_nonzero = true;
    }
    TEST_CHECK(has_nonzero);

    // Frame slicing across frame boundary [1000, 1300) + native I16
    auto dec_slice = decode_mp3(stream,
                                AudioDecodeOptions{
                                    .start_frame = 1000,
                                    .end_frame = 1300,
                                    .native_sample_type = true,
                                    .layout = AudioChannelLayout::Planar,
                                });
    TEST_CHECK(dec_slice.is_ok());
    TEST_CHECK(dec_slice->frames == 300 && dec_slice->total_source_frames == 2304);
    TEST_CHECK(dec_slice->sample_type == AudioSampleType::I16);

    // MPEG-2 Layer III Mono (22050 Hz, 64 kbps => frame_size = 72000 * 64 / 22050 = 208 bytes, 576 samples)
    Array<u8> mpeg2(208, u8{0});
    mpeg2[0] = 0xFFu;
    mpeg2[1] = 0xF3u; // MPEG-2 (version=2), Layer III (layer=1), no CRC
    mpeg2[2] = 0x80u; // bitrate_idx=8 (64 kbps in MPEG-2 L3), sr_idx=0 (22050 Hz), no pad
    mpeg2[3] = 0xC0u; // Mono (channel_mode=3)
    auto m2_info = peek_mp3(mpeg2);
    TEST_CHECK(m2_info.is_ok() && m2_info->channels == 1 && m2_info->sample_rate == 22050.0 &&
               m2_info->bit_rate_kbps == 64);
    auto m2_dec = decode_mp3(mpeg2);
    TEST_CHECK(m2_dec.is_ok() && m2_dec->frames == 576 && m2_dec->channels == 1);
}

// ============================================================================
// 5. AIFF / AIFC Codec Tests (8/16/24/32 PCM, Float32/64 AIFC, ulaw/alaw/sowt, Metadata)
// ============================================================================
void test_aiff_codec() {
    const u64 frames = 255; // odd frame count to test SSND & metadata word-alignment
    const f64 fs = 48000.0;
    Array<f64> stereo(keel::as<usize>(frames * 2));
    for (u64 t = 0; t < frames; ++t) {
        stereo[keel::as<usize>(t * 2 + 0)] = 0.7 * keel::sin(2.0 * kPi * 300.0 * keel::as<f64>(t) / fs);
        stereo[keel::as<usize>(t * 2 + 1)] = -0.4 * keel::sin(2.0 * kPi * 600.0 * keel::as<f64>(t) / fs);
    }
    AudioView view = AudioView::from_f64(stereo, frames, 2, fs, AudioChannelLayout::Interleaved);

    for (u16 bps : {u16{8}, u16{16}, u16{24}, u16{32}}) {
        auto enc = encode_aiff(view,
                               AudioEncodeOptions{
                                   .bits_per_sample = bps,
                                   .title = "Apple AIFF",
                                   .artist = "Cupertino",
                                   .comment = "Big Endian",
                               });
        TEST_CHECK(enc.is_ok() && is_aiff_bytes(*enc));

        auto info = peek_aiff(*enc);
        TEST_CHECK(info.is_ok());
        TEST_CHECK(info->format == AudioFormat::Aiff && info->channels == 2);
        TEST_CHECK(near_f64(info->sample_rate, fs, 1e-6));
        TEST_CHECK(info->bits_per_sample == bps && info->total_frames == frames);
        TEST_CHECK(info->metadata.title == "Apple AIFF");
        TEST_CHECK(info->metadata.artist == "Cupertino");
        TEST_CHECK(info->metadata.comment == "Big Endian");

        auto dec = decode_aiff(*enc);
        TEST_CHECK(dec.is_ok() && dec->frames == frames);
        const f64 tol = (bps == 8) ? 0.02 : 1e-4;
        for (usize i = 0; i < stereo.size(); ++i) {
            TEST_CHECK(near_f64(dec->samples_f64()[i], stereo[i], tol));
        }

        auto dec_nat = decode_aiff(*enc, AudioDecodeOptions{.native_sample_type = true});
        TEST_CHECK(dec_nat.is_ok());
        if (bps == 8) TEST_CHECK(dec_nat->sample_type == AudioSampleType::I8);
        if (bps == 16) TEST_CHECK(dec_nat->sample_type == AudioSampleType::I16);
        if (bps >= 24) TEST_CHECK(dec_nat->sample_type == AudioSampleType::I32);
    }

    // Float32 & Float64 AIFC round-trip
    for (u16 fbps : {u16{32}, u16{64}}) {
        auto enc_f = encode_aiff(view, AudioEncodeOptions{.bits_per_sample = fbps, .float_samples = true});
        TEST_CHECK(enc_f.is_ok() && is_aiff_bytes(*enc_f));
        auto info_f = peek_aiff(*enc_f);
        TEST_CHECK(info_f.is_ok() && info_f->bits_per_sample == fbps);
        auto dec_f = decode_aiff(*enc_f, AudioDecodeOptions{.native_sample_type = true});
        TEST_CHECK(dec_f.is_ok());
        if (fbps == 32) {
            TEST_CHECK(dec_f->sample_type == AudioSampleType::F32);
            for (usize i = 0; i < stereo.size(); ++i) {
                TEST_CHECK(near_f32(dec_f->samples_f32()[i], keel::as<f32>(stereo[i]), 1e-6f));
            }
        } else {
            TEST_CHECK(dec_f->sample_type == AudioSampleType::F64);
            for (usize i = 0; i < stereo.size(); ++i) {
                TEST_CHECK(dec_f->samples_f64()[i] == stereo[i]);
            }
        }
    }

    // Synthetic AIFC with 'ulaw', 'alaw', and 'sowt'
    const u8 aifc_ulaw[] = {
        'F',  'O',  'R',  'M',  0,    0,    0,    50,   'A',  'I',  'F',  'C',
        'C',  'O',  'M',  'M',  0,    0,    0,    22,   0,    1,    0,    0,
        0,    4,    0,    8,    0x40, 0x0B, 0xFA, 0x00, 0,    0,    0,    0,
        0,    0,    'u',  'l',  'a',  'w',  'S',  'S',  'N',  'D',  0,    0,
        0,    12,   0,    0,    0,    0,    0,    0,    0,    0,    0xFF, 0x80,
        0x00, 0x7F,
    };
    auto d_ulaw = decode_aiff(aifc_ulaw, AudioDecodeOptions{.native_sample_type = true});
    TEST_CHECK(d_ulaw.is_ok() && d_ulaw->sample_type == AudioSampleType::I16 && d_ulaw->frames == 4);
    TEST_CHECK(d_ulaw->samples_i16()[0] == 0 && d_ulaw->samples_i16()[1] > 30000 &&
               d_ulaw->samples_i16()[2] < -30000);

    Array<u8> aifc_alaw = Array<u8>::from(aifc_ulaw);
    keel::mem_copy(aifc_alaw.data() + 38, "alaw", 4);
    aifc_alaw[58] = 0xD5u; // A-law near-zero positive (+8)
    aifc_alaw[59] = 0x55u; // A-law near-zero negative (-8)
    aifc_alaw[60] = 0xAAu; // A-law loud positive (> +30000)
    aifc_alaw[61] = 0x2Au; // A-law loud negative (< -30000)
    auto d_alaw = decode_aiff(aifc_alaw, AudioDecodeOptions{.native_sample_type = true});
    TEST_CHECK(d_alaw.is_ok() && d_alaw->sample_type == AudioSampleType::I16);
    TEST_CHECK(d_alaw->samples_i16()[0] == 8 && d_alaw->samples_i16()[1] == -8);
    TEST_CHECK(d_alaw->samples_i16()[2] > 30000 && d_alaw->samples_i16()[3] < -30000);
}

// ============================================================================
// 6. AU / SND Codec Tests (8/16/24/32 PCM, Float32/64, Mu-law, A-law, Comment)
// ============================================================================
void test_au_codec() {
    const u64 frames = 200;
    const f64 fs = 16000.0;
    Array<f64> samples(keel::as<usize>(frames * 2));
    for (u64 t = 0; t < frames; ++t) {
        samples[keel::as<usize>(0 * frames + t)] = 0.6 * keel::sin(2.0 * kPi * 500.0 * keel::as<f64>(t) / fs);
        samples[keel::as<usize>(1 * frames + t)] = -0.3 * keel::cos(2.0 * kPi * 900.0 * keel::as<f64>(t) / fs);
    }
    AudioView view_planar = AudioView::from_f64(samples, frames, 2, fs, AudioChannelLayout::Planar);

    for (u16 bps : {u16{8}, u16{16}, u16{24}, u16{32}}) {
        auto enc = encode_au(view_planar, AudioEncodeOptions{.bits_per_sample = bps, .comment = "Sun NeXT Audio"});
        TEST_CHECK(enc.is_ok() && is_au_bytes(*enc));

        auto info = peek_au(*enc);
        TEST_CHECK(info.is_ok() && info->format == AudioFormat::Au);
        TEST_CHECK(info->channels == 2 && info->sample_rate == fs && info->bits_per_sample == bps);
        TEST_CHECK(info->total_frames == frames && info->metadata.comment == "Sun NeXT Audio");

        auto dec = decode_au(*enc, AudioDecodeOptions{.layout = AudioChannelLayout::Planar});
        TEST_CHECK(dec.is_ok() && dec->frames == frames);
        const f64 tol = (bps == 8) ? 0.02 : 1e-4;
        for (usize i = 0; i < samples.size(); ++i) {
            TEST_CHECK(near_f64(dec->samples_f64()[i], samples[i], tol));
        }

        auto dec_nat = decode_au(*enc, AudioDecodeOptions{.native_sample_type = true});
        TEST_CHECK(dec_nat.is_ok());
        if (bps == 8) TEST_CHECK(dec_nat->sample_type == AudioSampleType::I8);
        if (bps == 16) TEST_CHECK(dec_nat->sample_type == AudioSampleType::I16);
        if (bps >= 24) TEST_CHECK(dec_nat->sample_type == AudioSampleType::I32);
    }

    // Float32 & Float64 AU round-trip
    for (u16 fbps : {u16{32}, u16{64}}) {
        auto enc_f = encode_au(view_planar, AudioEncodeOptions{.bits_per_sample = fbps, .float_samples = true});
        TEST_CHECK(enc_f.is_ok());
        auto dec_f = decode_au(*enc_f,
                               AudioDecodeOptions{
                                   .native_sample_type = true,
                                   .layout = AudioChannelLayout::Planar,
                               });
        TEST_CHECK(dec_f.is_ok());
        if (fbps == 32) {
            TEST_CHECK(dec_f->sample_type == AudioSampleType::F32);
            for (usize i = 0; i < samples.size(); ++i) {
                TEST_CHECK(near_f32(dec_f->samples_f32()[i], keel::as<f32>(samples[i]), 1e-6f));
            }
        } else {
            TEST_CHECK(dec_f->sample_type == AudioSampleType::F64);
            for (usize i = 0; i < samples.size(); ++i) {
                TEST_CHECK(dec_f->samples_f64()[i] == samples[i]);
            }
        }
    }

    // Synthetic AU Mu-law (encoding=1) & A-law (encoding=27) with unknown streaming data_size (0xFFFFFFFF)
    const u8 au_mulaw[] = {
        '.',  's',  'n',  'd',  0,    0,    0,    24,   0xFF, 0xFF, 0xFF, 0xFF, 0,    0,
        0,    1,    0,    0,    0x1F, 0x40, 0,    0,    0,    1,    0xFF, 0x80, 0x00, 0x7F,
    };
    auto d_mu = decode_au(au_mulaw, AudioDecodeOptions{.native_sample_type = true});
    TEST_CHECK(d_mu.is_ok() && d_mu->sample_type == AudioSampleType::I16 && d_mu->frames == 4);
    TEST_CHECK(d_mu->samples_i16()[0] == 0 && d_mu->samples_i16()[1] > 30000 && d_mu->samples_i16()[2] < -30000);

    Array<u8> au_alaw = Array<u8>::from(au_mulaw);
    au_alaw[15] = 27u;
    au_alaw[24] = 0xD5u;
    auto d_al = decode_au(au_alaw, AudioDecodeOptions{.native_sample_type = true});
    TEST_CHECK(d_al.is_ok() && keel::abs(d_al->samples_i16()[0]) <= 16);
}

// ============================================================================
// 7. MIDI Codec Tests (SMF 0 & 1, Tempo Changes, Running Status, Round-Trip)
// ============================================================================
void test_midi_codec() {
    // (A) SMF Format 0 round-trip
    const MidiNote notes_in[] = {
        MidiNote{
            .track = 0, .channel = 0, .note_number = 60, .velocity = 80, .start_time_sec = 0.0, .end_time_sec = 0.5},
        MidiNote{
            .track = 0, .channel = 1, .note_number = 64, .velocity = 90, .start_time_sec = 0.5, .duration_sec = 0.5},
        MidiNote{
            .track = 0, .channel = 2, .note_number = 67, .velocity = 100, .start_time_sec = 1.0, .end_time_sec = 1.5},
    };

    auto encoded = encode_midi(notes_in, MidiEncodeOptions{.ticks_per_quarter_note = 480, .tempo_bpm = 120.0});
    TEST_CHECK(encoded.is_ok() && is_midi_bytes(*encoded));
    TEST_CHECK(sniff_audio_format(*encoded) == AudioFormat::Midi);

    auto info = peek_midi(*encoded);
    TEST_CHECK(info.is_ok());
    TEST_CHECK(info->format == 0 && info->num_tracks == 1 && info->ticks_per_quarter_note == 480);
    TEST_CHECK(near_f64(info->initial_tempo_bpm, 120.0, 1e-6));
    TEST_CHECK(info->total_notes == 3 && near_f64(info->duration_sec, 1.5, 1e-4));

    auto decoded = decode_midi(*encoded);
    TEST_CHECK(decoded.is_ok() && decoded->size() == 3);
    for (usize i = 0; i < 3; ++i) {
        const MidiNote& got = (*decoded)[i];
        TEST_CHECK(got.channel == notes_in[i].channel);
        TEST_CHECK(got.note_number == notes_in[i].note_number);
        TEST_CHECK(got.velocity == notes_in[i].velocity);
        TEST_CHECK(near_f64(got.start_time_sec, 0.5 * keel::as<f64>(i), 1e-4));
        TEST_CHECK(near_f64(got.end_time_sec, 0.5 * keel::as<f64>(i + 1), 1e-4));
        TEST_CHECK(near_f64(got.duration_sec, 0.5, 1e-4));
    }

    // (B) SMF Format 1 with Track 0 conductor (Track Name + Tempo change at tick 480: 120 BPM -> 240 BPM)
    // and Track 1 melody using Running Status (0x90) + Note-On vel=0 Note-Off!
    // Delta 480 in VLQ: 480 = (3 << 7) | 96 -> 0x83, 0x60
    const u8 smf1[] = {
        // MThd: len=6, format=1, num_tracks=2, tpqn=480 (0x01E0)
        'M', 'T', 'h', 'd', 0, 0, 0, 6, 0, 1, 0, 2, 0x01, 0xE0,
        // Track 0 (conductor): Track Name "Lead", Tempo 500000 us (120 BPM) at tick 0,
        // Tempo 250000 us (240 BPM = 0x03D090) at tick 480, End of Track
        'M', 'T', 'r', 'k', 0, 0, 0, 27,
        0x00, 0xFF, 0x03, 0x04, 'L', 'e', 'a', 'd',
        0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20,
        0x83, 0x60, 0xFF, 0x51, 0x03, 0x03, 0xD0, 0x90,
        0x00, 0xFF, 0x2F, 0x00,
        // Track 1 (notes on ch 3 = status 0x93, using Running Status for all subsequent events!):
        // tick 0:    0x93, 60, 100 (Note 60 ON)
        // +480:      60, 0         (Running status 0x93 vel=0 => Note 60 OFF at 0.50s)
        // +0:        72, 110       (Running status 0x93 => Note 72 ON at 0.50s)
        // +480:      72, 0         (Running status 0x93 vel=0 => Note 72 OFF at 0.50 + 0.25 = 0.75s!)
        // +0:        EOT
        'M', 'T', 'r', 'k', 0, 0, 0, 19,
        0x00, 0x93, 60, 100,
        0x83, 0x60, 60, 0,
        0x00, 72, 110,
        0x83, 0x60, 72, 0,
        0x00, 0xFF, 0x2F, 0x00,
    };

    auto info1 = peek_midi(smf1);
    TEST_CHECK(info1.is_ok());
    TEST_CHECK(info1->format == 1 && info1->num_tracks == 2);
    TEST_CHECK(info1->track_name == "Lead");
    TEST_CHECK(near_f64(info1->initial_tempo_bpm, 120.0, 1e-6));
    TEST_CHECK(info1->total_notes == 2);
    TEST_CHECK(near_f64(info1->duration_sec, 0.75, 1e-6));

    auto notes1 = decode_midi(smf1);
    TEST_CHECK(notes1.is_ok() && notes1->size() == 2);
    TEST_CHECK((*notes1)[0].track == 1 && (*notes1)[0].channel == 3 && (*notes1)[0].note_number == 60);
    TEST_CHECK(near_f64((*notes1)[0].start_time_sec, 0.0, 1e-6));
    TEST_CHECK(near_f64((*notes1)[0].end_time_sec, 0.5, 1e-6));
    TEST_CHECK((*notes1)[1].track == 1 && (*notes1)[1].channel == 3 && (*notes1)[1].note_number == 72);
    TEST_CHECK(near_f64((*notes1)[1].start_time_sec, 0.5, 1e-6));
    TEST_CHECK(near_f64((*notes1)[1].end_time_sec, 0.75, 1e-6));
    TEST_CHECK(near_f64((*notes1)[1].duration_sec, 0.25, 1e-6));
}

// ============================================================================
// 8. File I/O, Robustness, Safety Caps, Multi-Channel & Custom Arena Tests
// ============================================================================
void test_file_io_and_robustness() {
    // File I/O round-trip (.wav, .aiff, .au, .mid)
    const f64 pcm_vals[] = {0.25, -0.25, 0.5, -0.5, 0.75, -0.75};
    AudioView v = AudioView::from_f64(pcm_vals, 3, 2, 44100.0);

    auto tmp_dir_res = fs_temp_dir();
    TEST_CHECK(tmp_dir_res.is_ok());
    const StringRef tmp_dir = tmp_dir_res.is_ok() ? StringRef(*tmp_dir_res) : StringRef(".");
    const String wav_path = path_join(tmp_dir, "keel_audiofile_test_io.wav");
    const String aiff_path = path_join(tmp_dir, "keel_audiofile_test_io.aiff");
    const String au_path = path_join(tmp_dir, "keel_audiofile_test_io.au");
    const String mid_path = path_join(tmp_dir, "keel_audiofile_test_io.mid");

    TEST_CHECK(write_audio(wav_path, v, AudioEncodeOptions{.bits_per_sample = 24, .title = "FileWav"}).is_ok());
    TEST_CHECK(write_audio(aiff_path, v, AudioEncodeOptions{.bits_per_sample = 24, .title = "FileAiff"}).is_ok());
    TEST_CHECK(write_audio(au_path, v, AudioEncodeOptions{.bits_per_sample = 24, .comment = "FileAu"}).is_ok());

    auto wf_info = peek_audio_file(wav_path);
    auto wf_buf = read_audio(wav_path);
    TEST_CHECK(wf_info.is_ok() && wf_info->format == AudioFormat::Wav && wf_info->metadata.title == "FileWav");
    TEST_CHECK(wf_buf.is_ok() && wf_buf->frames == 3 && wf_buf->channels == 2);

    auto af_buf = read_audio(aiff_path);
    auto auf_buf = read_audio(au_path);
    TEST_CHECK(af_buf.is_ok() && af_buf->frames == 3 && af_buf->metadata.title == "FileAiff");
    TEST_CHECK(auf_buf.is_ok() && auf_buf->frames == 3 && auf_buf->metadata.comment == "FileAu");

    const MidiNote mnote{.note_number = 69, .velocity = 100, .start_time_sec = 0.0, .end_time_sec = 0.25};
    TEST_CHECK(write_midi(mid_path, ArrayRef<MidiNote>(&mnote, 1)).is_ok());
    auto mf_info = peek_midi_file(mid_path);
    auto mf_notes = read_midi(mid_path);
    TEST_CHECK(mf_info.is_ok() && mf_info->total_notes == 1);
    TEST_CHECK(mf_notes.is_ok() && mf_notes->size() == 1 && (*mf_notes)[0].note_number == 69);

    (void)fs_remove(wav_path);
    (void)fs_remove(aiff_path);
    (void)fs_remove(au_path);
    (void)fs_remove(mid_path);

    // Empty & tiny buffer errors
    ArrayRef<u8> empty{};
    const u8 tiny[] = {'R', 'I', 'F'};
    TEST_CHECK(peek_audio(empty).error().code == ErrorCode::EndOfFile);
    TEST_CHECK(decode_audio(tiny).error().code == ErrorCode::EndOfFile);
    TEST_CHECK(peek_wav(tiny).error().code == ErrorCode::EndOfFile);
    TEST_CHECK(decode_wav(tiny).error().code == ErrorCode::EndOfFile);
    TEST_CHECK(peek_flac(tiny).error().code == ErrorCode::EndOfFile);
    TEST_CHECK(decode_flac(tiny).error().code == ErrorCode::EndOfFile);
    TEST_CHECK(peek_mp3(tiny).error().code == ErrorCode::EndOfFile);
    TEST_CHECK(decode_mp3(tiny).error().code == ErrorCode::EndOfFile);
    TEST_CHECK(peek_aiff(tiny).error().code == ErrorCode::EndOfFile);
    TEST_CHECK(decode_aiff(tiny).error().code == ErrorCode::EndOfFile);
    TEST_CHECK(peek_au(tiny).error().code == ErrorCode::EndOfFile);
    TEST_CHECK(decode_au(tiny).error().code == ErrorCode::EndOfFile);
    TEST_CHECK(peek_midi(tiny).error().code == ErrorCode::EndOfFile);
    TEST_CHECK(decode_midi(tiny).error().code == ErrorCode::EndOfFile);

    // Unknown magic -> NotSupported on generic dispatcher, InvalidData on format-specific decoders
    const u8 unknown_magic[] = {'N', 'O', 'P', 'E', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    TEST_CHECK(peek_audio(unknown_magic).error().code == ErrorCode::NotSupported);
    TEST_CHECK(decode_audio(unknown_magic).error().code == ErrorCode::NotSupported);
    TEST_CHECK(decode_wav(unknown_magic).error().code == ErrorCode::InvalidData);
    TEST_CHECK(decode_flac(unknown_magic).error().code == ErrorCode::InvalidData);
    TEST_CHECK(decode_mp3(unknown_magic).error().code == ErrorCode::InvalidData);
    TEST_CHECK(decode_aiff(unknown_magic).error().code == ErrorCode::InvalidData);
    TEST_CHECK(decode_au(unknown_magic).error().code == ErrorCode::InvalidData);
    TEST_CHECK(decode_midi(unknown_magic).error().code == ErrorCode::InvalidData);

    // Truncated payloads & max_frames cap (OutOfRange)
    Array<f64> sig100(100, 0.25);
    AudioView v100 = AudioView::from_f64(sig100, 100, 1, 44100.0);

    auto valid_wav = encode_wav(v100);
    auto valid_aiff = encode_aiff(v100);
    auto valid_au = encode_au(v100);
    auto valid_mid = encode_midi(ArrayRef<MidiNote>(&mnote, 1));

    TEST_CHECK(decode_wav(valid_wav->slice(0, valid_wav->size() - 50)).error().code == ErrorCode::EndOfFile);
    TEST_CHECK(decode_aiff(valid_aiff->slice(0, valid_aiff->size() - 50)).error().code == ErrorCode::EndOfFile);
    TEST_CHECK(decode_au(valid_au->slice(0, valid_au->size() - 50)).error().code == ErrorCode::EndOfFile);
    TEST_CHECK(decode_midi(valid_mid->slice(0, valid_mid->size() - 5)).error().code == ErrorCode::EndOfFile);

    TEST_CHECK(decode_wav(*valid_wav, AudioDecodeOptions{.max_frames = 50}).error().code == ErrorCode::OutOfRange);
    TEST_CHECK(decode_aiff(*valid_aiff, AudioDecodeOptions{.max_frames = 50}).error().code == ErrorCode::OutOfRange);
    TEST_CHECK(decode_au(*valid_au, AudioDecodeOptions{.max_frames = 50}).error().code == ErrorCode::OutOfRange);

    // Invalid encode options & unsupported encoders
    TEST_CHECK(encode_audio(v100, AudioFormat::Flac).error().code == ErrorCode::NotSupported);
    TEST_CHECK(encode_audio(v100, AudioFormat::Mp3).error().code == ErrorCode::NotSupported);
    TEST_CHECK(encode_wav(v100, AudioEncodeOptions{.bits_per_sample = 12}).error().code == ErrorCode::InvalidArgument);
    TEST_CHECK(encode_wav(v100, AudioEncodeOptions{.bits_per_sample = 16, .float_samples = true}).error().code ==
               ErrorCode::InvalidArgument);
    TEST_CHECK(encode_wav(AudioView{.sample_rate = 44100.0, .channels = 0, .frames = 10}).error().code ==
               ErrorCode::InvalidArgument);

    // 6-channel 5.1 surround @ 192 kHz round-trip
    Array<f64> surround(100 * 6);
    for (usize ch = 0; ch < 6; ++ch) {
        for (usize t = 0; t < 100; ++t) {
            surround[ch * 100 + t] = 0.1 * keel::as<f64>(ch + 1);
        }
    }
    AudioView sur_view = AudioView::from_f64(surround, 100, 6, 192000.0, AudioChannelLayout::Planar);
    auto sur_wav = encode_wav(sur_view, AudioEncodeOptions{.bits_per_sample = 24});
    auto sur_dec = decode_wav(*sur_wav, AudioDecodeOptions{.layout = AudioChannelLayout::Planar});
    TEST_CHECK(sur_dec.is_ok() && sur_dec->channels == 6 && sur_dec->sample_rate == 192000.0);
    for (usize i = 0; i < surround.size(); ++i) {
        TEST_CHECK(near_f64(sur_dec->samples_f64()[i], surround[i], 1e-5));
    }

    // Custom keel::Arena allocator routing + intentionally misaligned Arena cursor check!
    Arena custom_arena(2 * 1024 * 1024);
    (void)custom_arena.alloc(3, 1); // Leave arena bump offset at odd alignment (offset == 3)
    TEST_CHECK((custom_arena.used() & 7u) != 0u);
    const usize used_before = custom_arena.used();

    auto arena_wav = encode_wav(v100, AudioEncodeOptions{.title = "ArenaMeta", .allocator = custom_arena.allocator()});
    TEST_CHECK(arena_wav.is_ok() && arena_wav->allocator() == custom_arena.allocator());

    (void)custom_arena.alloc(1, 1); // Force odd offset again right before decoding F64 samples
    auto arena_dec = decode_wav(*arena_wav, AudioDecodeOptions{.allocator = custom_arena.allocator()});
    TEST_CHECK(arena_dec.is_ok());
    TEST_CHECK(arena_dec->data.allocator() == custom_arena.allocator());
    TEST_CHECK((reinterpret_cast<usize>(arena_dec->samples_f64().data()) & (alignof(f64) - 1u)) == 0u);
    TEST_CHECK(arena_dec->metadata.title == "ArenaMeta");
    TEST_CHECK(custom_arena.used() > used_before);

    Arena clone_arena(512 * 1024);
    (void)clone_arena.alloc(5, 1);
    AudioBuffer cloned = arena_dec->clone(clone_arena.allocator());
    TEST_CHECK((reinterpret_cast<usize>(cloned.samples_f64().data()) & (alignof(f64) - 1u)) == 0u);
    TEST_CHECK(ArrayRef<f64>(cloned.samples_f64()) == ArrayRef<f64>(arena_dec->samples_f64()));
    TEST_CHECK(cloned.metadata.title == "ArenaMeta");

    // Hardening regressions:
    // 1. AIFF non-finite extended80 sample rate (exp = 0x7FFE -> +Inf) must be rejected.
    {
        const u8 inf_aiff[] = {
            'F', 'O', 'R', 'M', 0, 0, 0, 30, 'A', 'I', 'F', 'F',
            'C', 'O', 'M', 'M', 0, 0, 0, 18, 0, 1, 0, 0, 0, 0, 0, 16,
            0x7F, 0xFE, 0x80, 0, 0, 0, 0, 0, 0, 0,
        };
        TEST_CHECK(peek_aiff(inf_aiff).error().code == ErrorCode::InvalidData);
        TEST_CHECK(decode_aiff(inf_aiff).error().code == ErrorCode::InvalidData);
    }

    // 2. MIDI huge timestamp must fail with OutOfRange instead of UB cast.
    {
        const MidiNote huge{.note_number = 60, .velocity = 80, .start_time_sec = 1e18, .end_time_sec = 1e18 + 1.0};
        TEST_CHECK(encode_midi(ArrayRef<MidiNote>(&huge, 1)).error().code == ErrorCode::OutOfRange);
    }
}

// ============================================================================
// CLI Mode Helpers for Reference & Adversarial Differential Testing
// ============================================================================

const char* sample_type_str(AudioSampleType st) noexcept {
    switch (st) {
        case AudioSampleType::U8: return "u8";
        case AudioSampleType::I8: return "i8";
        case AudioSampleType::I16: return "i16";
        case AudioSampleType::I32: return "i32";
        case AudioSampleType::F32: return "f32";
        case AudioSampleType::F64: return "f64";
    }
    return "unknown";
}

bool parse_u64_arg(StringRef s, u64& out) noexcept {
    if (s.empty()) return false;
    u64 v = 0;
    for (usize i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (c < '0' || c > '9') return false;
        v = v * 10ULL + keel::as<u64>(c - '0');
    }
    out = v;
    return true;
}

bool parse_stype_arg(StringRef s, AudioSampleType& out) noexcept {
    if (s == "u8") { out = AudioSampleType::U8; return true; }
    if (s == "i8") { out = AudioSampleType::I8; return true; }
    if (s == "i16") { out = AudioSampleType::I16; return true; }
    if (s == "i32") { out = AudioSampleType::I32; return true; }
    if (s == "f32") { out = AudioSampleType::F32; return true; }
    if (s == "f64") { out = AudioSampleType::F64; return true; }
    return false;
}

i32 cli_decode(StringRef in_path, StringRef out_raw_path, bool native, u64 start_frame, u64 end_frame) {
    auto bytes_res = fs_read_file(in_path);
    if (!bytes_res) {
        keel::eprintln("read error");
        return 2;
    }
    const AudioFormat fmt = sniff_audio_format(*bytes_res);
    AudioDecodeOptions opts;
    opts.native_sample_type = native;
    opts.start_frame = start_frame;
    opts.end_frame = end_frame;
    opts.max_frames = 10000000ULL;

    auto dec = decode_audio(*bytes_res, opts);
    if (!dec) {
        keel::eprintln("decode error: {}", keel::as<i32>(dec.error().code));
        return 3;
    }
    if (fs_write_file(out_raw_path, dec->data).is_err()) {
        keel::eprintln("write error");
        return 4;
    }
    keel::println(
        "{{\"format\":\"{}\",\"channels\":{},\"sample_rate\":{},\"frames\":{},\"total_source_frames\":{},\"bits_per_sample\":{},\"sample_type\":\"{}\",\"title\":\"{}\",\"artist\":\"{}\",\"comment\":\"{}\"}}",
        audio_format_name(fmt), dec->channels, dec->sample_rate, dec->frames, dec->total_source_frames,
        dec->bits_per_sample, sample_type_str(dec->sample_type), dec->metadata.title, dec->metadata.artist,
        dec->metadata.comment);
    return 0;
}

i32 cli_encode(StringRef fmt_str, u64 sr, u64 ch, u64 frames, AudioSampleType st, u64 bps, bool flt,
               StringRef in_raw, StringRef out_path, StringRef title, StringRef artist, StringRef comment) {
    auto raw_res = fs_read_file(in_raw);
    if (!raw_res) {
        keel::eprintln("read raw error");
        return 2;
    }
    AudioFormat fmt = AudioFormat::Unknown;
    if (fmt_str == "wav") fmt = AudioFormat::Wav;
    else if (fmt_str == "aiff") fmt = AudioFormat::Aiff;
    else if (fmt_str == "au") fmt = AudioFormat::Au;
    else return 2;

    AudioView view{
        .sample_rate = keel::as<f64>(sr),
        .channels = keel::as<u16>(ch),
        .frames = frames,
        .sample_type = st,
        .layout = AudioChannelLayout::Interleaved,
        .data = *raw_res,
    };
    AudioEncodeOptions opts;
    opts.bits_per_sample = keel::as<u16>(bps);
    opts.float_samples = flt;
    opts.title = title;
    opts.artist = artist;
    opts.comment = comment;

    auto enc = encode_audio(view, fmt, opts);
    if (!enc) {
        keel::eprintln("encode error: {}", keel::as<i32>(enc.error().code));
        return 3;
    }
    if (fs_write_file(out_path, *enc).is_err()) {
        keel::eprintln("write error");
        return 4;
    }
    return 0;
}

i32 cli_decode_midi(StringRef in_path) {
    auto bytes_res = fs_read_file(in_path);
    if (!bytes_res) return 2;
    auto info = peek_midi(*bytes_res);
    auto notes = decode_midi(*bytes_res);
    if (!info || !notes) return 3;
    keel::print(
        "{{\"format\":{},\"num_tracks\":{},\"tpqn\":{},\"initial_tempo_bpm\":{},\"duration_sec\":{},\"total_notes\":{},\"track_name\":\"{}\",\"notes\":[",
        info->format, info->num_tracks, info->ticks_per_quarter_note, info->initial_tempo_bpm, info->duration_sec,
        info->total_notes, info->track_name);
    for (usize i = 0; i < notes->size(); ++i) {
        const MidiNote& n = (*notes)[i];
        if (i > 0) keel::print(",");
        keel::print(
            "{{\"track\":{},\"channel\":{},\"note\":{},\"vel\":{},\"start\":{},\"end\":{},\"dur\":{}}}",
            n.track, n.channel, n.note_number, n.velocity, n.start_time_sec, n.end_time_sec, n.duration_sec);
    }
    keel::println("]}");
    return 0;
}

void test_spec_completeness() {
    // 1. MIDI SMPTE timecode division (25fps x 40 ticks/frame = 1000 ticks/sec, and 29.97fps x 100) + EOT dangling Note-On
    {
        // 25 fps => high byte = -25 (0xE7), sub_ticks = 40 (0x28) => 1000 ticks/sec
        const u8 smpte25_mid[] = {
            'M',  'T',  'h',  'd',  0,    0,    0,    6,    0,    0,    0,    1,    0xE7u, 0x28u,
            'M',  'T',  'r',  'k',  0,    0,    0,    18,
            0x00u, 0x90u, 60u, 100u,             // tick 0 (0.0s): Note 60 ON
            0x83u, 0x74u, 0x80u, 60u, 0u,        // +500 ticks (VLQ 500 = 0x83 0x74) -> 0.5s: Note 60 OFF
            0x00u, 0x90u, 64u, 90u,              // tick 500 (0.5s): Note 64 ON (intentionally unclosed!)
            0x87u, 0x68u, 0xFFu, 0x2Fu, 0x00u,   // +1000 ticks (VLQ 1000 = 0x87 0x68) -> 1.5s: End of Track!
        };
        auto info = peek_midi(smpte25_mid);
        TEST_CHECK(info.is_ok() && near_f64(info->duration_sec, 1.5, 1e-9) && info->total_notes == 2u);

        auto notes = decode_midi(smpte25_mid);
        TEST_CHECK(notes.is_ok() && notes->size() == 2u);
        if (notes && notes->size() == 2u) {
            TEST_CHECK((*notes)[0].note_number == 60u && near_f64((*notes)[0].start_time_sec, 0.0, 1e-9) &&
                       near_f64((*notes)[0].end_time_sec, 0.5, 1e-9));
            TEST_CHECK((*notes)[1].note_number == 64u && near_f64((*notes)[1].start_time_sec, 0.5, 1e-9) &&
                       near_f64((*notes)[1].end_time_sec, 1.5, 1e-9) && near_f64((*notes)[1].duration_sec, 1.0, 1e-9));
        }

        // 29.97 fps => high byte = -29 (0xE3), sub_ticks = 100 (0x64) => 2997 ticks/sec
        const u8 smpte2997_mid[] = {
            'M',  'T',  'h',  'd',  0,    0,    0,    6,    0,    0,    0,    1,    0xE3u, 0x64u,
            'M',  'T',  'r',  'k',  0,    0,    0,    9,
            0x00u, 0x90u, 69u, 80u,
            0x97u, 0x35u, 0xFFu, 0x2Fu, 0x00u, // VLQ(2997) = 0x97 0x35 => exactly 1.0 second!
        };
        auto info29 = peek_midi(smpte2997_mid);
        TEST_CHECK(info29.is_ok() && near_f64(info29->duration_sec, 1.0, 1e-9));

        // Invalid SMPTE fps (-20 = 0xEC) must be rejected
        const u8 bad_smpte[] = {
            'M', 'T', 'h', 'd', 0, 0, 0, 6, 0, 0, 0, 1, 0xECu, 0x28u,
            'M', 'T', 'r', 'k', 0, 0, 0, 4, 0, 0xFFu, 0x2Fu, 0x00u,
        };
        TEST_CHECK(decode_midi(bad_smpte).is_err());
    }

    // 2. MP3 free-format bitrate (bitrate_idx == 0) frame detection & decoding
    {
        // Construct 3 consecutive MPEG-1 Layer III free-format frames (bitrate_idx=0, 44100 Hz, mono, frame_size=200 bytes)
        const usize fsz = 200u;
        Array<u8> free_mp3(fsz * 3u, u8{0});
        for (usize f = 0; f < 3u; ++f) {
            u8* fp = free_mp3.data() + f * fsz;
            fp[0] = 0xFFu;
            fp[1] = 0xFBu; // MPEG-1, Layer III, no CRC
            fp[2] = 0x00u; // bitrate_idx = 0 (free-format!), sr_idx = 0 (44100 Hz), padding = 0
            fp[3] = 0xC0u; // Mono
        }
        TEST_CHECK(is_mp3_bytes(free_mp3));
        auto info = peek_mp3(free_mp3);
        TEST_CHECK(info.is_ok() && info->channels == 1u && info->sample_rate == 44100.0);
        auto dec = decode_mp3(free_mp3);
        TEST_CHECK(dec.is_ok() && dec->frames == 3u * 1152u && dec->channels == 1u);
    }

    // 3. MPEG-1 Layer I frame sniffing & decoding (384 samples/frame)
    {
        // MPEG-1 Layer I (layer=3 -> 0xFF 0xFF), 128 kbps (bitrate_idx=4), 48000 Hz (sr_idx=1), mono:
        // frame_size = ((384 * 128 * 125) / 48000) & ~3 = 128 bytes
        const usize fsz = 128u;
        Array<u8> mp1(fsz * 2u, u8{0});
        for (usize f = 0; f < 2u; ++f) {
            u8* fp = mp1.data() + f * fsz;
            fp[0] = 0xFFu;
            fp[1] = 0xFFu; // MPEG-1, Layer I, no CRC
            fp[2] = 0x44u; // bitrate_idx = 4 (128 kbps), sr_idx = 1 (48000 Hz), pad = 0
            fp[3] = 0xC0u; // Mono
        }
        TEST_CHECK(is_mp3_bytes(mp1));
        auto info = peek_mp3(mp1);
        TEST_CHECK(info.is_ok() && info->compression_method == "MPEG Layer I" && info->sample_rate == 48000.0);
        auto dec = decode_mp3(mp1);
        TEST_CHECK(dec.is_ok() && dec->frames == 2u * 384u && dec->channels == 1u);
    }
}

i32 cli_encode_midi(StringRef out_path) {
    const MidiNote notes[] = {
        MidiNote{.track = 0, .channel = 0, .note_number = 60, .velocity = 100, .start_time_sec = 0.0, .end_time_sec = 0.5},
        MidiNote{.track = 0, .channel = 0, .note_number = 64, .velocity = 90, .start_time_sec = 0.25, .end_time_sec = 0.75},
        MidiNote{.track = 0, .channel = 2, .note_number = 67, .velocity = 110, .start_time_sec = 0.5, .end_time_sec = 1.25},
        MidiNote{.track = 0, .channel = 9, .note_number = 36, .velocity = 127, .start_time_sec = 1.0, .end_time_sec = 1.5},
    };
    auto enc = encode_midi(notes, MidiEncodeOptions{.ticks_per_quarter_note = 480, .tempo_bpm = 120.0});
    if (!enc) return 3;
    if (fs_write_file(out_path, *enc).is_err()) return 4;
    return 0;
}

i32 cli_fuzz_dir(StringRef dir_path) {
    usize total = 0;
    usize ok_count = 0;
    usize err_count = 0;
    AudioDecodeOptions opts;
    opts.max_frames = 250000ULL;

    Error err = fs_read_dir(dir_path, [&](const DirEntry& ent) -> bool {
        if (!ent.is_file) return true;
        auto bytes_res = fs_read_file(ent.full_path);
        if (!bytes_res) return true;
        ++total;
        const AudioFormat fmt = sniff_audio_format(*bytes_res);
        if (fmt == AudioFormat::Midi) {
            (void)peek_midi(*bytes_res);
            auto m = decode_midi(*bytes_res);
            if (m.is_ok()) ++ok_count;
            else ++err_count;
        } else {
            (void)peek_audio(*bytes_res);
            auto d = decode_audio(*bytes_res, opts);
            if (d.is_ok()) ++ok_count;
            else ++err_count;
        }
        return true;
    });
    if (err.is_err()) {
        keel::eprintln("read_dir error");
        return 2;
    }
    keel::println("Fuzzed {} audio/midi files (ok={}, rejected={}, crashes=0)", total, ok_count, err_count);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 4 && StringRef(argv[1]) == "--decode") {
        bool native = (argc >= 5 && StringRef(argv[4]) == "1");
        u64 sf = 0;
        u64 ef = Limits<u64>::kMax;
        if (argc >= 6) (void)parse_u64_arg(StringRef(argv[5]), sf);
        if (argc >= 7 && StringRef(argv[6]) != "end") (void)parse_u64_arg(StringRef(argv[6]), ef);
        return cli_decode(StringRef(argv[2]), StringRef(argv[3]), native, sf, ef);
    }
    if (argc >= 11 && StringRef(argv[1]) == "--encode") {
        u64 sr = 0, ch = 0, frames = 0, bps = 16;
        AudioSampleType st = AudioSampleType::F64;
        if (!parse_u64_arg(StringRef(argv[3]), sr) || !parse_u64_arg(StringRef(argv[4]), ch) ||
            !parse_u64_arg(StringRef(argv[5]), frames) || !parse_stype_arg(StringRef(argv[6]), st) ||
            !parse_u64_arg(StringRef(argv[7]), bps)) {
            return 2;
        }
        const bool flt = (StringRef(argv[8]) == "1");
        StringRef title = (argc >= 12) ? StringRef(argv[11]) : StringRef{};
        StringRef artist = (argc >= 13) ? StringRef(argv[12]) : StringRef{};
        StringRef comment = (argc >= 14) ? StringRef(argv[13]) : StringRef{};
        return cli_encode(StringRef(argv[2]), sr, ch, frames, st, bps, flt, StringRef(argv[9]), StringRef(argv[10]),
                          title, artist, comment);
    }
    if (argc == 3 && StringRef(argv[1]) == "--decode-midi") {
        return cli_decode_midi(StringRef(argv[2]));
    }
    if (argc == 3 && StringRef(argv[1]) == "--encode-midi") {
        return cli_encode_midi(StringRef(argv[2]));
    }
    if (argc == 3 && StringRef(argv[1]) == "--fuzz") {
        return cli_fuzz_dir(StringRef(argv[2]));
    }

    test_sniffing_and_helpers();
    test_wav_codec();
    test_flac_codec();
    test_mp3_codec();
    test_aiff_codec();
    test_au_codec();
    test_midi_codec();
    test_file_io_and_robustness();
    test_spec_completeness();

    if (g_tests_failed != 0) {
        keel::eprintln("{} test check(s) FAILED ({} passed)", g_tests_failed, g_tests_passed);
        return 1;
    }
    keel::println("All {} keel::audiofile test checks passed.", g_tests_passed);
    return 0;
}
