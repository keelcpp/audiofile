# keelcpp/audiofile

Lightweight C++20 audio & MIDI container codecs built on [`keelcpp/keel`](https://github.com/keelcpp/keel) under the `0BSD` license.

## Features

- **Audio Formats:** RIFF/WAVE (`WAV`: 8/16/24/32-bit PCM, 32/64-bit IEEE float, Mu-law, A-law, `WAVE_FORMAT_EXTENSIBLE`), `FLAC` (lossless Constant/Verbatim/Fixed/Adaptive LPC, Mid-Side/Left-Side/Right-Side stereo, 1–8 channels, RFC 9639 CRC-8/CRC-16 verification, Vorbis comments), MPEG-1/2/2.5 Layer I/II/III (`MP3`/`MP2`/`MP1`: full ISO/IEC 11172-3 & 13818-3 decoder with bit reservoir, all 32 Huffman tables, MS & intensity stereo, long/short/mixed IMDCT, 32-band polyphase synthesis, free-format bitrates, ID3v1/v2, Xing/Info/VBRI & LAME/Lavc gapless trimming), `AIFF` / `AIFF-C`, and Sun/NeXT `AU` / `SND`.
- **MIDI:** Standard MIDI File (`SMF` Format 0 & 1) peeking, note decoding with tempo-map, SMPTE timecode division, running-status, and End-of-Track note closing, plus Format 0 encoding.
- **Flexible Buffers:** Interleaved and planar channel layouts, normalized `f64` or native `u8` / `i8` / `i16` / `i32` / `f32` / `f64` sample types, frame-range slicing, and embedded metadata (`title`, `artist`, `comment`).
- **Zero STL header bloat & exception-neutral:** compiles cleanly with `-fno-exceptions -fno-rtti`, uses explicit `keel::Allocator*` and thread-local `keel::Arena` scratch memory.

## Reference Verification

Run `tests/verify_reference.py <path-to-test_audiofile>` to execute bidirectional differential checks against FFmpeg (`libmp3lame`/FLAC/WAV/AIFF/AU), SciPy (`scipy.io.wavfile`), MATLAB (`audioread`/`audiowrite`), an independent Python SMF 1.0 parser, and an adversarial fuzzing corpus across all 6 formats.

## CMake Integration

Links as `keel::audiofile` (`find_package(keel_audiofile)` or `FetchContent`).
