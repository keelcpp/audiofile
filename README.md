# keelcpp/audiofile

Lightweight C++20 audio & MIDI container codecs built on [`keelcpp/keel`](https://github.com/keelcpp/keel) under the `0BSD` license.

## Features

- **Audio Formats:** RIFF/WAVE (`WAV`), `FLAC`, MPEG-1/2/2.5 Layer III (`MP3`), `AIFF` / `AIFF-C`, and Sun/NeXT `AU` / `SND`.
- **MIDI:** Standard MIDI File (`SMF` Format 0 & 1) peeking, note decoding with tempo-map & running-status support, and Format 0 encoding.
- **Flexible Buffers:** Interleaved and planar channel layouts, normalized `f64` or native `u8` / `i8` / `i16` / `i32` / `f32` / `f64` sample types, frame-range slicing, and embedded metadata (`title`, `artist`, `comment`).
- **Zero STL header bloat & exception-neutral:** compiles cleanly with `-fno-exceptions -fno-rtti`, uses explicit `keel::Allocator*` and thread-local `keel::Arena` scratch memory.

## CMake Integration

Links as `keel::audiofile` (`find_package(keel_audiofile)` or `FetchContent`).
