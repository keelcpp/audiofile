#pragma once

#include "keel/array.h"
#include "keel/result.h"
#include "keel/string.h"

namespace keel::audiofile {

struct MidiNote {
    u16 track = 0;
    u8 channel = 0;
    u8 note_number = 60;
    u8 velocity = 64;
    f64 start_time_sec = 0.0;
    f64 end_time_sec = 0.0;
    f64 duration_sec = 0.0;
};

struct MidiInfo {
    u16 format = 0;
    u16 num_tracks = 0;
    u16 ticks_per_quarter_note = 480;
    f64 initial_tempo_bpm = 120.0;
    f64 duration_sec = 0.0;
    u64 total_notes = 0;
    String track_name;

    [[nodiscard]] MidiInfo clone(Allocator* alloc = nullptr) const;
};

struct MidiEncodeOptions {
    u16 ticks_per_quarter_note = 480;
    f64 tempo_bpm = 120.0;
    Allocator* allocator = nullptr;
};

[[nodiscard]] bool is_midi_bytes(ArrayRef<u8> bytes) noexcept;
[[nodiscard]] Result<MidiInfo> peek_midi(ArrayRef<u8> bytes, Allocator* alloc = nullptr);
[[nodiscard]] Result<Array<MidiNote>> decode_midi(ArrayRef<u8> bytes, Allocator* alloc = nullptr);
[[nodiscard]] Result<Array<u8>> encode_midi(ArrayRef<MidiNote> notes, const MidiEncodeOptions& opts = {});

[[nodiscard]] Result<MidiInfo> peek_midi_file(StringRef path, Allocator* alloc = nullptr);
[[nodiscard]] Result<Array<MidiNote>> read_midi(StringRef path, Allocator* alloc = nullptr);
Error write_midi(StringRef path, ArrayRef<MidiNote> notes, const MidiEncodeOptions& opts = {});

} // namespace keel::audiofile
