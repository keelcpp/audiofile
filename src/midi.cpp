#include "audio_internal.h"

#include "keel/algo.h"
#include "keel/arena.h"
#include "keel/fs.h"

namespace keel::audiofile {

MidiInfo MidiInfo::clone(Allocator* alloc) const {
    MidiInfo copy;
    copy.format = format;
    copy.num_tracks = num_tracks;
    copy.ticks_per_quarter_note = ticks_per_quarter_note;
    copy.initial_tempo_bpm = initial_tempo_bpm;
    copy.duration_sec = duration_sec;
    copy.total_notes = total_notes;
    copy.track_name = track_name.clone(alloc);
    return copy;
}

bool is_midi_bytes(ArrayRef<u8> bytes) noexcept {
    if (bytes.size() < 8) return false;
    return mem_eq(bytes.data(), "MThd", 4);
}

namespace {
namespace midi_detail {

[[nodiscard]] ErrorCode read_vlq(const u8* data, usize end, usize& pos, u32& out_val) noexcept {
    u32 val = 0;
    for (i32 i = 0; i < 4; ++i) {
        if (pos >= end) [[unlikely]] {
            return ErrorCode::EndOfFile;
        }
        const u8 b = data[pos++];
        val = (val << 7) | as<u32>(b & 0x7Fu);
        if ((b & 0x80u) == 0u) {
            out_val = val;
            return ErrorCode::Ok;
        }
    }
    return ErrorCode::InvalidData;
}

template <typename ByteBuf> void write_vlq(ByteBuf& buf, u32 val) {
    u8 tmp[5];
    i32 count = 0;
    tmp[count++] = as<u8>(val & 0x7Fu);
    while ((val >>= 7) > 0u) {
        tmp[count++] = as<u8>((val & 0x7Fu) | 0x80u);
    }
    for (i32 i = count - 1; i >= 0; --i) {
        buf.push_back(tmp[i]);
    }
}

struct RawTempoEvent {
    u64 tick = 0;
    u32 order = 0;
    u32 us_per_quarter = 500000;
};

struct TempoSegment {
    u64 tick = 0;
    f64 time_sec = 0.0;
    u32 us_per_quarter = 500000;
};

struct ActiveNoteNode {
    u64 start_tick = 0;
    u8 velocity = 64;
    ActiveNoteNode* next = nullptr;
};

struct ActiveNoteQueue {
    ActiveNoteNode* head = nullptr;
    ActiveNoteNode* tail = nullptr;
};

struct PendingNote {
    u16 track = 0;
    u8 channel = 0;
    u8 note_number = 60;
    u8 velocity = 64;
    u64 start_tick = 0;
    u64 end_tick = 0;
};

[[nodiscard]] f64 tick_to_seconds(u64 tick, ArrayRef<TempoSegment> segments, u16 tpqn) noexcept {
    if (segments.empty() || tpqn == 0u) return 0.0;
    const usize upper = keel::upper_bound(segments, tick, [](u64 t, const TempoSegment& seg) { return t < seg.tick; });
    const usize idx = (upper > 0u) ? (upper - 1u) : 0u;
    const TempoSegment& seg = segments[idx];
    const u64 delta_ticks = tick - seg.tick;
    const f64 sec_per_tick = (as<f64>(seg.us_per_quarter) / 1000000.0) / as<f64>(tpqn);
    return seg.time_sec + as<f64>(delta_ticks) * sec_per_tick;
}

struct ParsedSmf {
    u16 format = 0;
    u16 num_tracks = 0;
    u16 tpqn = 480;
    f64 initial_tempo_bpm = 120.0;
    f64 duration_sec = 0.0;
    u64 note_on_count = 0;
    String track_name;
    Array<MidiNote> notes;
};

[[nodiscard]] Result<ParsedSmf> parse_smf(ArrayRef<u8> bytes, bool build_notes, Allocator* dst_alloc) {
    if (bytes.size() < 8) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }
    if (!is_midi_bytes(bytes)) [[unlikely]] {
        return ErrorCode::InvalidData;
    }
    if (bytes.size() < 14) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }

    const u8* d = bytes.data();
    const usize len = bytes.size();
    const u32 header_len = audio_detail::read_u32_be(d + 4);
    if (header_len < 6u) [[unlikely]] {
        return ErrorCode::InvalidData;
    }
    if (header_len > len - 8u) [[unlikely]] {
        return ErrorCode::EndOfFile;
    }

    const u16 format = audio_detail::read_u16_be(d + 8);
    const u16 num_tracks = audio_detail::read_u16_be(d + 10);
    const u16 raw_div = audio_detail::read_u16_be(d + 12);
    if (format > 2u) [[unlikely]] {
        return ErrorCode::InvalidData;
    }

    u16 tpqn = 480u;
    if ((raw_div & 0x8000u) == 0u) {
        if (raw_div == 0u) [[unlikely]] {
            return ErrorCode::InvalidData;
        }
        tpqn = raw_div;
    }

    ArenaTemp scratch = get_scratch(dst_alloc);
    Arena& arena = scratch.arena();

    ArenaArray<RawTempoEvent> raw_tempos(arena);
    raw_tempos.push_back(RawTempoEvent{0ULL, 0u, 500000u});
    u32 tempo_order = 1u;

    ArenaArray<PendingNote> pending_notes(arena);
    ActiveNoteQueue* active_slots = build_notes ? arena.alloc_array<ActiveNoteQueue>(16u * 128u) : nullptr;
    ActiveNoteNode* free_list = nullptr;

    usize pos = 8u + as<usize>(header_len);
    u64 max_tick = 0;
    u64 note_on_count = 0;
    String first_track_name(dst_alloc);

    for (u16 trk = 0; trk < num_tracks; ++trk) {
        if (pos + 8u > len) [[unlikely]] {
            return ErrorCode::EndOfFile;
        }
        if (!mem_eq(d + pos, "MTrk", 4)) [[unlikely]] {
            return ErrorCode::InvalidData;
        }
        const u32 trk_len = audio_detail::read_u32_be(d + pos + 4);
        pos += 8u;
        if (trk_len > len - pos) [[unlikely]] {
            return ErrorCode::EndOfFile;
        }
        const usize trk_end = pos + as<usize>(trk_len);

        if (build_notes) {
            mem_zero(active_slots, sizeof(ActiveNoteQueue) * 16u * 128u);
        }

        u64 current_tick = 0;
        u8 running_status = 0;

        while (pos < trk_end) {
            u32 dt = 0;
            const ErrorCode dt_ec = read_vlq(d, trk_end, pos, dt);
            if (dt_ec != ErrorCode::Ok) [[unlikely]] {
                return dt_ec;
            }
            current_tick += dt;
            if (pos >= trk_end) [[unlikely]] {
                return ErrorCode::EndOfFile;
            }

            u8 status = d[pos];
            if ((status & 0x80u) != 0u) {
                ++pos;
                if (status <= 0xEFu) {
                    running_status = status;
                } else if (status <= 0xF7u) {
                    running_status = 0;
                }
            } else {
                if (running_status == 0u) [[unlikely]] {
                    return ErrorCode::InvalidData;
                }
                status = running_status;
            }

            if (status == 0xFFu) {
                if (pos >= trk_end) [[unlikely]] {
                    return ErrorCode::EndOfFile;
                }
                const u8 meta_type = d[pos++];
                u32 meta_len = 0;
                const ErrorCode ml_ec = read_vlq(d, trk_end, pos, meta_len);
                if (ml_ec != ErrorCode::Ok) [[unlikely]] {
                    return ml_ec;
                }
                if (meta_len > trk_end - pos) [[unlikely]] {
                    return ErrorCode::EndOfFile;
                }
                if (meta_type == 0x51u && meta_len == 3u) {
                    const u32 us = (as<u32>(d[pos]) << 16) | (as<u32>(d[pos + 1]) << 8) |
                                   as<u32>(d[pos + 2]);
                    if (us > 0u) {
                        raw_tempos.push_back(RawTempoEvent{current_tick, tempo_order++, us});
                    }
                } else if (meta_type == 0x03u && meta_len > 0u && first_track_name.empty()) {
                    first_track_name = audio_detail::make_trimmed_string(d + pos, meta_len, dst_alloc);
                } else if (meta_type == 0x2Fu) {
                    pos = trk_end;
                    break;
                }
                pos += meta_len;
            } else if (status == 0xF0u || status == 0xF7u) {
                u32 sysex_len = 0;
                const ErrorCode sl_ec = read_vlq(d, trk_end, pos, sysex_len);
                if (sl_ec != ErrorCode::Ok) [[unlikely]] {
                    return sl_ec;
                }
                if (sysex_len > trk_end - pos) [[unlikely]] {
                    return ErrorCode::EndOfFile;
                }
                pos += sysex_len;
            } else if (status >= 0xF1u) {
                // System Common / Real-Time single/multi-byte messages inside stream
                if (status == 0xF2u) {
                    if (trk_end - pos < 2u) return ErrorCode::EndOfFile;
                    pos += 2u;
                } else if (status == 0xF1u || status == 0xF3u) {
                    if (trk_end - pos < 1u) return ErrorCode::EndOfFile;
                    pos += 1u;
                }
            } else {
                const u8 msg_type = as<u8>(status & 0xF0u);
                const u8 ch = as<u8>(status & 0x0Fu);
                if (pos >= trk_end) [[unlikely]] {
                    return ErrorCode::EndOfFile;
                }
                const u8 d1 = as<u8>(d[pos++] & 0x7Fu);
                u8 d2 = 0;
                if (msg_type != 0xC0u && msg_type != 0xD0u) {
                    if (pos >= trk_end) [[unlikely]] {
                        return ErrorCode::EndOfFile;
                    }
                    d2 = as<u8>(d[pos++] & 0x7Fu);
                }

                if (msg_type == 0x90u && d2 > 0u) {
                    ++note_on_count;
                    if (build_notes) {
                        ActiveNoteNode* node = nullptr;
                        if (free_list) {
                            node = free_list;
                            free_list = free_list->next;
                        } else {
                            node = arena.create<ActiveNoteNode>();
                        }
                        node->start_tick = current_tick;
                        node->velocity = d2;
                        node->next = nullptr;

                        ActiveNoteQueue& q = active_slots[(as<usize>(ch) << 7) | as<usize>(d1)];
                        if (q.tail) {
                            q.tail->next = node;
                            q.tail = node;
                        } else {
                            q.head = node;
                            q.tail = node;
                        }
                    }
                } else if (build_notes && (msg_type == 0x80u || (msg_type == 0x90u && d2 == 0u))) {
                    ActiveNoteQueue& q = active_slots[(as<usize>(ch) << 7) | as<usize>(d1)];
                    if (q.head) {
                        ActiveNoteNode* node = q.head;
                        q.head = node->next;
                        if (!q.head) {
                            q.tail = nullptr;
                        }
                        const u64 end_tick = (current_tick >= node->start_tick) ? current_tick : node->start_tick;
                        pending_notes.push_back(
                            PendingNote{trk, ch, d1, node->velocity, node->start_tick, end_tick});

                        node->next = free_list;
                        free_list = node;
                    }
                }
            }
        }

        if (current_tick > max_tick) {
            max_tick = current_tick;
        }
        pos = trk_end;
    }

    keel::sort(raw_tempos, [](const RawTempoEvent& a, const RawTempoEvent& b) {
        if (a.tick != b.tick) return a.tick < b.tick;
        return a.order < b.order;
    });

    ArenaArray<TempoSegment> segments(arena, raw_tempos.size());
    for (usize i = 0; i < raw_tempos.size(); ++i) {
        const RawTempoEvent& ev = raw_tempos[i];
        if (!segments.empty() && segments.back().tick == ev.tick) {
            segments.back().us_per_quarter = ev.us_per_quarter;
        } else if (segments.empty()) {
            segments.push_back(TempoSegment{ev.tick, 0.0, ev.us_per_quarter});
        } else {
            const TempoSegment& prev = segments.back();
            const u64 dt = ev.tick - prev.tick;
            const f64 sec_per_tick = (as<f64>(prev.us_per_quarter) / 1000000.0) / as<f64>(tpqn);
            segments.push_back(
                TempoSegment{ev.tick, prev.time_sec + as<f64>(dt) * sec_per_tick, ev.us_per_quarter});
        }
    }

    ParsedSmf out;
    out.format = format;
    out.num_tracks = num_tracks;
    out.tpqn = tpqn;
    out.initial_tempo_bpm = 60000000.0 / as<f64>(segments[0].us_per_quarter);
    out.duration_sec = tick_to_seconds(max_tick, segments.as_ref(), tpqn);
    out.note_on_count = note_on_count;
    out.track_name = keel::move(first_track_name);
    out.notes = Array<MidiNote>(dst_alloc);

    if (build_notes && !pending_notes.empty()) {
        out.notes.reserve(pending_notes.size());
        for (usize i = 0; i < pending_notes.size(); ++i) {
            const PendingNote& pn = pending_notes[i];
            MidiNote n;
            n.track = pn.track;
            n.channel = pn.channel;
            n.note_number = pn.note_number;
            n.velocity = pn.velocity;
            n.start_time_sec = tick_to_seconds(pn.start_tick, segments.as_ref(), tpqn);
            n.end_time_sec = tick_to_seconds(pn.end_tick, segments.as_ref(), tpqn);
            if (n.end_time_sec < n.start_time_sec) {
                n.end_time_sec = n.start_time_sec;
            }
            n.duration_sec = n.end_time_sec - n.start_time_sec;
            out.notes.push_back(n);
        }
    }

    return out;
}

} // namespace midi_detail
} // namespace

Result<MidiInfo> peek_midi(ArrayRef<u8> bytes, Allocator* alloc) {
    Allocator* dst_alloc = alloc ? alloc : default_allocator();
    KEEL_TRY_ASSIGN(midi_detail::ParsedSmf smf, midi_detail::parse_smf(bytes, false, dst_alloc));

    MidiInfo info;
    info.format = smf.format;
    info.num_tracks = smf.num_tracks;
    info.ticks_per_quarter_note = smf.tpqn;
    info.initial_tempo_bpm = smf.initial_tempo_bpm;
    info.duration_sec = smf.duration_sec;
    info.total_notes = smf.note_on_count;
    info.track_name = keel::move(smf.track_name);
    return info;
}

Result<Array<MidiNote>> decode_midi(ArrayRef<u8> bytes, Allocator* alloc) {
    Allocator* dst_alloc = alloc ? alloc : default_allocator();
    KEEL_TRY_ASSIGN(midi_detail::ParsedSmf smf, midi_detail::parse_smf(bytes, true, dst_alloc));
    return keel::move(smf.notes);
}

Result<Array<u8>> encode_midi(ArrayRef<MidiNote> notes, const MidiEncodeOptions& opts) {
    if (opts.ticks_per_quarter_note == 0u || (opts.ticks_per_quarter_note & 0x8000u) != 0u) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }
    if (!keel::is_finite(opts.tempo_bpm) || !(opts.tempo_bpm > 0.0)) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }

    const f64 us_raw = keel::round(60000000.0 / opts.tempo_bpm);
    if (!(us_raw >= 1.0 && us_raw <= 16777215.0)) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }
    const u32 us_per_quarter = as<u32>(us_raw);
    const f64 seconds_per_tick =
        (as<f64>(us_per_quarter) / 1000000.0) / as<f64>(opts.ticks_per_quarter_note);

    struct MidiOutEvent {
        u64 tick = 0;
        u32 order = 0;
        u8 status = 0;
        u8 data1 = 0;
        u8 data2 = 0;
    };

    Allocator* dst_alloc = opts.allocator ? opts.allocator : default_allocator();
    ArenaTemp scratch = get_scratch(dst_alloc);
    ArenaArray<MidiOutEvent> events(scratch.arena(), notes.size() * 2u);

    u32 order_seq = 0;
    for (usize i = 0; i < notes.size(); ++i) {
        const MidiNote& n = notes[i];
        if (n.channel > 15u || n.note_number > 127u || n.velocity > 127u) [[unlikely]] {
            return ErrorCode::InvalidArgument;
        }
        if (!keel::is_finite(n.start_time_sec) || !(n.start_time_sec >= 0.0) || !keel::is_finite(n.end_time_sec) ||
            !keel::is_finite(n.duration_sec)) [[unlikely]] {
            return ErrorCode::InvalidArgument;
        }

        f64 end_sec = n.start_time_sec;
        if (n.end_time_sec > n.start_time_sec) {
            end_sec = n.end_time_sec;
        } else if (n.duration_sec > 0.0) {
            end_sec = n.start_time_sec + n.duration_sec;
        }

        const u64 start_tick = as<u64>(keel::round_i64(n.start_time_sec / seconds_per_tick));
        u64 end_tick = as<u64>(keel::round_i64(end_sec / seconds_per_tick));
        if (end_tick <= start_tick) {
            end_tick = start_tick + 1ULL;
        }

        const u8 ch = as<u8>(n.channel & 0x0Fu);
        const u8 vel = (n.velocity == 0u) ? u8{1} : n.velocity;
        events.push_back(MidiOutEvent{start_tick, order_seq++, as<u8>(0x90u | ch), n.note_number, vel});
        events.push_back(MidiOutEvent{end_tick, order_seq++, as<u8>(0x80u | ch), n.note_number, 0u});
    }

    keel::sort(events, [](const MidiOutEvent& a, const MidiOutEvent& b) {
        if (a.tick != b.tick) return a.tick < b.tick;
        const u8 ta = as<u8>(a.status & 0xF0u);
        const u8 tb = as<u8>(b.status & 0xF0u);
        if (ta != tb) return ta < tb; // Note-Off (0x80) before Note-On (0x90) at identical tick
        return a.order < b.order;
    });

    ArenaArray<u8> trk_payload(scratch.arena(), events.size() * 4u + 16u);

    // Tempo Meta Event at tick 0
    midi_detail::write_vlq(trk_payload, 0u);
    trk_payload.push_back(0xFFu);
    trk_payload.push_back(0x51u);
    trk_payload.push_back(0x03u);
    trk_payload.push_back(as<u8>((us_per_quarter >> 16) & 0xFFu));
    trk_payload.push_back(as<u8>((us_per_quarter >> 8) & 0xFFu));
    trk_payload.push_back(as<u8>(us_per_quarter & 0xFFu));

    u64 last_tick = 0;
    for (usize i = 0; i < events.size(); ++i) {
        const MidiOutEvent& ev = events[i];
        const u64 delta = ev.tick - last_tick;
        if (delta > 0x0FFFFFFFULL) [[unlikely]] {
            return ErrorCode::OutOfRange;
        }
        midi_detail::write_vlq(trk_payload, as<u32>(delta));
        trk_payload.push_back(ev.status);
        trk_payload.push_back(ev.data1);
        trk_payload.push_back(ev.data2);
        last_tick = ev.tick;
    }

    // End-of-Track Meta Event
    midi_detail::write_vlq(trk_payload, 0u);
    trk_payload.push_back(0xFFu);
    trk_payload.push_back(0x2Fu);
    trk_payload.push_back(0x00u);

    if (trk_payload.size() > 0xFFFFFFFFu) [[unlikely]] {
        return ErrorCode::OutOfRange;
    }

    Array<u8> out = Array<u8>::with_capacity(22u + trk_payload.size(), dst_alloc);
    audio_detail::write_fourcc(out, "MThd");
    audio_detail::write_u32_be(out, 6u);
    audio_detail::write_u16_be(out, 0u); // SMF Format 0
    audio_detail::write_u16_be(out, 1u); // 1 track
    audio_detail::write_u16_be(out, opts.ticks_per_quarter_note);

    audio_detail::write_fourcc(out, "MTrk");
    audio_detail::write_u32_be(out, as<u32>(trk_payload.size()));
    out.append(ArrayRef<u8>(trk_payload.data(), trk_payload.size()));

    return out;
}

Result<MidiInfo> peek_midi_file(StringRef path, Allocator* alloc) {
    if (path.empty()) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }
    ArenaTemp scratch = get_scratch(alloc);
    KEEL_TRY_ASSIGN(Array<u8> file_bytes, fs_read_file(path, scratch.allocator()));
    return peek_midi(file_bytes, alloc);
}

Result<Array<MidiNote>> read_midi(StringRef path, Allocator* alloc) {
    if (path.empty()) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }
    ArenaTemp scratch = get_scratch(alloc);
    KEEL_TRY_ASSIGN(Array<u8> file_bytes, fs_read_file(path, scratch.allocator()));
    return decode_midi(file_bytes, alloc);
}

Error write_midi(StringRef path, ArrayRef<MidiNote> notes, const MidiEncodeOptions& opts) {
    if (path.empty()) [[unlikely]] {
        return ErrorCode::InvalidArgument;
    }
    ArenaTemp scratch = get_scratch(opts.allocator);
    MidiEncodeOptions temp_opts = opts;
    if (!temp_opts.allocator) {
        temp_opts.allocator = scratch.allocator();
    }
    KEEL_TRY_ASSIGN(Array<u8> encoded, encode_midi(notes, temp_opts));
    return fs_write_file(path, encoded);
}

} // namespace keel::audiofile
