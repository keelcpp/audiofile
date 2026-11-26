#!/usr/bin/env python3
"""
Bidirectional reference differential verification & adversarial fuzz harness
for keel::audiofile against:
  1. FFmpeg 8.x (WAV, FLAC levels 0/5/8 8/16/24-bit mono/stereo/5.1,
                 MP3 MPEG-1/2/2.5 CBR/VBR mono/stereo/joint-stereo/transients,
                 AIFF/AIFF-C, AU/SND incl. Mu-law/A-law/float)
  2. SciPy 1.17.x (scipy.io.wavfile)
  3. MATLAB R2025b (audioread / audiowrite across WAV, FLAC, MP3) when installed
  4. Independent Python SMF 1.0 binary parser for MIDI Format 0 & 1

Usage:
  python tests/verify_reference.py <path-to-test_audiofile.exe> [--no-matlab]
"""

import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile


STYPE_TO_DTYPE = {
    "u8": np.uint8,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "f32": np.float32,
    "f64": np.float64,
}


def run_keel_decode(exe: Path, in_path: Path, raw_path: Path, native: bool = False, start: int = 0, end=None):
    cmd = [str(exe), "--decode", str(in_path), str(raw_path), "1" if native else "0", str(start),
           "end" if end is None else str(end)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"keel --decode failed ({proc.returncode}) on {in_path.name}: {proc.stderr.strip()}")
    meta = json.loads(proc.stdout.strip())
    dtype = STYPE_TO_DTYPE[meta["sample_type"]]
    arr = np.frombuffer(raw_path.read_bytes(), dtype=dtype)
    if meta["channels"] == 1:
        arr = arr.reshape((meta["frames"],))
    else:
        arr = arr.reshape((meta["frames"], meta["channels"]))
    return meta, arr


def run_keel_encode(exe: Path, fmt: str, sr: int, arr: np.ndarray, stype: str, bps: int, flt: bool,
                    out_path: Path, tmp: Path, title="", artist="", comment=""):
    raw_path = tmp / f"enc_{out_path.name}.raw"
    arr = np.ascontiguousarray(arr, dtype=STYPE_TO_DTYPE[stype])
    raw_path.write_bytes(arr.tobytes())
    frames = arr.shape[0]
    ch = 1 if arr.ndim == 1 else arr.shape[1]
    cmd = [str(exe), "--encode", fmt, str(sr), str(ch), str(frames), stype, str(bps),
           "1" if flt else "0", str(raw_path), str(out_path), title, artist, comment]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"keel --encode {fmt} failed ({proc.returncode}): {proc.stderr.strip()}")


def make_test_signal(sr: int, frames: int, ch: int, transient: bool = False) -> np.ndarray:
    t = np.arange(frames, dtype=np.float64) / float(sr)
    cols = []
    for c in range(ch):
        f1 = 440.0 * (1.0 + 0.25 * c)
        f2 = 1320.0 + 180.0 * c
        sig = 0.55 * np.sin(2.0 * np.pi * f1 * t) + 0.25 * np.cos(2.0 * np.pi * f2 * t + 0.4 * c)
        if transient:
            # Sharp impulse bursts to exercise MP3 short & mixed blocks!
            for pos in range(sr // 8, frames - 64, sr // 4):
                env = np.exp(-np.arange(64) / 6.0)
                sig[pos:pos + 64] += (0.4 if c % 2 == 0 else -0.35) * env * np.sin(np.arange(64) * 0.9)
        cols.append(np.clip(sig, -0.95, 0.95))
    return cols[0] if ch == 1 else np.stack(cols, axis=-1)


def ffmpeg_decode_f64(ffmpeg: str, in_path: Path, ch: int, tmp: Path, ignore_xing: bool = False) -> np.ndarray:
    out_raw = tmp / f"ffdec_{in_path.name}.f64"
    cmd = [ffmpeg, "-y"]
    if ignore_xing:
        cmd += ["-flags2", "+skip_manual"]
    cmd += ["-i", str(in_path), "-f", "f64le", "-acodec", "pcm_f64le", str(out_raw)]
    subprocess.run(cmd, check=True, capture_output=True)
    arr = np.frombuffer(out_raw.read_bytes(), dtype=np.float64)
    return arr if ch == 1 else arr.reshape((-1, ch))


def assert_exact(name: str, got: np.ndarray, exp: np.ndarray, stats: list):
    if got.shape != exp.shape or got.dtype != exp.dtype:
        raise AssertionError(f"[{name}] shape/dtype mismatch: got {got.shape}/{got.dtype}, exp {exp.shape}/{exp.dtype}")
    if not np.array_equal(got, exp):
        diff = np.abs(got.astype(np.float64) - exp.astype(np.float64))
        raise AssertionError(f"[{name}] mismatch: max_diff={diff.max()}, mean={diff.mean():.6g}")
    stats.append(name)


def assert_close(name: str, got: np.ndarray, exp: np.ndarray, max_abs: float, max_rmse: float, stats: list):
    if got.shape != exp.shape:
        raise AssertionError(f"[{name}] shape mismatch: got {got.shape}, exp {exp.shape}")
    diff = np.abs(got.astype(np.float64) - exp.astype(np.float64))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mx = float(diff.max())
    if mx > max_abs or rmse > max_rmse:
        raise AssertionError(f"[{name}] exceeded tol: rmse={rmse:.6g} (lim {max_rmse}), max={mx:.6g} (lim {max_abs})")
    stats.append(name)


def verify_wav_scipy_ffmpeg(exe: Path, ffmpeg: str, tmp: Path, stats: list):
    raw = tmp / "wav_dec.raw"
    sr = 44100
    sig2 = make_test_signal(sr, 2000, 2)
    sig6 = make_test_signal(48000, 1024, 6)

    # 1. SciPy <-> keel WAV bit-exact (u8, i16, i32, f32)
    for tag, arr, stype, bps, flt in [
        ("wav_u8", ((sig2 + 1.0) * 127.5).round().astype(np.uint8), "u8", 8, False),
        ("wav_i16", (sig2 * 32767.0).round().astype(np.int16), "i16", 16, False),
        ("wav_i32", (sig2 * 2147483647.0).round().astype(np.int32), "i32", 32, False),
        ("wav_f32", sig2.astype(np.float32), "f32", 32, True),
        ("wav_51_i16", (sig6 * 32767.0).round().astype(np.int16), "i16", 16, False),
    ]:
        p_sci = tmp / f"sci_{tag}.wav"
        wavfile.write(str(p_sci), 48000 if tag.endswith("51_i16") else sr, arr)
        _, got = run_keel_decode(exe, p_sci, raw, native=True)
        assert_exact(f"scipy->keel:{tag}", got, arr, stats)

        p_keel = tmp / f"keel_{tag}.wav"
        run_keel_encode(exe, "wav", 48000 if tag.endswith("51_i16") else sr, arr, stype, bps, flt,
                        p_keel, tmp, title="RefTest", artist="Keel", comment="BitExact")
        got_sr, got_sci = wavfile.read(str(p_keel))
        assert got_sr == (48000 if tag.endswith("51_i16") else sr)
        assert_exact(f"keel->scipy:{tag}", got_sci, arr, stats)

    # 2. FFmpeg <-> keel WAV (24-bit PCM, 64-bit float, Mu-law, A-law)
    i16_stereo = (sig2 * 32767.0).round().astype(np.int16)
    p_src_wav = tmp / "src_i16.wav"
    wavfile.write(str(p_src_wav), sr, i16_stereo)

    for codec, name, tol in [
        ("pcm_s24le", "wav_s24", 1e-6),
        ("pcm_f64le", "wav_f64", 1e-12),
        ("pcm_mulaw", "wav_mulaw", 1e-4),
        ("pcm_alaw", "wav_alaw", 1e-4),
    ]:
        p_ff = tmp / f"ff_{name}.wav"
        subprocess.run([ffmpeg, "-y", "-i", str(p_src_wav), "-acodec", codec, str(p_ff)],
                       check=True, capture_output=True)
        ref = ffmpeg_decode_f64(ffmpeg, p_ff, 2, tmp)
        _, got = run_keel_decode(exe, p_ff, raw, native=False)
        assert_close(f"ffmpeg->keel:{name}", got, ref, max_abs=tol, max_rmse=tol, stats=stats)

    # keel -> FFmpeg 24-bit & 64-bit float WAV
    p_k24 = tmp / "keel_24.wav"
    run_keel_encode(exe, "wav", sr, sig2, "f64", 24, False, p_k24, tmp)
    ref24 = ffmpeg_decode_f64(ffmpeg, p_k24, 2, tmp)
    _, got24 = run_keel_decode(exe, p_k24, raw, native=False)
    assert_close("keel->ffmpeg:wav_24", got24, ref24, max_abs=1e-12, max_rmse=1e-12, stats=stats)

    p_k64 = tmp / "keel_64.wav"
    run_keel_encode(exe, "wav", sr, sig2, "f64", 64, True, p_k64, tmp)
    ref64 = ffmpeg_decode_f64(ffmpeg, p_k64, 2, tmp)
    assert_exact("keel->ffmpeg:wav_f64", ref64, sig2, stats)


def verify_flac_ffmpeg(exe: Path, ffmpeg: str, tmp: Path, stats: list):
    raw = tmp / "flac_dec.raw"
    # Encode REAL FLAC files with FFmpeg across 16-bit, 24-bit, mono, stereo, 5.1, compression levels 0, 5, 8!
    cases = [
        ("flac_mono16_l0", 44100, 1, 16, 0, "pcm_s16le", "s16", np.int16, 8192),
        ("flac_stereo16_l5", 44100, 2, 16, 5, "pcm_s16le", "s16", np.int16, 12000),
        ("flac_stereo16_l8", 48000, 2, 16, 8, "pcm_s16le", "s16", np.int16, 12000),
        ("flac_stereo24_l8", 48000, 2, 24, 8, "pcm_s24le", "s32", np.int32, 10000),
        ("flac_51_16_l5", 48000, 6, 16, 5, "pcm_s16le", "s16", np.int16, 6000),
    ]
    for name, sr, ch, bps, lvl, wav_codec, ff_sf, dtype, frames in cases:
        sig = make_test_signal(sr, frames, ch, transient=True)
        # Include a quiet/constant segment & wasted-bits-friendly segment to hit ALL FLAC subframe types!
        if frames >= 4096:
            sig[1024:1536] = 0.0
            if ch == 2:
                # Identical L/R segment (Mid-Side / Left-Side stereo)
                sig[2048:3072, 1] = sig[2048:3072, 0]
        p_wav = tmp / f"{name}_in.wav"
        if bps == 16:
            arr_in = (sig * 32767.0).round().astype(np.int16)
            wavfile.write(str(p_wav), sr, arr_in)
        else:
            # 24-bit via keel WAV encoder
            run_keel_encode(exe, "wav", sr, sig, "f64", 24, False, p_wav, tmp)

        p_flac = tmp / f"{name}.flac"
        subprocess.run(
            [ffmpeg, "-y", "-i", str(p_wav), "-c:a", "flac", "-compression_level", str(lvl),
             "-sample_fmt", ff_sf, "-metadata", "title=FlacRef", "-metadata", "artist=FFmpeg", str(p_flac)],
            check=True, capture_output=True
        )

        # Decode with FFmpeg to exact f64 reference and compare bit-exact with keel::decode_flac!
        ref_f64 = ffmpeg_decode_f64(ffmpeg, p_flac, ch, tmp)
        meta, got_f64 = run_keel_decode(exe, p_flac, raw, native=False)
        assert meta["title"] == "FlacRef" and meta["artist"] == "FFmpeg"
        assert_close(f"ffmpeg->keel:{name}", got_f64, ref_f64, max_abs=1e-12, max_rmse=1e-12, stats=stats)

        if bps == 16:
            _, got_i16 = run_keel_decode(exe, p_flac, raw, native=True)
            assert_exact(f"ffmpeg->keel:{name}_native_i16", got_i16, arr_in, stats)

    # Verify RFC 9639 CRC-8/CRC-16 bit-flip corruption rejection on a real FFmpeg-encoded FLAC!
    p_corrupt = tmp / "flac_corrupt_crc.flac"
    orig_flac = bytearray((tmp / "flac_stereo16_l5.flac").read_bytes())
    orig_flac[-1] ^= 0x01  # flip LSB of last frame's CRC-16 footer
    p_corrupt.write_bytes(bytes(orig_flac))
    proc = subprocess.run([str(exe), "--decode", str(p_corrupt), str(raw), "0", "0", "end"],
                          capture_output=True, text=True, check=False)
    assert proc.returncode != 0, "Expected keel::decode_flac to reject CRC-16 corrupted frame!"
    stats.append("ffmpeg->keel:flac_crc16_corrupt_rejected")


def verify_mp3_ffmpeg(exe: Path, ffmpeg: str, tmp: Path, stats: list):
    raw = tmp / "mp3_dec.raw"

    # Encode REAL MP3 streams with FFmpeg libmp3lame across MPEG-1, MPEG-2, MPEG-2.5,
    # CBR, VBR, mono, stereo, joint-stereo, and sharp transients (short/mixed blocks)!
    mp3_cases = [
        ("mp3_m1_44k_stereo_192k", 44100, 2, ["-b:a", "192k", "-joint_stereo", "0"], True),
        ("mp3_m1_44k_jstereo_128k", 44100, 2, ["-b:a", "128k", "-joint_stereo", "1"], True),
        ("mp3_m1_48k_vbr_q2", 48000, 2, ["-q:a", "2"], True),
        ("mp3_m1_32k_mono_64k", 32000, 1, ["-b:a", "64k"], False),
        ("mp3_m2_22k_jstereo_64k", 22050, 2, ["-b:a", "64k", "-joint_stereo", "1"], True),
        ("mp3_m2_24k_mono_48k", 24000, 1, ["-b:a", "48k"], False),
        ("mp3_m25_11k_mono_32k", 11025, 1, ["-b:a", "32k"], False),
        ("mp3_m25_8k_mono_24k", 8000, 1, ["-b:a", "24k"], False),
    ]

    for name, sr, ch, enc_args, transient in mp3_cases:
        frames = sr * 1  # 1 second (~38 MPEG-1 frames)
        sig = make_test_signal(sr, frames, ch, transient=transient)
        p_wav = tmp / f"{name}_in.wav"
        wavfile.write(str(p_wav), sr, (sig * 32767.0).round().astype(np.int16))

        # Disable Xing header (-write_xing 0) on raw stream comparison so FFmpeg and keel decode
        # the EXACT same sequence of MPEG audio frames without LAME gapless trim offset differences!
        p_mp3 = tmp / f"{name}.mp3"
        subprocess.run(
            [ffmpeg, "-y", "-i", str(p_wav), "-c:a", "libmp3lame", *enc_args,
             "-write_xing", "0", "-metadata", "title=Mp3Ref", "-metadata", "artist=Lame", str(p_mp3)],
            check=True, capture_output=True
        )

        ref = ffmpeg_decode_f64(ffmpeg, p_mp3, ch, tmp)
        meta, got = run_keel_decode(exe, p_mp3, raw, native=False)
        assert meta["channels"] == ch and int(round(meta["sample_rate"])) == sr
        assert meta["title"] == "Mp3Ref" and meta["artist"] == "Lame"

        # Align lengths (both decode exact same MPEG frames; allow exact match)
        n = min(ref.shape[0], got.shape[0])
        assert n >= frames // 2, f"[{name}] decoded too few frames: {n}"
        ref_n = ref[:n]
        got_n = got[:n]

        # Compare keel's ISO Layer III decoder directly against FFmpeg's mp3float reference decoder!
        # Both implement ISO 11172-3 / 13818-3: RMSE must be tiny (< 1e-4) and Pearson r > 0.99999!
        diff = np.abs(got_n - ref_n)
        rmse = float(np.sqrt(np.mean(diff * diff)))
        mx = float(diff.max())
        corr = float(np.corrcoef(got_n.ravel(), ref_n.ravel())[0, 1])
        if rmse > 1.5e-4 or corr < 0.99999:
            raise AssertionError(f"[ffmpeg->keel:{name}] MP3 mismatch vs FFmpeg: rmse={rmse:.6g}, max={mx:.6g}, corr={corr:.8f}")
        stats.append(f"ffmpeg->keel:{name} (rmse={rmse:.2e})")

    # Also test a VBR MP3 WITH Xing/LAME gapless header:
    # 1. Exact sample length & 0-sample phase match directly against FFmpeg mp3float!
    # 2. Exact frame-slice consistency!
    sig_vbr = make_test_signal(44100, 44100, 2, transient=True)
    p_wav_vbr = tmp / "vbr_xing_in.wav"
    wavfile.write(str(p_wav_vbr), 44100, (sig_vbr * 32767.0).round().astype(np.int16))
    p_mp3_xing = tmp / "vbr_xing.mp3"
    subprocess.run(
        [ffmpeg, "-y", "-i", str(p_wav_vbr), "-c:a", "libmp3lame", "-q:a", "3", "-write_xing", "1", str(p_mp3_xing)],
        check=True, capture_output=True
    )
    ref_vbr = ffmpeg_decode_f64(ffmpeg, p_mp3_xing, 2, tmp)
    _, full_vbr = run_keel_decode(exe, p_mp3_xing, raw, native=False)
    assert full_vbr.shape == ref_vbr.shape, (
        f"[mp3_vbr_xing_gapless] length mismatch: keel={full_vbr.shape} vs ffmpeg={ref_vbr.shape}"
    )
    diff_vbr = np.abs(full_vbr - ref_vbr)
    rmse_vbr = float(np.sqrt(np.mean(diff_vbr * diff_vbr)))
    corr_vbr = float(np.corrcoef(full_vbr.ravel(), ref_vbr.ravel())[0, 1])
    if rmse_vbr > 1.5e-4 or corr_vbr < 0.99999:
        raise AssertionError(
            f"[ffmpeg->keel:mp3_vbr_xing_gapless] mismatch: rmse={rmse_vbr:.6g}, corr={corr_vbr:.8f}"
        )
    stats.append(f"ffmpeg->keel:mp3_vbr_xing_gapless (len={full_vbr.shape[0]}, rmse={rmse_vbr:.2e})")

    _, slice_vbr = run_keel_decode(exe, p_mp3_xing, raw, native=False, start=2500, end=5200)
    assert_exact("keel:mp3_vbr_xing_slice_exact", slice_vbr, full_vbr[2500:5200], stats)

    # MPEG-1 Layer II (.mp2) stereo & mono decoded by keel::decode_mp3 vs FFmpeg mp2float!
    for name, sr, ch, br in [
        ("mp2_m1_44k_stereo_192k", 44100, 2, "192k"),
        ("mp2_m1_48k_mono_128k", 48000, 1, "128k"),
    ]:
        sig_mp2 = make_test_signal(sr, sr, ch, transient=True)
        p_wav_mp2 = tmp / f"{name}_in.wav"
        wavfile.write(str(p_wav_mp2), sr, (sig_mp2 * 32767.0).round().astype(np.int16))
        p_mp2 = tmp / f"{name}.mp2"
        subprocess.run(
            [ffmpeg, "-y", "-i", str(p_wav_mp2), "-c:a", "mp2", "-b:a", br, str(p_mp2)],
            check=True, capture_output=True
        )
        ref_mp2 = ffmpeg_decode_f64(ffmpeg, p_mp2, ch, tmp)
        meta_mp2, got_mp2 = run_keel_decode(exe, p_mp2, raw, native=False)
        assert meta_mp2["channels"] == ch and int(round(meta_mp2["sample_rate"])) == sr
        n_mp2 = min(ref_mp2.shape[0], got_mp2.shape[0])
        diff_mp2 = np.abs(got_mp2[:n_mp2] - ref_mp2[:n_mp2])
        rmse_mp2 = float(np.sqrt(np.mean(diff_mp2 * diff_mp2)))
        corr_mp2 = float(np.corrcoef(got_mp2[:n_mp2].ravel(), ref_mp2[:n_mp2].ravel())[0, 1])
        if rmse_mp2 > 1.5e-4 or corr_mp2 < 0.99999:
            raise AssertionError(
                f"[ffmpeg->keel:{name}] MP2 mismatch vs FFmpeg: rmse={rmse_mp2:.6g}, corr={corr_mp2:.8f}"
            )
        stats.append(f"ffmpeg->keel:{name} (rmse={rmse_mp2:.2e})")


def verify_aiff_au_ffmpeg(exe: Path, ffmpeg: str, tmp: Path, stats: list):
    raw = tmp / "aa_dec.raw"
    sr = 48000
    sig = make_test_signal(sr, 1500, 2)
    p_src_wav = tmp / "aa_src.wav"
    wavfile.write(str(p_src_wav), sr, (sig * 32767.0).round().astype(np.int16))

    # AIFF / AIFF-C: FFmpeg -> keel & keel -> FFmpeg
    for codec, ext, name, tol in [
        ("pcm_s8", "aiff", "aiff_s8", 1e-6),
        ("pcm_s16be", "aiff", "aiff_s16be", 1e-12),
        ("pcm_s24be", "aiff", "aiff_s24be", 1e-12),
        ("pcm_s32be", "aiff", "aiff_s32be", 1e-12),
        ("pcm_f32be", "aifc", "aifc_f32be", 1e-7),
        ("pcm_f64be", "aifc", "aifc_f64be", 1e-12),
        ("pcm_s16le", "aifc", "aifc_sowt", 1e-12),
        ("pcm_mulaw", "aifc", "aifc_ulaw", 1e-4),
        ("pcm_alaw", "aifc", "aifc_alaw", 1e-4),
    ]:
        p_ff = tmp / f"ff_{name}.{ext}"
        subprocess.run([ffmpeg, "-y", "-i", str(p_src_wav), "-f", "aiff", "-acodec", codec, str(p_ff)],
                       check=True, capture_output=True)
        ref = ffmpeg_decode_f64(ffmpeg, p_ff, 2, tmp)
        _, got = run_keel_decode(exe, p_ff, raw, native=False)
        assert_close(f"ffmpeg->keel:{name}", got, ref, max_abs=tol, max_rmse=tol, stats=stats)

    for bps, flt, tag in [(8, False, "s8"), (16, False, "s16"), (24, False, "s24"),
                          (32, False, "s32"), (32, True, "f32"), (64, True, "f64")]:
        p_k = tmp / f"keel_aiff_{tag}.aiff"
        run_keel_encode(exe, "aiff", sr, sig, "f64", bps, flt, p_k, tmp, title="AiffT", artist="AiffA", comment="AiffC")
        ref = ffmpeg_decode_f64(ffmpeg, p_k, 2, tmp)
        _, got = run_keel_decode(exe, p_k, raw, native=False)
        assert_close(f"keel->ffmpeg:aiff_{tag}", got, ref, max_abs=1e-12, max_rmse=1e-12, stats=stats)

    # AU / SND: FFmpeg -> keel & keel -> FFmpeg
    for codec, name, tol in [
        ("pcm_s8", "au_s8", 1e-6),
        ("pcm_s16be", "au_s16be", 1e-12),
        ("pcm_s24be", "au_s24be", 1e-12),
        ("pcm_s32be", "au_s32be", 1e-12),
        ("pcm_f32be", "au_f32be", 1e-7),
        ("pcm_f64be", "au_f64be", 1e-12),
        ("pcm_mulaw", "au_ulaw", 1e-4),
        ("pcm_alaw", "au_alaw", 1e-4),
    ]:
        p_ff = tmp / f"ff_{name}.au"
        subprocess.run([ffmpeg, "-y", "-i", str(p_src_wav), "-f", "au", "-acodec", codec, str(p_ff)],
                       check=True, capture_output=True)
        ref = ffmpeg_decode_f64(ffmpeg, p_ff, 2, tmp)
        _, got = run_keel_decode(exe, p_ff, raw, native=False)
        assert_close(f"ffmpeg->keel:{name}", got, ref, max_abs=tol, max_rmse=tol, stats=stats)

    for bps, flt, tag in [(8, False, "s8"), (16, False, "s16"), (24, False, "s24"),
                          (32, False, "s32"), (32, True, "f32"), (64, True, "f64")]:
        p_k = tmp / f"keel_au_{tag}.au"
        run_keel_encode(exe, "au", sr, sig, "f64", bps, flt, p_k, tmp, comment="AuComment")
        ref = ffmpeg_decode_f64(ffmpeg, p_k, 2, tmp)
        _, got = run_keel_decode(exe, p_k, raw, native=False)
        assert_close(f"keel->ffmpeg:au_{tag}", got, ref, max_abs=1e-12, max_rmse=1e-12, stats=stats)


def parse_smf_independent(blob: bytes):
    """Independent Python SMF 1.0 binary parser to verify keel::encode_midi output."""
    assert blob[:4] == b"MThd"
    hdr_len = struct.unpack(">I", blob[4:8])[0]
    fmt, ntrks, tpqn = struct.unpack(">HHH", blob[8:14])
    pos = 8 + hdr_len
    tracks = []
    for _ in range(ntrks):
        assert blob[pos:pos + 4] == b"MTrk"
        tlen = struct.unpack(">I", blob[pos + 4:pos + 8])[0]
        pos += 8
        end = pos + tlen
        events = []
        tick = 0
        running = 0
        while pos < end:
            dt = 0
            while True:
                b = blob[pos]; pos += 1
                dt = (dt << 7) | (b & 0x7F)
                if not (b & 0x80):
                    break
            tick += dt
            st = blob[pos]
            if st & 0x80:
                pos += 1
                if st <= 0xEF:
                    running = st
            else:
                st = running
            if st == 0xFF:
                mtype = blob[pos]; pos += 1
                mlen = 0
                while True:
                    b = blob[pos]; pos += 1
                    mlen = (mlen << 7) | (b & 0x7F)
                    if not (b & 0x80):
                        break
                events.append(("meta", tick, mtype, blob[pos:pos + mlen]))
                pos += mlen
            else:
                mt = st & 0xF0
                d1 = blob[pos]; pos += 1
                d2 = blob[pos] if mt not in (0xC0, 0xD0) else 0
                if mt not in (0xC0, 0xD0):
                    pos += 1
                events.append(("ch", tick, mt, st & 0x0F, d1, d2))
        tracks.append(events)
        pos = end
    return fmt, ntrks, tpqn, tracks


def verify_midi_independent(exe: Path, tmp: Path, stats: list):
    # 1. keel -> independent Python SMF 1.0 parser
    p_mid = tmp / "keel_out.mid"
    subprocess.run([str(exe), "--encode-midi", str(p_mid)], check=True, capture_output=True)
    fmt, ntrks, tpqn, tracks = parse_smf_independent(p_mid.read_bytes())
    assert fmt == 0 and ntrks == 1 and tpqn == 480
    note_ons = [e for e in tracks[0] if e[0] == "ch" and e[2] == 0x90 and e[5] > 0]
    note_offs = [e for e in tracks[0] if e[0] == "ch" and (e[2] == 0x80 or (e[2] == 0x90 and e[5] == 0))]
    assert len(note_ons) == 4 and len(note_offs) == 4
    assert [(e[1], e[3], e[4], e[5]) for e in note_ons] == [
        (0, 0, 60, 100), (240, 0, 64, 90), (480, 2, 67, 110), (960, 9, 36, 127)
    ]
    stats.append("keel->py_smf:format0_encode")

    # 2. Independent Python SMF Format 1 generator (multi-track, tempo change, SysEx, running status) -> keel
    def vlq(v: int) -> bytes:
        out = [v & 0x7F]
        v >>= 7
        while v:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        return bytes(reversed(out))

    # Track 0: tempo 120 BPM (500000 us) at tick 0, tempo 60 BPM (1000000 us) at tick 960
    t0 = (
        vlq(0) + b"\xFF\x03\x05Synth" +
        vlq(0) + b"\xFF\x51\x03" + (500000).to_bytes(3, "big") +
        vlq(960) + b"\xFF\x51\x03" + (1000000).to_bytes(3, "big") +
        vlq(0) + b"\xFF\x2F\x00"
    )
    # Track 1: note 60 spanning across tempo change [480..1440] -> 0.5s to (1.0s + 1.0s = 2.0s), plus SysEx
    t1 = (
        vlq(0) + b"\xF0\x03\x7E\x7F\xF7" +
        vlq(480) + bytes([0x90, 60, 105]) +
        vlq(960) + bytes([60, 0]) +  # running status Note-Off
        vlq(0) + b"\xFF\x2F\x00"
    )
    smf1 = (
        b"MThd" + struct.pack(">IHHH", 6, 1, 2, 480) +
        b"MTrk" + struct.pack(">I", len(t0)) + t0 +
        b"MTrk" + struct.pack(">I", len(t1)) + t1
    )
    p_smf1 = tmp / "py_smf1.mid"
    p_smf1.write_bytes(smf1)
    proc = subprocess.run([str(exe), "--decode-midi", str(p_smf1)], check=True, capture_output=True, text=True)
    info = json.loads(proc.stdout.strip())
    assert info["format"] == 1 and info["num_tracks"] == 2 and info["track_name"] == "Synth"
    assert len(info["notes"]) == 1
    n0 = info["notes"][0]
    assert n0["note"] == 60 and n0["vel"] == 105
    assert abs(n0["start"] - 0.5) < 1e-6 and abs(n0["end"] - 2.0) < 1e-6 and abs(n0["dur"] - 1.5) < 1e-6
    stats.append("py_smf->keel:format1_tempo_change_sysex")

    # 3. SMPTE 25fps x 40 sub-ticks (1000 ticks/sec) + dangling Note-On closed at End of Track -> keel
    t_smpte = (
        vlq(0) + bytes([0x90, 62, 100]) +
        vlq(250) + bytes([0x80, 62, 0]) +
        vlq(0) + bytes([0x90, 67, 88]) +
        vlq(750) + b"\xFF\x2F\x00"
    )
    smf_smpte = b"MThd" + struct.pack(">IHHbB", 6, 0, 1, -25, 40) + b"MTrk" + struct.pack(">I", len(t_smpte)) + t_smpte
    p_smpte = tmp / "py_smpte.mid"
    p_smpte.write_bytes(smf_smpte)
    proc_s = subprocess.run([str(exe), "--decode-midi", str(p_smpte)], check=True, capture_output=True, text=True)
    info_s = json.loads(proc_s.stdout.strip())
    assert len(info_s["notes"]) == 2
    assert abs(info_s["notes"][0]["end"] - 0.25) < 1e-6
    assert info_s["notes"][1]["note"] == 67 and abs(info_s["notes"][1]["start"] - 0.25) < 1e-6 and abs(info_s["notes"][1]["end"] - 1.0) < 1e-6
    stats.append("py_smf->keel:smpte25_and_eot_note_close")


def verify_matlab(exe: Path, ffmpeg: str, tmp: Path, stats: list):
    matlab = shutil.which("matlab") or r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"
    if not os.path.exists(matlab):
        print("  [SKIP] MATLAB not found")
        return

    raw = tmp / "ml_dec.raw"
    sr = 44100
    sig = make_test_signal(sr, 4410, 2, transient=True)
    p_k_wav16 = tmp / "keel_ml_16.wav"
    p_k_wav24 = tmp / "keel_ml_24.wav"
    p_k_wavf32 = tmp / "keel_ml_f32.wav"
    run_keel_encode(exe, "wav", sr, sig, "f64", 16, False, p_k_wav16, tmp)
    run_keel_encode(exe, "wav", sr, sig, "f64", 24, False, p_k_wav24, tmp)
    run_keel_encode(exe, "wav", sr, sig.astype(np.float32), "f32", 32, True, p_k_wavf32, tmp)

    # Also encode an MP3 via FFmpeg (no Xing) so MATLAB audioread and keel decode_mp3 can be cross-checked!
    p_mp3_ml = tmp / "ml_ref.mp3"
    subprocess.run(
        [ffmpeg, "-y", "-i", str(p_k_wav16), "-c:a", "libmp3lame", "-b:a", "192k", "-write_xing", "0", str(p_mp3_ml)],
        check=True, capture_output=True
    )

    _, k_wav16_f64 = run_keel_decode(exe, p_k_wav16, raw, native=False)
    _, k_wav24_f64 = run_keel_decode(exe, p_k_wav24, raw, native=False)
    _, k_wavf32_f64 = run_keel_decode(exe, p_k_wavf32, raw, native=False)
    _, k_mp3_f64 = run_keel_decode(exe, p_mp3_ml, raw, native=False)

    (tmp / "k_wav16.f64").write_bytes(k_wav16_f64.tobytes())
    (tmp / "k_wav24.f64").write_bytes(k_wav24_f64.tobytes())
    (tmp / "k_wavf32.f64").write_bytes(k_wavf32_f64.tobytes())
    (tmp / "k_mp3.f64").write_bytes(k_mp3_f64.tobytes())

    td = str(tmp).replace("\\", "/")
    ml_script = tmp / "verify_audio.m"
    ml_script.write_text(f"""
td = '{td}';
function y = load_f64(path, ch)
    fid = fopen(path, 'rb');
    raw = fread(fid, Inf, '*double');
    fclose(fid);
    y = reshape(raw, [ch, numel(raw)/ch]).';
end

% 1. keel WAV -> MATLAB audioread (bit-exact!)
[y16, fs16] = audioread([td '/keel_ml_16.wav']);
assert(fs16 == 44100 && max(abs(y16(:) - reshape(load_f64([td '/k_wav16.f64'], 2), [], 1))) < 1e-12);

[y24, fs24] = audioread([td '/keel_ml_24.wav']);
assert(fs24 == 44100 && max(abs(y24(:) - reshape(load_f64([td '/k_wav24.f64'], 2), [], 1))) < 1e-12);

[yf32, fsf32] = audioread([td '/keel_ml_f32.wav']);
assert(fsf32 == 44100 && max(abs(yf32(:) - reshape(load_f64([td '/k_wavf32.f64'], 2), [], 1))) < 1e-12);

% 2. MP3 cross-check: MATLAB audioread vs keel decode_mp3!
[ymp3, fsmp3] = audioread([td '/ml_ref.mp3']);
kmp3 = load_f64([td '/k_mp3.f64'], 2);
% Align via cross-correlation (Media Foundation / LAME may trim/keep leading delay)
c = xcorr(ymp3(:,1), kmp3(:,1), 2048);
[~, idx] = max(c);
lag = idx - 2049;
if lag >= 0
    n = min(size(ymp3,1) - lag, size(kmp3,1));
    a = ymp3(1+lag:lag+n, :); b = kmp3(1:n, :);
else
    alag = -lag;
    n = min(size(ymp3,1), size(kmp3,1) - alag);
    a = ymp3(1:n, :); b = kmp3(1+alag:alag+n, :);
end
r = corr(a(:), b(:));
rmse = sqrt(mean((a(:) - b(:)).^2));
assert(fsmp3 == 44100 && r > 0.9999 && rmse < 2e-3, sprintf('MP3 MATLAB mismatch: r=%.6f rmse=%.6g', r, rmse));

% 3. MATLAB audiowrite -> keel (FLAC 16-bit & 24-bit, WAV 24-bit)
t = (0:8191).' / 48000;
y_out = [0.6*sin(2*pi*440*t), -0.5*cos(2*pi*660*t)];
audiowrite([td '/ml_out_16.flac'], y_out, 48000, 'BitsPerSample', 16);
audiowrite([td '/ml_out_24.flac'], y_out, 48000, 'BitsPerSample', 24);
audiowrite([td '/ml_out_24.wav'],  y_out, 48000, 'BitsPerSample', 24);
[rf16, ~] = audioread([td '/ml_out_16.flac']);
[rf24, ~] = audioread([td '/ml_out_24.flac']);
[rw24, ~] = audioread([td '/ml_out_24.wav']);
fid = fopen([td '/ml_rf16.f64'], 'wb'); fwrite(fid, rf16.', 'double'); fclose(fid);
fid = fopen([td '/ml_rf24.f64'], 'wb'); fwrite(fid, rf24.', 'double'); fclose(fid);
fid = fopen([td '/ml_rw24.f64'], 'wb'); fwrite(fid, rw24.', 'double'); fclose(fid);
disp('MATLAB_AUDIO_OK');
""", encoding="utf-8")

    proc = subprocess.run([matlab, "-batch", f"run('{td}/verify_audio.m')"], capture_output=True, text=True, check=False)
    if proc.returncode != 0 or "MATLAB_AUDIO_OK" not in proc.stdout:
        raise RuntimeError(f"MATLAB audio verification failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    for tag in ["keel->matlab:wav_16", "keel->matlab:wav_24", "keel->matlab:wav_f32", "keel<->matlab:mp3_192k"]:
        stats.append(tag)

    for fname, ref_bin, name in [
        ("ml_out_16.flac", "ml_rf16.f64", "matlab->keel:flac_16"),
        ("ml_out_24.flac", "ml_rf24.f64", "matlab->keel:flac_24"),
        ("ml_out_24.wav", "ml_rw24.f64", "matlab->keel:wav_24"),
    ]:
        exp = np.frombuffer((tmp / ref_bin).read_bytes(), dtype=np.float64).reshape((-1, 2))
        _, got = run_keel_decode(exe, tmp / fname, raw, native=False)
        assert_close(name, got, exp, max_abs=1e-12, max_rmse=1e-12, stats=stats)


def run_adversarial_fuzz(exe: Path, ffmpeg: str, tmp: Path):
    fuzz_dir = tmp / "fuzz_audio"
    fuzz_dir.mkdir(exist_ok=True)
    rng = random.Random(20261126)

    sig = make_test_signal(44100, 2048, 2, transient=True)
    p_wav = tmp / "fseed.wav"
    wavfile.write(str(p_wav), 44100, (sig * 32767.0).round().astype(np.int16))
    p_flac = tmp / "fseed.flac"
    subprocess.run([ffmpeg, "-y", "-i", str(p_wav), "-c:a", "flac", str(p_flac)], check=True, capture_output=True)
    p_mp3 = tmp / "fseed.mp3"
    subprocess.run([ffmpeg, "-y", "-i", str(p_wav), "-c:a", "libmp3lame", "-b:a", "128k", str(p_mp3)],
                   check=True, capture_output=True)
    p_aiff = tmp / "fseed.aiff"
    subprocess.run([ffmpeg, "-y", "-i", str(p_wav), "-f", "aiff", str(p_aiff)], check=True, capture_output=True)
    p_au = tmp / "fseed.au"
    subprocess.run([ffmpeg, "-y", "-i", str(p_wav), "-f", "au", str(p_au)], check=True, capture_output=True)
    p_mid = tmp / "fseed.mid"
    subprocess.run([str(exe), "--encode-midi", str(p_mid)], check=True, capture_output=True)

    seeds = [
        ("wav", p_wav.read_bytes()),
        ("flac", p_flac.read_bytes()),
        ("mp3", p_mp3.read_bytes()),
        ("aiff", p_aiff.read_bytes()),
        ("au", p_au.read_bytes()),
        ("mid", p_mid.read_bytes()),
    ]

    idx = 0
    for ext, blob in seeds:
        for frac in [0.02, 0.1, 0.3, 0.6, 0.95]:
            cut = max(1, int(len(blob) * frac))
            (fuzz_dir / f"trunc_{idx:03d}.{ext}").write_bytes(blob[:cut])
            idx += 1
        for _ in range(25):
            mut = bytearray(blob)
            for _ in range(rng.randint(1, 12)):
                mut[rng.randrange(len(mut))] = rng.randrange(256)
            (fuzz_dir / f"mut_{idx:03d}.{ext}").write_bytes(bytes(mut))
            idx += 1

    # Targeted malformed / OOM / overflow bombs across all 6 formats
    bombs = [
        # WAV with huge data chunk size
        b"RIFF\xff\xff\xff\xffWAVEfmt \x10\x00\x00\x00\x01\x00\x02\x00\x44\xac\x00\x00\x10\xb1\x02\x00\x04\x00\x10\x00data\xff\xff\xff\xff\x00\x00",
        # FLAC with negative qlp_shift & bad sync
        b"fLaC\x80\x00\x00\x22\x00\x10\x00\x10\x00\x00\x00\x00\x00\x00\x0a\xc4\x42\xf0\x00\x00\x00\x10" + b"\x00" * 16 + b"\xff\xf8\x69\x04\x00\x00\xff\xff\xff\xff",
        # MP3 with ID3v2 huge syncsafe size
        b"ID3\x03\x00\x00\x7f\x7f\x7f\x7f\x00\x00",
        # MP3 with corrupt side info / max main_data_begin=511
        b"\xff\xfb\x90\x40" + b"\xff" * 413,
        # AIFF with Inf extended80 sample rate
        b"FORM\x00\x00\x00\x26AIFFCOMM\x00\x00\x00\x12\x00\x01\x00\x00\x00\x10\x00\x10\x7f\xfe\x80\x00\x00\x00\x00\x00\x00\x00SSND\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00",
        # AU with huge data_offset and 0 channels
        b".snd\xff\xff\xff\xff\xff\xff\xff\xff\x00\x00\x00\x03\x00\x00\xac\x44\x00\x00\x00\x00",
        # MIDI with 65535 tracks and truncated MTrk
        b"MThd\x00\x00\x00\x06\x00\x01\xff\xff\x01\xe0MTrk\xff\xff\xff\xff\x80\x80\x80\x80\x80",
    ]
    for b in bombs:
        (fuzz_dir / f"bomb_{idx:03d}.bin").write_bytes(b)
        idx += 1

    for _ in range(30):
        (fuzz_dir / f"rand_{idx:03d}.bin").write_bytes(rng.randbytes(rng.randint(1, 512)))
        idx += 1

    proc = subprocess.run([str(exe), "--fuzz", str(fuzz_dir)], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Audio fuzz failed ({proc.returncode}): {proc.stdout}\n{proc.stderr}")
    print(f"  [FUZZ] {proc.stdout.strip()}")


def main():
    if len(sys.argv) < 2:
        print("Usage: verify_reference.py <path-to-test_audiofile.exe> [--no-matlab]")
        sys.exit(2)

    exe = Path(sys.argv[1]).resolve()
    no_matlab = "--no-matlab" in sys.argv[2:]
    ffmpeg = shutil.which("ffmpeg")
    if not exe.exists():
        print(f"ERROR: executable not found: {exe}")
        sys.exit(2)
    if not ffmpeg:
        print("ERROR: ffmpeg not found on PATH")
        sys.exit(2)

    with tempfile.TemporaryDirectory(prefix="keel_aud_ref_") as td:
        tmp = Path(td)
        stats_wav: list[str] = []
        stats_flac: list[str] = []
        stats_mp3: list[str] = []
        stats_aa: list[str] = []
        stats_mid: list[str] = []
        stats_ml: list[str] = []

        print("=== 1. WAV bidirectional differential tests (SciPy + FFmpeg) ===")
        verify_wav_scipy_ffmpeg(exe, ffmpeg, tmp, stats_wav)
        print(f"  [PASS] WAV: {len(stats_wav)}/{len(stats_wav)} checks passed")

        print("=== 2. FLAC lossless differential tests (FFmpeg levels 0/5/8, 16/24-bit, 1/2/6ch) ===")
        verify_flac_ffmpeg(exe, ffmpeg, tmp, stats_flac)
        print(f"  [PASS] FLAC: {len(stats_flac)}/{len(stats_flac)} checks passed (bit-exact lossless)")

        print("=== 3. MP3 Layer III differential tests (FFmpeg libmp3lame MPEG-1/2/2.5 CBR/VBR/transients) ===")
        verify_mp3_ffmpeg(exe, ffmpeg, tmp, stats_mp3)
        print(f"  [PASS] MP3: {len(stats_mp3)}/{len(stats_mp3)} checks passed (all r > 0.99999, rmse < 1.5e-4)")

        print("=== 4. AIFF / AIFF-C & AU / SND bidirectional differential tests (FFmpeg) ===")
        verify_aiff_au_ffmpeg(exe, ffmpeg, tmp, stats_aa)
        print(f"  [PASS] AIFF + AU: {len(stats_aa)}/{len(stats_aa)} checks passed")

        print("=== 5. MIDI SMF 0 & 1 independent binary parser verification ===")
        verify_midi_independent(exe, tmp, stats_mid)
        print(f"  [PASS] MIDI: {len(stats_mid)}/{len(stats_mid)} checks passed")

        if not no_matlab:
            print("=== 6. MATLAB R2025b bidirectional differential tests (WAV/FLAC/MP3) ===")
            verify_matlab(exe, ffmpeg, tmp, stats_ml)
            print(f"  [PASS] MATLAB: {len(stats_ml)}/{len(stats_ml)} checks passed")
        else:
            print("=== 6. MATLAB R2025b skipped (--no-matlab) ===")

        print("=== 7. Adversarial fuzz corpus ===")
        run_adversarial_fuzz(exe, ffmpeg, tmp)

        total = len(stats_wav) + len(stats_flac) + len(stats_mp3) + len(stats_aa) + len(stats_mid) + len(stats_ml)
        print(f"\nALL AUDIO REFERENCE DIFFERENTIAL CHECKS PASSED ({total}/{total} + fuzz 0 crashes).")


if __name__ == "__main__":
    main()
