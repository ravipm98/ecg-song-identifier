"""
fingerprint.py — core Shazam-style audio fingerprinting functions.

Pipeline: load audio -> spectrogram -> sparse peak picking (constellation)
-> pair nearby peaks into (f1, f2, dt) hashes -> store hash->(song, time) in
a database -> match a query by histogramming offsets.
"""

import os
import subprocess
import numpy as np
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
from collections import defaultdict, Counter


# ---------------------------------------------------------------------
# Audio loading (via ffmpeg, since librosa/soundfile are unavailable)
# ---------------------------------------------------------------------
def _find_ffmpeg():
    """Locate the ffmpeg binary robustly. shutil.which() respects PATH the
    same way the shell does; if that fails, fall back to a couple of common
    install locations seen on Linux/cloud containers."""
    import shutil
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    for candidate in ('/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg', '/bin/ffmpeg'):
        if os.path.exists(candidate):
            return candidate
    raise RuntimeError(
        "ffmpeg was not found on this system (checked PATH and common install "
        "locations). On Streamlit Community Cloud, make sure 'ffmpeg' is listed "
        "in packages.txt at the repo root and that the app was rebooted after "
        "adding it (Manage app -> Reboot app)."
    )


def load_audio(path, sr=11025, mono=True, max_seconds=None, offset_seconds=None):
    """Decode any audio file to a float32 mono waveform at sample rate `sr`
    using ffmpeg as a subprocess (works for mp3/wav/m4a/etc. without extra
    Python audio libraries)."""
    ffmpeg_exe = _find_ffmpeg()
    cmd = [ffmpeg_exe]
    if offset_seconds:
        cmd += ['-ss', str(offset_seconds)]
    cmd += ['-i', path, '-f', 'f32le', '-ar', str(sr), '-ac', '1' if mono else '2',
            '-loglevel', 'error']
    if max_seconds:
        cmd += ['-t', str(max_seconds)]
    cmd += ['-']

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr_text = result.stderr.decode('utf-8', errors='replace').strip()
        raise RuntimeError(
            f"ffmpeg failed to decode '{path}' (exit code {result.returncode}). "
            f"ffmpeg stderr: {stderr_text or '(empty)'}"
        )

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(
            f"ffmpeg produced no audio data for '{path}'. The file may be "
            f"corrupted, empty, or in an unsupported format."
        )
    return audio, sr


# ---------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------
def compute_spectrogram(audio, sr, nperseg=1024, noverlap=768):
    f, t, Sxx = spectrogram(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db


# ---------------------------------------------------------------------
# Sparse peak-picking ("constellation")
# ---------------------------------------------------------------------
def find_peaks_2d(S, neighborhood=(20, 20), threshold_percentile=92):
    """Local maxima within a (freq, time) neighborhood that also exceed an
    amplitude percentile threshold -- the 'standout' time-frequency peaks."""
    local_max = maximum_filter(S, size=neighborhood) == S
    threshold = np.percentile(S, threshold_percentile)
    peaks_mask = local_max & (S > threshold)
    freq_idx, time_idx = np.where(peaks_mask)
    return freq_idx, time_idx


# ---------------------------------------------------------------------
# Hashing: pair nearby peaks into compact (f1, f2, dt) fingerprints
# ---------------------------------------------------------------------
def generate_hashes(freq_idx, time_idx, fan_out=8, max_time_bins=100, min_time_bins=1):
    """For each anchor peak, pair it with up to `fan_out` later peaks within
    a time window, producing hashes of the form (f1, f2, dt) tagged with the
    anchor's absolute time bin t1."""
    peaks = sorted(zip(time_idx, freq_idx))
    hashes = []
    n = len(peaks)
    for i in range(n):
        t1, f1 = peaks[i]
        count = 0
        for j in range(i + 1, n):
            t2, f2 = peaks[j]
            dt = t2 - t1
            if dt < min_time_bins:
                continue
            if dt > max_time_bins:
                break
            hashes.append(((int(f1), int(f2), int(dt)), int(t1)))
            count += 1
            if count >= fan_out:
                break
    return hashes


def generate_single_peak_hashes(freq_idx, time_idx):
    """Baseline alternative: treat each single peak's frequency as its own
    'hash', with no time-pairing. Used for the single-peak vs paired-hash
    comparison requested in the assignment."""
    return [((int(f),), int(t)) for f, t in zip(freq_idx, time_idx)]


# ---------------------------------------------------------------------
# Full fingerprinting convenience wrapper
# ---------------------------------------------------------------------
def fingerprint_audio(audio, sr, nperseg=1024, noverlap=768,
                       neighborhood=(20, 20), threshold_percentile=95,
                       fan_out=3, max_time_bins=100):
    f, t, Sxx_db = compute_spectrogram(audio, sr, nperseg, noverlap)
    freq_idx, time_idx = find_peaks_2d(Sxx_db, neighborhood, threshold_percentile)
    hashes = generate_hashes(freq_idx, time_idx, fan_out, max_time_bins)
    return {
        'f': f, 't': t, 'Sxx_db': Sxx_db,
        'freq_idx': freq_idx, 'time_idx': time_idx,
        'hashes': hashes
    }


# ---------------------------------------------------------------------
# Database build & matching
# ---------------------------------------------------------------------
def build_database(song_hashes_dict):
    """song_hashes_dict: {song_name: [(hash_key, t1), ...], ...}
    Returns: {hash_key: [(song_name, t1), ...], ...}"""
    db = defaultdict(list)
    for song_name, hashes in song_hashes_dict.items():
        for hash_key, t1 in hashes:
            db[hash_key].append((song_name, t1))
    return db


def match_query(query_hashes, db, top_k=5):
    """For each song that shares any hash with the query, build a histogram
    of (query_time - song_time) offsets. A true match produces one
    dominant, sharply-peaked offset; a wrong song gives scattered offsets."""
    offset_counts = defaultdict(Counter)
    for hash_key, t_query in query_hashes:
        if hash_key in db:
            for song_name, t_song in db[hash_key]:
                offset = t_query - t_song
                offset_counts[song_name][offset] += 1

    scores = {}
    best_offsets = {}
    for song_name, counter in offset_counts.items():
        best_offset, best_count = counter.most_common(1)[0]
        scores[song_name] = best_count
        best_offsets[song_name] = best_offset

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return ranked, offset_counts, best_offsets
