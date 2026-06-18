import os, time, pickle
import sys
sys.path.insert(0, '.')
from fingerprint import load_audio, fingerprint_audio, build_database

SONGS_DIR = 'songs'
OUT_DB = 'song_database.pkl'

song_files = sorted([f for f in os.listdir(SONGS_DIR) if f.lower().endswith('.mp3')])
print(f"Found {len(song_files)} songs to index.")

song_hashes_dict = {}
song_durations = {}
t_start = time.time()

for i, fname in enumerate(song_files):
    song_name = os.path.splitext(fname)[0]
    path = os.path.join(SONGS_DIR, fname)
    t0 = time.time()
    audio, sr = load_audio(path, sr=11025)
    fp = fingerprint_audio(audio, sr)
    song_hashes_dict[song_name] = fp['hashes']
    song_durations[song_name] = len(audio) / sr
    t1 = time.time()
    print(f"[{i+1:2d}/{len(song_files)}] {song_name:45s} "
          f"dur={song_durations[song_name]:6.1f}s  "
          f"peaks={len(fp['freq_idx']):5d}  hashes={len(fp['hashes']):6d}  "
          f"({t1-t0:.2f}s)")

t_end = time.time()
print(f"\nTotal indexing time: {t_end-t_start:.1f}s for {len(song_files)} songs")

db = build_database(song_hashes_dict)
print(f"Database has {len(db)} unique hash buckets")

with open(OUT_DB, 'wb') as f:
    pickle.dump({
        'db': db,
        'song_hashes_dict': song_hashes_dict,
        'song_durations': song_durations,
        'song_list': song_files,
    }, f)

print(f"Saved database to {OUT_DB}")
