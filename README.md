# Sonic Signatures — Audio Fingerprint Identifier (Q3B)

A Shazam-style audio identifier: spectrogram → sparse constellation of peaks
→ paired `(f1, f2, Δt)` hashes → offset-histogram matching against an
indexed database of 50 songs.

## Files

- `app.py` — the Streamlit app (single-clip mode + batch mode)
- `fingerprint.py` — core fingerprinting library (loading, spectrogram,
  peak-picking, hashing, database build/match)
- `song_database.pkl` — pre-built fingerprint database for all 50 provided
  songs (ships with the app so it works immediately on deploy — no need to
  re-index on startup)
- `build_database.py` — script used to (re)build `song_database.pkl` from a
  `songs/` folder of mp3 files, in case the song list changes
- `requirements.txt` / `packages.txt` — Python and system dependencies

## Running locally

```bash
pip install -r requirements.txt
# ffmpeg must also be installed and on PATH (apt install ffmpeg / brew install ffmpeg)
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Push this folder to a public GitHub repository (include `song_database.pkl`
   — it's ~50 MB, under GitHub's 100 MB file limit, so a normal `git add` /
   `git push` works without Git LFS).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and click
   "New app".
3. Point it at your repository, branch, and `app.py` as the entry point.
4. Streamlit Cloud automatically installs `requirements.txt` (Python
   packages) and `packages.txt` (system packages — this is what installs
   `ffmpeg` on the server, since audio decoding depends on it).
5. Deploy. The first load may take a minute while dependencies install; after
   that, `song_database.pkl` loads once (cached via `@st.cache_resource`) and
   the app is ready to use immediately, with no re-indexing needed.

## Usage

**Single-clip mode:** upload one query audio clip (mp3/wav/m4a/ogg). The app
displays the identified song, the score margin, and three tabs showing the
spectrogram, the constellation of peaks, and the offset histograms for the
top candidate songs.

**Batch mode:** upload multiple query clips at once. The app processes all of
them and produces a downloadable `results.csv` with exactly two columns,
`filename` and `prediction` (the matched song's filename without extension,
or `NO_MATCH` if no song scored above zero).

## Rebuilding the database with a different song set

Put your `.mp3` files in a `songs/` subfolder next to `build_database.py`,
then run:

```bash
python build_database.py
```

This regenerates `song_database.pkl`, which `app.py` will pick up automatically.
