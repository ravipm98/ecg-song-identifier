"""
EE200 Course Project — Q3B: Signals to Softwares ("Zapptain America")
Streamlit app wrapping the Shazam-style audio fingerprinting identifier
built in Q3A.

Two modes (selectable in the sidebar):
  1. Single-clip mode — upload one query clip, see the spectrogram,
     constellation of peaks, offset histogram, and the identified song.
  2. Batch mode — upload multiple query clips, get a results.csv with
     columns: filename, prediction (matched song's filename, no extension).

Run locally with:  streamlit run app.py
"""

import io
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from fingerprint import (
    load_audio,
    fingerprint_audio,
    match_query,
)

st.set_page_config(page_title="Sonic Signatures — Song Identifier", layout="wide")

DB_PATH = os.path.join(os.path.dirname(__file__), "song_database.pkl")


@st.cache_resource
def load_database():
    with open(DB_PATH, "rb") as f:
        data = pickle.load(f)
    return data


def save_uploaded_to_tempfile(uploaded_file):
    """Streamlit's UploadedFile has no real path on disk; write it to a temp
    file so ffmpeg (used inside fingerprint.load_audio) can read it."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp3"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def plot_spectrogram(fp):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.pcolormesh(fp["t"], fp["f"], fp["Sxx_db"], shading="gouraud", cmap="magma")
    ax.set_ylim(0, 4000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram")
    fig.tight_layout()
    return fig


def plot_constellation(fp):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.pcolormesh(fp["t"], fp["f"], fp["Sxx_db"], shading="gouraud", cmap="gray_r", alpha=0.5)
    ax.scatter(
        fp["t"][fp["time_idx"]], fp["f"][fp["freq_idx"]],
        s=14, c="red", marker="o", edgecolors="black", linewidths=0.3,
    )
    ax.set_ylim(0, 4000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Constellation map ({len(fp['freq_idx'])} peaks)")
    fig.tight_layout()
    return fig


def plot_offset_histogram(offset_counts, best_offsets, top_songs):
    fig, axes = plt.subplots(1, min(3, len(top_songs)), figsize=(13, 4), squeeze=False)
    axes = axes[0]
    for ax, song in zip(axes, top_songs):
        counter = offset_counts[song]
        best_off = best_offsets[song]
        window = 60
        offs = np.arange(best_off - window, best_off + window)
        counts = [counter.get(o, 0) for o in offs]
        ax.bar(offs, counts, width=1.0)
        ax.set_title(f"{song}\npeak={counter[best_off]}", fontsize=9)
        ax.set_xlabel("Offset (time bins)")
        ax.set_ylabel("Count")
    fig.suptitle("Offset histograms — top candidate songs")
    fig.tight_layout()
    return fig


st.title("🎵 Sonic Signatures — Audio Fingerprint Identifier")
st.caption(
    "A Shazam-style identifier: spectrogram → sparse constellation of peaks → "
    "paired (f1,f2,Δt) hashes → offset-histogram matching against an indexed song database."
)

data = load_database()
db = data["db"]
song_list = [os.path.splitext(s)[0] for s in data["song_list"]]

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Single-clip mode", "Batch mode"])
    st.markdown("---")
    st.write(f"**Indexed songs:** {len(song_list)}")
    with st.expander("Show song list"):
        for s in song_list:
            st.write("- " + s)

if mode == "Single-clip mode":
    st.subheader("Single-clip identification")
    uploaded = st.file_uploader(
        "Upload a query clip (mp3/wav/m4a)", type=["mp3", "wav", "m4a", "ogg"]
    )

    if uploaded is not None:
        tmp_path = None
        ranked = []
        try:
            with st.spinner("Loading and fingerprinting audio..."):
                tmp_path = save_uploaded_to_tempfile(uploaded)
                st.write(f"Debug: saved to {tmp_path}, size={os.path.getsize(tmp_path)} bytes")
                audio, sr = load_audio(tmp_path, sr=11025)
                st.write(f"Debug: decoded audio shape={audio.shape}, sr={sr}")
                fp = fingerprint_audio(audio, sr)
                st.write(f"Debug: found {len(fp['freq_idx'])} peaks, {len(fp['hashes'])} hashes")
                ranked, offset_counts, best_offsets = match_query(fp["hashes"], db, top_k=5)
                st.write(f"Debug: top match = {ranked[0] if ranked else 'none'}")
        except Exception as e:
            import traceback
            st.error(f"Failed to process this file: {e}")
            with st.expander("Full error details (debug)"):
                st.code(traceback.format_exc())
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if ranked and ranked[0][1] > 0:
            best_song, best_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else 0
            margin = best_score / max(second_score, 1)

            st.success(f"🎯 **Identified song: {best_song}**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Top match score", best_score)
            col2.metric("Runner-up score", second_score)
            col3.metric("Decisiveness margin", f"{margin:.1f}×")

            st.markdown("##### Top candidates")
            st.table(pd.DataFrame(ranked, columns=["Song", "Score"]))

            st.markdown("##### Intermediate steps")
            tab1, tab2, tab3 = st.tabs(["Spectrogram", "Constellation", "Offset histogram"])
            with tab1:
                st.pyplot(plot_spectrogram(fp))
            with tab2:
                st.pyplot(plot_constellation(fp))
            with tab3:
                top_songs = [s for s, _ in ranked[:3]]
                st.pyplot(plot_offset_histogram(offset_counts, best_offsets, top_songs))

else:
    st.subheader("Batch identification")
    st.write(
        "Upload multiple query clips. Each will be identified and written to "
        "`results.csv` with columns **filename, prediction** (matched song's "
        "filename without extension)."
    )
    uploaded_files = st.file_uploader(
        "Upload query clips (mp3/wav/m4a)",
        type=["mp3", "wav", "m4a", "ogg"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button(f"Run batch identification on {len(uploaded_files)} clips"):
            rows = []
            progress = st.progress(0.0)
            status = st.empty()

            for i, uploaded in enumerate(uploaded_files):
                status.write(f"Processing: {uploaded.name}")
                tmp_path = save_uploaded_to_tempfile(uploaded)
                try:
                    audio, sr = load_audio(tmp_path, sr=11025)
                    fp = fingerprint_audio(audio, sr)
                    ranked, _, _ = match_query(fp["hashes"], db, top_k=1)
                    prediction = ranked[0][0] if ranked and ranked[0][1] > 0 else "NO_MATCH"
                finally:
                    os.unlink(tmp_path)

                rows.append({"filename": uploaded.name, "prediction": prediction})
                progress.progress((i + 1) / len(uploaded_files))

            status.write("Done.")
            results_df = pd.DataFrame(rows, columns=["filename", "prediction"])
            st.dataframe(results_df, use_container_width=True)

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results.csv",
                data=csv_bytes,
                file_name="results.csv",
                mime="text/csv",
            )
