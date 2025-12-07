#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pickle

import gdown
import numpy as np
import streamlit as st
from nilearn.plotting import plot_stat_map
import matplotlib.pyplot as plt

# ======================= CONFIG =======================

# Root directory for the data
# Can be overridden with an environment variable, useful on Streamlit Cloud
DATA_ROOT_ENV = os.environ.get("DNN_DATASET_ROOT", "DFG_data/dnn_datasets")
DNN_DATASET_ROOT = Path(DATA_ROOT_ENV)
DNN_DATASET_ROOT.mkdir(parents=True, exist_ok=True)

# Google Drive folder that contains:
#   ridge_var_all_voxels_stimXcog.pkl
#   ridge_all_voxels_stimXcog.pkl
#   session_1_cog2bold.pkl
#   session_2_cog2bold.pkl
#   session_3_cog2bold.pkl
COG2BOLD_FOLDER_ID = "1rauLwY1OSwBu2KppNDJhofVSSobdKnat"

REQUIRED_FILENAMES = [
    "ridge_var_all_voxels_stimXcog.pkl",
    "ridge_all_voxels_stimXcog.pkl",
    "session_1_cog2bold.pkl",
    "session_2_cog2bold.pkl",
    "session_3_cog2bold.pkl",
]

RIDGE_MEAN_WEIGHTS_PATH = DNN_DATASET_ROOT / "ridge_all_voxels_stimXcog.pkl"
RIDGE_VAR_WEIGHTS_PATH = DNN_DATASET_ROOT / "ridge_var_all_voxels_stimXcog.pkl"

CHUNK_SIZE = 256
RESET_SESSIONS = [1, 2, 3]


# ======================= DATA DOWNLOAD FROM GOOGLE DRIVE =======================

@st.cache_data
def ensure_cog2bold_files():
    """
    Make sure all required .pkl files exist locally.
    If any are missing, download the whole folder from Google Drive.
    This is intended to run at most once per deployment.
    """
    missing = [f for f in REQUIRED_FILENAMES if not (DNN_DATASET_ROOT / f).exists()]

    if missing:
        st.info(
            "Downloading model and cog2BOLD files from Google Drive. "
            "This should happen only once per environment."
        )
        # Download all files from the folder into DNN_DATASET_ROOT
        gdown.download_folder(
            id=COG2BOLD_FOLDER_ID,
            output=str(DNN_DATASET_ROOT),
            quiet=False,
            use_cookies=False,
        )

    # Check again
    still_missing = [f for f in REQUIRED_FILENAMES if not (DNN_DATASET_ROOT / f).exists()]
    if still_missing:
        st.error(
            "Could not find or download the following required files:\n"
            + "\n".join(still_missing)
            + f"\n\nChecked in: {DNN_DATASET_ROOT}"
        )
        st.stop()

    # Return something trivial just so Streamlit can cache it
    return True


# ======================= CACHED LOADERS =======================

@st.cache_resource
def load_masker_and_cog_columns():
    """Load masker and column names. Cached to run only once."""
    pkl_candidates = sorted(DNN_DATASET_ROOT.glob("session_*_cog2bold.pkl"))
    if not pkl_candidates:
        st.error(f"No session_*_cog2bold.pkl found in {DNN_DATASET_ROOT}")
        st.stop()

    pkl_path = pkl_candidates[0]
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)

    meta = bundle["meta"]
    masker = meta["masker"]
    cog_columns_session1 = meta["cog_columns"]
    stim_labels = meta["stimulus_labels"]

    base_cog_columns = []
    seen = set()
    for col in cog_columns_session1:
        base = col.split("#")[0]
        if base not in seen:
            seen.add(base)
            base_cog_columns.append(base)

    return masker, base_cog_columns, stim_labels


@st.cache_resource
def load_models():
    """Load the heavy Ridge models. Cached to run only once."""
    if not RIDGE_MEAN_WEIGHTS_PATH.exists():
        st.error(f"Model file not found: {RIDGE_MEAN_WEIGHTS_PATH}")
        st.stop()
    if not RIDGE_VAR_WEIGHTS_PATH.exists():
        st.error(f"Model file not found: {RIDGE_VAR_WEIGHTS_PATH}")
        st.stop()

    with open(RIDGE_MEAN_WEIGHTS_PATH, "rb") as f:
        mean_dict = pickle.load(f)
    with open(RIDGE_VAR_WEIGHTS_PATH, "rb") as f:
        var_dict = pickle.load(f)

    return mean_dict, var_dict


@st.cache_data
def get_reset_statistics(n_cog_features, _target_sessions):
    """Computes global and per-session means."""
    session_files = sorted(DNN_DATASET_ROOT.glob("session_*_cog2bold.pkl"))

    total_sum_all = np.zeros(n_cog_features, dtype=np.float64)
    total_n_all = 0
    sums_per_sess = {s: np.zeros(n_cog_features, dtype=np.float64) for s in _target_sessions}
    n_per_sess = {s: 0 for s in _target_sessions}

    for p in session_files:
        with open(p, "rb") as f:
            d = pickle.load(f)

        meta = d["meta"]
        sess_id = int(meta["session"])
        paths = d["paths"]
        base_dir = p.parent

        cog_path = base_dir / paths["cognitive"]
        if not cog_path.exists():
            continue

        cog_mem = np.load(cog_path, mmap_mode="r")
        N_total = cog_mem.shape[0]
        for start in range(0, N_total, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, N_total)
            if end <= start:
                continue

            cog_raw = cog_mem[start:end, :].astype(np.float64, copy=False)
            total_sum_all += cog_raw.sum(axis=0)
            total_n_all += cog_raw.shape[0]
            if sess_id in _target_sessions:
                sums_per_sess[sess_id] += cog_raw.sum(axis=0)
                n_per_sess[sess_id] += cog_raw.shape[0]

    if total_n_all == 0:
        return np.zeros(n_cog_features), {s: np.zeros(n_cog_features) for s in _target_sessions}

    global_mean = total_sum_all / total_n_all
    session_means = {}
    for s in _target_sessions:
        if n_per_sess[s] > 0:
            session_means[s] = sums_per_sess[s] / n_per_sess[s]
        else:
            session_means[s] = global_mean.copy()

    return global_mean, session_means


# ======================= HELPERS =======================

def get_default_hrf_weights(n_trs):
    base = np.array([0.0, 0.15, 1.0, 0.8, 0.65, 0.35, 0.15, 0.05], dtype=np.float64)
    if n_trs <= len(base):
        w = base[:n_trs].copy()
    else:
        w = np.concatenate([base, np.full(n_trs - len(base), base[-1])])
    if w.sum() > 0:
        w /= w.sum()
    return w


def build_z(cog_raw, stim_label, cog_mean, cog_std, label2id, n_cog, n_stim):
    c_std = (cog_raw - cog_mean) / cog_std
    s_idx = label2id[stim_label]
    stim_onehot = np.zeros(n_stim, dtype=np.float64)
    stim_onehot[s_idx] = 1.0
    cog_blocks = np.zeros(n_stim * n_cog, dtype=np.float64)
    start = s_idx * n_cog
    cog_blocks[start:start + n_cog] = c_std
    return np.concatenate([cog_blocks, stim_onehot, [1.0]], axis=0)


def aggregate_over_tr(values_2d, mode, tr_index, hrf_weights, custom_weights_str):
    V, T = values_2d.shape
    if mode == "no_agg":
        idx = int(np.clip(tr_index, 0, T - 1))
        return values_2d[:, idx], f"TR={idx}"
    elif mode == "hrf":
        return values_2d @ hrf_weights, "HRF-weighted"
    elif mode == "custom":
        try:
            parts = [float(p) for p in custom_weights_str.split(",") if p.strip()]
            w = np.array(parts, dtype=np.float64)
            if len(w) < T:
                w = np.concatenate([w, np.zeros(T - len(w))])
            else:
                w = w[:T]
            if w.sum() > 0:
                w /= w.sum()
            return values_2d @ w, "Custom-weighted"
        except Exception:
            return np.zeros(V), "Error parsing weights"
    elif mode == "max":
        idx = np.abs(values_2d).argmax(axis=1)
        return values_2d[np.arange(V), idx], "Max over TRs"
    return values_2d[:, 0], "Unknown"


# ======================= MAIN UI =======================

def main():
    st.set_page_config(layout="wide", page_title="fMRI Explorer")

    # Make sure model and session files are present locally
    ensure_cog2bold_files()

    # 1. LOAD DATA
    with st.spinner("Loading models..."):
        masker, cog_labels, global_labels = load_masker_and_cog_columns()
        mean_dict, var_dict = load_models()

        # Unpack
        W = mean_dict["W"]
        W_var = var_dict["W_var"]
        n_cog = mean_dict["n_cog_features"]
        n_stim = mean_dict["n_stimuli"]
        n_trs = mean_dict["n_trs"]
        cog_mean = mean_dict["cog_mean"]
        cog_std = mean_dict["cog_std"]
        label2id = {lab: i for i, lab in enumerate(global_labels)}

        # Load Reset Stats
        global_stats, sess_stats = get_reset_statistics(n_cog, RESET_SESSIONS)

    # 2. INIT STATE
    if "cog_sliders" not in st.session_state:
        st.session_state.cog_sliders = global_stats.copy()
    if "vmin" not in st.session_state:
        st.session_state.vmin = -2.0
    if "vmax" not in st.session_state:
        st.session_state.vmax = 2.0

    st.title("fMRI Activation Viewer")

    # =========================================================
    # SPLIT SCREEN: LEFT (Settings) | RIGHT (Sliders)
    # =========================================================
    col_left, col_right = st.columns([1, 2])

    # ------------------ LEFT COLUMN (SECTION 1) ------------------
    with col_left:
        with st.expander("🛠️ 1. Settings", expanded=True):
            st.markdown("##### Analysis Mode")
            mode = st.radio("Mode", ["Single stimulus", "Contrast (t-like)"], label_visibility="collapsed")

            if mode == "Single stimulus":
                stim = st.selectbox("Stimulus", global_labels)
            else:
                stimA = st.selectbox("Stim A", global_labels, index=0)
                stimB = st.selectbox("Stim B", global_labels, index=1 if len(global_labels) > 1 else 0)

            st.markdown("---")
            st.markdown("##### Time Aggregation")
            agg_mode = st.selectbox("Method", ["hrf", "no_agg", "custom", "max"])

            custom_w_str = ""
            if agg_mode == "custom":
                custom_w_str = st.text_input("Weights", "0,0.2,1,0.6,0.3")

    # ------------------ RIGHT COLUMN (SECTION 2) ------------------
    with col_right:
        with st.expander("🎛️ 2. Cognitive Feature Sliders", expanded=True):

            # --- RESET TOOLBAR ---
            c_reset1, c_reset2 = st.columns([2, 1])
            with c_reset1:
                reset_choice = st.selectbox(
                    "Reset Source",
                    ["Global Mean", "Session 1", "Session 2", "Session 3"],
                    label_visibility="collapsed"
                )
            with c_reset2:
                if st.button("Reset Now", use_container_width=True):
                    if reset_choice == "Global Mean":
                        st.session_state.cog_sliders = global_stats.copy()
                    elif reset_choice == "Session 1":
                        st.session_state.cog_sliders = sess_stats[1].copy()
                    elif reset_choice == "Session 2":
                        st.session_state.cog_sliders = sess_stats[2].copy()
                    elif reset_choice == "Session 3":
                        st.session_state.cog_sliders = sess_stats[3].copy()
                    st.rerun()

            st.markdown("---")

            # --- SLIDERS ---
            grid_cols = st.columns(2)

            for i, name in enumerate(cog_labels):
                mu = global_stats[i]
                sigma = cog_std[i]
                if sigma <= 0:
                    low, high = mu - 1.0, mu + 1.0
                else:
                    low, high = mu - 3 * sigma, mu + 3 * sigma

                low = max(0, low)
                col_idx = i % 2
                with grid_cols[col_idx]:
                    val = st.slider(
                        f"{name}",
                        float(low), float(high),
                        float(st.session_state.cog_sliders[i]),
                        step=float(high - low) / 100.0,
                        key=f"slider_{i}"
                    )
                    st.session_state.cog_sliders[i] = val

    # =========================================================
    # SECTION 3: BRAIN MAP (BOTTOM - FULL WIDTH)
    # =========================================================
    st.markdown("---")
    st.subheader("🧠 3. Brain Activation Map")

    # --- DYNAMIC CONTROLS (TR SLIDER) ---
    tr_val = 0
    if agg_mode == "no_agg":
        st.info("Time Series Mode: Select TR to view")
        tr_val = st.slider("TR Index", 0, n_trs - 1, 0)

    # --- CALCULATION ---
    cog_raw_current = st.session_state.cog_sliders
    hrf_w = get_default_hrf_weights(n_trs)
    display_map = None
    title_str = ""

    if mode == "Single stimulus":
        z = build_z(cog_raw_current, stim, cog_mean, cog_std, label2id, n_cog, n_stim)
        Y_hat = np.tensordot(z, W, axes=(0, 0))
        display_map, agg_info = aggregate_over_tr(Y_hat, agg_mode, tr_val, hrf_w, custom_w_str)
        title_str = f"BOLD: {stim} ({agg_info})"
    else:
        zA = build_z(cog_raw_current, stimA, cog_mean, cog_std, label2id, n_cog, n_stim)
        zB = build_z(cog_raw_current, stimB, cog_mean, cog_std, label2id, n_cog, n_stim)
        muA = np.tensordot(zA, W, axes=(0, 0))
        muB = np.tensordot(zB, W, axes=(0, 0))
        varA = np.tensordot(zA, W_var, axes=(0, 0))
        varB = np.tensordot(zB, W_var, axes=(0, 0))
        se = np.sqrt(np.maximum(varA, 1e-8) + np.maximum(varB, 1e-8))
        display_map, agg_info = aggregate_over_tr((muA - muB) / se, agg_mode, tr_val, hrf_w, custom_w_str)
        title_str = f"Contrast: {stimA} vs {stimB} ({agg_info})"

    # --- VISUALIZATION CONTROLS ---
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        nslices = st.number_input("# Slices (0=Auto)", 0, 30, 7)
    with c2:
        vmin_input = st.number_input("vmin", value=st.session_state.vmin)
    with c3:
        vmax_input = st.number_input("vmax", value=st.session_state.vmax)
    with c4:
        st.write("")
        if st.button("Auto Scale (95%)"):
            finite_vals = display_map[np.isfinite(display_map)]
            if finite_vals.size > 0:
                vmax_auto = np.percentile(np.abs(finite_vals), 95)
                if vmax_auto <= 0:
                    vmax_auto = 1.0
                st.session_state.vmin = -vmax_auto
                st.session_state.vmax = vmax_auto
                st.rerun()

    st.session_state.vmin = vmin_input
    st.session_state.vmax = vmax_input

    # --- PLOTTING ---
    mask_img = masker.mask_img_
    img_to_plot = masker.inverse_transform(display_map.astype(np.float32))

    cut_coords = None
    if nslices > 0:
        affine = mask_img.affine
        z_bounds = (affine[2, 3], affine[2, 3] + affine[2, 2] * mask_img.shape[2])
        cut_coords = np.linspace(z_bounds[0], z_bounds[1], nslices + 2)[1:-1]

    fig = plt.figure(figsize=(12, 5))
    plot_stat_map(
        img_to_plot,
        bg_img=None,
        display_mode="z",
        cut_coords=cut_coords,
        cmap="cold_hot",
        vmin=st.session_state.vmin,
        vmax=st.session_state.vmax,
        title=title_str,
        figure=fig,
        symmetric_cbar=True,
    )
    st.pyplot(fig, clear_figure=True)


if __name__ == "__main__":
    main()
