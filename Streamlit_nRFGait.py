import streamlit as st
import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import sqlite3
import os
import uuid
import json
from io import StringIO
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="IMU Gait Analysis - Revised",
    page_icon="🚶",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_COLS = [
    "module",
    "unix_time_ms",
    "timestamp_iso",
    "device_ms",
    "ax_g",
    "ay_g",
    "az_g",
    "gx_dps",
    "gy_dps",
    "gz_dps"
]

NUMERIC_COLS = [
    "unix_time_ms",
    "device_ms",
    "ax_g",
    "ay_g",
    "az_g",
    "gx_dps",
    "gy_dps",
    "gz_dps"
]

GAIT_TRIAL_TYPES = [
    "Gait trial 1",
    "Gait trial 2",
    "Gait trial 3"
]

CALIBRATION_TYPES = [
    "Disc calibration",
    "On-body static calibration"
]

TRIAL_TYPES = CALIBRATION_TYPES + GAIT_TRIAL_TYPES


# ============================================================
# HELPERS
# ============================================================

def generate_assessment_id() -> str:
    return (
        datetime.now().strftime("%Y%m%d_%H%M%S") +
        "_" +
        uuid.uuid4().hex[:6]
    )


def new_assessment_id():
    st.session_state.assessment_id = generate_assessment_id()


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"

    try:
        if pd.isna(value):
            return "-"

        return f"{float(value):.{digits}f}"

    except Exception:
        return str(value)


# ============================================================
# DATA LOADING
# ============================================================

def load_imu_dataframe(file) -> pd.DataFrame:
    """
    Loads IMU CSV data.

    Supports:
    - CSV with header
    - CSV without header
    - Export issue where header and first row are concatenated
    """

    file.seek(0)
    raw = file.read()
    text = raw.decode("utf-8-sig", errors="ignore").strip()

    if not text:
        raise ValueError("The uploaded file is empty.")

    header = ",".join(EXPECTED_COLS)
    header_lower = header.lower()

    if text.lower().startswith(header_lower):
        idx = len(header)

        if idx < len(text) and text[idx] not in ("\n", "\r"):
            text = text[:idx] + "\n" + text[idx:]

    lines = text.splitlines()

    if len(lines) == 0:
        raise ValueError("No readable lines found in the file.")

    first_line = lines[0].lower()

    has_header = (
        "module" in first_line and
        "unix_time_ms" in first_line
    )

    if has_header:
        df = pd.read_csv(StringIO(text))
        df.columns = [str(c).strip().lower() for c in df.columns]
    else:
        df = pd.read_csv(StringIO(text), header=None, names=EXPECTED_COLS)

    if df.shape[1] < 10:
        raise ValueError(
            "Expected at least 10 columns: "
            "module, unix_time_ms, timestamp_iso, device_ms, "
            "ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps"
        )

    df = df.iloc[:, :10]
    df.columns = EXPECTED_COLS

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["unix_time_ms", "ax_g", "ay_g", "az_g"]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid IMU rows found after cleaning.")

    return df


# ============================================================
# SAMPLING RATE
# ============================================================

def estimate_sampling_rate(df: pd.DataFrame, override_hz: float = 0.0) -> float:
    if override_hz and override_hz > 0:
        return float(override_hz)

    if len(df) < 3:
        return 200.0

    diffs = np.diff(df["unix_time_ms"].astype(float))
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        return 200.0

    median_dt_ms = float(np.median(diffs))

    if median_dt_ms <= 0:
        return 200.0

    fs = 1000.0 / median_dt_ms

    if fs < 10 or fs > 1000:
        return 200.0

    return float(fs)


# ============================================================
# GAIT PROCESSING
# ============================================================

def process_gait(
    df: pd.DataFrame,
    cutoff_hz: float = 15.0,
    min_stride_s: float = 0.6,
    prominence_g: float = 0.20,
    fs_override_hz: float = 0.0
):
    df = df.copy()

    if df.empty:
        return df, {}, np.array([], dtype=int), 200.0

    df = df.sort_values("unix_time_ms").reset_index(drop=True)

    t0 = df["unix_time_ms"].iloc[0]
    df["time_s"] = (df["unix_time_ms"] - t0) / 1000.0

    fs = estimate_sampling_rate(df, override_hz=fs_override_hz)

    df["accel_res_g"] = np.sqrt(
        df["ax_g"] ** 2 +
        df["ay_g"] ** 2 +
        df["az_g"] ** 2
    )

    df["gyro_res_dps"] = np.sqrt(
        df["gx_dps"] ** 2 +
        df["gy_dps"] ** 2 +
        df["gz_dps"] ** 2
    )

    if len(df) > 30:
        nyquist = fs / 2.0
        wn = cutoff_hz / nyquist
        wn = min(max(wn, 0.01), 0.99)

        b, a = signal.butter(4, wn, btype="low")

        try:
            accel_filt = signal.filtfilt(b, a, df["accel_res_g"].values)
        except Exception:
            accel_filt = df["accel_res_g"].values
    else:
        accel_filt = df["accel_res_g"].values

    df["accel_res_filt_g"] = accel_filt

    peaks = np.array([], dtype=int)

    if len(df) > 10:
        distance_samples = max(1, int(min_stride_s * fs))

        peaks, _ = signal.find_peaks(
            df["accel_res_filt_g"],
            distance=distance_samples,
            prominence=prominence_g
        )

    heel_strike_times = df["time_s"].values[peaks]
    stride_times = np.diff(heel_strike_times)

    duration_s = float(df["time_s"].iloc[-1]) if len(df) else 0.0

    if stride_times.size > 0:
        mean_stride_time_s = float(np.mean(stride_times))
        median_stride_time_s = float(np.median(stride_times))
        min_stride_time_s = float(np.min(stride_times))
        max_stride_time_s = float(np.max(stride_times))
        cadence_steps_per_min = float(120.0 / mean_stride_time_s)
    else:
        mean_stride_time_s = None
        median_stride_time_s = None
        min_stride_time_s = None
        max_stride_time_s = None
        cadence_steps_per_min = None

    if stride_times.size > 1:
        sd_stride_time_s = float(np.std(stride_times, ddof=1))

        if mean_stride_time_s and mean_stride_time_s > 0:
            stride_time_cv_pct = float(
                100.0 * sd_stride_time_s / mean_stride_time_s
            )
        else:
            stride_time_cv_pct = None
    else:
        sd_stride_time_s = None
        stride_time_cv_pct = None

    metrics = {
        "record_type": "gait",
        "duration_s": duration_s,
        "estimated_fs_hz": float(fs),
        "n_samples": int(len(df)),

        "heel_strikes_detected": int(len(peaks)),
        "strides_detected": int(max(0, len(peaks) - 1)),

        "mean_stride_time_s": mean_stride_time_s,
        "median_stride_time_s": median_stride_time_s,
        "min_stride_time_s": min_stride_time_s,
        "max_stride_time_s": max_stride_time_s,
        "sd_stride_time_s": sd_stride_time_s,
        "stride_time_cv_pct": stride_time_cv_pct,
        "cadence_steps_per_min": cadence_steps_per_min,

        "filter_cutoff_hz": float(cutoff_hz),
        "min_stride_s": float(min_stride_s),
        "peak_prominence_g": float(prominence_g)
    }

    return df, metrics, peaks, fs


# ============================================================
# CALIBRATION PROCESSING
# ============================================================

def process_calibration(
    df: pd.DataFrame,
    calibration_type: str,
    fs_override_hz: float = 0.0
):
    df = df.copy()

    if df.empty:
        return df, {}

    df = df.sort_values("unix_time_ms").reset_index(drop=True)

    t0 = df["unix_time_ms"].iloc[0]
    df["time_s"] = (df["unix_time_ms"] - t0) / 1000.0

    fs = estimate_sampling_rate(df, override_hz=fs_override_hz)

    df["accel_res_g"] = np.sqrt(
        df["ax_g"] ** 2 +
        df["ay_g"] ** 2 +
        df["az_g"] ** 2
    )

    df["gyro_res_dps"] = np.sqrt(
        df["gx_dps"] ** 2 +
        df["gy_dps"] ** 2 +
        df["gz_dps"] ** 2
    )

    duration_s = float(df["time_s"].iloc[-1]) if len(df) else 0.0

    mean_ax = float(df["ax_g"].mean())
    mean_ay = float(df["ay_g"].mean())
    mean_az = float(df["az_g"].mean())

    std_ax = float(df["ax_g"].std(ddof=1)) if len(df) > 1 else 0.0
    std_ay = float(df["ay_g"].std(ddof=1)) if len(df) > 1 else 0.0
    std_az = float(df["az_g"].std(ddof=1)) if len(df) > 1 else 0.0

    mean_gx = float(df["gx_dps"].mean())
    mean_gy = float(df["gy_dps"].mean())
    mean_gz = float(df["gz_dps"].mean())

    std_gx = float(df["gx_dps"].std(ddof=1)) if len(df) > 1 else 0.0
    std_gy = float(df["gy_dps"].std(ddof=1)) if len(df) > 1 else 0.0
    std_gz = float(df["gz_dps"].std(ddof=1)) if len(df) > 1 else 0.0

    accel_res_mean = float(df["accel_res_g"].mean())
    accel_res_std = float(df["accel_res_g"].std(ddof=1)) if len(df) > 1 else 0.0

    gyro_res_mean = float(df["gyro_res_dps"].mean())
    gyro_res_std = float(df["gyro_res_dps"].std(ddof=1)) if len(df) > 1 else 0.0

    accel_vec = np.array([mean_ax, mean_ay, mean_az], dtype=float)
    accel_norm = float(np.linalg.norm(accel_vec))

    if accel_norm > 0:
        gravity_vector = (accel_vec / accel_norm).tolist()
    else:
        gravity_vector = [None, None, None]

    static_quality = "INFO"

    if calibration_type == "On-body static calibration":
        if 0.90 <= accel_res_mean <= 1.10 and gyro_res_mean <= 15.0:
            static_quality = "PASS"
        else:
            static_quality = "CHECK"

    calibration_json = {
        "calibration_type": calibration_type,
        "accel_mean_g": [mean_ax, mean_ay, mean_az],
        "accel_std_g": [std_ax, std_ay, std_az],
        "gyro_mean_dps": [mean_gx, mean_gy, mean_gz],
        "gyro_std_dps": [std_gx, std_gy, std_gz],
        "gravity_vector_sensor": gravity_vector,
        "static_quality": static_quality
    }

    metrics = {
        "record_type": "calibration",
        "calibration_type": calibration_type,

        "duration_s": duration_s,
        "estimated_fs_hz": float(fs),
        "n_samples": int(len(df)),

        "mean_ax_g": mean_ax,
        "mean_ay_g": mean_ay,
        "mean_az_g": mean_az,

        "std_ax_g": std_ax,
        "std_ay_g": std_ay,
        "std_az_g": std_az,

        "mean_gx_dps": mean_gx,
        "mean_gy_dps": mean_gy,
        "mean_gz_dps": mean_gz,

        "std_gx_dps": std_gx,
        "std_gy_dps": std_gy,
        "std_gz_dps": std_gz,

        "accel_res_mean_g": accel_res_mean,
        "accel_res_std_g": accel_res_std,

        "gyro_res_mean_dps": gyro_res_mean,
        "gyro_res_std_dps": gyro_res_std,

        "static_quality": static_quality,

        "calibration_json": json.dumps(calibration_json)
    }

    return df, metrics


# ============================================================
# SPATIAL METRICS FROM WALKWAY DISTANCE
# ============================================================

def add_spatial_metrics(
    metrics: dict,
    total_distance_m: float,
    walking_duration_override_s: float
):
    if not total_distance_m or total_distance_m <= 0:
        return metrics

    file_duration_s = metrics.get("duration_s")

    if walking_duration_override_s and walking_duration_override_s > 0:
        effective_duration_s = float(walking_duration_override_s)
    else:
        effective_duration_s = float(file_duration_s) if file_duration_s else None

    if not effective_duration_s or effective_duration_s <= 0:
        return metrics

    speed = float(total_distance_m / effective_duration_s)

    metrics["effective_walking_duration_s"] = effective_duration_s
    metrics["walking_speed_m_s"] = speed

    cadence = metrics.get("cadence_steps_per_min")

    if cadence and cadence > 0:
        step_freq_hz = cadence / 60.0
        stride_freq_hz = cadence / 120.0

        if step_freq_hz > 0:
            metrics["estimated_step_length_m"] = speed / step_freq_hz

        if stride_freq_hz > 0:
            metrics["estimated_stride_length_m"] = speed / stride_freq_hz

    return metrics


# ============================================================
# EVENTS
# ============================================================

def make_events_df(
    df: pd.DataFrame,
    peaks: np.ndarray,
    metadata: dict
) -> pd.DataFrame:
    if len(peaks) == 0:
        return pd.DataFrame(
            columns=["uhid", "event_type", "time_s", "unix_time_ms"]
        )

    events = pd.DataFrame({
        "uhid": metadata.get("UHID", ""),
        "event_type": "heel_strike",
        "time_s": df["time_s"].iloc[peaks].values,
        "unix_time_ms": df["unix_time_ms"].iloc[peaks].values
    })

    return events


# ============================================================
# STORAGE
# ============================================================

def save_raw_file(storage_dir: str, session_id: str, uploaded_file) -> str:
    raw_dir = os.path.join(storage_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    safe_name = os.path.basename(uploaded_file.name).replace(" ", "_")
    dest_path = os.path.join(raw_dir, f"{session_id}_{safe_name}")

    uploaded_file.seek(0)

    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    return dest_path


def save_analysis(
    storage_mode: str,
    storage_dir: str,
    metadata: dict,
    metrics: dict,
    events_df: pd.DataFrame,
    raw_file=None
):
    os.makedirs(storage_dir, exist_ok=True)

    session_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S") +
        "_" +
        uuid.uuid4().hex[:6]
    )

    raw_file_path = None

    if raw_file is not None:
        raw_file_path = save_raw_file(storage_dir, session_id, raw_file)

    session_row = {
        "session_id": session_id,
        "saved_at": datetime.now().isoformat(),
        **metadata,
        **metrics,
        "raw_file_path": raw_file_path
    }

    if storage_mode == "SQLite":
        db_path = os.path.join(storage_dir, "gait_database.db")

        conn = sqlite3.connect(db_path)

        try:
            pd.DataFrame([session_row]).to_sql(
                "sessions",
                conn,
                if_exists="append",
                index=False
            )

            if not events_df.empty:
                ev = events_df.copy()
                ev.insert(0, "session_id", session_id)

                ev.to_sql(
                    "heel_strike_events",
                    conn,
                    if_exists="append",
                    index=False
                )

            conn.commit()

        finally:
            conn.close()

        return session_id, db_path

    else:
        sessions_path = os.path.join(storage_dir, "sessions.csv")
        events_path = os.path.join(storage_dir, "heel_strike_events.csv")

        sessions_header_needed = (
            not os.path.exists(sessions_path) or
            os.path.getsize(sessions_path) == 0
        )

        events_header_needed = (
            not os.path.exists(events_path) or
            os.path.getsize(events_path) == 0
        )

        pd.DataFrame([session_row]).to_csv(
            sessions_path,
            mode="a",
            header=sessions_header_needed,
            index=False
        )

        if not events_df.empty:
            ev = events_df.copy()
            ev.insert(0, "session_id", session_id)

            ev.to_csv(
                events_path,
                mode="a",
                header=events_header_needed,
                index=False
            )

        return session_id, sessions_path


def load_records(storage_mode: str, storage_dir: str) -> pd.DataFrame:
    if storage_mode == "SQLite":
        db_path = os.path.join(storage_dir, "gait_database.db")

        if not os.path.exists(db_path):
            return pd.DataFrame()

        conn = sqlite3.connect(db_path)

        try:
            df = pd.read_sql_query(
                "SELECT * FROM sessions ORDER BY saved_at DESC",
                conn
            )
        except Exception:
            df = pd.DataFrame()
        finally:
            conn.close()

        return df

    else:
        sessions_path = os.path.join(storage_dir, "sessions.csv")

        if not os.path.exists(sessions_path):
            return pd.DataFrame()

        try:
            return pd.read_csv(sessions_path)
        except Exception:
            return pd.DataFrame()


def load_events(
    storage_mode: str,
    storage_dir: str,
    session_id: str = None
) -> pd.DataFrame:
    empty = pd.DataFrame(
        columns=[
            "session_id",
            "uhid",
            "event_type",
            "time_s",
            "unix_time_ms"
        ]
    )

    if storage_mode == "SQLite":
        db_path = os.path.join(storage_dir, "gait_database.db")

        if not os.path.exists(db_path):
            return empty

        conn = sqlite3.connect(db_path)

        try:
            if session_id:
                df = pd.read_sql_query(
                    """
                    SELECT *
                    FROM heel_strike_events
                    WHERE session_id = ?
                    ORDER BY time_s
                    """,
                    conn,
                    params=(session_id,)
                )
            else:
                df = pd.read_sql_query(
                    """
                    SELECT *
                    FROM heel_strike_events
                    ORDER BY session_id, time_s
                    """,
                    conn
                )

        except Exception:
            df = empty
        finally:
            conn.close()

        return df

    else:
        events_path = os.path.join(storage_dir, "heel_strike_events.csv")

        if not os.path.exists(events_path):
            return empty

        try:
            df = pd.read_csv(events_path)
        except Exception:
            return empty

        if session_id is not None and "session_id" in df.columns:
            df = df[df["session_id"] == session_id]

        return df


# ============================================================
# PLOTTING
# ============================================================

def downsample_for_plot(df: pd.DataFrame, max_points: int = 20000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df

    step = len(df) // max_points + 1

    return df.iloc[::step].copy()


def plot_gait(df: pd.DataFrame, peaks: np.ndarray):
    fig, ax = plt.subplots(figsize=(12, 4))

    plot_df = downsample_for_plot(df)

    ax.plot(
        plot_df["time_s"],
        plot_df["accel_res_g"],
        alpha=0.35,
        label="Raw resultant acceleration"
    )

    ax.plot(
        plot_df["time_s"],
        plot_df["accel_res_filt_g"],
        label="Filtered resultant acceleration"
    )

    if len(peaks) > 0:
        ax.plot(
            df["time_s"].iloc[peaks],
            df["accel_res_filt_g"].iloc[peaks],
            "rx",
            markersize=8,
            label="Detected heel strikes"
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (g)")
    ax.set_title("Gait trial: resultant acceleration and heel strikes")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    return fig


def plot_calibration(df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        sharex=True
    )

    plot_df = downsample_for_plot(df)

    ax1.plot(
        plot_df["time_s"],
        plot_df["accel_res_g"],
        label="Resultant acceleration"
    )

    ax1.axhline(
        1.0,
        color="red",
        linestyle="--",
        alpha=0.6,
        label="1 g reference"
    )

    ax1.set_ylabel("Acceleration (g)")
    ax1.set_title("Calibration / static check")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(
        plot_df["time_s"],
        plot_df["gyro_res_dps"],
        color="tab:orange",
        label="Resultant gyroscope"
    )

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angular velocity (deg/s)")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()

    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Storage and processing settings")

storage_mode = st.sidebar.radio(
    "Storage backend",
    options=["SQLite", "CSV files"]
)

storage_dir = st.sidebar.text_input(
    "Storage directory",
    value="gait_storage"
)

st.sidebar.caption(
    "If you used an older version of this app, use a new storage folder "
    "or delete/rename the old database, because new fields have been added."
)

st.sidebar.subheader("Gait event detection")

fs_override_hz = st.sidebar.number_input(
    "Sampling rate override (Hz), 0 = auto-detect",
    min_value=0.0,
    max_value=1000.0,
    value=0.0,
    step=1.0
)

cutoff_hz = st.sidebar.slider(
    "Low-pass filter cutoff (Hz)",
    min_value=5.0,
    max_value=30.0,
    value=15.0,
    step=0.5
)

min_stride_s = st.sidebar.slider(
    "Minimum stride time (s)",
    min_value=0.4,
    max_value=2.0,
    value=0.6,
    step=0.05
)

prominence_g = st.sidebar.number_input(
    "Heel-strike peak prominence (g)",
    min_value=0.0,
    max_value=5.0,
    value=0.20,
    step=0.05,
    format="%.2f"
)


# ============================================================
# TABS
# ============================================================

tab_analyze, tab_records, tab_about = st.tabs(
    ["Analyze & Save", "Saved Records", "About"]
)


# ============================================================
# TAB 1: ANALYZE AND SAVE
# ============================================================

with tab_analyze:

    st.header("IMU Gait Analysis - Revised")

    st.write(
        "Upload disc calibration, on-body static calibration, "
        "or gait trial files. Use the same Assessment ID to link "
        "all files from the same patient visit."
    )

    if "assessment_id" not in st.session_state:
        st.session_state.assessment_id = generate_assessment_id()

    st.subheader("Assessment")

    col_id_1, col_id_2 = st.columns([3, 1])

    with col_id_2:
        st.button(
            "New Assessment ID",
            on_click=new_assessment_id
        )

    with col_id_1:
        assessment_id = st.text_input(
            "Assessment ID",
            key="assessment_id"
        )

    st.subheader("Upload IMU file")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv", "txt"]
    )

    df_raw = None

    if uploaded_file is not None:
        try:
            df_raw = load_imu_dataframe(uploaded_file)

            st.success(
                f"Loaded {len(df_raw)} valid rows from `{uploaded_file.name}`."
            )

            with st.expander("Preview raw data", expanded=False):
                st.dataframe(df_raw.head(100), use_container_width=True)

                c1, c2, c3 = st.columns(3)

                c1.write(f"Rows: {len(df_raw)}")
                c2.write(f"Module: {df_raw['module'].iloc[0] if len(df_raw) else '-'}")

                if len(df_raw):
                    c3.write(
                        f"Time: {df_raw['timestamp_iso'].iloc[0]} "
                        f"to {df_raw['timestamp_iso'].iloc[-1]}"
                    )

        except Exception as e:
            st.error(f"Could not load CSV file: {e}")

    st.subheader("Patient, trial and walkway details")

    with st.form("patient_trial_form"):

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            uhid = st.text_input("UHID *")
            name = st.text_input("Name")

        with col2:
            age = st.number_input(
                "Age (years)",
                min_value=0,
                max_value=120,
                value=30,
                step=1
            )

            sex = st.selectbox(
                "Sex",
                options=[
                    "Female",
                    "Male",
                    "Other",
                    "Not specified"
                ]
            )

        with col3:
            weight_kg = st.number_input(
                "Weight (kg)",
                min_value=0.0,
                max_value=500.0,
                value=70.0,
                step=0.1
            )

            height_cm = st.number_input(
                "Height (cm)",
                min_value=0.0,
                max_value=250.0,
                value=170.0,
                step=0.1
            )

        with col4:
            sensor_distance_cm = st.number_input(
                "Sensor distance from ground (cm)",
                min_value=0.0,
                max_value=250.0,
                value=12.0,
                step=0.1
            )

            diagnosis = st.text_area(
                "Clinical diagnosis",
                height=68
            )

        st.markdown("### Trial and walkway details")

        trial_col1, trial_col2 = st.columns(2)

        with trial_col1:
            trial_type = st.selectbox(
                "Data type",
                options=TRIAL_TYPES
            )

            walkway_length_m = st.number_input(
                "Walkway one-way distance (m)",
                min_value=0.0,
                max_value=500.0,
                value=0.0,
                step=0.1
            )

            number_of_passes = st.number_input(
                "Number of one-way passes",
                min_value=0,
                max_value=200,
                value=0,
                step=1
            )

            total_distance_override_m = st.number_input(
                "Total walked distance override (m), 0 = auto",
                min_value=0.0,
                max_value=2000.0,
                value=0.0,
                step=0.1
            )

        with trial_col2:
            walking_duration_override_s = st.number_input(
                "Walking duration override (s), 0 = use file duration",
                min_value=0.0,
                max_value=3600.0,
                value=0.0,
                step=0.5
            )

            include_turns = st.checkbox(
                "Include turns in walking duration",
                value=True
            )

            notes = st.text_area(
                "Notes",
                height=100
            )

        save_choice = st.checkbox(
            "Save processed record",
            value=True
        )

        save_raw_choice = st.checkbox(
            "Also save raw IMU CSV file",
            value=False
        )

        submitted = st.form_submit_button("Process & Save")

    if submitted:

        if df_raw is None:
            st.warning("Please upload an IMU CSV file first.")

        elif not uhid.strip():
            st.warning("UHID is required.")

        else:

            try:

                if not assessment_id.strip():
                    assessment_id = generate_assessment_id()

                if total_distance_override_m > 0:
                    total_distance_m = float(total_distance_override_m)
                else:
                    total_distance_m = float(walkway_length_m) * float(number_of_passes)

                is_gait = trial_type in GAIT_TRIAL_TYPES

                if is_gait:
                    df_proc, metrics, peaks, fs = process_gait(
                        df=df_raw,
                        cutoff_hz=cutoff_hz,
                        min_stride_s=min_stride_s,
                        prominence_g=prominence_g,
                        fs_override_hz=fs_override_hz
                    )

                    metrics = add_spatial_metrics(
                        metrics=metrics,
                        total_distance_m=total_distance_m,
                        walking_duration_override_s=walking_duration_override_s
                    )

                    events_df = None

                    metadata = {
                        "assessment_id": assessment_id.strip(),
                        "trial_type": trial_type,

                        "UHID": uhid.strip(),
                        "Name": name.strip(),
                        "Age": int(age),
                        "Sex": sex,
                        "Weight_kg": float(weight_kg),
                        "Height_cm": float(height_cm),
                        "Sensor_distance_from_ground_cm": float(sensor_distance_cm),
                        "Clinical_diagnosis": diagnosis.strip(),

                        "walkway_one_way_length_m": float(walkway_length_m),
                        "number_of_passes": int(number_of_passes),
                        "total_distance_m": float(total_distance_m),
                        "walking_duration_override_s": float(walking_duration_override_s),
                        "include_turns": bool(include_turns),

                        "notes": notes.strip(),
                        "source_file_name": uploaded_file.name
                    }

                    events_df = make_events_df(
                        df=df_proc,
                        peaks=peaks,
                        metadata=metadata
                    )

                else:
                    df_proc, metrics = process_calibration(
                        df=df_raw,
                        calibration_type=trial_type,
                        fs_override_hz=fs_override_hz
                    )

                    peaks = np.array([], dtype=int)

                    metadata = {
                        "assessment_id": assessment_id.strip(),
                        "trial_type": trial_type,

                        "UHID": uhid.strip(),
                        "Name": name.strip(),
                        "Age": int(age),
                        "Sex": sex,
                        "Weight_kg": float(weight_kg),
                        "Height_cm": float(height_cm),
                        "Sensor_distance_from_ground_cm": float(sensor_distance_cm),
                        "Clinical_diagnosis": diagnosis.strip(),

                        "walkway_one_way_length_m": float(walkway_length_m),
                        "number_of_passes": int(number_of_passes),
                        "total_distance_m": float(total_distance_m),
                        "walking_duration_override_s": float(walking_duration_override_s),
                        "include_turns": bool(include_turns),

                        "notes": notes.strip(),
                        "source_file_name": uploaded_file.name
                    }

                    events_df = pd.DataFrame()

                session_id = None
                save_location = None
                save_error = None

                if save_choice:
                    raw_file_to_save = uploaded_file if save_raw_choice else None

                    try:
                        session_id, save_location = save_analysis(
                            storage_mode=storage_mode,
                            storage_dir=storage_dir,
                            metadata=metadata,
                            metrics=metrics,
                            events_df=events_df,
                            raw_file=raw_file_to_save
                        )

                    except Exception as e:
                        save_error = str(e)

                st.session_state["analysis"] = {
                    "df": df_proc,
                    "metrics": metrics,
                    "peaks": peaks,
                    "events": events_df,
                    "metadata": metadata,
                    "session_id": session_id,
                    "save_location": save_location,
                    "saved": bool(save_choice and save_error is None),
                    "save_error": save_error
                }

            except Exception as e:
                st.error(f"Processing failed: {e}")

    # Display result
    if "analysis" in st.session_state:

        res = st.session_state["analysis"]

        st.subheader("Results")

        metadata = res.get("metadata", {})
        metrics = res.get("metrics", {})

        st.write(
            f"Assessment ID: `{metadata.get('assessment_id', '-')}`  \n"
            f"Trial type: `{metadata.get('trial_type', '-')}`"
        )

        if res.get("saved"):
            st.success(
                "Saved session "
                f"`{res.get('session_id')}` "
                f"using {storage_mode}. "
                f"Location: `{res.get('save_location')}`"
            )

        elif res.get("save_error"):
            st.error(
                "Processing completed, but saving failed: "
                f"{res['save_error']}"
            )

        else:
            st.info("Processed successfully, but record was not saved.")

        if metadata.get("trial_type") in GAIT_TRIAL_TYPES:

            metric_cols = st.columns(6)

            metric_cols[0].metric(
                "Heel strikes",
                metrics.get("heel_strikes_detected", 0)
            )

            metric_cols[1].metric(
                "Strides",
                metrics.get("strides_detected", 0)
            )

            metric_cols[2].metric(
                "Mean stride time (s)",
                fmt(metrics.get("mean_stride_time_s"))
            )

            metric_cols[3].metric(
                "Cadence (steps/min)",
                fmt(metrics.get("cadence_steps_per_min"), 1)
            )

            metric_cols[4].metric(
                "Stride CV (%)",
                fmt(metrics.get("stride_time_cv_pct"), 2)
            )

            metric_cols[5].metric(
                "Speed (m/s)",
                fmt(metrics.get("walking_speed_m_s"), 2)
            )

            if metrics.get("strides_detected", 0) < 3:
                st.warning(
                    "Fewer than 3 strides were detected. "
                    "Consider adjusting prominence or checking the file."
                )

            if (
                metadata.get("total_distance_m", 0) <= 0 and
                metrics.get("walking_speed_m_s") is None
            ):
                st.info(
                    "No walkway distance was entered, so speed and "
                    "stride length were not estimated."
                )

        else:

            metric_cols = st.columns(4)

            metric_cols[0].metric(
                "Duration (s)",
                fmt(metrics.get("duration_s"), 1)
            )

            metric_cols[1].metric(
                "Mean accel resultant (g)",
                fmt(metrics.get("accel_res_mean_g"), 3)
            )

            metric_cols[2].metric(
                "Mean gyro resultant (deg/s)",
                fmt(metrics.get("gyro_res_mean_dps"), 1)
            )

            metric_cols[3].metric(
                "Quality",
                metrics.get("static_quality", "-")
            )

        plot_tab, data_tab, events_tab = st.tabs(
            ["Plot", "Processed data", "Events"]
        )

        with plot_tab:
            if metadata.get("trial_type") in GAIT_TRIAL_TYPES:
                fig = plot_gait(res["df"], res["peaks"])
            else:
                fig = plot_calibration(res["df"])

            st.pyplot(fig)

        with data_tab:
            st.dataframe(res["df"].tail(200), use_container_width=True)

            processed_csv = res["df"].to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download processed signal CSV",
                data=processed_csv,
                file_name="processed_imu_gait.csv",
                mime="text/csv"
            )

        with events_tab:
            if res["events"] is None or res["events"].empty:
                st.info("No heel-strike events for this record.")
            else:
                st.dataframe(res["events"], use_container_width=True)

                events_csv = res["events"].to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download heel-strike events CSV",
                    data=events_csv,
                    file_name="heel_strike_events.csv",
                    mime="text/csv"
                )


# ============================================================
# TAB 2: SAVED RECORDS
# ============================================================

with tab_records:

    st.header("Saved Records")

    records = load_records(storage_mode, storage_dir)

    if records.empty:
        st.info(
            f"No saved records found for {storage_mode} in `{storage_dir}`."
        )

    else:

        search_text = st.text_input(
            "Search UHID, name, diagnosis, assessment ID, session ID"
        )

        if search_text:
            mask = (
                records.astype(str)
                .apply(
                    lambda col: col.str.contains(
                        search_text,
                        case=False,
                        na=False
                    )
                )
                .any(axis=1)
            )

            records_view = records[mask]

        else:
            records_view = records

        st.dataframe(records_view, use_container_width=True)

        records_csv = records_view.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download filtered records CSV",
            data=records_csv,
            file_name="gait_records_filtered.csv",
            mime="text/csv"
        )

        st.subheader("Assessment view")

        if "assessment_id" in records.columns:

            assessment_options = (
                records["assessment_id"]
                .dropna()
                .unique()
                .tolist()
            )

            selected_assessment = st.selectbox(
                "Select assessment ID",
                options=assessment_options
            )

            assessment_records = records[
                records["assessment_id"] == selected_assessment
            ]

            st.dataframe(assessment_records, use_container_width=True)

            gait_records = assessment_records[
                assessment_records["trial_type"].isin(GAIT_TRIAL_TYPES)
            ]

            if not gait_records.empty:

                st.markdown("#### Gait trial summary")

                summary_cols = [
                    "trial_type",
                    "heel_strikes_detected",
                    "strides_detected",
                    "mean_stride_time_s",
                    "cadence_steps_per_min",
                    "stride_time_cv_pct",
                    "walking_speed_m_s",
                    "estimated_step_length_m",
                    "estimated_stride_length_m"
                ]

                existing_summary_cols = [
                    c for c in summary_cols if c in gait_records.columns
                ]

                st.dataframe(
                    gait_records[existing_summary_cols],
                    use_container_width=True
                )

                numeric_summary_cols = [
                    "mean_stride_time_s",
                    "cadence_steps_per_min",
                    "stride_time_cv_pct",
                    "walking_speed_m_s",
                    "estimated_step_length_m",
                    "estimated_stride_length_m"
                ]

                existing_numeric_cols = [
                    c for c in numeric_summary_cols if c in gait_records.columns
                ]

                if existing_numeric_cols:
                    st.markdown("#### Average across gait trials")

                    summary_df = (
                        gait_records[existing_numeric_cols]
                        .mean(numeric_only=True)
                        .to_frame("mean")
                        .T
                    )

                    st.dataframe(summary_df, use_container_width=True)

            if "session_id" in assessment_records.columns:

                st.markdown("#### Heel-strike events for selected session")

                session_options = assessment_records["session_id"].tolist()

                selected_session = st.selectbox(
                    "Select session ID",
                    options=session_options
                )

                events = load_events(
                    storage_mode=storage_mode,
                    storage_dir=storage_dir,
                    session_id=selected_session
                )

                if events.empty:
                    st.info("No heel-strike events found for this session.")
                else:
                    st.dataframe(events, use_container_width=True)

                    events_csv = events.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download events for selected session",
                        data=events_csv,
                        file_name=f"events_{selected_session}.csv",
                        mime="text/csv"
                    )

        else:
            st.info("Loaded records do not contain an assessment_id column.")


# ============================================================
# TAB 3: ABOUT
# ============================================================

with tab_about:

    st.header("About this revised app")

    st.write(
        """
        This revised Streamlit app supports a structured gait-analysis workflow.

        ### Record types

        1. **Disc calibration**
        2. **On-body static calibration**
        3. **Gait trial 1**
        4. **Gait trial 2**
        5. **Gait trial 3**

        ### Assessment ID

        Use the same **Assessment ID** for all files belonging to one patient visit.

        Example workflow:

        ```text
        Assessment ID: 20260920_visit_001
        - Disc calibration file
        - On-body static calibration file
        - Gait trial 1 file
        - Gait trial 2 file
        - Gait trial 3 file
        ```

        ### Walkway distance

        For gait trials, enter either:

        - Walkway one-way distance and number of passes, or
        - Total walked distance override.

        The app can then estimate:

        ```text
        walking_speed_m_s
        estimated_step_length_m
        estimated_stride_length_m
        ```

        If the trial includes turns, speed and stride length are approximate unless
        you enter a walking duration override that excludes turns.

        ### Storage

        Records can be stored as:

        - SQLite database
        - CSV files

        Stored fields include:

        ```text
        assessment_id
        trial_type
        UHID
        Name
        Age
        Sex
        Weight_kg
        Height_cm
        Sensor_distance_from_ground_cm
        Clinical_diagnosis
        walkway_one_way_length_m
        number_of_passes
        total_distance_m
        walking_duration_override_s
        include_turns
        notes
        source_file_name
        ```

        ### Important note

        If you used an older version of this app, start with a new storage folder
        or delete/rename the old database. The new version adds additional columns.
        """
    )