import pandas as pd
import numpy as np


def load_pose_data(pose_file_path: str) -> pd.DataFrame:
    """
    Load and process pose data from a feather file.
    Computes velocity, acceleration, and jerk based on translation along x-axis (tx_m).
    Timestamps are handled in nanoseconds and converted to seconds for precision.
    """
    df = pd.read_feather(pose_file_path)

    if "timestamp_ns" not in df.columns or "tx_m" not in df.columns:
        raise ValueError("Missing required columns in pose data.")

    # Convert timestamp to seconds for physics calculations
    df["timestamp_ns"] = df["timestamp_ns"].astype("int64")
    df["timestamp_s"] = df["timestamp_ns"].astype("float64") * 1e-9

    # Optional: Min-max scaled timestamp (0 to 1) for plotting or ML inputs
    ts_min = df["timestamp_s"].min()
    ts_max = df["timestamp_s"].max()
    df["timestamp_scaled"] = (df["timestamp_s"] - ts_min) / (ts_max - ts_min)

    # Compute time differences in seconds
    df["dt"] = df["timestamp_s"].diff()
    df["dt"] = df["dt"].fillna(1e-6)

    # Ensure tx_m is float for diff
    df["tx_m"] = df["tx_m"].astype("float64")

    # Compute velocity, acceleration, and jerk
    df["velocity"] = df["tx_m"].diff() / df["dt"]
    df["acceleration"] = df["velocity"].diff() / df["dt"]
    df["jerk"] = df["acceleration"].diff() / df["dt"]

    # Replace NaNs with 0 for clean usage
    df["velocity"] = df["velocity"].fillna(0)
    df["acceleration"] = df["acceleration"].fillna(0)
    df["jerk"] = df["jerk"].fillna(0)

    return df
