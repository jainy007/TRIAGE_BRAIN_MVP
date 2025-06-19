import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

DATASET_ROOT = Path("/mnt/db/av_dataset/")
SENSOR_DIR = DATASET_ROOT / "sensor" / "train"
POSE_FILES_LIST = DATASET_ROOT / "feather_files.txt"
PWD = os.getcwd()
NUM_WORKERS = 16

def collect_feather_ids():
    if not POSE_FILES_LIST.exists():
        return set()
    with open(POSE_FILES_LIST, "r") as f:
        paths = [line.strip() for line in f if line.strip().endswith(".feather")]
    feather_ids = {Path(p).parts[-2] for p in paths}
    return feather_ids

def scan_scene(scene_path):
    scene_id = scene_path.name
    result = {
        "scene_id": scene_id,
        "path": str(scene_path),
        "has_pose": False,
        "has_metadata": False,
        "lidar_sweeps": 0,
        "radar_sweeps": 0,
        "camera_views": [],
        "camera_frames": {},
    }
    # Global feather map from txt file
    pose_map = scan_scene.feather_path_map
    pose_path = pose_map.get(scene_id)
    result["has_pose"] = pose_path is not None
    result["pose_path"] = pose_path if pose_path else ""




    # Metadata check: look for at least one .json in /state/
    state_dir = scene_path / "state"
    if state_dir.exists():
        result["has_metadata"] = any(state_dir.glob("*.json"))

    # Lidar: count feather, bin, or pcd sweeps
    lidar_dir = scene_path / "sensors" / "lidar"
    if lidar_dir.exists():
        result["lidar_sweeps"] = len(list(lidar_dir.glob("*.feather"))) + \
                                 len(list(lidar_dir.glob("*.pcd"))) + \
                                 len(list(lidar_dir.glob("*.bin")))

    # Radar
    radar_dir = scene_path / "sensors" / "radar"
    if radar_dir.exists():
        result["radar_sweeps"] = len(list(radar_dir.glob("*.feather"))) + \
                                 len(list(radar_dir.glob("*.bin")))

    # Camera check — enhanced
    cam_root = scene_path / "sensors" / "cameras"
    if cam_root.exists():
        for cam_view in cam_root.glob("*"):
            if cam_view.is_dir():
                jpg_count = len(list(cam_view.glob("*.jpg")))
                if jpg_count > 0:
                    result["camera_views"].append(cam_view.name)
                    result["camera_frames"][cam_view.name] = jpg_count

    return result

def load_feather_map():
    if not POSE_FILES_LIST.exists():
        return {}
    with open(POSE_FILES_LIST, "r") as f:
        paths = [line.strip() for line in f if line.strip()]
    mapping = {}
    for path in paths:
        p = Path(path)
        parent = p.parent.name
        if any(k in p.name for k in ['egovehicle', 'pose', 'city_SE3']):
            mapping[parent] = path
    return mapping


def main():
    all_scenes = [p for p in SENSOR_DIR.iterdir() if p.is_dir()]
    print(f"[INFO] Found {len(all_scenes)} scenes to scan.")

    # Feather list as lookup
    scan_scene.feather_ids = collect_feather_ids()
    scan_scene.feather_path_map = load_feather_map()


    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(scan_scene, all_scenes), total=len(all_scenes)))

    # Write all 3 jsonls
    with open(os.path.join(PWD, "data_exploration", "scene_core_map.jsonl"), "w") as f_core, \
         open(os.path.join(PWD, "data_exploration", "scene_sensor_map.jsonl"), "w") as f_sensor, \
         open(os.path.join(PWD, "data_exploration", "scene_full_map.jsonl"), "w") as f_full:

        for entry in results:
            json.dump({
                "scene_id": entry["scene_id"],
                "has_pose": entry["has_pose"],
                "has_metadata": entry["has_metadata"]
            }, f_core)
            f_core.write("\n")

            json.dump({
                "scene_id": entry["scene_id"],
                "camera_views": entry["camera_views"],
                "camera_frames": entry["camera_frames"],
                "lidar_sweeps": entry["lidar_sweeps"],
                "radar_sweeps": entry["radar_sweeps"]
            }, f_sensor)

            f_sensor.write("\n")

            json.dump(entry, f_full)
            f_full.write("\n")

    # Optional summary
    df = pd.DataFrame(results)
    df["num_cameras"] = df["camera_views"].apply(len)
    df["num_frames_total"] = df["camera_frames"].apply(lambda d: sum(d.values()))

    df.to_csv(os.path.join(PWD, "data_exploration", "scene_summary.csv"), index=False)

    print(f"[DONE] Wrote 3 JSONLs and summary CSV with {len(results)} entries.")

if __name__ == "__main__":
    main()
