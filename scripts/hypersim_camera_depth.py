#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hypersim camera intrinsics/extrinsics and metric depth utilities.

This script extracts:
- Camera extrinsics (pose): R_cw, C_w, and 3x4 matrix E = [R_cw | -R_cw*C_w]
- Camera intrinsics/projection per scene: 4x4 M_proj, 3x3 M_cam_from_uv, image size (W,H)
- Convert metric depth (meters to camera optical center) to planar depth z_cam (asset units)

Directory assumptions (run from repo root: ml-hypersim):
- Scenes under: evermotion_dataset/scenes/ai_VVV_NNN
- Camera data under: _detail/cam_XX/{camera_keyframe_orientations.hdf5, camera_keyframe_positions.hdf5, camera_keyframe_frame_indices.hdf5?}
- Depth under: images/scene_cam_XX_geometry_hdf5/frame.IIII.depth_meters.hdf5
- Scene scale under: _detail/metadata_scene.csv (meters_per_asset_unit)
- Intrinsics CSV under: contrib/mikeroberts3000/metadata_camera_parameters.csv

Usage examples:
  # Extrinsics (single frame)
  python scripts/hypersim_camera_depth.py get-extrinsics --scene_path evermotion_dataset/scenes/ai_001_001 --cam cam_00 --frame 0000

  # Extrinsics (batch: all frames to JSONL)
  python scripts/hypersim_camera_depth.py get-extrinsics --scene_path evermotion_dataset/scenes/ai_001_001 --cam cam_00 --all --format jsonl --out evermotion_dataset/scenes/ai_001_001/_detail/cam_00/extrinsics.jsonl

  # Intrinsics (per scene, by name)
  python scripts/hypersim_camera_depth.py get-intrinsics --scene_name ai_001_001

  # Depth conversion (requires scene_name for intrinsics, scene_path for depth/scale)
  python scripts/hypersim_camera_depth.py convert-depth --scene_name ai_001_001 --scene_path evermotion_dataset/scenes/ai_001_001 --cam cam_00 --frame 0000 --out evermotion_dataset/scenes/ai_001_001/images/scene_cam_00_geometry_hdf5/frame.0000.z_cam_asset.npz

Notes:
- Coordinate conventions follow Hypersim README.
- Positions/lengths are in asset coordinates/units; depth_meters is in meters. Use meters_per_asset_unit to unify units.
"""

import os
import sys
import json
import h5py
import numpy as np
import pandas as pd
import argparse
import csv


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def scene_dir(scene_name: str) -> str:
    return os.path.join(REPO_ROOT, "evermotion_dataset", "scenes", scene_name)


def load_scene_scale(scene_dir_path: str) -> float:
    """Return meters_per_asset_unit from _detail/metadata_scene.csv"""
    meta_scene_csv = os.path.join(scene_dir_path, "_detail", "metadata_scene.csv")
    df = pd.read_csv(meta_scene_csv)
    m_per_asset = float(df["meters_per_asset_unit"].values[0])
    return m_per_asset


def load_extrinsics_arrays(scene_dir_path: str, cam_name: str):
    """
    Load arrays needed for extrinsics and optional frame index mapping.

    Returns:
    - R_wc_all: N x 3 x 3 (camera->world rotation)
    - C_w_all:  N x 3     (camera center in world/asset coords)
    - frame_indices: None or (N,) int array mapping keyframe i -> image frame index IIII
    """
    cam_dir = os.path.join(scene_dir_path, "_detail", cam_name)

    with h5py.File(
        os.path.join(cam_dir, "camera_keyframe_orientations.hdf5"), "r"
    ) as f:
        R_wc_all = np.array(f["dataset"])  # N x 3 x 3
    with h5py.File(os.path.join(cam_dir, "camera_keyframe_positions.hdf5"), "r") as f:
        C_w_all = np.array(f["dataset"])  # N x 3

    frame_map_path = os.path.join(cam_dir, "camera_keyframe_frame_indices.hdf5")
    if os.path.exists(frame_map_path):
        with h5py.File(frame_map_path, "r") as f:
            frame_indices = np.array(f["dataset"])  # N (int frame numbers)
    else:
        frame_indices = None

    return R_wc_all, C_w_all, frame_indices


def enumerate_frames(frame_indices: np.ndarray, N: int):
    """
    Enumerate all frames as list of tuples (i, IIII_str).
    If frame_indices is provided, IIII_str = zero-padded frame_indices[i].
    Otherwise, IIII_str = zero-padded i.
    """
    frames = []
    if frame_indices is None:
        for i in range(N):
            frames.append((i, str(i).zfill(4)))
    else:
        for i in range(N):
            frames.append((i, str(int(frame_indices[i])).zfill(4)))
    return frames


def compute_extrinsics(R_wc_all: np.ndarray, C_w_all: np.ndarray, i: int):
    """
    Compute R_cw, C_w, E for keyframe index i.
    - R_cw: 3x3 (world->camera rotation)
    - C_w:  3-vector (camera center in world/asset coords)
    - E_3x4: [R_cw | -R_cw @ C_w]
    """
    R_wc = R_wc_all[i]
    C_w = C_w_all[i]
    R_cw = R_wc.T
    E = np.hstack([R_cw, (-R_cw @ C_w.reshape(3, 1))])  # 3x4
    return R_cw, C_w, E


def load_extrinsics(scene_dir_path: str, cam_name: str, frame_iiii: str):
    """
    Return extrinsics for the given frame:
    - R_cw: 3x3 (world->camera rotation)
    - C_w:  3-vector (camera center in world/asset coords)
    - E_3x4: [R_cw | -R_cw @ C_w]
    """
    R_wc_all, C_w_all, frame_indices = load_extrinsics_arrays(scene_dir_path, cam_name)

    # Resolve frame IIII -> keyframe index i
    if frame_indices is not None:
        matches = np.where(frame_indices == int(frame_iiii))[0]
        if len(matches) == 0:
            raise RuntimeError(
                f"Frame {frame_iiii} not found in camera_keyframe_frame_indices.hdf5"
            )
        i = int(matches[0])
    else:
        # Fallback: assume i == IIII (use cautiously; verify for your data)
        i = int(frame_iiii)

    return compute_extrinsics(R_wc_all, C_w_all, i)


def load_intrinsics_projection(scene_name: str):
    """
    Load per-scene camera parameters from contrib/mikeroberts3000/metadata_camera_parameters.csv.
    Returns:
    - M_proj: 4x4 numpy array
    - (W, H): image width/height
    - M_cam_from_uv: 3x3 numpy array
    """
    csv_path = os.path.join(
        REPO_ROOT, "contrib", "mikeroberts3000", "metadata_camera_parameters.csv"
    )
    df = pd.read_csv(csv_path)
    rows = df[df["scene_name"] == scene_name]
    if rows.empty:
        raise RuntimeError(
            f"Scene '{scene_name}' not found in metadata_camera_parameters.csv"
        )
    row = rows.iloc[0]

    W = int(float(row["settings_output_img_width"]))
    H = int(float(row["settings_output_img_height"]))

    M_proj = np.array(
        [
            [row["M_proj_00"], row["M_proj_01"], row["M_proj_02"], row["M_proj_03"]],
            [row["M_proj_10"], row["M_proj_11"], row["M_proj_12"], row["M_proj_13"]],
            [row["M_proj_20"], row["M_proj_21"], row["M_proj_22"], row["M_proj_23"]],
            [row["M_proj_30"], row["M_proj_31"], row["M_proj_32"], row["M_proj_33"]],
        ],
        dtype=np.float64,
    )

    M_cam_from_uv = np.array(
        [
            [row["M_cam_from_uv_00"], row["M_cam_from_uv_01"], row["M_cam_from_uv_02"]],
            [row["M_cam_from_uv_10"], row["M_cam_from_uv_11"], row["M_cam_from_uv_12"]],
            [row["M_cam_from_uv_20"], row["M_cam_from_uv_21"], row["M_cam_from_uv_22"]],
        ],
        dtype=np.float64,
    )

    return M_proj, (W, H), M_cam_from_uv


def ndc_to_pixels(x_ndc: np.ndarray, W: int, H: int):
    """Hypersim convention: image origin at top-left, y increases downward."""
    u = (x_ndc[0] * 0.5 + 0.5) * W
    v = (1.0 - (x_ndc[1] * 0.5 + 0.5)) * H
    return u, v


def project_world_points(
    X_w: np.ndarray,
    R_cw: np.ndarray,
    C_w: np.ndarray,
    M_proj: np.ndarray,
    W: int,
    H: int,
):
    """Project world points (asset coords) to pixel coordinates."""
    X_cam = (R_cw @ (X_w.T - C_w.reshape(3, 1))).T  # (N,3)
    X_cam_h = np.hstack([X_cam, np.ones((X_cam.shape[0], 1))])  # (N,4)
    x_clip = (M_proj @ X_cam_h.T).T  # (N,4)
    x_ndc = x_clip[:, :3] / x_clip[:, [3]]
    uv = np.stack(
        [ndc_to_pixels(x_ndc[i], W, H) for i in range(x_ndc.shape[0])], axis=0
    )
    return uv  # (N,2)


def depth_path(scene_dir_path: str, cam_name: str, frame_iiii: str) -> str:
    return os.path.join(
        scene_dir_path,
        "images",
        f"scene_{cam_name}_geometry_hdf5",
        f"frame.{frame_iiii}.depth_meters.hdf5",
    )


def load_metric_depth(
    scene_dir_path: str, cam_name: str, frame_iiii: str
) -> np.ndarray:
    """Load metric depth (meters) as numpy array (H x W)."""
    path = depth_path(scene_dir_path, cam_name, frame_iiii)
    with h5py.File(path, "r") as f:
        depth_m = np.array(f["dataset"])  # H x W
    return depth_m


def pixel_grid_ndc(W: int, H: int):
    """Return NDC coordinate grids (x_ndc, y_ndc) with pixel centers."""
    u = (np.arange(W) + 0.5) / W  # [0,1]
    v = (np.arange(H) + 0.5) / H  # [0,1]
    x_ndc = u[None, :] * 2.0 - 1.0  # H x W after broadcast
    y_ndc = 1.0 - (v[:, None] * 2.0)
    return x_ndc, y_ndc


def depth_euclidean_to_planar_z_vectorized(
    depth_m: np.ndarray,
    meters_per_asset_unit: float,
    W: int,
    H: int,
    M_cam_from_uv: np.ndarray,
) -> np.ndarray:
    """
    Convert Euclidean distance (meters) to planar depth z_cam (asset units).
    z_cam = d_asset * v_cam_z, where v_cam is unit direction in camera space.
    Vectorized implementation.
    """
    if depth_m.shape != (H, W):
        raise ValueError(f"depth_m shape {depth_m.shape} does not match H,W ({H},{W})")

    d_asset = depth_m / meters_per_asset_unit  # H x W

    x_ndc, y_ndc = pixel_grid_ndc(W, H)  # H x W

    # Form uv1 vectors: [x_ndc, y_ndc, 1] for all pixels
    uv1 = np.stack([x_ndc, y_ndc, np.ones_like(x_ndc)], axis=0)  # 3 x H x W
    uv1 = uv1.reshape(3, -1)  # 3 x (H*W)

    d_cam = M_cam_from_uv @ uv1  # 3 x (H*W)
    # Normalize directions
    norms = np.linalg.norm(d_cam, axis=0, keepdims=True) + 1e-12
    d_cam_unit = d_cam / norms
    v_cam_z = d_cam_unit[2, :]  # (H*W,)

    z_cam = (d_asset.reshape(-1) * v_cam_z).reshape(H, W)
    return z_cam


def _format_extrinsics_records_jsonl(records):
    lines = []
    for rec in records:
        lines.append(json.dumps(rec, ensure_ascii=False))
    return "\n".join(lines)


def _format_extrinsics_records_json(records):
    return json.dumps(records, ensure_ascii=False, indent=2)


def _format_extrinsics_records_csv(records):
    """
    Flatten matrices for CSV:
    - R_cw_00..R_cw_22 (row-major)
    - C_w_0..C_w_2
    - E_00..E_23 (row-major 3x4)
    """
    header = [
        "scene_path",
        "cam",
        "frame",
    ]
    # R_cw
    header += [f"R_cw_{r}{c}" for r in range(3) for c in range(3)]
    # C_w
    header += [f"C_w_{k}" for k in range(3)]
    # E (3x4)
    header += [f"E_{r}{c}" for r in range(3) for c in range(4)]

    rows = []
    for rec in records:
        row = [rec["scene_path"], rec["cam"], rec["frame"]]
        R = np.array(rec["R_cw"], dtype=np.float64).reshape(3, 3)
        C = np.array(rec["C_w"], dtype=np.float64).reshape(
            3,
        )
        E = np.array(rec["E_3x4"], dtype=np.float64).reshape(3, 4)
        row += list(R.flatten())
        row += list(C.flatten())
        row += list(E.flatten())
        rows.append((header, row))
    return header, [row for _, row in rows]


def cmd_get_extrinsics(args):
    sdir = args.scene_path
    R_wc_all, C_w_all, frame_indices = load_extrinsics_arrays(sdir, args.cam)
    all_frames = enumerate_frames(frame_indices, R_wc_all.shape[0])

    # Selection
    selected = []
    if getattr(args, "frame", None) is not None:
        # Single frame IIII string
        target = str(args.frame)
        matches = [tpl for tpl in all_frames if tpl[1] == target]
        if not matches:
            raise RuntimeError(f"Requested frame {target} not found for {args.cam}")
        selected = matches
    elif getattr(args, "all", False):
        selected = all_frames
    elif getattr(args, "range", None) is not None:
        start, end = args.range
        s_i = int(start)
        e_i = int(end)
        selected = [tpl for tpl in all_frames if s_i <= int(tpl[1]) <= e_i]
        if not selected:
            raise RuntimeError(f"No frames in range [{start}, {end}] for {args.cam}")
    elif getattr(args, "list", None) is not None:
        wanted = set([x.strip() for x in str(args.list).split(",") if x.strip() != ""])
        selected = [tpl for tpl in all_frames if tpl[1] in wanted]
        missing = wanted.difference(set([tpl[1] for tpl in selected]))
        if missing:
            raise RuntimeError(
                f"Frames not found for {args.cam}: {sorted(list(missing))}"
            )
    else:
        # Shouldn't happen due to mutually exclusive group requirement
        raise RuntimeError(
            "No selection provided. Use --frame or --all or --range or --list."
        )

    # Build records
    records = []
    for i, IIII in selected:
        R_cw, C_w, E = compute_extrinsics(R_wc_all, C_w_all, i)
        rec = {
            "scene_path": sdir,
            "cam": args.cam,
            "frame": IIII,
            "R_cw": R_cw.tolist(),
            "C_w": C_w.tolist(),
            "E_3x4": E.tolist(),
        }
        records.append(rec)

    # Output formatting
    fmt = getattr(args, "format", "jsonl").lower()
    out_path = getattr(args, "out", None)

    if fmt == "jsonl":
        text = _format_extrinsics_records_jsonl(records)
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text + ("\n" if text else ""))
        else:
            print(text)
    elif fmt == "json":
        text = _format_extrinsics_records_json(records)
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            print(text)
    elif fmt == "csv":
        header, rows = _format_extrinsics_records_csv(records)
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
        else:
            # Write to stdout
            print(",".join(header))
            for row in rows:
                # Convert floats to repr
                print(",".join([str(x) for x in row]))
    else:
        raise ValueError(f"Unsupported --format '{fmt}'. Choose from jsonl,json,csv.")


def cmd_get_intrinsics(args):
    M_proj, (W, H), M_cam_from_uv = load_intrinsics_projection(args.scene_name)
    result = {
        "scene_name": args.scene_name,
        "image_size": {"width": W, "height": H},
        "M_proj_4x4": M_proj.tolist(),
        "M_cam_from_uv_3x3": M_cam_from_uv.tolist(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_convert_depth(args):
    sdir = args.scene_path
    depth_m = load_metric_depth(sdir, args.cam, args.frame)
    M_proj, (W, H), M_cam_from_uv = load_intrinsics_projection(args.scene_name)
    m_per_asset = load_scene_scale(sdir)

    z_cam_asset = depth_euclidean_to_planar_z_vectorized(
        depth_m, m_per_asset, W, H, M_cam_from_uv
    )

    out_path = (
        args.out
        if args.out
        else os.path.join(
            sdir,
            "images",
            f"scene_{args.cam}_geometry_hdf5",
            f"frame.{args.frame}.z_cam_asset.npz",
        )
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, z_cam_asset=z_cam_asset)

    # Basic stats for quick sanity check
    stats = {
        "scene_name": args.scene_name,
        "scene_path": args.scene_path,
        "cam": args.cam,
        "frame": args.frame,
        "output": out_path,
        "shape": list(z_cam_asset.shape),
        "min": float(np.min(z_cam_asset)),
        "max": float(np.max(z_cam_asset)),
        "mean": float(np.mean(z_cam_asset)),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def build_parser():
    p = argparse.ArgumentParser(description="Hypersim camera/depth utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Extrinsics: scene_path + selection (frame/all/range/list) + formatting
    p_ext = sub.add_parser(
        "get-extrinsics", help="Dump extrinsics (R_cw, C_w, E) for one or many frames"
    )
    p_ext.add_argument(
        "--scene_path",
        required=True,
        help="Filesystem path to scene, e.g., evermotion_dataset/scenes/ai_001_001",
    )
    p_ext.add_argument("--cam", required=True, help="Camera name, e.g., cam_00")

    g_sel = p_ext.add_mutually_exclusive_group(required=True)
    g_sel.add_argument("--frame", required=False, help="Frame index 'IIII', e.g., 0000")
    g_sel.add_argument(
        "--all", action="store_true", help="Dump all available frames for the camera"
    )
    g_sel.add_argument(
        "--range",
        nargs=2,
        metavar=("START", "END"),
        help="Closed interval of frame indices 'IIII', e.g., 0000 0123",
    )
    g_sel.add_argument(
        "--list",
        required=False,
        help="Comma-separated list of frame indices, e.g., 0000,0005,0010",
    )

    p_ext.add_argument(
        "--format",
        default="jsonl",
        choices=["jsonl", "json", "csv"],
        help="Output format (default: jsonl)",
    )
    p_ext.add_argument("--out", required=False, help="Output path (default: stdout)")
    p_ext.set_defaults(func=cmd_get_extrinsics)

    # Intrinsics: scene_name (string key into CSV)
    p_int = sub.add_parser(
        "get-intrinsics",
        help="Dump intrinsics (M_proj, M_cam_from_uv, image size) as JSON",
    )
    p_int.add_argument(
        "--scene_name", required=True, help="Scene name, e.g., ai_001_001"
    )
    p_int.set_defaults(func=cmd_get_intrinsics)

    # Depth conversion: requires both scene_name (intrinsics) and scene_path (depth/scale)
    p_conv = sub.add_parser(
        "convert-depth",
        help="Convert metric depth (meters) to planar depth z_cam (asset units)",
    )
    p_conv.add_argument(
        "--scene_name", required=True, help="Scene name, e.g., ai_001_001"
    )
    p_conv.add_argument(
        "--scene_path",
        required=True,
        help="Filesystem path to scene, e.g., evermotion_dataset/scenes/ai_001_001",
    )
    p_conv.add_argument("--cam", required=True, help="Camera name, e.g., cam_00")
    p_conv.add_argument("--frame", required=True, help="Frame index 'IIII', e.g., 0000")
    p_conv.add_argument(
        "--out", required=False, help="Output .npz path (default: scene path)"
    )
    p_conv.set_defaults(func=cmd_convert_depth)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
