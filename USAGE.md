# Hypersim 相机内外参与 metric depth 使用说明

本文面向已下载好的 Hypersim 场景数据，说明如何在不重新渲染的前提下，直接从场景目录读取：
- 每张图片的相机外参（位姿）
- 相机内参（投影模型/矩阵）
- metric depth（到相机光心的欧氏距离，单位米）

并提供可运行的脚本与示例命令。

注意：本文遵循 Hypersim 的坐标/单位约定（见顶层 README “Coordinate conventions”“Camera trajectories”“Camera intrinsics”“Lossless HDR images”）。

---

## 一、数据布局与坐标/单位约定

以某一场景 `evermotion_dataset/scenes/ai_VVV_NNN` 为例：

- 相机轨迹与位姿（外参）：
  - `_detail/cam_XX/camera_keyframe_orientations.hdf5`：形状 N×3×3，记为 `R_wc[i]`，将相机坐标系向量（列向量）映射到世界坐标系（资产坐标）。
  - `_detail/cam_XX/camera_keyframe_positions.hdf5`：形状 N×3，记为世界坐标系中的相机中心 `C_w[i]`（资产坐标）。
  - `_detail/cam_XX/camera_keyframe_frame_indices.hdf5`（如存在）：将 keyframe 索引映射到图像帧号 `frame.IIII`。

- 图像（含几何/深度）：
  - `images/scene_cam_XX_geometry_hdf5/frame.IIII.depth_meters.hdf5`：单位米的欧氏距离（到光心）。
  - `images/scene_cam_XX_geometry_hdf5/frame.IIII.position.hdf5`：世界空间位置（资产坐标），可用于校验投影。
  - 其他几何通道（法线/实体ID/语义等）同目录。

- 场景级单位转换：
  - `_detail/metadata_scene.csv`：字段 `meters_per_asset_unit`。资产单位到米的比例：
    - 距离（资产单位） × meters_per_asset_unit = 距离（米）
    - 或者：距离（米） / meters_per_asset_unit = 距离（资产单位）

- 场景相机内参/投影模型（每场景一行）：
  - `contrib/mikeroberts3000/metadata_camera_parameters.csv`
    - `scene_name`、`settings_output_img_width`、`settings_output_img_height`
    - `M_proj_00..33`（4×4）：“修正的透视投影矩阵”，可替代标准 OpenGL perspective 使用，已考虑 tilt‑shift 等非标准参数。
    - `M_cam_from_uv_00..22`（3×3）：从归一化像素坐标 (u,v,1) 到相机空间方向的线性映射（常用于从像素反推相机射线）。

Hypersim 相机坐标系约定：x 右、y 上、z 沿相机视线方向“远离”相机（z>0 远离光心）。

---

## 二、外参（位姿）的读取与组装

已知每帧的
- `R_wc`：相机→世界的旋转（3×3）
- `C_w`：世界中的相机中心（3×1）

则世界→相机的旋转为：
- `R_cw = R_wc.T`

以常用的 3×4 外参矩阵（世界点到相机点）表示：
- 对任意世界点 `X_w`（3×1），先平移到相机中心：`X_w - C_w`
- 相机空间坐标：`X_cam = R_cw @ (X_w - C_w)`
- 若需 3×4 矩阵 `E = [R_cw | -R_cw @ C_w]`，则 `X_cam = E @ [X_w; 1]`

注意：所有位置/长度默认是“资产坐标/资产单位”。与以米为单位的深度联用时，须用 `meters_per_asset_unit` 做单位统一。

帧号匹配：
- 如存在 `_detail/cam_XX/camera_keyframe_frame_indices.hdf5`，可用其中的 `frame_indices[i] = IIII` 将 `R_wc[i]`、`C_w[i]` 与图像 `frame.IIII.*` 一一对应。
- 如该文件缺失，通常 `i` 与 `IIII` 同步（按生成流程），建议实际读取校验。

---

## 三、内参/投影模型的读取与使用

Hypersim 每个场景可能使用不同的相机参数（含可能的倾斜/平移等特殊设置）。官方提供了“修正的透视投影矩阵”，推荐直接使用。

- 从 `contrib/mikeroberts3000/metadata_camera_parameters.csv` 中取该场景的行（按 `scene_name` 过滤）。
- 获取输出分辨率：`W = settings_output_img_width`、`H = settings_output_img_height`
- 组装 4×4 投影矩阵 `M_proj`（按列主序或行主序需与代码一致，下文示例按行主序填入）。

两种常见用途：
1) 世界点 → 像素投影（正向）
   - 先用外参得到 `X_cam = R_cw @ (X_w - C_w)`
   - 齐次裁剪空间：`x_clip = M_proj @ [X_cam; 1]`
   - NDC：`x_ndc = x_clip[:3] / x_clip[3]`（范围约 [-1,1]）
   - 像素：`u = (x_ndc[0]*0.5 + 0.5) * W`
             `v = (1 - (x_ndc[1]*0.5 + 0.5)) * H`（图像原点在左上，y 轴向下）
   - 可用 `position.hdf5` 的世界点做投影验证。

2) 像素 → 相机射线（逆向）
   - 将像素归一化到 uv（如 NDC 或自定义归一化），再用 `M_cam_from_uv` 得到相机空间方向 `d_cam`。
   - 常用于把 metric depth 转为平面深度（见下一节）或做点云回投。

备注：
- `M_proj` 与 `M_cam_from_uv` 已包含场景的特殊内参/畸变等修正，避免手动拆解为 (fx, fy, cx, cy)。确需经典 K 时，可根据该矩阵反解或参考 `contrib/mikeroberts3000` 示例代码。

---

## 四、metric depth 的读取与单位转换

- 路径：`images/scene_cam_XX_geometry_hdf5/frame.IIII.depth_meters.hdf5`
- 含义：3D物点到相机光心的欧氏距离（单位米），不是相机空间的平面深度 z。
- 与外参/位置等（资产坐标/单位）联合计算时，需单位统一：
  - `d_asset = depth_meters / meters_per_asset_unit`

将欧氏距离转为“平面深度 z_cam”（相机空间）的方法之一：
- 取该像素的相机空间单位方向向量 `v_cam`（用 `M_cam_from_uv` 从像素获得）。
- 因 `depth_meters` 是到光心的欧氏距离 `d_meter`，则在资产单位下的欧氏距离为 `d_asset`。
- Hypersim 相机 z 轴沿视线远离相机，因此平面深度为：
  - `z_cam = d_asset * v_cam[2]`
- 若需传统“前向为 -z”的习惯，可取 `z_planar = -z_cam`。

更严格的推导与代码片段可参考 README 指向的 issue（Simon Niklaus 的示例）。

---

## 五、脚本使用（scripts/hypersim_camera_depth.py）

依赖：`h5py、numpy、pandas`。确保在仓库根目录 `ml-hypersim` 下运行。

- 获取某帧外参（R_cw、C_w、E）：
```
python scripts/hypersim_camera_depth.py get-extrinsics --scene ai_001_001 --cam cam_00 --frame 0000
```

- 获取场景内参/投影（M_proj、M_cam_from_uv、分辨率）：
```
python scripts/hypersim_camera_depth.py get-intrinsics --scene ai_001_001
```

- 将欧氏深度（米）转换为平面深度 z_cam（资产单位），并保存为 NPZ：
```
python scripts/hypersim_camera_depth.py convert-depth --scene ai_001_001 --cam cam_00 --frame 0000 --out evermotion_dataset/scenes/ai_001_001/images/scene_cam_00_geometry_hdf5/frame.0000.z_cam_asset.npz
```

- 作为库函数使用（示例）：
```python
from scripts.hypersim_camera_depth import (
    scene_dir, load_extrinsics, load_intrinsics_projection,
    load_metric_depth, load_scene_scale, depth_euclidean_to_planar_z_vectorized
)

scene_name = "ai_001_001"
cam_name   = "cam_00"
frame_iiii = "0000"
sdir = scene_dir(scene_name)

R_cw, C_w, E = load_extrinsics(sdir, cam_name, frame_iiii)
M_proj, (W,H), M_cam_from_uv = load_intrinsics_projection(scene_name)
depth_m = load_metric_depth(sdir, cam_name, frame_iiii)
m_per_asset = load_scene_scale(sdir)

z_cam_asset = depth_euclidean_to_planar_z_vectorized(depth_m, m_per_asset, W, H, M_cam_from_uv)
```

---

## 六、常见注意事项与建议

- 单位统一：
  - 相机位置/世界点/ bounding boxes 等均为资产坐标/单位；metric depth 为米。
  - 在任何涉及两者的计算中，务必使用 `meters_per_asset_unit` 做单位转换。

- z 轴约定：
  - Hypersim 的相机 z 轴“朝远离相机”的方向（z>0）。传统 CV 常将“前向”为 -z。若需要传统约定，请使用 `z_planar = -z_cam`。

- 帧索引对齐：
  - 优先使用 `camera_keyframe_frame_indices.hdf5` 做映射；没有时再假定 i 与 IIII 同步。

- 投影矩阵与畸变：
  - 推荐直接使用 `contrib/mikeroberts3000` 提供的 `M_proj` 和示例代码，而非手动构造 K，因为部分场景使用了 tilt‑shift 等非标准参数。

- 速度与准确性：
  - 深度转换建议使用向量化/并行化实现（脚本已提供向量化版本）。
  - 若只需将欧氏深度用于点云回投，可直接用 `d_asset * v_cam` 得到相机空间的 3D 点。

- 参考资料：
  - README “Camera intrinsics”“Camera trajectories”“Lossless high-dynamic range images”
  - Simon Niklaus 的欧氏深度→平面深度示例（README 链接的 issue #9）
  - `contrib/mikeroberts3000` 的 Python/Jupyter 示例

---
