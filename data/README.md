# data 目录说明

本目录 `data/` 存放 RarpSora 项目使用的数据集文件，主要是手术动作视频数据、标注信息和输出结果数据。


## 目录概览

```
data/
|-- knotting_videos/
|-- knotting_videos.zip
|-- needleGrasping_videos/
|-- needleGrasping_videos.zip
|-- needlePuncture_videos/
|-- needlePuncture_videos.zip
|-- outputs-suturePulling/
`-- suturePulling_videos.zip
```

## 各数据集简介

### knotting_videos
- 目录数量：118 个子任务文件夹
- 子目录示例：`SARRARP502022_knotting_100_video40`
- 每个样本包含：
  - `frames/`：RGB 图像序列（`00001.png`、`00002.png` 等）
  - `depth image/`：深度图像序列（PNG）
  - `flow image/`：光流图像序列（PNG）
  - `instance1/`、`instance2/`：实例级标注，包含 `keypoints.json`、`points_gripper.json`、`points_overall.json`、`points_shaft.json`、`points_wrist.json`
  - 可选 3D 点云文件：`l_gripper_points3d.ply`、`r_gripper_points3d.ply`、`shaft_points3d.ply`、`wrist_points3d.ply`

### needleGrasping_videos
- 目录数量：675 个子任务文件夹
- 子目录示例：`SARRARP502022_needleGrasping_0_video1`
- 每个样本通常包含：
  - `frames/`：RGB 图像序列
  - `depth image/`：深度图像序列
  - `flow image/`：光流图像序列
  - `instance1/`、`instance2/`：实例级标注，常见为 `points_overall.json`

### needlePuncture_videos
- 目录数量：863 个子任务文件夹
- 子目录示例：`SARRARP502022_needlePuncture_0_video1`
- 每个样本通常包含：
  - `frames/`：RGB 图像序列
  - `depth image/`：深度图像序列
  - `flow image/`：光流图像序列
  - `instance1/`、`instance2/`：实例级标注，包含关键点和点云信息
  - 可能包含 3D 点云文件：`l_gripper_points3d.ply`、`r_gripper_points3d.ply`、`shaft_points3d.ply`、`wrist_points3d.ply`

### outputs-suturePulling
- 目录数量：210 个子任务文件夹
- 子目录示例：`SARRARP502022_suturePulling_1017_video3`
- 每个样本通常包含：
  - `frames_v2/`：生成或后处理后的图像序列
  - `instance1/`、`instance2/`：输出结果相关的实例数据，例如掩码、分割结果等


## 常见子目录结构示例

```
SARRARP502022_knotting_100_video40/
|-- depth image/
|   |-- 00001.png
|   |-- 00002.png
|   `-- ...
|-- flow image/
|   |-- 00001.png
|   |-- 00002.png
|   `-- ...
|-- frames/
|   |-- 00001.png
|   |-- 00002.png
|   `-- ...
|-- instance1/
|   |-- keypoints.json
|   |-- points_gripper.json
|   |-- points_overall.json
|   |-- points_shaft.json
|   `-- points_wrist.json
|-- instance2/
|   |-- keypoints.json
|   |-- points_gripper.json
|   |-- points_overall.json
|   |-- points_shaft.json
|   `-- points_wrist.json
|-- l_gripper_points3d.ply
|-- r_gripper_points3d.ply
|-- shaft_points3d.ply
`-- wrist_points3d.ply
```

如果需要，我也可以继续帮你补充每个子数据集的具体下载方式或说明。