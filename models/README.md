# models 目录说明

本目录 `models/` 包含 RarpSora 项目中模型定义、支持代码，以及与预训练权重相关的引用信息。

> 注意：大型预训练模型权重文件不建议直接提交到 GitHub。如果它们不在仓库中，应当从官方来源单独下载。

## 需要下载的模型文件及来源

- `models/dav2/depth_anything_v2_vitl.pth`
  - 说明：Depth Anything V2 ViTL 权重，用于深度估计/视觉任务。
  - 建议下载地址：https://huggingface.co/IDEA-CCNL/Depth-Anything-V2

- `models/sam/sam_vit_h_4b8939.pth`
  - 说明：Segment Anything Model (SAM) ViT-H 权重，用于图像分割和掩码生成。
  - 建议下载地址：
    - https://github.com/facebookresearch/segment-anything
    - https://huggingface.co/facebook/segment-anything

- `models/searaft/SEA-RAFT-main/`
  - 说明：SEA-RAFT 光流/运动估计代码库。该子目录包含代码，必要时可以从官方仓库获取对应预训练权重。
  - 建议来源：https://github.com/Sea-Raft/sea-raft

- `models/svd/Training/ckpts/`
  - 说明：SVD 相关训练检查点目录。大型检查点文件不应直接提交到 GitHub。

## 当前 `models/` 代码结构

```
models/
|-- Control_Backbone.py
|-- Control_Encoder.py
|-- controlnet_sdv.py
|-- softsplat.py
|-- cmp/
|   |-- losses.py
|   |-- experiments/
|   |-- models/
|   `-- utils/
|-- dav2/
|   |-- Depth-Anything-V2/
|   `-- depth_anything_v2_vitl.pth
|-- sam/
|   `-- sam_vit_h_4b8939.pth
|-- searaft/
|   `-- SEA-RAFT-main/
|-- svd/
|   `-- Training/
`-- __pycache__/  # Python 缓存，不上传
```

### 结构说明

- `Control_Backbone.py`、`Control_Encoder.py`、`controlnet_sdv.py`、`softsplat.py` 是核心模型源码文件。
- `cmp/` 包含比较模型的工具代码、实验代码和训练辅助模块。
- `dav2/` 包含 Depth Anything V2 相关目录以及一个权重文件。
- `sam/` 包含 SAM ViT-H 权重文件。
- `searaft/SEA-RAFT-main/` 包含 SEA-RAFT 代码库。
- `svd/Training/` 应用于 SVD 相关训练检查点，不建议直接上传大型文件。
- `__pycache__/` 是 Python 缓存目录，应忽略。

## GitHub 推荐处理方式

- 将代码文件和子目录纳入版本控制。
- 不要将大型预训练权重文件提交到 GitHub 仓库。
- 建议在 `.gitignore` 中添加如下内容：

```gitignore
__pycache__/
*.pth
*.pt
*.safetensors
models/dav2/*.pth
models/sam/*.pth
models/svd/Training/ckpts/
```

如果项目确实需要这些检查点文件，请在 README 中提供下载说明，并仅保留文件名或占位符在版本控制中。
