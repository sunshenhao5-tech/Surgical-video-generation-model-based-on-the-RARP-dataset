# Surgical-video-generation-model-based-on-the-RARP-dataset

<video width="640" height="360" controls><source src="https://github.com/samho2gyb/Surgical-video-generation-model-based-on-the-RARP-dataset/raw/main/demo/demo.mp4" type="video/mp4">您的浏览器不支持 video 标签，请点击下方链接观看：https://github.com/samho2gyb/Surgical-video-generation-model-based-on-the-RARP-dataset/raw/main/demo/demo.mp4</video>

Surgical-video-generation-model-based-on-the-RARP-dataset 是一个用于可控手术视频生成的解耦 RGBD-Flow 扩散模型。通过深度、光流和分割信息，实现精确的运动控制和视频合成。

## 功能特性

- **可控视频生成**：基于用户绘制的轨迹和运动刷子生成手术视频。
- **多模态融合**：结合 RGB、深度、光流和分割信息。
- **交互式界面**：使用 Gradio 提供直观的 Web 界面。
- **模块化设计**：支持不同组件的独立训练和推理。

## 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+
- 推荐 GPU：至少 8GB VRAM

### 安装依赖

```bash
pip install -r requirements.txt
```

### 下载权重

项目需要预训练权重，请下载并放置到相应目录：

- [Depth Anything V2](https://huggingface.co/IDEA-CCNL/Depth-Anything-V2) → `models/dav2/`
- [Segment Anything Model](https://github.com/facebookresearch/segment-anything) → `models/sam/`
- [SEA-RAFT](https://github.com/Sea-Raft/sea-raft) → `models/searaft/SEA-RAFT-main/models/`
- [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) → 指定路径

训练权重（如 `seg_encoder.pth`）需要通过训练脚本生成。

### 运行 Demo

```bash
python gradio_demo_run.py
```

打开浏览器访问显示的地址，使用界面上传图像、绘制轨迹并生成视频。

## 项目结构

```
RarpSora/
|-- gradio_demo_run.py          # Gradio Web 界面
|-- requirements.txt            # Python 依赖
|-- pipeline/                   # 推理管道
|-- Training/                   # 训练脚本和配置
|-- utils/                      # 工具函数
|-- models/                     # 模型定义和权重
|-- data/                       # 数据集（不上传）
|-- logs/                       # 训练日志和权重（不上传）
`-- README.md                   # 项目说明
```

## 数据集

项目使用手术动作视频数据集，包括：

- `knotting_videos`：结扎动作视频
- `needleGrasping_videos`：针抓取动作视频
- `needlePuncture_videos`：针穿刺动作视频
- `outputs-suturePulling`：缝合拉拽输出

数据集包含 RGB 图像序列、深度图、光流图和标注信息。详情见 `data/README.md`。

## 训练

### 第一阶段训练

```bash
python Training/train_stage1.py
```

生成 `seg_encoder.pth` 等基础权重。

### 第二阶段训练

```bash
python Training/train_stage2.py
```

使用第一阶段权重进行完整训练。

训练输出保存在 `logs/` 目录。详情见 `logs/README.md`。

## 使用说明

1. **准备数据**：下载数据集并放置到 `data/` 目录。
2. **下载权重**：按上述链接下载预训练权重。
3. **运行训练**：按顺序执行训练脚本。
4. **推理**：使用 `gradio_demo_run.py` 进行交互式生成。

## 致谢

本项目基于 SurgSora 开源代码开发，衷心感谢 SurgSora 社区为手术视频生成领域做出的贡献。SurgSora 提供了基础的扩散模型框架和训练 pipeline，本项目在此基础上进行了扩展和优化，包括添加 RGBD-Flow 解耦机制和可控生成功能。


